// The offline evaluation-register CLI (ADR-010 O1, protocol I2 and I6). It is
// the logic behind `scripts/evaluation-register.ts`; a pure function of its
// arguments, the files they name and the clock, so tests call it directly.
// It never touches the network and never loads the engine. Exit 0 ok, 1 a
// refusal (a record, a chain or a published head is wrong), 2 usage.

import { appendFileSync, existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { alphaPrimePpm, EvaluationConfigError, isComplete, readEvaluationConfig, requireComplete, type EvaluationConfig } from "./evaluation-config.ts";
import { checkCommitment, checkPlan, commitmentHash, planHash, readEvaluationPlan, readSealedSetCommitment } from "./evaluation-plan.ts";
import { appendEntry, assertPlanPublished, checkHeads, type EntryDraft, readHead, RegisterError, verifyRegister } from "./evaluation-register.ts";
import { jcs, type Json } from "./canonical-json.ts";

export interface CliResult {
  code: 0 | 1 | 2;
  out: string;
  err: string;
}

export const USAGE = `usage: evaluation-register <command> [options]
  config-check   --parameters F --rule F [--complete]
  commit-set     --register F --commitment F --parameters F --rule F --operator NAME [--recorded-at T]
  register-plan  --register F --plan F --parameters F --rule F --operator NAME [--recorded-at T]
  append         --register F --entry F --operator NAME [--recorded-at T]
  verify         --register F [--heads DIR] [--published PLAN_HASH]
  head           --register F
  heads-check    --dir DIR
`;

class UsageError extends Error {}

type Options = Map<string, string | true>;

const FLAGS = new Set(["complete"]);

function parseOptions(args: readonly string[]): Options {
  const options: Options = new Map();
  for (let i = 0; i < args.length; i++) {
    const arg = args[i] as string;
    if (!arg.startsWith("--")) throw new UsageError(`unexpected argument ${arg}`);
    const name = arg.slice(2);
    if (FLAGS.has(name)) options.set(name, true);
    else {
      const value = args[++i];
      if (value === undefined) throw new UsageError(`--${name} needs a value`);
      options.set(name, value);
    }
  }
  return options;
}

const need = (options: Options, name: string): string => {
  const value = options.get(name);
  if (typeof value !== "string") throw new UsageError(`--${name} is required`);
  return value;
};

const readJson = (path: string): unknown => {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new EvaluationConfigError(`cannot read ${path}: ${(error as Error).message}`);
  }
};

const readRegister = (path: string): string => (existsSync(path) ? readFileSync(path, "utf8") : "");

const configOf = (options: Options): EvaluationConfig => readEvaluationConfig(readJson(need(options, "parameters")), readJson(need(options, "rule")));

const stamp = (options: Options, now: () => Date): string => {
  const given = options.get("recorded-at");
  return typeof given === "string" ? given : `${now().toISOString().slice(0, 19)}Z`;
};

const line = (value: unknown): string => `${jcs(value as Json)}\n`;

function readHeads(dir: string): unknown[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => {
      const value = readJson(join(dir, f));
      const seq = (value as { seq?: unknown } | null)?.seq;
      if (f !== `${String(seq).padStart(10, "0")}.json`) throw new RegisterError(`${f}: the file name must be the zero-padded position (${String(seq).padStart(10, "0")}.json)`);
      return value;
    });
}

function execute(command: string, options: Options, now: () => Date): string {
  switch (command) {
    case "config-check": {
      const config = configOf(options);
      if (options.has("complete")) requireComplete(config.parameters);
      return line({ parameters_hash: config.parameters_hash, rule_hash: config.rule_hash, alpha_prime_ppm: alphaPrimePpm(config.parameters), complete: isComplete(config.parameters) });
    }
    case "commit-set": {
      const config = configOf(options);
      const commitment = readSealedSetCommitment(readJson(need(options, "commitment")));
      checkCommitment(commitment, config);
      const register = need(options, "register");
      const done = appendEntry(readRegister(register), { kind: "set_commitment", recorded_at: stamp(options, now), operator: need(options, "operator"), commitment });
      appendFileSync(register, line(done.entry));
      return line({ commitment_hash: commitmentHash(commitment), head: done.head });
    }
    case "register-plan": {
      const config = configOf(options);
      const plan = readEvaluationPlan(readJson(need(options, "plan")));
      const register = need(options, "register");
      const text = readRegister(register);
      const set = verifyRegister(text).sets.get(plan.set_id);
      if (set === undefined) throw new RegisterError(`set ${plan.set_id} has no commitment in the register`);
      checkPlan(plan, { config, commitment: set.commitment });
      const done = appendEntry(text, { kind: "plan_registration", recorded_at: stamp(options, now), operator: need(options, "operator"), plan });
      appendFileSync(register, line(done.entry));
      return line({ plan_hash: planHash(plan), head: done.head });
    }
    case "append": {
      const register = need(options, "register");
      const body = readJson(need(options, "entry")) as Record<string, unknown>;
      if (body === null || typeof body !== "object" || Array.isArray(body)) throw new UsageError("--entry must hold a JSON object");
      if (body.kind === "set_commitment" || body.kind === "plan_registration") throw new UsageError(`use ${body.kind === "set_commitment" ? "commit-set" : "register-plan"}, which also checks the records`);
      const draft = { recorded_at: stamp(options, now), operator: need(options, "operator"), ...body } as unknown as EntryDraft;
      const done = appendEntry(readRegister(register), draft);
      appendFileSync(register, line(done.entry));
      return line({ head: done.head });
    }
    case "verify": {
      const state = verifyRegister(readRegister(need(options, "register")));
      const heads = options.has("heads") ? readHeads(need(options, "heads")) : null;
      const published = heads === null ? null : checkHeads(state, heads);
      const plan = options.get("published");
      if (typeof plan === "string") {
        if (published === null) throw new UsageError("--published needs --heads");
        assertPlanPublished(state, published, plan);
      }
      return line({ entries: state.entries.length, head: state.head, sets: state.sets.size, published_through: published?.published_through ?? null, unpublished: published?.unpublished ?? null });
    }
    case "head": {
      const state = verifyRegister(readRegister(need(options, "register")));
      if (state.head === null) throw new RegisterError("the register is empty: it has no head");
      return line(state.head);
    }
    case "heads-check": {
      const heads = readHeads(need(options, "dir")).map(readHead);
      return line({ heads: heads.length, latest: heads[heads.length - 1]?.seq ?? null });
    }
    default:
      throw new UsageError(`unknown command ${command}`);
  }
}

export function runEvaluationCli(argv: readonly string[], now: () => Date = () => new Date()): CliResult {
  const [command, ...rest] = argv;
  try {
    if (command === undefined) throw new UsageError("a command is required");
    return { code: 0, out: execute(command, parseOptions(rest), now), err: "" };
  } catch (error) {
    if (error instanceof UsageError) return { code: 2, out: "", err: `${error.message}\n${USAGE}` };
    if (error instanceof RegisterError || error instanceof EvaluationConfigError) return { code: 1, out: "", err: `${error.message}\n` };
    throw error;
  }
}
