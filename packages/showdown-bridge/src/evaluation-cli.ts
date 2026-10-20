// The offline evaluation-register CLI (ADR-010 O1, protocol I2 and I6). It is
// the logic behind `scripts/evaluation-register.ts`; a pure function of its
// arguments, the files they name and the clock, so tests call it directly.
// It never touches the network and never loads the engine. Exit 0 ok, 1 a
// refusal (a record, a chain or a published head is wrong), 2 usage.

import { execFileSync } from "node:child_process";
import { closeSync, existsSync, fsyncSync, openSync, readdirSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  alphaPrimePpm,
  EvaluationConfigError,
  isComplete,
  PARAMETERS_FILE,
  readEvaluationConfig,
  requireComplete,
  RULE_FILE,
  type EvaluationConfig,
} from "./evaluation-config.ts";
import { commitmentHash, generateSecret, keyCommitment, planHash, readEvaluationPlan, readSealedSetCommitment } from "./evaluation-plan.ts";
import { appendEntry, assertPlanPublished, checkHeads, checkHeadsAddOnly, type EntryDraft, readHead, RegisterError, verifyRegister } from "./evaluation-register.ts";
import { jcs, type Json } from "./canonical-json.ts";

export interface CliResult {
  code: 0 | 1 | 2;
  out: string;
  err: string;
}

export const USAGE = `usage: evaluation-register <command> [options]
  Records default to the committed config/evaluation v1 files; --parameters and
  --rule may be repeated (paired in order) when a register spans versions.
  Heads are read from the working tree (--heads DIR) or, for the trusted copy,
  from a git ref (--heads DIR --heads-ref origin/main).
  init           --register F                      (the only command that starts a register)
  config-check   --parameters F --rule F [--complete]
  commit-set     --register F --commitment F --operator NAME [--recorded-at T] [records]
  register-plan  --register F --plan F --operator NAME [--recorded-at T] [records]
  append         --register F --entry F --operator NAME [--recorded-at T] [records] [--heads DIR [--heads-ref REF]]
                 (l2a_recheck and run_outcome for a plan need the heads: the plan must be published)
  verify         --register F [--heads DIR [--heads-ref REF]] [--published PLAN_HASH] [records]
  head           --register F
  heads-check    --dir DIR [--base REF]            (with --base: add-only against that ref)
  gen-secret     --out F                           (256 random bits, mode 0600, never overwrites; prints its SHA-256)
`;

class UsageError extends Error {}

type Options = Map<string, string[]>;

const FLAGS = new Set(["complete"]);

function parseOptions(args: readonly string[]): Options {
  const options: Options = new Map();
  for (let i = 0; i < args.length; i++) {
    const arg = args[i] as string;
    if (!arg.startsWith("--")) throw new UsageError(`unexpected argument ${arg}`);
    const name = arg.slice(2);
    let value = "true";
    if (!FLAGS.has(name)) {
      const next = args[++i];
      if (next === undefined) throw new UsageError(`--${name} needs a value`);
      value = next;
    }
    options.set(name, [...(options.get(name) ?? []), value]);
  }
  return options;
}

const need = (options: Options, name: string): string => {
  const value = options.get(name)?.[0];
  if (value === undefined) throw new UsageError(`--${name} is required`);
  return value;
};

const readJson = (path: string): unknown => {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new EvaluationConfigError(`cannot read ${path}: ${(error as Error).message}`);
  }
};

/** A register that is not there is refused: a mistyped path must not start a new genesis chain. */
const readRegister = (path: string): string => {
  if (!existsSync(path)) throw new RegisterError(`${path} does not exist (a new register is started only with init)`);
  return readFileSync(path, "utf8");
};

/** The records to check against: every --parameters/--rule pair, else the committed v1 files. */
function configsOf(options: Options): EvaluationConfig[] {
  const parameters = options.get("parameters") ?? [PARAMETERS_FILE];
  const rules = options.get("rule") ?? [RULE_FILE];
  if (parameters.length !== rules.length) throw new UsageError("give as many --rule as --parameters");
  return parameters.map((file, i) => readEvaluationConfig(readJson(file), readJson(rules[i] as string)));
}

const stamp = (options: Options, now: () => Date): string => options.get("recorded-at")?.[0] ?? `${now().toISOString().slice(0, 19)}Z`;

const line = (value: unknown): string => `${jcs(value as Json)}\n`;

/** The new line is on disk before the command reports it: written, then fsynced. One writer at a time (the runbook says so). */
function appendDurably(path: string, text: string): void {
  const fd = openSync(path, "a");
  try {
    writeSync(fd, text);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

const HEAD_FILE = /\.json$/;

function headsInTree(dir: string): Map<string, string> {
  const files = new Map<string, string>();
  if (!existsSync(dir)) return files;
  for (const name of readdirSync(dir).filter((f) => HEAD_FILE.test(f)).sort()) files.set(name, readFileSync(join(dir, name), "utf8"));
  return files;
}

/** The head files as a git ref holds them: the copy a merge made, not whatever is in the working tree. */
function headsAtRef(ref: string, dir: string, cwd: string): Map<string, string> {
  const git = (...args: string[]): string => {
    try {
      return execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      throw new EvaluationConfigError(`git ${args[0]} ${ref} failed: ${((error as { stderr?: string }).stderr ?? "").trim()}`);
    }
  };
  let trimmed = dir;
  while (trimmed.endsWith("/")) trimmed = trimmed.slice(0, -1);
  const prefix = `${trimmed}/`;
  const files = new Map<string, string>();
  for (const path of git("ls-tree", "--name-only", ref, "--", prefix).split("\n").filter((p) => HEAD_FILE.test(p))) {
    files.set(path.slice(prefix.length), git("show", `${ref}:${path}`));
  }
  return files;
}

function headsOf(options: Options, cwd: string): unknown[] | null {
  const dir = options.get("heads")?.[0];
  if (dir === undefined) {
    if (options.has("heads-ref")) throw new UsageError("--heads-ref needs --heads (the directory)");
    return null;
  }
  const ref = options.get("heads-ref")?.[0];
  const files = ref === undefined ? headsInTree(resolve(cwd, dir)) : headsAtRef(ref, dir, cwd);
  checkHeadsAddOnly(new Map(), files);
  return [...files.values()].map((text) => JSON.parse(text) as unknown);
}

/** A run starts or is recorded only for a plan whose own entry a published head covers (I2). */
function requirePublished(options: Options, cwd: string, register: string, planHashValue: string): void {
  const heads = headsOf(options, cwd);
  if (heads === null) throw new UsageError("this entry records a run: give --heads (and --heads-ref origin/main) so the plan's publication can be checked");
  const state = verifyRegister(register, configsOf(options));
  assertPlanPublished(state, checkHeads(state, heads), planHashValue);
}

function execute(command: string, options: Options, now: () => Date, cwd: string): string {
  switch (command) {
    case "init": {
      const register = need(options, "register");
      if (existsSync(register)) throw new RegisterError(`${register} already exists`);
      writeFileSync(register, "", { flag: "wx", mode: 0o600 });
      return line({ register, entries: 0 });
    }
    case "config-check": {
      const [config] = configsOf(options) as [EvaluationConfig];
      if (options.has("complete")) requireComplete(config.parameters);
      return line({ parameters_hash: config.parameters_hash, rule_hash: config.rule_hash, alpha_prime_ppm: alphaPrimePpm(config.parameters), complete: isComplete(config.parameters) });
    }
    case "commit-set": {
      const configs = configsOf(options);
      const commitment = readSealedSetCommitment(readJson(need(options, "commitment")));
      const register = need(options, "register");
      const done = appendEntry(readRegister(register), { kind: "set_commitment", recorded_at: stamp(options, now), operator: need(options, "operator"), commitment }, configs);
      appendDurably(register, line(done.entry));
      return line({ commitment_hash: commitmentHash(commitment), head: done.head });
    }
    case "register-plan": {
      const configs = configsOf(options);
      const plan = readEvaluationPlan(readJson(need(options, "plan")));
      const register = need(options, "register");
      const text = readRegister(register);
      const set = verifyRegister(text, configs).sets.get(plan.set_id);
      if (set === undefined) throw new RegisterError(`set ${plan.set_id} has no commitment in the register`);
      const done = appendEntry(text, { kind: "plan_registration", recorded_at: stamp(options, now), operator: need(options, "operator"), plan }, configs);
      appendDurably(register, line(done.entry));
      return line({ plan_hash: planHash(plan), head: done.head });
    }
    case "append": {
      const register = need(options, "register");
      const body = readJson(need(options, "entry")) as Record<string, unknown>;
      if (body === null || typeof body !== "object" || Array.isArray(body)) throw new UsageError("--entry must hold a JSON object");
      if (body.kind === "set_commitment" || body.kind === "plan_registration") throw new UsageError(`use ${body.kind === "set_commitment" ? "commit-set" : "register-plan"}, which also checks the records`);
      const text = readRegister(register);
      if ((body.kind === "l2a_recheck" || body.kind === "run_outcome") && typeof body.plan_hash === "string") requirePublished(options, cwd, text, body.plan_hash);
      const draft = { recorded_at: stamp(options, now), operator: need(options, "operator"), ...body } as unknown as EntryDraft;
      const done = appendEntry(text, draft, configsOf(options));
      appendDurably(register, line(done.entry));
      return line({ head: done.head });
    }
    case "verify": {
      const state = verifyRegister(readRegister(need(options, "register")), configsOf(options));
      const heads = headsOf(options, cwd);
      const published = heads === null ? null : checkHeads(state, heads);
      const plan = options.get("published")?.[0];
      if (plan !== undefined) {
        if (published === null) throw new UsageError("--published needs --heads");
        assertPlanPublished(state, published, plan);
      }
      return line({ entries: state.entries.length, head: state.head, sets: state.sets.size, published_through: published?.published_through ?? null, unpublished: published?.unpublished ?? null });
    }
    case "head": {
      const state = verifyRegister(readRegister(need(options, "register")), configsOf(options));
      if (state.head === null) throw new RegisterError("the register is empty: it has no head");
      return line(state.head);
    }
    case "heads-check": {
      const dir = need(options, "dir");
      const current = headsInTree(resolve(cwd, dir));
      const base = options.get("base")?.[0];
      checkHeadsAddOnly(base === undefined ? new Map() : headsAtRef(base, dir, cwd), current);
      const heads = [...current.values()].map((text) => readHead(JSON.parse(text)));
      return line({ heads: heads.length, latest: heads.reduce<number | null>((top, h) => (top === null || h.seq > top ? h.seq : top), null) });
    }
    case "gen-secret": {
      const secret = generateSecret();
      try {
        writeFileSync(need(options, "out"), secret, { flag: "wx", mode: 0o600 });
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "EEXIST") throw new EvaluationConfigError(`${need(options, "out")} already exists: a secret is never overwritten`);
        throw error;
      }
      return line({ sha256: keyCommitment(secret) });
    }
    default:
      throw new UsageError(`unknown command ${command}`);
  }
}

export function runEvaluationCli(argv: readonly string[], now: () => Date = () => new Date(), cwd: string = process.cwd()): CliResult {
  const [command, ...rest] = argv;
  try {
    if (command === undefined) throw new UsageError("a command is required");
    return { code: 0, out: execute(command, parseOptions(rest), now, cwd), err: "" };
  } catch (error) {
    if (error instanceof UsageError) return { code: 2, out: "", err: `${error.message}\n${USAGE}` };
    if (error instanceof RegisterError || error instanceof EvaluationConfigError) return { code: 1, out: "", err: `${error.message}\n` };
    throw error;
  }
}
