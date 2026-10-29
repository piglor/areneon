// The command line of sealed generation, run as the store user inside the
// isolated evaluation store (docs/runbooks/evaluation-store.md section 9):
//
//   generate     --store DIR --set-id ID --generation N --public-manifest FILE
//                --parameters FILE --rule FILE --config FILE --labels FILE --format ID
//                [--plan-id ID] [--vocabulary FILE] [--known FILE]... [--max-attempts N]
//                [--generator-commit SHA] [--progress-every DRAWS]
//   reveal-hash  --store DIR --set-id ID
//   retire       --store DIR --set-id ID --out DIR --register FILE
//                --parameters FILE --rule FILE   (repeat the pair for a register that spans versions)
//                --heads DIR [--heads-ref REF]   (the published chain heads, checked before anything is written)
//   verify       --dir DIR --public-manifest FILE --register FILE --parameters FILE --rule FILE
//                --heads DIR [--heads-ref REF]   (the published chain heads; use --heads-ref origin/main)
//                --config FILE --labels FILE --format ID
//                [--vocabulary FILE] [--known FILE]...
//
// `--known` files are JSON Lines of teams (records or revealed sealed teams)
// that carry the engine's `paste`. No option takes a key or a salt, and no
// output carries one: a run prints one JSON line of hashes and counts, and
// counts on stderr. The key and the salt are made from the OS CSPRNG inside
// `generate` and written only below `--store`. Exit codes: 0 done, 1 refused,
// 2 a usage or input error.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import {
  assertRetirementPublished,
  checkHeads,
  commitmentHash,
  type Engine,
  exportEngineData,
  exportTeamVocabulary,
  loadEngine,
  readEvaluationConfig,
  readPublishedHeads,
  readTeamDatasetManifest,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  readTeamVocabulary,
  resolveFormat,
  sealedManifestHash,
  verifyRegister,
} from "@areneon/showdown-bridge";
import { generateSealedSet, type KnownTeam } from "./sealed.ts";
import { parseSecretJson } from "./sealed-store.ts";
import { readRevealDir, retireSealedSet, revealHash, verifyRetiredSealed } from "./sealed-retire.ts";

export interface SealedCliIo {
  out(line: string): void;
  err(line: string): void;
  /** The engine to use; the pinned one when omitted. */
  engine?: Engine;
  /** The 256-bit source; the OS CSPRNG when omitted. Tests only. */
  secret?: () => string;
}

class UsageError extends Error {
  override name = "UsageError";
}

const OPTIONS = {
  store: { type: "string" },
  "set-id": { type: "string" },
  generation: { type: "string" },
  "plan-id": { type: "string" },
  "public-manifest": { type: "string" },
  parameters: { type: "string", multiple: true },
  rule: { type: "string", multiple: true },
  config: { type: "string" },
  labels: { type: "string" },
  format: { type: "string" },
  vocabulary: { type: "string" },
  known: { type: "string", multiple: true },
  "max-attempts": { type: "string" },
  "generator-commit": { type: "string" },
  "progress-every": { type: "string" },
  out: { type: "string" },
  register: { type: "string" },
  dir: { type: "string" },
  heads: { type: "string" },
  "heads-ref": { type: "string" },
} as const;

type Values = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>["values"];

const need = (values: Values, name: keyof Values & string): string => {
  const value = values[name];
  if (typeof value !== "string") throw new UsageError(`--${name} is required`);
  return value;
};

/** `parseSecretJson` with a usage error: it refuses without quoting the text, which may hold a key, a salt or a paste (a `SyntaxError` message can quote it). */
function parseInput(text: string, what: string): unknown {
  try {
    return parseSecretJson(text, what);
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
}

/** Every JSON file the command reads goes through `parseInput`. */
const json = (path: string): unknown => parseInput(readFileSync(path, "utf8"), path);

function count(text: string | undefined, name: string, fallback?: number): number {
  if (text === undefined) {
    if (fallback === undefined) throw new UsageError(`--${name} is required`);
    return fallback;
  }
  if (!/^[0-9]+$/.test(text) || !Number.isSafeInteger(Number(text))) throw new UsageError(`--${name} must be a whole number`);
  return Number(text);
}

/** The parameters and rule pairs: `--parameters` and `--rule` repeated in step. */
function evaluationConfigs(values: Values) {
  const parameters = values.parameters ?? [];
  const rules = values.rule ?? [];
  if (parameters.length === 0 || parameters.length !== rules.length) throw new UsageError("--parameters and --rule are required, once each per version");
  return parameters.map((p, i) => readEvaluationConfig(json(p), json(rules[i] as string)));
}

/** Teams from JSON Lines files: every line must carry the engine's paste. */
function knownTeams(paths: readonly string[]): KnownTeam[] {
  return paths.flatMap((path) =>
    readFileSync(path, "utf8")
      .split("\n")
      .filter((line) => line !== "")
      .map((line, n) => {
        const paste = (parseInput(line, `${path} line ${n + 1}`) as { paste?: unknown } | null)?.paste;
        if (typeof paste !== "string") throw new UsageError(`${path} line ${n + 1} has no paste`);
        return { paste };
      }),
  );
}

function generatorInputs(values: Values, engine: Engine) {
  const formatId = need(values, "format");
  const table = exportEngineData(engine, formatId);
  const config = readTeamGeneratorConfig(json(need(values, "config")), table);
  const labelSet = readTeamLabelSet(json(need(values, "labels")), table);
  const vocabulary = values.vocabulary === undefined ? exportTeamVocabulary(engine, formatId, table) : readTeamVocabulary(json(values.vocabulary), table);
  return { format: resolveFormat(engine, formatId), table, config, labelSet, vocabulary };
}

function commit(values: Values): string {
  const given = values["generator-commit"] ?? execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  if (!/^[0-9a-f]{40}$/.test(given)) throw new UsageError("the generator commit is not a 40-digit lowercase hex SHA");
  return given;
}

function runGenerate(values: Values, io: SealedCliIo, engine: Engine): number {
  const evaluation = evaluationConfigs(values);
  if (evaluation.length !== 1) throw new UsageError("generate takes one --parameters and one --rule");
  const inputs = generatorInputs(values, engine);
  const setId = need(values, "set-id");
  const publicManifest = readTeamDatasetManifest(json(need(values, "public-manifest")), { config: inputs.config, labelSet: inputs.labelSet });
  const result = generateSealedSet({
    engine,
    ...inputs,
    evaluation: evaluation[0] as (typeof evaluation)[number],
    publicManifest,
    known: knownTeams(values.known ?? []),
    storeDir: need(values, "store"),
    setId,
    generation: count(values.generation, "generation"),
    planId: values["plan-id"] ?? setId,
    generatorCommit: commit(values),
    maxAttempts: count(values["max-attempts"], "max-attempts", 3),
    secret: io.secret,
    progressEvery: count(values["progress-every"], "progress-every", 10_000),
    onProgress: ({ attempt, draws, open }) => io.err(`attempt ${attempt}: ${draws} draws, ${open} bins open`),
  });
  for (const warning of result.warnings) io.err(`warning: ${warning}`);
  io.out(
    JSON.stringify({
      set_id: setId,
      team_count: result.manifest.team_count,
      attempts: result.discarded + 1,
      discarded: result.discarded,
      set_manifest_hash: sealedManifestHash(result.manifest),
      commitment_hash: commitmentHash(result.commitment),
      set_commitment: result.commitment.set_commitment,
      key_commitment: result.commitment.key_commitment,
    }),
  );
  return 0;
}

function runRevealHash(values: Values, io: SealedCliIo): number {
  io.out(JSON.stringify({ set_id: need(values, "set-id"), reveal_hash: revealHash(need(values, "store"), need(values, "set-id")) }));
  return 0;
}

/** Register authenticity is enforced in code, not only by the runbook: the chain must verify and match every published head (no rollback, no fork). Used before anything is written or read from a bundle. */
function checkPublishedHeads(values: Values, registerText: string, configs: ReturnType<typeof evaluationConfigs>, retiringSet?: string): void {
  const heads = readPublishedHeads(need(values, "heads"), values["heads-ref"], process.cwd(), { requireHeads: true });
  const state = verifyRegister(registerText, configs);
  const check = checkHeads(state, heads);
  if (retiringSet !== undefined) assertRetirementPublished(state, check, retiringSet);
}

function runRetire(values: Values, io: SealedCliIo): number {
  const registerText = readFileSync(need(values, "register"), "utf8");
  const configs = evaluationConfigs(values);
  // Before the store is touched: a rolled-back or forked register must not unlock a reveal bundle.
  checkPublishedHeads(values, registerText, configs, need(values, "set-id"));
  const hash = retireSealedSet({
    storeDir: need(values, "store"),
    setId: need(values, "set-id"),
    outDir: need(values, "out"),
    registerText,
    configs,
  });
  io.out(JSON.stringify({ set_id: need(values, "set-id"), reveal_hash: hash, out: need(values, "out") }));
  return 0;
}

function runVerify(values: Values, io: SealedCliIo, engine: Engine): number {
  const configs = evaluationConfigs(values);
  if (configs.length !== 1) throw new UsageError("verify takes one --parameters and one --rule");
  const evaluation = configs[0] as (typeof configs)[number];
  const inputs = generatorInputs(values, engine);
  const bundle = readRevealDir(need(values, "dir"));
  const publicManifest = readTeamDatasetManifest(json(need(values, "public-manifest")), { config: inputs.config, labelSet: inputs.labelSet });
  const registerText = readFileSync(need(values, "register"), "utf8");
  checkPublishedHeads(values, registerText, configs);
  const report = verifyRetiredSealed({ engine, ...inputs, evaluation, bundle, publicManifest, registerText, known: knownTeams(values.known ?? []) });
  io.out(JSON.stringify({ verified: true, ...report }));
  return 0;
}

const COMMANDS = ["generate", "reveal-hash", "retire", "verify"];

/** Run the command line; returns the exit code. */
export function runSealedCli(argv: readonly string[], io: SealedCliIo): number {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true });
    const [command, ...rest] = positionals;
    if (rest.length > 0 || command === undefined || !COMMANDS.includes(command)) throw new UsageError(`usage: <${COMMANDS.join("|")}> ... (see docs/runbooks/evaluation-store.md section 9)`);
    if (command === "reveal-hash") return runRevealHash(values, io);
    if (command === "retire") return runRetire(values, io);
    const engine = io.engine ?? loadEngine();
    return command === "generate" ? runGenerate(values, io, engine) : runVerify(values, io, engine);
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    const input = error instanceof Error && (INPUT_ERRORS.has(error.name) || (typeof code === "string" && (code === "ENOENT" || code.startsWith("ERR_PARSE_ARGS"))));
    const refused = error instanceof Error && REFUSALS.has(error.name);
    if (!input && !refused) throw error;
    io.err(`${(error as Error).name}: ${(error as Error).message}`);
    return input ? 2 : 1;
  }
}

/** Errors that mean the inputs are wrong: exit 2. */
const INPUT_ERRORS = new Set(["UsageError", "SyntaxError", "TeamRecordError", "StreamError", "TeamVocabularyError", "EngineDataError", "UnsupportedFormatError", "EngineError", "EvaluationConfigError"]);
/** Errors that mean the store, the plan or the run refused: exit 1. */
const REFUSALS = new Set(["SealedError", "SealedShortfallError", "SealedStoreError", "SealedRetireError", "SealedVerifyError", "PlanError", "RegisterError", "DedupeError"]);
