// The command line of the team generator:
//
//   generate  --config FILE --labels FILE --format ID --out DIR
//             --dataset-id ID [--plan-id ID] --generation-key HEX --assignment-key HEX
//             [--dev N] [--validation N] [--training N]
//             [--reference FILE]... [--reference-on-disk] [--pilot FILE]
//             [--uniform-pilot FILE] [--d-team N] [--evaluation-only] [--publish]
//             [--vocabulary FILE] [--generator-commit SHA] [--progress-every DRAWS]
//   pilot     --config FILE --labels FILE --format ID --out FILE --generation-key HEX [--vocabulary FILE] [--draws N] [--without-proposal]
//             (--draws runs a shorter trial pilot; `generate` refuses a pilot whose draws are not the config's)
//
// Mixture datasets (ADR-020): `--uniform-pilot` is the pilot of the same config without its `proposal` (made with
// `pilot --without-proposal`), the reference plan of gate P4; `--d-team N` adds the cross-split near-duplicate counts
// to coverage.json; `--evaluation-only` (needs `--d-team` and a `--reference`) drops every team within N differing
// fields of a reference team, logs each as `near_reference` and marks the manifest evaluation-only. A PUBLISHING run
// is one given `--publish`: if any pilot gate (P1 to P4) fails or cannot be evaluated, the files are still written
// and the exit code is 1. Without `--publish` the gates are reported on stderr and in coverage.json only.
//
// `--config` is a TeamGeneratorConfig and `--labels` a TeamLabelSet (data
// records, read through the bridge's readers against the pinned engine).
// `--reference` files are JSON Lines of earlier datasets' records, whose
// canonical hashes the new dataset must not repeat. Without `--pilot`,
// `generate` runs the config's pilot first. The vocabulary is exported from
// the pinned engine unless a file is given (it is checked against the engine's
// data). Output is deterministic: the same commit, config, label set, keys and
// references give the same bytes. Exit codes: 0 done, 1 refused (the plan or
// the run cannot be completed), 2 a usage or input error.

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import {
  type Engine,
  exportEngineData,
  exportTeamVocabulary,
  importPaste,
  loadEngine,
  readTeamGeneratorConfig,
  type TeamGeneratorConfig,
  readTeamLabelSet,
  readTeamVocabulary,
  recordHash,
  resolveFormat,
  teamVocabularyHash,
  ROLE_SETS_FILE,
} from "@areneon/showdown-bridge";
import { recordHashes, recordPastes, writeDataset, writeReferenceHashes } from "./dataset-io.ts";
import { FileStage, KnownHashes, referencePoolHash, sortedUnique } from "./dedupe.ts";
import { Drawer } from "./draw.ts";
import { NearIndex } from "./near-index.ts";
import { type CoverageReport, generate, type ReferencePool } from "./generate.ts";
import { type Pilot, readPilot, runPilot } from "./pilot.ts";
import { SPLITS, type SplitSizes } from "./plan.ts";
import { Proposal } from "./proposal.ts";
import { roleSetsSource } from "./role-set-proposal.ts";
import { checkKey } from "./stream.ts";

export interface CliIo {
  out(line: string): void;
  err(line: string): void;
  /** The engine to use; the pinned one when omitted. */
  engine?: Engine;
}

class UsageError extends Error {
  override name = "UsageError";
}

const OPTIONS = {
  config: { type: "string" },
  labels: { type: "string" },
  format: { type: "string" },
  out: { type: "string" },
  "dataset-id": { type: "string" },
  "plan-id": { type: "string" },
  "generation-key": { type: "string" },
  "assignment-key": { type: "string" },
  dev: { type: "string" },
  validation: { type: "string" },
  training: { type: "string" },
  reference: { type: "string", multiple: true },
  "reference-on-disk": { type: "boolean" },
  pilot: { type: "string" },
  vocabulary: { type: "string" },
  "role-sets": { type: "string" },
  "uniform-pilot": { type: "string" },
  "d-team": { type: "string" },
  "evaluation-only": { type: "boolean" },
  publish: { type: "boolean" },
  "without-proposal": { type: "boolean" },
  draws: { type: "string" },
  "progress-every": { type: "string" },
  "generator-commit": { type: "string" },
} as const;

type Values = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>["values"];

const need = (values: Values, name: keyof Values & string): string => {
  const value = values[name];
  if (typeof value !== "string") throw new UsageError(`--${name} is required`);
  return value;
};

const json = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));

function count(text: string | undefined, name: string): number | undefined {
  if (text === undefined) return undefined;
  if (!/^[0-9]+$/.test(text) || !Number.isSafeInteger(Number(text))) throw new UsageError(`--${name} must be a whole number`);
  return Number(text);
}

/** What the operator should know about a finished run, one line each. */
export function notices(coverage: Pick<CoverageReport, "provisioning" | "replan" | "splits" | "proposal">, dMax: number): string[] {
  const lines: string[] = [];
  const { provisioning } = coverage;
  if (provisioning.exceeds_d_max) lines.push(`warning: the ${provisioning.planned_draws}-draw fill quantile is beyond d_max ${dMax}; a bin may fall short and re-plan`);
  if (coverage.replan > 0) lines.push(`re-planned ${coverage.replan} time(s) after a shortfall: see coverage.json`);
  for (const s of coverage.splits) if (s.teams !== s.requested) lines.push(`${s.split}: ${s.requested} requested, ${s.teams} delivered (proportional quotas with a floor of n_stratum_min per stratum)`);
  if (coverage.proposal !== undefined && coverage.proposal.failed.length > 0) lines.push(`pilot gates not met or not evaluated: ${coverage.proposal.failed.join(", ")} (see coverage.json)`);
  return lines;
}

/** The pilot's length: `--draws` for a trial, else the config's `pilot_draws`. */
export function trialDraws(draws: string | undefined, config: { pilot_draws: number }): number {
  return count(draws, "draws") ?? config.pilot_draws;
}

function commit(values: Values): string {
  const given = values["generator-commit"] ?? execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  if (!/^[0-9a-f]{40}$/.test(given)) throw new UsageError("the generator commit is not a 40-digit lowercase hex SHA");
  return given;
}

/** Everything a run reads: the engine and what is exported or checked against it. */
function inputs(values: Values, engine: Engine) {
  const formatId = need(values, "format");
  const table = exportEngineData(engine, formatId);
  const config = readTeamGeneratorConfig(json(need(values, "config")), table);
  const labelSet = readTeamLabelSet(json(need(values, "labels")), table);
  const vocabulary = values.vocabulary === undefined ? exportTeamVocabulary(engine, formatId, table) : readTeamVocabulary(json(values.vocabulary), table);
  // The role-set table is read only when the config's mixture has a role_sets component; the default is the golden export.
  const roleSets = config.proposal?.components.some((c) => c.kind === "role_sets")
    ? roleSetsSource(engine, vocabulary, table, json(values["role-sets"] ?? join(fileURLToPath(new URL("../../../fixtures/showdown", import.meta.url)), formatId, ROLE_SETS_FILE)))
    : undefined;
  return { format: resolveFormat(engine, formatId), table, config, labelSet, vocabulary, roleSets };
}

function referencePool(values: Values, scratch: string): { pool: ReferencePool; sorted: string[]; close: () => void } {
  const sorted = sortedUnique(recordHashes(values.reference ?? []));
  const stage = values["reference-on-disk"] === true ? FileStage.write(join(scratch, "reference.bin"), sorted) : undefined;
  const known = new KnownHashes(sorted, stage);
  return { pool: { hash: referencePoolHash(sorted), known }, sorted, close: () => known.close() };
}

function runGenerate(values: Values, io: CliIo, engine: Engine): number {
  const sizes: SplitSizes = {};
  for (const split of SPLITS) {
    const n = count(values[split], split);
    if (n !== undefined) sizes[split] = n;
  }
  const out = need(values, "out");
  const generationKey = checkKey(need(values, "generation-key"), "--generation-key");
  const assignmentKey = checkKey(need(values, "assignment-key"), "--assignment-key");
  const datasetId = need(values, "dataset-id");
  const generatorCommit = commit(values);
  const { format, table, config, labelSet, vocabulary, roleSets } = inputs(values, engine);
  let pilot: Pilot | undefined;
  if (values.pilot !== undefined) {
    pilot = readPilot(json(values.pilot), { config_hash: recordHash(config), label_set_hash: recordHash(labelSet), vocabulary_hash: teamVocabularyHash(vocabulary) }, labelSet);
  }
  let uniformPilot: Pilot | undefined;
  if (values["uniform-pilot"] !== undefined) {
    uniformPilot = readPilot(json(values["uniform-pilot"]), { config_hash: recordHash(withoutProposal(config)), label_set_hash: recordHash(labelSet), vocabulary_hash: teamVocabularyHash(vocabulary) }, labelSet);
  }
  const dTeam = count(values["d-team"], "d-team");
  const evaluationOnly = values["evaluation-only"] === true;
  if (evaluationOnly && (dTeam === undefined || (values.reference ?? []).length === 0)) throw new UsageError("--evaluation-only needs --d-team and at least one --reference");
  const exclude =
    evaluationOnly && dTeam !== undefined
      ? { d_team: dTeam, index: new NearIndex(recordPastes(values.reference ?? []).map((paste) => importPaste(engine, paste)), dTeam) }
      : undefined;
  mkdirSync(out, { recursive: true });
  const scratch = mkdtempSync(join(tmpdir(), "team-dataset-"));
  const reference = referencePool(values, scratch);
  try {
    const started = Date.now();
    const result = generate({
      engine,
      format,
      table,
      vocabulary,
      labelSet,
      config,
      roleSets,
      generationKey,
      assignmentKey,
      datasetId,
      planId: values["plan-id"] ?? datasetId,
      sizes,
      generatorCommit,
      reference: reference.pool,
      pilot,
      uniformPilot,
      dTeam,
      exclude,
      logDir: out,
      progressEvery: count(values["progress-every"], "progress-every") ?? 10_000,
      onProgress: ({ draws, open }) => io.err(`${draws} draws, ${open} bins open`),
    });
    const files = [...writeDataset(out, result), ...writeReferenceHashes(out, reference.sorted)];
    const seconds = Math.max((Date.now() - started) / 1000, 0.001);
    const { stream } = result.coverage;
    for (const line of notices(result.coverage, config.d_max)) io.err(line);
    const total = stream.draws + result.pilotDrawsRun;
    io.err(`${total} draws (${result.pilotDrawsRun} of them the pilot) in ${seconds.toFixed(1)} s: ${Math.round(total / seconds)} draws/s`);
    const gates = result.coverage.proposal;
    io.out(
      JSON.stringify({
        dataset_id: result.datasetId,
        plan_id: result.planId,
        manifest_hash: recordHash(result.manifest),
        teams: Object.fromEntries(result.plan.splits.map((s) => [s, (result.records[s] as unknown[]).length])),
        strata: result.plan.strata.length,
        attempts: result.attempts.length,
        files,
        ...(gates === undefined ? {} : { gates: { publishable: gates.publishable, failed: gates.failed } }),
      }),
    );
    if (values.publish === true && gates === undefined) io.err("--publish: the config has no proposal, so there are no pilot gates to check");
    if (values.publish === true && gates !== undefined && !gates.publishable) {
      io.err(`not publishable: ${gates.failed.join(", ")}`);
      return 1;
    }
    return 0;
  } finally {
    reference.close();
    rmSync(scratch, { recursive: true, force: true });
  }
}

/** The config without its `proposal`: the uniform config a mixture is compared with (gate P4). */
function withoutProposal(config: TeamGeneratorConfig): TeamGeneratorConfig {
  const plain = structuredClone(config);
  delete plain.proposal;
  return plain;
}

function runPilotCommand(values: Values, io: CliIo, engine: Engine): number {
  const out = need(values, "out");
  const generationKey = checkKey(need(values, "generation-key"), "--generation-key");
  const inputsOf = inputs(values, engine);
  const { format, table, labelSet, vocabulary } = inputsOf;
  // The uniform reference pilot (gate P4) is the pilot of the same config without its proposal.
  const config = values["without-proposal"] === true ? withoutProposal(inputsOf.config) : inputsOf.config;
  const roleSets = values["without-proposal"] === true ? undefined : inputsOf.roleSets;
  const identity = { config_hash: recordHash(config), label_set_hash: recordHash(labelSet), vocabulary_hash: teamVocabularyHash(vocabulary) };
  const drawer = new Drawer({ engine, format, proposal: new Proposal(vocabulary, table, config, roleSets), labelSet, generationKey });
  const started = Date.now();
  const pilot = runPilot(drawer, identity, labelSet, trialDraws(values.draws, config));
  writeFileSync(out, `${JSON.stringify(pilot, null, 2)}\n`);
  io.err(`${pilot.draws} pilot draws in ${((Date.now() - started) / 1000).toFixed(1)} s`);
  io.out(JSON.stringify({ pilot: out, draws: pilot.draws, rejected: pilot.rejected }));
  return 0;
}

/** Run the command line; returns the exit code. */
export function runCli(argv: readonly string[], io: CliIo): number {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true });
    const [command, ...rest] = positionals;
    if (rest.length > 0 || (command !== "generate" && command !== "pilot")) throw new UsageError("usage: <generate|pilot> --config FILE --labels FILE --format ID ...");
    const engine = io.engine ?? loadEngine();
    return command === "generate" ? runGenerate(values, io, engine) : runPilotCommand(values, io, engine);
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    const input = error instanceof Error && (INPUT_ERRORS.has(error.name) || (typeof code === "string" && (code === "ENOENT" || code.startsWith("ERR_PARSE_ARGS"))));
    const refused = error instanceof Error && (REFUSALS.has(error.name) || OVERSIZE.has(String(code)) || (error instanceof RangeError && /Set maximum size|Invalid (string|array) length/.test(error.message)));
    if (!input && !refused) throw error;
    io.err(`${(error as Error).name}: ${(error as Error).message}`);
    return input ? 2 : 1;
  }
}

/** Errors that mean the inputs are wrong: exit 2. */
const INPUT_ERRORS = new Set(["UsageError", "SyntaxError", "TeamRecordError", "StreamError", "PilotError", "TeamVocabularyError", "EngineDataError", "UnsupportedFormatError", "EngineError", "RoleSetError"]);
/** Node errors for data beyond what one string or one Set can hold: this slice builds its indexes in memory (the shard slice needs a binary builder). */
const OVERSIZE = new Set(["ERR_STRING_TOO_LONG"]);
/** Errors that mean the plan or the run cannot be completed: exit 1. */
const REFUSALS = new Set(["PlanError", "GenerateError", "ProposalError", "DedupeError"]);
