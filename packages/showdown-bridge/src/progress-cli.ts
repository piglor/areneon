// `scripts/progress.ts`: one generation's progress evaluation for the Python
// command (ml/areneon_ml/progress). It takes everything on the command line,
// starts the policy worker (CPU torch, stdio only) with the frozen pair,
// plays the matches and writes `progress.json` and the stored games into the
// directory it is given. It prints one canonical JSON line (the counts and the
// wall-clock per match). Exit 0 ok, 1 an evaluation that failed or was refused,
// 2 a usage or configuration error. Nothing is retried, nothing is committed
// here: the Python side makes the directory real.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { type Json, jcs } from "./canonical-json.ts";
import { engineDataHash, readEngineData } from "./engine-data.ts";
import { loadEngine } from "./engine.ts";
import type { Engine } from "./engine.ts";
import { UsageError } from "./generation-cli.ts";
import { damageAwareSetup, type FeatureSetup, featureSetup, GenerationError } from "./league-generation.ts";
import { PolicyWorkerClient } from "./policy-worker.ts";
import { EVAL_DATASET_FLAGS, readEvalDataset, withEvalDataset } from "./eval-dataset.ts";
import { type ProgressMode, type ProgressPool, ProgressError, playProgress, writeProgress, PROGRESS_MODES } from "./progress-evaluation.ts";
import { progressRecordHash } from "./progress-records.ts";
import { readTrainingRunConfig, runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

export { UsageError };

/** Where the validation teams come from: the port the team-dataset script fills (the bridge cannot import it). */
export type ProgressPoolSource = (engine: Engine, config: TrainingRunConfig) => ProgressPool;

const FLAGS = {
  config: { type: "string" },
  generation: { type: "string" },
  preview: { type: "string" },
  battle: { type: "string" },
  "policy-id-sample": { type: "string" },
  "policy-id-argmax": { type: "string" },
  "generation-manifest-sha256": { type: "string" },
  "team-set": { type: "string" },
  "run-dir": { type: "string" },
  "engine-data": { type: "string" },
  "encoder-version": { type: "string" },
  "worker-python": { type: "string" },
  games: { type: "string" },
  "record-games": { type: "string" },
  out: { type: "string" },
  // The previous generation's frozen pair (G >= 1): all four together, or none (G = 0).
  "previous-preview": { type: "string" },
  "previous-battle": { type: "string" },
  "previous-policy-id-sample": { type: "string" },
  "previous-policy-id-argmax": { type: "string" },
  // Anchors (fixed earlier generations): each `--anchor <generation>,<preview>,<battle>,<policy-id-sample>,<policy-id-argmax>`.
  anchor: { type: "string", multiple: true },
  "anchor-games": { type: "string" },
  // Another run's frozen pair (a cross-run comparison): `--versus <run-id>,<run-config-hash>,<generation>,<preview>,<battle>,
  // <policy-id-sample>,<policy-id-argmax>`, with that run's directory (read-only: its checkpoints are the worker's), its
  // encoder version and its `team-set.json` (a validation team it trained on is refused) and the games per match.
  versus: { type: "string" },
  "versus-dir": { type: "string" },
  "versus-encoder-version": { type: "string" },
  "versus-team-set": { type: "string" },
  "run-games": { type: "string" },
  // The Preview ablation (#940): `--ablation-preview <generation>,<preview-checkpoint>,<policy-id-argmax>` (this
  // Battle with an earlier generation's Preview) and `--ablation-random-policy-id` (this Battle with a random Preview).
  "ablation-preview": { type: "string" },
  "ablation-random-policy-id": { type: "string" },
  // The damage-aware baseline as a third scripted opponent (ADR-018 slice 4): given, its two matches are played, and
  // each may take at most this many wall-clock milliseconds or the evaluation fails and commits nothing.
  "damage-aware-budget-ms": { type: "string" },
  // The validation split of another verified dataset instead of the run's own (ADR-020 E-17.1): both flags, or none.
  ...EVAL_DATASET_FLAGS,
  // Key the validation pairs and battle seeds by this id instead of the run key (#1177).
  "evaluation-key": { type: "string" },
} as const;
const OPTIONAL = new Set<string>(["previous-preview", "previous-battle", "previous-policy-id-sample", "previous-policy-id-argmax", "anchor", "anchor-games", "versus", "versus-dir", "versus-encoder-version", "versus-team-set", "run-games", "ablation-preview", "ablation-random-policy-id", "damage-aware-budget-ms", "eval-dataset-dir", "eval-dataset-hash", "evaluation-key"]);
const VERSUS = ["versus", "versus-dir", "versus-encoder-version", "versus-team-set", "run-games"] as const;
const PREVIOUS = ["previous-preview", "previous-battle", "previous-policy-id-sample", "previous-policy-id-argmax"] as const;

const integer = (name: string, text: string, least: number): number => {
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < least || !/^\d+$/.test(text)) throw new UsageError(`--${name} must be an integer of at least ${least}`);
  return value;
};

/** Every battle seed of the run's committed `plan.json` files (generations 1 and up). */
export function committedPlanSeeds(runDir: string): Set<string> {
  const seeds = new Set<string>();
  const base = join(runDir, "generations");
  for (const name of readdirSync(base).sort()) {
    if (!/^g\d{6}$/.test(name)) continue;
    let text: string;
    try {
      text = readFileSync(join(base, name, "plan.json"), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue; // generation 0 plays nothing
      throw error;
    }
    for (const entry of JSON.parse(text) as { seed: string }[]) seeds.add(entry.seed);
  }
  return seeds;
}

/** The team hashes of a run's `team-set.json`; anything else is refused (this list is what keeps trained teams out of the pool). */
function readTeamSet(path: string): string[] {
  const teams = (JSON.parse(readFileSync(path, "utf8")) as { teams?: unknown }).teams;
  if (!Array.isArray(teams) || teams.some((t) => typeof t !== "string" || !/^[0-9a-f]{64}$/.test(t))) throw new UsageError("--versus-team-set must hold a teams list of SHA-256 hashes");
  return teams as string[];
}

/** What a frozen pair's worker needs: one worker session holds one pair, CPU torch, stdio only, one query at a time. */
export interface FrozenPairClientOptions {
  python: string;
  runDir: string;
  pair: { preview: string; battle: string };
  encoderVersion: string;
  engineDataFile: string;
  engineDataHash: string;
  max_line_bytes: number;
  handshake_timeout_ms: number;
  /** The feature set the pair's checkpoints read (`featureSetup(...).statement`; absent: none): the worker may state exactly this. */
  features?: FeatureSetup["statement"];
}

/**
 * The candidate feature setup of one run config (absent: the pair reads none), failing as a usage error when this
 * checkout does not compute the hash the config names. Each run's seats are set up from that run's own config.
 */
export function pairFeatureSetup(engine: Engine, config: TrainingRunConfig): FeatureSetup | undefined {
  try {
    return featureSetup(engine, config);
  } catch (error) {
    if (error instanceof GenerationError) throw new UsageError(error.message);
    throw error;
  }
}

/** The policy worker client of one frozen pair, read-only from a run directory's checkpoint store. */
export function frozenPairClient(o: FrozenPairClientOptions): PolicyWorkerClient {
  return new PolicyWorkerClient({
    command: o.python,
    args: ["-I", "-m", "areneon_ml.worker", "--run-dir", o.runDir, "--preview", o.pair.preview, "--battle", o.pair.battle, "--engine-data", o.engineDataFile, "--max-line-bytes", String(o.max_line_bytes)],
    env: {},
    max_line_bytes: o.max_line_bytes,
    n_parallel: 1, // evaluation never batches (ADR-010 section 2)
    handshake_timeout_ms: o.handshake_timeout_ms,
    ...(o.features === undefined ? {} : { features: o.features }),
    expect: {
      encoder_version: o.encoderVersion,
      engine_data_hash: o.engineDataHash,
      checkpoints: [
        { role: "battle", checkpoint_id: o.pair.battle },
        { role: "preview", checkpoint_id: o.pair.preview },
      ],
    },
  });
}

/** Run one generation's progress evaluation; returns the process exit code and writes its one JSON line with `print`. */
export async function runProgressCli(argv: readonly string[], print: (line: string) => void, poolSource: ProgressPoolSource): Promise<number> {
  let values: Record<Exclude<keyof typeof FLAGS, "anchor">, string | undefined> & { anchor?: string[] };
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: Exclude<keyof typeof FLAGS, "anchor">): string => values[name] ?? ((): never => { throw new UsageError(`--${name} is required`); })();
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) if (!OPTIONAL.has(name)) need(name as Exclude<keyof typeof FLAGS, "anchor">);
  const anchorSpecs = (values.anchor ?? []).map((text) => {
    const parts = text.split(",");
    const [generation, preview, battle, sample, argmax] = parts;
    if (parts.length !== 5 || generation === undefined || !/^\d+$/.test(generation)) throw new UsageError("--anchor must be <generation>,<preview>,<battle>,<policy-id-sample>,<policy-id-argmax>");
    return { generation: Number(generation), checkpoints: { preview: preview as string, battle: battle as string }, policy_ids: { sample: sample as string, argmax: argmax as string } };
  });
  if ((anchorSpecs.length === 0) !== (values["anchor-games"] === undefined)) throw new UsageError("--anchor and --anchor-games go together");
  const anchor_games = values["anchor-games"] === undefined ? undefined : integer("anchor-games", values["anchor-games"], 2);
  if ((values["ablation-preview"] === undefined) !== (values["ablation-random-policy-id"] === undefined)) throw new UsageError("--ablation-preview and --ablation-random-policy-id go together");
  const ablationParts = values["ablation-preview"]?.split(",");
  if (ablationParts !== undefined && (ablationParts.length !== 3 || !/^\d+$/.test(ablationParts[0] as string))) throw new UsageError("--ablation-preview must be <generation>,<preview-checkpoint>,<policy-id-argmax>");
  if (ablationParts !== undefined && (anchorSpecs.length > 0 || PREVIOUS.some((name) => values[name] !== undefined))) throw new UsageError("the Preview ablation is played alone: no anchors and no previous-generation flags");
  const damageBudget = values["damage-aware-budget-ms"] === undefined ? undefined : integer("damage-aware-budget-ms", values["damage-aware-budget-ms"], 1);
  if (damageBudget !== undefined && ablationParts !== undefined) throw new UsageError("the Preview ablation is played alone: no damage-aware matches");
  const versusGiven = VERSUS.filter((name) => values[name] !== undefined);
  if (versusGiven.length !== 0 && versusGiven.length !== VERSUS.length) throw new UsageError(`the cross-run flags go together: ${VERSUS.map((n) => `--${n}`).join(", ")}`);
  const versusParts = values.versus?.split(",");
  if (versusParts !== undefined && (versusParts.length !== 7 || !/^\d+$/.test(versusParts[2] as string))) throw new UsageError("--versus must be <run-id>,<run-config-hash>,<generation>,<preview>,<battle>,<policy-id-sample>,<policy-id-argmax>");
  const run_games = values["run-games"] === undefined ? undefined : integer("run-games", values["run-games"], 2);
  if (versusParts !== undefined && (ablationParts !== undefined)) throw new UsageError("the Preview ablation is played alone: no other run");
  const given = PREVIOUS.filter((name) => values[name] !== undefined);
  if (given.length !== 0 && given.length !== PREVIOUS.length) throw new UsageError(`the previous-generation flags go together: ${PREVIOUS.map((n) => `--${n}`).join(", ")}`);
  const generation = integer("generation", need("generation"), 0);
  const games = integer("games", need("games"), 1);
  const record_games = integer("record-games", need("record-games"), 0);
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (config.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  if (config.team_dataset === undefined) {
    console.error("progress: the run config names no team dataset: a progress evaluation needs the validation split of one");
    return 1;
  }
  const evalDataset = readEvalDataset(values);
  if (evalDataset !== undefined && evalDataset.dataset_hash === config.team_dataset.dataset_hash) throw new UsageError("--eval-dataset-hash is the run's own training dataset: leave the option out to evaluate on its validation split");
  const teamSet = JSON.parse(readFileSync(need("team-set"), "utf8")) as { teams: string[] };
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const { max_line_bytes, handshake_timeout_ms } = config.generation;
  // The candidate features of the run's pair (none: nothing below changes) and, for another run's pair, of ITS config.
  const features = pairFeatureSetup(engine, config);
  const versusConfig = values["versus-dir"] === undefined ? undefined : readTrainingRunConfig(JSON.parse(readFileSync(join(values["versus-dir"], "run-config.json"), "utf8")));
  if (versusConfig !== undefined && versusParts !== undefined && (await runConfigHash(versusConfig)) !== versusParts[1]) throw new UsageError("--versus-dir holds another run config than the --versus run config hash names");
  const versusFeatures = versusConfig === undefined ? undefined : pairFeatureSetup(engine, versusConfig);
  const preview = need("preview");
  const battle = need("battle");
  // One worker session holds one frozen pair, so the previous generation's pair gets a worker of its own.
  const workerFor = (pair: { preview: string; battle: string }, runDir: string = need("run-dir"), encoderVersion: string = need("encoder-version"), setup: FeatureSetup | null = features ?? null): PolicyWorkerClient =>
    frozenPairClient({ python: need("worker-python"), runDir, pair, encoderVersion, engineDataFile: dataFile, engineDataHash: dataHash, max_line_bytes, handshake_timeout_ms, ...(setup === null ? {} : { features: setup.statement }) });
  const client = workerFor({ preview, battle });
  const previousPair = given.length === 0 ? undefined : { preview: need("previous-preview"), battle: need("previous-battle") };
  let previousClient: PolicyWorkerClient | undefined;
  // The ablation's third variant: this generation's Battle with an earlier generation's Preview, behind a worker of its own.
  const ablationClient = ablationParts === undefined ? undefined : workerFor({ preview: ablationParts[1] as string, battle });
  const anchorClients: PolicyWorkerClient[] = [];
  const versus =
    versusParts === undefined
      ? undefined
      : {
          run_id: versusParts[0] as string,
          run_config_hash: versusParts[1] as string,
          generation: Number(versusParts[2]),
          checkpoints: { preview: versusParts[3] as string, battle: versusParts[4] as string },
          policy_ids: { sample: versusParts[5] as string, argmax: versusParts[6] as string },
        };
  const versusClient = versus === undefined ? undefined : workerFor(versus.checkpoints, need("versus-dir"), need("versus-encoder-version"), versusFeatures ?? null);
  const versusTeamSet = values["versus-team-set"] === undefined ? [] : readTeamSet(values["versus-team-set"]);
  try {
    previousClient = previousPair === undefined ? undefined : workerFor(previousPair);
    const pool = poolSource(engine, withEvalDataset(config, evalDataset)); // the evaluation dataset, hash-checked by the pool; `config` stays the run's own
    await client.start();
    await previousClient?.start();
    await ablationClient?.start();
    await versusClient?.start();
    for (const spec of anchorSpecs) {
      const anchorClient = workerFor(spec.checkpoints);
      anchorClients.push(anchorClient); // closed in `finally` even if its start fails
      await anchorClient.start();
    }
    const policy_ids: Record<ProgressMode, string> = { sample: need("policy-id-sample"), argmax: need("policy-id-argmax") };
    const result = await playProgress({
      engine,
      config,
      generation,
      generation_manifest_sha256: need("generation-manifest-sha256"),
      checkpoints: { preview, battle },
      policy_ids,
      client,
      ...(features === undefined ? {} : { features: features.configs }),
      ...(previousPair === undefined || previousClient === undefined
        ? {}
        : {
            previous: {
              checkpoints: previousPair,
              policy_ids: { sample: need("previous-policy-id-sample"), argmax: need("previous-policy-id-argmax") },
              client: previousClient,
            },
          }),
      ...(anchor_games === undefined ? {} : { anchors: anchorSpecs.map((spec, index) => ({ ...spec, client: anchorClients[index] as PolicyWorkerClient })), anchor_games }),
      ...(versus === undefined || versusClient === undefined || run_games === undefined ? {} : { versus: { ...versus, client: versusClient, ...(versusFeatures === undefined ? {} : { features: versusFeatures.configs }) }, run_games }),
      ...(ablationParts === undefined || ablationClient === undefined
        ? {}
        : {
            ablation: {
              generation: { generation: Number(ablationParts[0]), checkpoint: ablationParts[1] as string, policy_id: ablationParts[2] as string, client: ablationClient },
              random_policy_id: need("ablation-random-policy-id"),
            },
          }),
      ...(damageBudget === undefined ? {} : { damage_aware: { setup: damageAwareSetup(engine, config.format_id), match_budget_ms: damageBudget } }),
      ...(values["evaluation-key"] === undefined ? {} : { evaluation_key: values["evaluation-key"] }),
      pool,
      games,
      record_games,
      training_teams: new Set([...teamSet.teams, ...versusTeamSet]),
      league_seeds: new Set([...committedPlanSeeds(need("run-dir")), ...(values["versus-dir"] === undefined ? [] : committedPlanSeeds(values["versus-dir"]))]),
    });
    writeProgress(need("out"), result);
    print(
      jcs({
        generation,
        record_hash: progressRecordHash(result.record),
        matches: result.record.matches.map((m) => ({ match_id: m.match_id, games: m.games, wins: m.wins, ties: m.ties, losses: m.losses })),
        wall_ms: result.wall_ms,
        game_ms: result.game_ms,
        modes: [...PROGRESS_MODES],
      } as unknown as Json),
    );
    return 0;
  } catch (error) {
    if (error instanceof ProgressError) {
      console.error(`progress: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    try {
      await client.close();
    } finally {
      try {
        await previousClient?.close(); // every worker is closed even if closing another throws
      } finally {
        await Promise.allSettled([...anchorClients, ...(ablationClient === undefined ? [] : [ablationClient]), ...(versusClient === undefined ? [] : [versusClient])].map((c) => c.close()));
      }
    }
  }
}
