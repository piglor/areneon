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
import { PolicyWorkerClient } from "./policy-worker.ts";
import { type ProgressMode, type ProgressPool, ProgressError, playProgress, writeProgress, PROGRESS_MODES } from "./progress-evaluation.ts";
import { progressRecordHash } from "./progress-records.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "./training-run-config.ts";

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
} as const;

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

/** Run one generation's progress evaluation; returns the process exit code and writes its one JSON line with `print`. */
export async function runProgressCli(argv: readonly string[], print: (line: string) => void, poolSource: ProgressPoolSource): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: keyof typeof FLAGS): string => values[name] ?? ((): never => { throw new UsageError(`--${name} is required`); })();
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) need(name);
  const generation = integer("generation", need("generation"), 0);
  const games = integer("games", need("games"), 1);
  const record_games = integer("record-games", need("record-games"), 0);
  const config = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (config.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  if (config.team_dataset === undefined) {
    console.error("progress: the run config names no team dataset: a progress evaluation needs the validation split of one");
    return 1;
  }
  const teamSet = JSON.parse(readFileSync(need("team-set"), "utf8")) as { teams: string[] };
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const dataFile = need("engine-data");
  const dataHash = engineDataHash(readEngineData(JSON.parse(readFileSync(dataFile, "utf8"))));
  const { max_line_bytes, handshake_timeout_ms } = config.generation;
  const preview = need("preview");
  const battle = need("battle");
  const client = new PolicyWorkerClient({
    command: need("worker-python"),
    args: ["-I", "-m", "areneon_ml.worker", "--run-dir", need("run-dir"), "--preview", preview, "--battle", battle, "--engine-data", dataFile, "--max-line-bytes", String(max_line_bytes)],
    env: {},
    max_line_bytes,
    n_parallel: 1, // evaluation never batches (ADR-010 section 2)
    handshake_timeout_ms,
    expect: {
      encoder_version: need("encoder-version"),
      engine_data_hash: dataHash,
      checkpoints: [
        { role: "battle", checkpoint_id: battle },
        { role: "preview", checkpoint_id: preview },
      ],
    },
  });
  try {
    const pool = poolSource(engine, config);
    await client.start();
    const policy_ids: Record<ProgressMode, string> = { sample: need("policy-id-sample"), argmax: need("policy-id-argmax") };
    const result = await playProgress({
      engine,
      config,
      generation,
      generation_manifest_sha256: need("generation-manifest-sha256"),
      checkpoints: { preview, battle },
      policy_ids,
      client,
      pool,
      games,
      record_games,
      training_teams: new Set(teamSet.teams),
      league_seeds: committedPlanSeeds(need("run-dir")),
    });
    writeProgress(need("out"), result);
    print(
      jcs({
        generation,
        record_hash: progressRecordHash(result.record),
        matches: result.record.matches.map((m) => ({ match_id: m.match_id, games: m.games, wins: m.wins, ties: m.ties, losses: m.losses })),
        wall_ms: result.wall_ms,
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
    await client.close();
  }
}
