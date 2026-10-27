// `scripts/scripted-h2h.ts`: the scripted head-to-head for the Python command (ml/areneon_ml/progress/scripted.py).
// It plays arm a against arm b on the validation pairs of the run's team dataset (or of the dataset it is given),
// every pair from both seats, over `--game-processes` processes, and writes `head-to-head.json` into `--out`. It
// prints one canonical JSON line (the counts and the wall-clock). Exit 0 ok, 1 a head-to-head that failed or was
// refused (an incident, the wall-time cap), 2 a usage or configuration error. Nothing is retried or committed here:
// the Python side makes the directory real and logs the look.
//
// Game processes follow the generation loop's sharding rule: process k plays the games `g mod S = k`, a game is a
// function of its index alone, so the merged rows (and the record) are the same for every S.

import { closeSync, fsyncSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { type Json, jcs } from "./canonical-json.ts";
import { EVAL_DATASET_FLAGS, readEvalDataset, withEvalDataset } from "./eval-dataset.ts";
import type { BeliefWorldsFactory } from "./damage-aware-worlds.ts";
import { loadEngine } from "./engine.ts";
import { startShard, UsageError, writeShardFile } from "./generation-cli.ts";
import { GenerationError } from "./league-generation.ts";
import type { PlayedEntry } from "./league-generation.ts";
import type { ProgressPoolSource } from "./progress-cli.ts";
import { ProgressError } from "./progress-evaluation.ts";
import { scriptedH2HRecordHash, type ScriptedGameRow } from "./scripted-h2h-record.ts";
import { buildScriptedRecord, playScriptedGames, ScriptedH2HError, scriptedArm } from "./scripted-h2h.ts";
import { readTrainingRunConfig } from "./training-run-config.ts";

export { UsageError };

const FLAGS = {
  config: { type: "string" },
  "arm-a": { type: "string" },
  "arm-b": { type: "string" },
  pairs: { type: "string" },
  out: { type: "string" },
  // Another team dataset's validation split instead of the config's (the eval-dataset option): both flags, or none.
  ...EVAL_DATASET_FLAGS,
  "game-processes": { type: "string" },
  "wall-time-cap-ms": { type: "string" },
  // Internal: set on the game processes this one starts (`INDEX/COUNT`); such a process plays its slice and writes `shard.jsonl`.
  shard: { type: "string" },
} as const;
const OPTIONAL = new Set<string>(["eval-dataset-dir", "eval-dataset-hash", "game-processes", "wall-time-cap-ms", "shard"]);
const MAX_PROCESSES = 64;
/** The record the command writes into `--out`. */
export const H2H_FILE = "head-to-head.json";

const integer = (name: string, text: string | undefined, least: number): number => {
  const value = Number(text);
  if (text === undefined || !/^\d+$/.test(text) || !Number.isSafeInteger(value) || value < least) throw new UsageError(`--${name} must be an integer of at least ${least}`);
  return value;
};

/** Run one scripted head-to-head; returns the process exit code and writes its one JSON line with `print`. */
export async function runScriptedH2HCli(argv: readonly string[], print: (line: string) => void, poolSource: ProgressPoolSource, beliefWorlds?: BeliefWorldsFactory, shardScript?: string): Promise<number> {
  let values: Record<keyof typeof FLAGS, string | undefined>;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as typeof values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const need = (name: keyof typeof FLAGS): string => values[name] ?? ((): never => { throw new UsageError(`--${name} is required`); })();
  for (const name of Object.keys(FLAGS) as (keyof typeof FLAGS)[]) if (!OPTIONAL.has(name)) need(name);
  const evalDataset = readEvalDataset(values);
  const pairs = integer("pairs", need("pairs"), 1);
  const processes = values["game-processes"] === undefined ? 1 : integer("game-processes", values["game-processes"], 1);
  if (processes > MAX_PROCESSES) throw new UsageError(`--game-processes is at most ${MAX_PROCESSES}`);
  const cap = values["wall-time-cap-ms"] === undefined ? undefined : integer("wall-time-cap-ms", values["wall-time-cap-ms"], 1);
  let own: { index: number; of: number } | undefined;
  if (values.shard !== undefined) {
    const match = /^(\d+)\/(\d+)$/.exec(values.shard);
    own = match === null ? undefined : { index: Number(match[1]), of: Number(match[2]) };
    if (own === undefined || processes !== 1 || own.of < 2 || own.of > MAX_PROCESSES || own.index >= own.of) throw new UsageError("--shard is INDEX/COUNT, and only for a process of its own");
  } else if (processes > 1) {
    own = { index: 0, of: processes };
    if (shardScript === undefined) throw new UsageError("--game-processes needs the script to start again");
  }
  const read = readTrainingRunConfig(JSON.parse(readFileSync(need("config"), "utf8")));
  if (read.seeds.run_key.startsWith("CHANGE-ME")) throw new UsageError("seeds.run_key is still the CHANGE-ME placeholder: give the run its own name");
  const config = withEvalDataset(read, evalDataset);
  if (config.team_dataset === undefined) {
    console.error("scripted-h2h: the run config names no team dataset: a head-to-head needs the validation split of one");
    return 1;
  }
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const scratch = values.shard === undefined && processes > 1 ? mkdtempSync(join(tmpdir(), "areneon-h2h-")) : undefined;
  const children: ReturnType<typeof startShard>[] = [];
  const started = performance.now();
  try {
    // Start the further game processes first: each loads its own pool while this one does.
    if (scratch !== undefined) {
      for (let index = 1; index < processes; index++) {
        const out = join(scratch, String(index));
        const rest = Object.entries(values).flatMap(([name, value]) => (value === undefined || ["out", "game-processes", "shard"].includes(name) ? [] : [`--${name}`, value]));
        children.push(startShard(shardScript as string, [...rest, "--out", out, "--shard", `${index}/${processes}`], out, index));
      }
    }
    const budget = { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
    const context = { engine, formatId: config.format_id, budget, beliefWorlds };
    const arms = { a: scriptedArm(need("arm-a"), context), b: scriptedArm(need("arm-b"), context) };
    const pool = poolSource(engine, config);
    const others = Promise.all(children.map((child) => child.entries));
    const playing = playScriptedGames({ engine, config, pool, pairs, arms, ...(own === undefined ? {} : { shard: own }), ...(cap === undefined ? {} : { wall_time_cap_ms: cap }) });
    if (values.shard !== undefined) {
      // A further game process: it ends if the process that started it dies (its stdin closes).
      process.stdin.once("end", process.exit.bind(process, 1));
      process.stdin.resume();
      (process.stdin as unknown as { unref?: () => void }).unref?.();
      mkdirSync(need("out"), { recursive: true });
      writeShardFile(join(need("out"), "shard.jsonl"), (await playing) as unknown as PlayedEntry[]);
      return 0;
    }
    // The first game process to fail fails this one at once; the `finally` below kills the others.
    playing.catch(() => undefined); // reported where it is awaited
    const mine = await Promise.race([playing, others.then(() => new Promise<never>(() => undefined))]);
    const rows: ScriptedGameRow[] = [...mine, ...(await others).flat().map((entry) => entry as unknown as ScriptedGameRow)];
    const record = buildScriptedRecord({ engine, config, pool, pairs, arms, rows, ...(cap === undefined ? {} : { wall_time_cap_ms: cap }) });
    mkdirSync(need("out"), { recursive: true });
    const descriptor = openSync(join(need("out"), H2H_FILE), "wx");
    try {
      writeSync(descriptor, `${jcs(record as unknown as Json)}\n`);
      fsyncSync(descriptor);
    } finally {
      closeSync(descriptor);
    }
    const { result } = record;
    print(jcs({ record_hash: scriptedH2HRecordHash(record), games: record.plan.games, wins_a: result.wins_a, wins_b: result.wins_b, ties: result.ties, wall_ms: Math.round(performance.now() - started) } as unknown as Json));
    return 0;
  } catch (error) {
    if (error instanceof ScriptedH2HError || error instanceof ProgressError || error instanceof GenerationError) {
      console.error(`scripted-h2h: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    for (const child of children) child.stop();
    if (scratch !== undefined) rmSync(scratch, { recursive: true, force: true });
  }
}
