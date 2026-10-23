// `packages/team-dataset/scripts/teacher-run.ts` (and this package's own script for the interim
// pool): play a teacher run (ADR-019 track 1, #965) and write its directory. Exit 0 ok, 1 a run that
// failed (a harness incident), 2 a usage or configuration error. Nothing is retried. The directory
// is built beside `--out` and renamed into place at the end, so `--out` is either absent or whole.
//
// One game process plays every game by default; `--game-processes N` starts N-1 more copies of the
// script, each on the games `g` with `g mod N = i`, and the bytes do not depend on N.

import { mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync, rmdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { parseArgs } from "node:util";
import { loadEngine } from "./engine.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { startShard, type ShardProcess, UsageError, writeShardFile, type PoolSource } from "./generation-cli.ts";
import { GenerationError, type PoolRequest, type Shard, trainingTeamPool } from "./league-generation.ts";
import { checkTeacherSettings, playTeacherEntries, playTeacherRun, teacherContext, type TeacherEntry, type TeacherSettings, writeTeacherRun } from "./teacher-generation.ts";

const MAX_PARALLEL = 64;
const SHARD_FILE = "shard.jsonl";

const FLAGS = {
  "run-key": { type: "string" },
  format: { type: "string" },
  games: { type: "string" },
  "random-games": { type: "string" },
  "actor-rule-games": { type: "string" },
  "max-turns": { type: "string" },
  "epsilon-ppm": { type: "string" },
  "preview-ms": { type: "string" },
  "battle-ms": { type: "string" },
  "excluded-abilities": { type: "string" },
  dataset: { type: "string" },
  "dataset-hash": { type: "string" },
  out: { type: "string" },
  "game-processes": { type: "string" },
  // Internal: set on the game processes this one starts (`INDEX/COUNT`); such a process plays its slice and writes `shard.jsonl`.
  shard: { type: "string" },
} as const;

type Values = Record<keyof typeof FLAGS, string | undefined>;

/** The interim pool (committed fixture teams); a dataset is the team-dataset script's pool source. */
export const interimTeacherPool: PoolSource = (engine, request) => {
  if (request.team_dataset !== undefined) throw new UsageError("--dataset needs packages/team-dataset/scripts/teacher-run.ts");
  return trainingTeamPool(engine, request.generation.excluded_abilities);
};

function integer(values: Values, name: keyof typeof FLAGS, fallback: string): number {
  const text = values[name] ?? fallback;
  if (!/^\d+$/.test(text)) throw new UsageError(`--${name} must be a non-negative integer`);
  return Number(text);
}

/** Run a teacher run; returns the process exit code and writes its one JSON line with `print`. */
export async function runTeacherCli(
  argv: readonly string[],
  print: (line: string) => void = console.log,
  poolSource: PoolSource = interimTeacherPool,
  /** The script that started this process: `--game-processes` above 1 runs it again for each further game process. */
  shardScript?: string,
): Promise<number> {
  let values: Values;
  try {
    values = parseArgs({ args: [...argv], options: FLAGS, strict: true }).values as Values;
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  for (const name of ["run-key", "format", "out"] as const) if (values[name] === undefined) throw new UsageError(`--${name} is required`);
  if ((values.dataset === undefined) !== (values["dataset-hash"] === undefined)) throw new UsageError("--dataset and --dataset-hash go together");
  const settings: TeacherSettings = {
    run_key: values["run-key"] as string,
    format_id: values.format as string,
    games: integer(values, "games", "0"),
    random_games: integer(values, "random-games", "0"),
    actor_rule_games: integer(values, "actor-rule-games", "0"),
    max_turns: integer(values, "max-turns", "100"),
    epsilon_ppm: integer(values, "epsilon-ppm", "0"),
    latency_budget_ms: { preview_ms: integer(values, "preview-ms", "30000"), battle_ms: integer(values, "battle-ms", "30000") },
  };
  try {
    checkTeacherSettings(settings);
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  const processes = integer(values, "game-processes", "1");
  if (processes < 1 || processes > MAX_PARALLEL) throw new UsageError(`--game-processes must be an integer from 1 to ${MAX_PARALLEL}`);
  let own: Shard | undefined;
  if (values.shard !== undefined) {
    const match = /^(\d+)\/(\d+)$/.exec(values.shard);
    if (match === null || processes !== 1) throw new UsageError("--shard is INDEX/COUNT, and only for a process of its own");
    own = { index: Number(match[1]), of: Number(match[2]) };
    if (own.of < 2 || own.of > MAX_PARALLEL || own.index >= own.of) throw new UsageError("--shard is INDEX/COUNT, and only for a process of its own");
  } else if (processes > 1) {
    own = { index: 0, of: processes };
    if (shardScript === undefined) throw new UsageError("--game-processes needs the script to start again");
  }
  const out = values.out as string;
  if (values.shard === undefined && existsSync(out) && readdirSync(out).length > 0) throw new UsageError("--out exists and is not empty");
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const excluded = (values["excluded-abilities"] ?? "Illusion").split(",").filter((name) => name !== "");
  const request: PoolRequest = {
    format_id: settings.format_id,
    generation: { excluded_abilities: excluded },
    ...(values.dataset === undefined ? {} : { team_dataset: { path: values.dataset, dataset_hash: values["dataset-hash"] as string } }),
  };
  const scratch = values.shard === undefined && processes > 1 ? mkdtempSync(join(tmpdir(), "areneon-teacher-shards-")) : undefined;
  const children: ShardProcess[] = [];
  let partialDir: string | undefined;
  try {
    // Start the further game processes first: each loads its own pool while this one does.
    if (scratch !== undefined) {
      for (let index = 1; index < processes; index++) {
        const shardOut = join(scratch, String(index));
        const rest = Object.entries(values).flatMap(([name, value]) =>
          value === undefined || ["out", "game-processes", "shard"].includes(name) ? [] : [`--${name}`, value],
        );
        children.push(startShard(shardScript as string, [...rest, "--out", shardOut, "--shard", `${index}/${processes}`], shardOut, index));
      }
    }
    const pool = poolSource(engine, request);
    if (values.shard !== undefined) {
      // A further game process: it ends if the process that started it dies (its stdin closes).
      process.stdin.once("end", () => process.exit(1));
      process.stdin.resume();
      (process.stdin as unknown as { unref?: () => void }).unref?.();
      const entries = await playTeacherEntries({ engine, settings, pool }, teacherContext(engine, settings), own);
      mkdirSync(out, { recursive: true });
      writeShardFile(join(out, SHARD_FILE), entries);
      return 0;
    }
    partialDir = join(dirname(out), `.${basename(out)}.partial`);
    const partial = partialDir;
    rmSync(partial, { recursive: true, force: true });
    mkdirSync(partial, { recursive: true });
    const others = Promise.all(children.map((child) => child.entries)).then((all) => all.flat() as TeacherEntry[]);
    const result = await playTeacherRun({ engine, settings, pool, stream_path: join(partial, "stream.jsonl"), ...(own === undefined ? {} : { shard: own }), others });
    writeTeacherRun(partial, result, request.team_dataset?.dataset_hash ?? null);
    if (existsSync(out)) rmdirSync(out); // empty (checked at the start): a rename onto it replaces it
    renameSync(partial, out);
    print(
      jcs({
        run_key: settings.run_key,
        teacher_policy_id: result.teacher.policy_id,
        teacher_identity_hash: result.teacher.identity_hash,
        games: result.games.length,
        episodes: result.episodes.length,
        examples: result.set.examples,
        training_set_hash: result.training_set_hash,
      } as Json),
    );
    return 0;
  } catch (error) {
    if (error instanceof GenerationError) {
      console.error(`teacher-run: ${error.message}`);
      return 1;
    }
    throw error;
  } finally {
    for (const child of children) child.stop();
    if (partialDir !== undefined) rmSync(partialDir, { recursive: true, force: true }); // gone already after a successful rename
    if (scratch !== undefined) rmSync(scratch, { recursive: true, force: true });
  }
}
