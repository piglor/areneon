// The evaluation-key option of the progress evaluation (#1177): a declared key replaces the run key in
// `progress-pairing/1` and `progress-seed/1`, so two runs evaluated with one key play the same pairs, seats, seeds and
// baseline streams. In-process stand-in worker; the Python command is driven by tests/ml/progress/test_evaluation_key.py.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  actorRule,
  jcs,
  type Json,
  type PolicyClient,
  playProgress,
  type ProgressOptions,
  ProgressError,
  progressGameStreams,
  progressPlan,
  progressRecordHash,
  readProgressEvaluationRecordV2,
  readProgressGameRecord,
  readTrainingRunConfig,
  teamHash,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const OWN = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "progress-key-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as { seeds: Record<string, unknown>; generation: object };
const runConfig = (run_key: string) => readTrainingRunConfig({ ...tiny, seeds: { ...tiny.seeds, run_key }, generation: { ...tiny.generation, max_turns: 100 }, team_dataset: { path: OWN, dataset_hash: verifiedDatasetHash(engine, OWN, FORMAT) } });
const configA = runConfig("key-run-a");
const configB = runConfig("key-run-b");
const pool = datasetValidationPool(engine, configA);
const PAIRS = Math.min(2, Math.floor(pool.teams.length / 2));
const KEY = "paired-eval-1";

const client = (): PolicyClient & { beginGame: () => Promise<void> } => ({
  beginGame: () => Promise.resolve(),
  ask(_battle, query) {
    const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
    const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
    return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
  },
});
const options = (config: typeof configA, change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config,
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints: { preview: "1".repeat(64), battle: "2".repeat(64) },
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client: client(),
  pool,
  games: 2 * PAIRS,
  record_games: 2 * PAIRS,
  training_teams: new Set(),
  league_seeds: new Set(),
  ...change,
});

describe("the plan", () => {
  it("is a function of the key and the teams alone, and of the run key without a key", () => {
    const base = { pool, games: 2 * PAIRS, training_teams: new Set<string>(), league_seeds: new Set<string>() };
    const a = progressPlan({ ...base, config: configA, evaluation_key: KEY });
    const b = progressPlan({ ...base, config: configB, evaluation_key: KEY });
    expect(b).toEqual(a);
    expect(a.seeds[0]).toBe(progressGameStreams(KEY, 0).seed);
    // Negative control: without the key the two runs plan differently.
    expect(progressPlan({ ...base, config: configB }).seeds).not.toEqual(progressPlan({ ...base, config: configA }).seeds);
    expect(progressPlan({ ...base, config: configA }).seeds[0]).toBe(progressGameStreams("key-run-a", 0).seed);
  });

  it("checks the key's seeds against the committed plans, not the run key's", () => {
    const base = { pool, games: 2, training_teams: new Set<string>() };
    expect(() => progressPlan({ ...base, config: configA, evaluation_key: KEY, league_seeds: new Set([progressGameStreams(KEY, 0).seed]) })).toThrow(/equals a seed of a committed plan/);
    expect(progressPlan({ ...base, config: configA, evaluation_key: KEY, league_seeds: new Set([progressGameStreams("key-run-a", 0).seed]) }).seeds).toHaveLength(1);
  });
});

describe("playProgress on one key", () => {
  it("gives two runs the same pair, seat, seed and team hashes in every stored game, and the same baseline stream", async () => {
    const a = await playProgress(options(configA, { evaluation_key: KEY }));
    const b = await playProgress(options(configB, { evaluation_key: KEY }));
    expect(a.record.run_id).toBe("key-run-a");
    expect(b.record.run_id).toBe("key-run-b");
    expect(a.record.plan.evaluation_key).toBe(KEY);
    expect(a.game_records.size).toBeGreaterThan(0);
    expect([...b.game_records.keys()]).toEqual([...a.game_records.keys()]);
    const seats = (r: typeof a.record) => r.games.map((g) => [g.match_id, g.game, g.pair, g.candidate_seat]);
    expect(seats(b.record)).toEqual(seats(a.record));
    for (const [name, left] of a.game_records) {
      const right = b.game_records.get(name);
      if (right === undefined) throw new Error(`${name} missing`);
      expect(left.evaluation_key).toBe(KEY);
      expect(readProgressGameRecord(JSON.parse(jcs(left as unknown as Json))).seed).toBe(left.seed);
      expect([right.seed, right.pair, right.teams.p1.team_hash, right.teams.p2.team_hash]).toEqual([left.seed, left.pair, left.teams.p1.team_hash, left.teams.p2.team_hash]);
      if (name.startsWith("random-legal-")) expect(right.omniscient_log_sha256).toBe(left.omniscient_log_sha256); // the same baseline stream
    }
    expect(readProgressEvaluationRecordV2(JSON.parse(jcs(a.record as unknown as Json))).plan.evaluation_key).toBe(KEY);
    // Negative control: without the key the runs differ in seed (so the pairing is not shared).
    const plainA = await playProgress(options(configA));
    const plainB = await playProgress(options(configB));
    expect(plainA.record.plan).not.toHaveProperty("evaluation_key");
    const firstSeed = (r: typeof plainA) => [...r.game_records.values()][0]?.seed;
    expect(firstSeed(plainB)).not.toBe(firstSeed(plainA));
    expect(firstSeed(plainA)).not.toBe(firstSeed(a));
    for (const game of plainA.game_records.values()) expect(game).not.toHaveProperty("evaluation_key");
    // Byte-identical rerun, and the key changes the record.
    expect(progressRecordHash((await playProgress(options(configA, { evaluation_key: KEY }))).record)).toBe(progressRecordHash(a.record));
    expect(progressRecordHash(plainA.record)).not.toBe(progressRecordHash(a.record));
  }, 480_000);

  it("keeps the teams of the pairs the plan names", async () => {
    const plan = progressPlan({ config: configA, pool, games: 2 * PAIRS, training_teams: new Set(), league_seeds: new Set(), evaluation_key: KEY });
    const result = await playProgress(options(configA, { evaluation_key: KEY, games: 2, record_games: 2 }));
    const game = [...result.game_records.values()][0];
    const [first, second] = plan.pairs[0] as NonNullable<(typeof plan.pairs)[number]>;
    expect([game?.teams.p1.team_hash, game?.teams.p2.team_hash]).toEqual([teamHash(first.sets), teamHash(second.sets)]);
  }, 240_000);

  it("refuses the run's own key and a malformed one", async () => {
    await expect(playProgress(options(configA, { evaluation_key: "key-run-a" }))).rejects.toThrow(ProgressError);
    await expect(playProgress(options(configA, { evaluation_key: "key-run-a" }))).rejects.toThrow(/omit the option/);
    await expect(playProgress(options(configA, { evaluation_key: "not a token" }))).rejects.toThrow(/id token/);
  });
});
