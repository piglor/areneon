// The eval-dataset option of the progress evaluation (ADR-020 E-17.1): the validation split of another verified
// dataset, a second tiny one generated here from the committed pilot and other keys. In-process stand-in worker; the
// Python command and the real worker are driven by tests/ml/progress/test_eval_dataset.py.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  actorRule,
  jcs,
  type Json,
  type PolicyClient,
  playProgress,
  type ProgressOptions,
  progressGameStreams,
  progressPlan,
  progressRecordHash,
  readEvalDataset,
  readProgressEvaluationRecordV2,
  readTrainingRunConfig,
  runConfigHash,
  runProgressCli,
  teamHash,
  UsageError,
  withEvalDataset,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR, loadFixtureInputs, readFixtureRun, recordedPilot, regenerate, writeFixtures } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const OWN = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const scratch = mkdtempSync(join(tmpdir(), "progress-eval-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const raw = { ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, team_dataset: { path: OWN, dataset_hash: verifiedDatasetHash(engine, OWN, FORMAT) } };
const config = readTrainingRunConfig(raw);

// A second tiny dataset: the committed fixture's sizes and pilot under other keys.
const inputs = loadFixtureInputs(engine, FORMAT);
const SECOND = join(scratch, "second");
writeFixtures(SECOND, regenerate(inputs, { ...readFixtureRun(OWN), dataset_id: "eval-v1", generation_key: "11".repeat(32), assignment_key: "22".repeat(32) }, recordedPilot(inputs, OWN)));
const secondHash = verifiedDatasetHash(engine, SECOND, FORMAT);
const evalConfig = withEvalDataset(config, { path: SECOND, dataset_hash: secondHash });

const client = (): PolicyClient & { beginGame: () => Promise<void> } => ({
  beginGame: () => Promise.resolve(),
  ask(_battle, query) {
    const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
    const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
    return Promise.resolve({ schema_version: 1 as const, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" as const });
  },
});
const options = (pool = datasetValidationPool(engine, evalConfig), change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config, // the run's own, unchanged
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints: { preview: "1".repeat(64), battle: "2".repeat(64) },
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client: client(),
  pool,
  games: 2,
  record_games: 0,
  training_teams: new Set(),
  league_seeds: new Set(),
  ...change,
});

describe("the option's flags", () => {
  it("go together and take a lower-case SHA-256", () => {
    expect(readEvalDataset({})).toBeUndefined();
    expect(readEvalDataset({ "eval-dataset-dir": "d", "eval-dataset-hash": "a".repeat(64) })).toEqual({ path: "d", dataset_hash: "a".repeat(64) });
    expect(() => readEvalDataset({ "eval-dataset-dir": "d" })).toThrow(UsageError);
    expect(() => readEvalDataset({ "eval-dataset-hash": "a".repeat(64) })).toThrow(/go together/);
    expect(() => readEvalDataset({ "eval-dataset-dir": "d", "eval-dataset-hash": "A".repeat(64) })).toThrow(/64 lower-case/);
    expect(withEvalDataset(config, undefined)).toBe(config);
  });
});

describe("the evaluation dataset's validation pool", () => {
  it("is the second dataset's validation split, hash-checked, and shares no team with the run's own pool", () => {
    expect(secondHash).not.toBe(config.team_dataset?.dataset_hash);
    const pool = datasetValidationPool(engine, evalConfig);
    expect(pool.dataset_hash).toBe(secondHash);
    const own = new Set(datasetValidationPool(engine, config).teams.map((t) => t.canonical_team_hash));
    expect(pool.teams.length).toBeGreaterThan(0);
    for (const team of pool.teams) expect(own.has(team.canonical_team_hash)).toBe(false);
    expect(() => datasetValidationPool(engine, withEvalDataset(config, { path: SECOND, dataset_hash: "0".repeat(64) }))).toThrow(/hash is not the one/);
  });
});

describe("playProgress on the evaluation dataset", () => {
  it("names the evaluation dataset beside the run's own, and is byte-identical when run twice", async () => {
    const a = await playProgress(options());
    const b = await playProgress(options());
    expect(a.record.teams).toEqual({ dataset_hash: config.team_dataset?.dataset_hash, evaluation_dataset_hash: secondHash, split: "validation", pairs_used: 1 });
    expect(jcs(a.record as unknown as Json)).toBe(jcs(b.record as unknown as Json));
    expect(progressRecordHash(a.record)).toBe(progressRecordHash(b.record));
    expect(() => readProgressEvaluationRecordV2(JSON.parse(jcs(a.record as unknown as Json)))).not.toThrow();
    // The same plan on the run's own pool is a different record that does not name an evaluation dataset.
    const own = await playProgress(options(datasetValidationPool(engine, config)));
    expect(own.record.teams).toEqual({ dataset_hash: config.team_dataset?.dataset_hash, split: "validation", pairs_used: 1 });
    expect(progressRecordHash(own.record)).not.toBe(progressRecordHash(a.record));
  }, 240_000);

  it("keeps the refusals: a trained team, a committed plan's seed and more games than pairs", () => {
    const pool = datasetValidationPool(engine, evalConfig);
    const base = { config, pool, games: 2, training_teams: new Set<string>(), league_seeds: new Set<string>() };
    expect(() => progressPlan({ ...base, training_teams: new Set(pool.teams.map((t) => teamHash(t.sets))) })).toThrow(/run trained on it/);
    expect(() => progressPlan({ ...base, league_seeds: new Set([progressGameStreams(config.seeds.run_key, 0).seed]) })).toThrow(/equals a seed of a committed plan/);
    expect(() => progressPlan({ ...base, games: 2 * pool.teams.length })).toThrow(/disjoint team pairs/);
    expect(progressPlan(base).pairs).toHaveLength(1);
  });
});

describe("the progress command's flags", () => {
  const files = (): { config: string; flags: string[] } => {
    const configFile = join(scratch, "config.json");
    const teamSet = join(scratch, "team-set.json");
    writeFileSync(configFile, JSON.stringify(raw));
    writeFileSync(teamSet, JSON.stringify({ teams: [] }));
    const flags = ["--config", configFile, "--generation", "1", "--preview", "1".repeat(64), "--battle", "2".repeat(64), "--policy-id-sample", "policy:s", "--policy-id-argmax", "policy:a", "--generation-manifest-sha256", "a".repeat(64), "--team-set", teamSet, "--run-dir", scratch, "--engine-data", join(scratch, "none.json"), "--encoder-version", "enc1:x", "--worker-python", "none", "--games", "2", "--record-games", "0", "--out", join(scratch, "out")];
    return { config: configFile, flags };
  };
  const run = (extra: string[]) => runProgressCli([...files().flags, ...extra], () => undefined, datasetValidationPool);

  it("refuses one flag without the other, a malformed hash and the run's own dataset passed through the option", async () => {
    await expect(run(["--eval-dataset-dir", SECOND])).rejects.toThrow(/go together/);
    await expect(run(["--eval-dataset-hash", secondHash])).rejects.toThrow(/go together/);
    await expect(run(["--eval-dataset-dir", SECOND, "--eval-dataset-hash", "xyz"])).rejects.toThrow(/lower-case/);
    await expect(run(["--eval-dataset-dir", OWN, "--eval-dataset-hash", config.team_dataset?.dataset_hash as string])).rejects.toThrow(/run's own training dataset/);
  });

  // The other run of a head to head trained on a third dataset: the evaluation dataset must be clear of ITS dev and training teams too.
  describe("with another run", () => {
    const THIRD = join(scratch, "third");
    writeFixtures(THIRD, regenerate(inputs, { ...readFixtureRun(OWN), dataset_id: "other-v1", generation_key: "33".repeat(32), assignment_key: "44".repeat(32) }, recordedPilot(inputs, OWN)));
    const thirdHash = verifiedDatasetHash(engine, THIRD, FORMAT);
    const versusFlags = async (dataset: { path: string; dataset_hash: string }): Promise<{ flags: string[]; config: ReturnType<typeof readTrainingRunConfig> }> => {
      const other = readTrainingRunConfig({ ...raw, seeds: { ...(tiny.seeds as object), run_key: "other-run" }, team_dataset: dataset });
      const dir = join(scratch, `versus-${dataset.dataset_hash.slice(0, 8)}`);
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "run-config.json"), JSON.stringify({ ...raw, seeds: { ...(tiny.seeds as object), run_key: "other-run" }, team_dataset: dataset }));
      const teamSet = join(scratch, "other-team-set.json");
      writeFileSync(teamSet, JSON.stringify({ teams: [] }));
      const h = "a".repeat(64);
      const versus = `other-run,${await runConfigHash(other)},1,${h},${h},policy:s,policy:a`;
      return { flags: ["--engine-data", join(ROOT, "fixtures/showdown", FORMAT, "engine-data.json"), "--versus", versus, "--versus-dir", dir, "--versus-encoder-version", "enc", "--versus-team-set", teamSet, "--run-games", "2", "--eval-dataset-dir", SECOND, "--eval-dataset-hash", secondHash], config: other };
    };

    it("asks for the other run's teams too, and refuses an evaluation dataset that is its training dataset", async () => {
      const asked: string[] = [];
      const ports = { run_teams: (_e: unknown, c: { seeds: { run_key: string } }) => (asked.push(c.seeds.run_key), new Set<string>()) };
      const ok = await versusFlags({ path: THIRD, dataset_hash: thirdHash });
      // The worker does not exist here: the command stops at its start, after both runs' teams were read.
      await expect(runProgressCli([...files().flags, ...ok.flags], () => undefined, datasetValidationPool, ports as never)).rejects.toThrow();
      expect(asked).toEqual([config.seeds.run_key, "other-run"]);
      const same = await versusFlags({ path: SECOND, dataset_hash: secondHash });
      asked.length = 0;
      await expect(runProgressCli([...files().flags, ...same.flags], () => undefined, datasetValidationPool, ports as never)).rejects.toThrow(/other run's own training dataset/);
      expect(asked).toEqual([config.seeds.run_key]);
    }, 120_000);

    it("refuses a validation team that is in the other run's dev or training split", () => {
      const pool = datasetValidationPool(engine, evalConfig);
      const trained = new Set(pool.teams.map((t) => teamHash(t.sets)));
      expect(() => progressPlan({ config, pool, games: 2, training_teams: new Set(["f".repeat(64), ...trained]), league_seeds: new Set<string>() })).toThrow(/run trained on it/);
    });
  });
});
