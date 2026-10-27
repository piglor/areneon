// The scripted head-to-head (ADR-025 section 13, #1130): two scripted baselines on the fixture dataset's validation pairs,
// every pair from both seats. Few pairs only: the full experiment is its own ticket.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildScriptedRecord,
  H2H_FILE,
  playScriptedGames,
  progressGameStreams,
  readScriptedH2HRecord,
  readTrainingRunConfig,
  runScriptedH2HCli,
  type ScriptedArm,
  ScriptedH2HError,
  scriptedArm,
  UsageError,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { beliefWorldsFactory } from "../src/damage-aware-worlds.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
const raw = { ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } };
const config = readTrainingRunConfig(raw);
const pool = datasetValidationPool(engine, config);
const scratch = mkdtempSync(join(tmpdir(), "h2h-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const configFile = join(scratch, "config.json");
writeFileSync(configFile, JSON.stringify(raw));

const context = { engine, formatId: FORMAT, budget: { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms }, beliefWorlds: beliefWorldsFactory };
const arms = (a: string, b: string): { a: ScriptedArm; b: ScriptedArm } => ({ a: scriptedArm(a, context), b: scriptedArm(b, context) });
const recordOf = async (pairs: number, pair: { a: ScriptedArm; b: ScriptedArm }) => {
  const rows = await playScriptedGames({ engine, config, pool, pairs, arms: pair });
  return buildScriptedRecord({ engine, config, pool, pairs, arms: pair, rows });
};

describe("a head-to-head of two scripted baselines", () => {
  it("plays each pair from both seats on the progress plan and reads back as a record", async () => {
    const pair = arms("actor-rule@v1", "damage-aware@v1");
    const record = await recordOf(2, pair);
    expect(record.plan).toMatchObject({ games: 4, seed_derivation: "progress-seed/1", pairing: "progress-pairing/1" });
    expect(record.games.map((g) => [g.game, g.pair, g.a_seat])).toEqual([[0, 0, "p1"], [1, 0, "p2"], [2, 1, "p1"], [3, 1, "p2"]]);
    expect(record.result.wins_a + record.result.wins_b + record.result.ties).toBe(4);
    expect(record.arms.a.identity_hash).toBeUndefined();
    expect(record.arms.b.identity_hash).toBe(pair.b.identity_hash);
    expect(record.arms.b.counts?.decisions).toBeGreaterThan(0);
    expect(record.arms.b.belief).toBeUndefined();
    expect(readScriptedH2HRecord(JSON.parse(JSON.stringify(record)))).toEqual(record);
    expect(progressGameStreams(config.seeds.run_key, 0).seed).toMatch(/^sodium,/); // the plan's seeds are the progress evaluation's
  });

  it("is byte-identical on rerun and for any split of the games over processes", async () => {
    const pair = arms("random-legal@v1", "damage-aware@v1");
    const whole = JSON.stringify(await recordOf(2, pair));
    expect(JSON.stringify(await recordOf(2, pair))).toBe(whole);
    const slices = await Promise.all([0, 1, 2].map((index) => playScriptedGames({ engine, config, pool, pairs: 2, arms: pair, shard: { index, of: 3 } })));
    expect(slices.map((rows) => rows.map((r) => r.game))).toEqual([[0, 3], [1], [2]]);
    expect(JSON.stringify(buildScriptedRecord({ engine, config, pool, pairs: 2, arms: pair, rows: slices.flat().reverse() }))).toBe(whole);
  });

  it("counts damage-aware@v2's belief and its refusals and fallbacks, and names both baselines by identity", async () => {
    const pair = arms("damage-aware@v1", "damage-aware@v2");
    const record = await recordOf(1, pair);
    const v1 = record.arms.a;
    const v2 = record.arms.b;
    expect(v1.identity_hash).not.toBe(v2.identity_hash);
    expect(v1.belief).toBeUndefined();
    expect(v2.belief).toMatchObject({ worlds: 8, identity_hash: pair.b.belief?.identity_hash });
    expect(v2.belief?.totals.decisions).toBeGreaterThan(0);
    expect(v2.belief?.totals.worlds_sampled).toBeGreaterThan(0);
    // the generator-matched prior and dataset teams: the belief refuses nothing here (ADR-025 section 13 expects 0)
    expect(v2.belief?.totals.refusals).toEqual([]);
    expect(v2.counts?.reader_fallbacks).toBe(0);
    expect(record.games.every((g) => g.a_belief === undefined && g.b_belief !== undefined && g.a_counts !== undefined && g.b_counts !== undefined)).toBe(true);
  }, 600_000);

  it("fails as a gate when the declared wall-time cap passes, with an injected clock", async () => {
    let now = 0;
    const clock = (): number => (now += 100);
    const pair = arms("actor-rule@v1", "random-legal@v1");
    await expect(playScriptedGames({ engine, config, pool, pairs: 2, arms: pair, wall_time_cap_ms: 150, clock })).rejects.toThrow(/wall-time cap of 150 ms passed after 2 games.*nothing is committed/);
    now = 0;
    const rows = await playScriptedGames({ engine, config, pool, pairs: 2, arms: pair, wall_time_cap_ms: 10_000, clock }); // 8 clock reads of 100 ms stay inside
    expect(rows).toHaveLength(4);
  });

  it("fails the run on a harness incident and refuses bad requests", async () => {
    const broken: ScriptedArm = {
      policy_id: "random-legal@v1",
      seat: () => ({
        actor: () => {
          throw new Error("boom");
        },
      }),
    };
    await expect(playScriptedGames({ engine, config, pool, pairs: 1, arms: { a: arms("actor-rule@v1", "random-legal@v1").a, b: broken } })).rejects.toThrow(/game 0: .*boom/);
    const same = arms("actor-rule@v1", "random-legal@v1");
    await expect(playScriptedGames({ engine, config, pool, pairs: 1, arms: { a: same.a, b: same.a } })).rejects.toThrow(/different baselines/);
    await expect(playScriptedGames({ engine, config, pool, pairs: 0, arms: same })).rejects.toThrow(ScriptedH2HError);
    await expect(playScriptedGames({ engine, config, pool, pairs: 10_000, arms: same })).rejects.toThrow(/disjoint team pairs/);
    await expect(playScriptedGames({ engine, config, pool, pairs: 1, arms: same, shard: { index: 3, of: 3 } })).rejects.toThrow(/INDEX\/COUNT/);
    await expect(playScriptedGames({ engine, config, pool, pairs: 1, arms: same, wall_time_cap_ms: 0 })).rejects.toThrow(/positive whole number/);
    expect(() => scriptedArm("learned@v1", context)).toThrow(/not a scripted baseline/);
    expect(() => scriptedArm("damage-aware@v1", { ...context, formatId: "gen9other" })).toThrow(/damage-aware@v1: .*vocabulary/);
    expect(() => scriptedArm("damage-aware@v2", { ...context, beliefWorlds: undefined })).toThrow(/belief port/);
    expect(() => buildScriptedRecord({ engine, config, pool, pairs: 2, arms: same, rows: [] })).toThrow(/missing or was played twice/);
  });
});

describe("the command", () => {
  const run = async (args: string[], out: string, extra: string[] = [], script?: string) => {
    const lines: string[] = [];
    const code = await runScriptedH2HCli(["--config", configFile, "--arm-a", "actor-rule@v1", "--arm-b", "random-legal@v1", "--pairs", "2", "--out", out, ...extra, ...args], (l) => lines.push(l), datasetValidationPool, beliefWorldsFactory, script);
    const file = join(out, H2H_FILE);
    return { code, line: lines[0], bytes: existsSync(file) ? readFileSync(file, "utf8") : "" };
  };

  it("writes the record, prints one line, and is the same for one process and for several", async () => {
    const one = await run([], join(scratch, "one"), ["--wall-time-cap-ms", "3600000"]);
    expect(one.code).toBe(0);
    const summary = JSON.parse(one.line as string) as { games: number; record_hash: string };
    expect(summary.games).toBe(4);
    expect(readScriptedH2HRecord(JSON.parse(one.bytes)).plan.wall_time_cap_ms).toBe(3_600_000);
    const two = await run(["--game-processes", "2"], join(scratch, "two"), ["--wall-time-cap-ms", "3600000"], join(ROOT, "packages/team-dataset/scripts/scripted-h2h.ts"));
    expect(two.code).toBe(0);
    expect(two.bytes).toBe(one.bytes);
    expect(JSON.parse(two.line as string).record_hash).toBe(summary.record_hash);
  }, 300_000);

  it("plays only its slice as a further game process and leaves it for the process that started it", async () => {
    const out = join(scratch, "slice");
    const { code, bytes } = await run(["--shard", "1/2"], out);
    expect(code).toBe(0);
    expect(bytes).toBe(""); // no head-to-head.json in a slice's directory: the parent merges its shard.jsonl
    const rows = readFileSync(join(out, "shard.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l) as { game: number });
    expect(rows.map((r) => r.game)).toEqual([1, 3]);
  }, 120_000);

  it("exits 1 for a config with no team dataset and rethrows an error it does not know", async () => {
    const bare = join(scratch, "bare.json");
    const { team_dataset: _dropped, ...without } = raw;
    writeFileSync(bare, JSON.stringify(without));
    const lines: string[] = [];
    const err = console.error;
    console.error = () => undefined;
    try {
      const args = ["--config", bare, "--arm-a", "actor-rule@v1", "--arm-b", "random-legal@v1", "--pairs", "1", "--out", join(scratch, "bare")];
      expect(await runScriptedH2HCli(args, (l) => lines.push(l), datasetValidationPool, beliefWorldsFactory)).toBe(1);
    } finally {
      console.error = err;
    }
    const boom = (): never => {
      throw new Error("unexpected");
    };
    await expect(runScriptedH2HCli(["--config", configFile, "--arm-a", "actor-rule@v1", "--arm-b", "random-legal@v1", "--pairs", "1", "--out", join(scratch, "boom")], () => undefined, boom)).rejects.toThrow("unexpected");
  });

  it("takes another dataset, exits 1 on a refusal and throws UsageError on bad flags", async () => {
    const err = console.error;
    console.error = () => undefined;
    try {
      expect((await run([], join(scratch, "wrong-hash"), ["--eval-dataset-dir", SOURCE, "--eval-dataset-hash", "0".repeat(64)])).code).toBe(1);
      expect((await run([], join(scratch, "too-many"), ["--pairs", "100000"])).code).toBe(1);
    } finally {
      console.error = err;
    }
    mkdirSync(join(scratch, "x"), { recursive: true });
    await expect(run([], join(scratch, "x"), ["--eval-dataset-dir", SOURCE])).rejects.toThrow(UsageError);
    await expect(run([], join(scratch, "y"), ["--pairs", "0"])).rejects.toThrow(UsageError);
    await expect(run([], join(scratch, "y"), ["--game-processes", "2"])).rejects.toThrow(/script to start again/);
    await expect(run([], join(scratch, "y"), ["--shard", "2/2"])).rejects.toThrow(/INDEX\/COUNT/);
    await expect(run([], join(scratch, "y"), ["--shard", "one/two"])).rejects.toThrow(/INDEX\/COUNT/);
    const placeholder = join(scratch, "placeholder.json");
    writeFileSync(placeholder, JSON.stringify({ ...raw, seeds: { ...((tiny.seeds as object)), run_key: "CHANGE-ME-run" } }));
    await expect(runScriptedH2HCli(["--config", placeholder, "--arm-a", "actor-rule@v1", "--arm-b", "random-legal@v1", "--pairs", "1", "--out", join(scratch, "ph")], () => undefined, datasetValidationPool)).rejects.toThrow(/CHANGE-ME/);
    await expect(run([], join(scratch, "y"), ["--bogus", "1"])).rejects.toThrow(UsageError);
    await expect(runScriptedH2HCli(["--pairs", "1"], () => undefined, datasetValidationPool)).rejects.toThrow(/--config is required/);
  });
});
