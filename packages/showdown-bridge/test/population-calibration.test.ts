// The anchor calibration of a population run (ADR-021 section 4, slice 7): the plan, the anchors a config names and a
// tiny calibration played on the pinned engine.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { calibrationGameStreams, loadEngine } from "../src/index.ts";
import { trainingTeamPool } from "../src/league-generation.ts";
import { runCalibrationCli } from "../src/population-cli.ts";
import { playCalibration, writeCalibration } from "../src/population-calibration.ts";
import { calibrationAnchors, NO_REFERENCE_REFUSAL, planCalibration, PopulationPlanError } from "../src/population-plan.ts";
import type { TrainingRunConfig } from "../src/training-run-config.ts";
import { populationConfig, populationOf } from "./population-support.ts";

const ANCHORS = [
  { actor: "random-legal", policy_id: "random-legal@v1", games: 2 },
  { actor: "actor-rule", policy_id: "actor-rule@v1", games: 2 },
] as const;

/** The tiny population config with the given baselines and `calibration_pairs`. */
function configWith(baselines: readonly { actor: "random-legal" | "actor-rule" | "damage-aware"; policy_id: string; games: number }[], pairs: number): TrainingRunConfig {
  const base = populationConfig(2, 2);
  return { ...base, generation: { ...base.generation, baseline_games: baselines.map((b) => ({ ...b })) }, population: { ...populationOf(base), rating: { ...populationOf(base).rating, calibration_pairs: pairs } } };
}

describe("calibrationAnchors", () => {
  it("lists the configured anchors sorted by policy id and needs actor-rule@v1", () => {
    expect(calibrationAnchors(configWith(ANCHORS, 2)).map((a) => a.policy_id)).toEqual(["actor-rule@v1", "random-legal@v1"]);
    const three = [...ANCHORS, { actor: "damage-aware", policy_id: "damage-aware@v1", games: 2 }] as const;
    expect(calibrationAnchors(configWith(three, 2)).map((a) => a.policy_id)).toEqual(["actor-rule@v1", "damage-aware@v1", "random-legal@v1"]);
    expect(() => calibrationAnchors(configWith([ANCHORS[0]], 2))).toThrow(new PopulationPlanError(NO_REFERENCE_REFUSAL));
  });

  it("leaves out a baseline that is no anchor: another policy id, another actor kind, or too few games", () => {
    const config = configWith([ANCHORS[1], { actor: "random-legal", policy_id: "random-legal@tiny", games: 2 }, { actor: "damage-aware", policy_id: "random-legal@v1", games: 2 }], 2);
    expect(calibrationAnchors(config).map((a) => a.policy_id)).toEqual(["actor-rule@v1"]);
    expect(() => calibrationAnchors(configWith([{ ...ANCHORS[1], games: 0 }], 2))).toThrow(PopulationPlanError);
  });
});

describe("planCalibration", () => {
  const config = configWith([...ANCHORS, { actor: "damage-aware", policy_id: "damage-aware@v1", games: 2 }], 3);
  const plan = planCalibration(config, 40);

  it("plays every pair of anchors for calibration_pairs team pairs, each twice with the anchors' seats swapped", () => {
    expect(plan).toHaveLength(2 * 3 * 3);
    const order = plan.filter((entry) => entry.half === 0).map((entry) => `${entry.a.policy_id}|${entry.b.policy_id}`);
    expect(order).toEqual([...Array(3).fill("actor-rule@v1|damage-aware@v1"), ...Array(3).fill("actor-rule@v1|random-legal@v1"), ...Array(3).fill("damage-aware@v1|random-legal@v1")]);
    for (const [index, entry] of plan.entries()) {
      expect([entry.game, entry.pair, entry.half]).toEqual([index, Math.floor(index / 2), index % 2]);
      expect(entry.seats.p1.policy_id).toBe(entry.half === 0 ? entry.a.policy_id : entry.b.policy_id);
      expect(entry.seats.p2.policy_id).toBe(entry.half === 0 ? entry.b.policy_id : entry.a.policy_id);
      expect(entry.teams).toEqual(plan[index - (index % 2)]?.teams);
    }
  });

  it("takes seeds, team draws and baseline streams from league-seed/1 at generation 0", () => {
    for (const entry of plan) {
      const streams = calibrationGameStreams(config.seeds.run_key, entry.pair);
      expect(entry.seed).toBe(streams.seed);
      expect(entry.teams).toEqual({ p1: streams.team_draws.p1 % 40, p2: streams.team_draws.p2 % 40 });
      expect(entry.baseline_seed).toBe(streams.baseline_seed);
    }
    expect(planCalibration(config, 40)).toEqual(plan);
  });

  it("refuses a config without population and an empty pool", () => {
    const { population: _population, ...bare } = config;
    expect(() => planCalibration(bare as TrainingRunConfig, 40)).toThrow(PopulationPlanError);
    expect(() => planCalibration(config, 0)).toThrow(RangeError);
  });
});

describe("a tiny calibration on the pinned engine", () => {
  const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  const config = configWith(ANCHORS, 2);
  const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
  const scratch = mkdtempSync(join(tmpdir(), "population-calibration-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));

  it("plays the plan, writes calibration.json and repeats byte for byte", async () => {
    const first = await playCalibration({ engine, config, pool });
    expect(first.anchors).toEqual(["actor-rule@v1", "random-legal@v1"]);
    expect(first.games).toHaveLength(2 * 2);
    for (const [index, row] of first.games.entries()) {
      expect([row.game, row.pair, row.half, row.a, row.b]).toEqual([index, Math.floor(index / 2), index % 2, "actor-rule@v1", "random-legal@v1"]);
      expect(row.seed).toBe(calibrationGameStreams(config.seeds.run_key, row.pair).seed);
      expect(["played", "forfeit"]).toContain(row.outcome);
    }
    expect(first.team_hashes).toEqual([...new Set(first.team_hashes)].sort());
    writeCalibration(join(scratch, "a"), first);
    writeCalibration(join(scratch, "b"), await playCalibration({ engine, config, pool }));
    const text = readFileSync(join(scratch, "a", "calibration.json"), "utf8");
    expect(text).toBe(readFileSync(join(scratch, "b", "calibration.json"), "utf8"));
    expect(JSON.parse(text)).toMatchObject({ schema_version: 1, format: "population-calibration/1", run_key: config.seeds.run_key, calibration_pairs: 2 });
  }, 120_000);

  it("is run by the CLI branch, which prints one line and reports a refused config", async () => {
    const lines: string[] = [];
    expect(await runCalibrationCli({ engine, config, pool, out: join(scratch, "cli") }, (line) => lines.push(line))).toBe(0);
    expect(JSON.parse(lines[0] as string)).toMatchObject({ generation: 0, anchors: ["actor-rule@v1", "random-legal@v1"], games: 4 });
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(await runCalibrationCli({ engine, config: configWith([ANCHORS[0]], 2), pool, out: join(scratch, "bad") }, () => undefined)).toBe(1);
    expect(error).toHaveBeenCalledWith(expect.stringContaining(NO_REFERENCE_REFUSAL));
    error.mockRestore();
  }, 120_000);
});
