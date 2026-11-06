// The outcomes of a calibration game that a real scripted game rarely produces: a forfeit, an incident and a played game without a record.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { populationConfig, populationOf } from "./population-support.ts";
import type { TrainingRunConfig } from "../src/training-run-config.ts";

const outcomes: unknown[] = [];
vi.mock("../src/policy-actor.ts", () => ({ policyGame: () => Promise.resolve(outcomes.shift()) }));

const { loadEngine } = await import("../src/index.ts");
const { trainingTeamPool } = await import("../src/league-generation.ts");
const { playCalibration } = await import("../src/population-calibration.ts");
const { runCalibrationCli } = await import("../src/population-cli.ts");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const base = populationConfig(2, 2);
const config: TrainingRunConfig = {
  ...base,
  generation: { ...base.generation, baseline_games: [{ actor: "random-legal", policy_id: "random-legal@v1", games: 2 }, { actor: "actor-rule", policy_id: "actor-rule@v1", games: 2 }] },
  population: { ...populationOf(base), rating: { ...populationOf(base).rating, calibration_pairs: 1 } },
};
const pool = trainingTeamPool(engine, config.generation.excluded_abilities);
const played = { outcome: { kind: "played", winner: null, turns: 3 }, record: { winner: "p1", turns: 3 } };

beforeEach(() => {
  outcomes.length = 0;
});

describe("calibration game outcomes", () => {
  it("records a forfeit with its loser, cause and the other side as winner", async () => {
    outcomes.push({ outcome: { kind: "forfeit", loser: "p1", winner: "p2", turns: 2, cause: "timeout", request_seq: 1, detail: "x" }, record: null }, played);
    const result = await playCalibration({ engine, config, pool });
    expect(result.games[0]).toMatchObject({ outcome: "forfeit", winner: "p2", forfeit: { loser: "p1", cause: "timeout" } });
    expect(result.games[1]).toMatchObject({ outcome: "played", winner: "p1", turns: 3 });
  });

  it("fails on an incident and on a played game with no record", async () => {
    outcomes.push({ outcome: { kind: "incident", incident: "engine_error", message: "boom" }, record: null });
    await expect(playCalibration({ engine, config, pool })).rejects.toThrow(/calibration game 0 .*engine_error: boom/);
    outcomes.push({ outcome: { kind: "played", winner: null, turns: 1 }, record: null });
    await expect(playCalibration({ engine, config, pool })).rejects.toThrow(/a played game has no record/);
  });

  it("reports a failed calibration as exit 1 and lets another error through", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    outcomes.push({ outcome: { kind: "incident", incident: "engine_error", message: "boom" }, record: null });
    expect(await runCalibrationCli({ engine, config, pool, out: "/tmp/population-calibration-outcomes" }, () => undefined)).toBe(1);
    error.mockRestore();
    outcomes.push(undefined);
    await expect(runCalibrationCli({ engine, config, pool, out: "/tmp/population-calibration-outcomes" }, () => undefined)).rejects.toThrow(TypeError);
  });
});
