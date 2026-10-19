import { describe, expect, it } from "vitest";
import {
  A2_BUDGET_CONFIG,
  battleBudget,
  type BudgetConfig,
  estimateRemainingDecisions,
  previewBudget,
  shadowCharge,
  shadowRequest,
  type ShadowTimer,
  startShadowTimer,
  type TimerSnapshot,
} from "../src/index.ts";
import { vgcTimer } from "./deadline-timer.ts";

const VGC = vgcTimer();

function snap(turn: number, bank: number, grace = 0): TimerSnapshot {
  return { timer_source: "server", turn_seconds_left: turn, bank_seconds_left: bank, grace_seconds_left: grace };
}

function shadow(turn: number, bank: number, grace = 0): TimerSnapshot {
  return { ...snap(turn, bank, grace), timer_source: "shadow" };
}

/** Rows are `[battle_turn, own_alive, opp_alive, own_bench]`; the first is Team Preview. */
type ScheduleRow = readonly [number, number, number, number];

/** `long_battle_schedule` of the #678 model: one faint every ~5 turns, plus forced switches. */
function longBattleSchedule(turns: number): ScheduleRow[] {
  const schedule: ScheduleRow[] = [[0, 4, 4, 2]];
  let [ownAlive, oppAlive, ownBench] = [4, 4, 2];
  for (let turn = 1; turn <= turns; turn++) {
    schedule.push([turn, ownAlive, oppAlive, ownBench]);
    if (turn % 10 === 5 && ownBench > 0) {
      ownAlive -= 1;
      ownBench -= 1;
      schedule.push([turn, ownAlive, oppAlive, ownBench]);
    } else if (turn % 10 === 0 && oppAlive > 1) {
      oppAlive -= 1;
    }
  }
  return schedule;
}

interface Row {
  turn_left: number;
  bank_before: number;
  remaining: number;
  submit_within_ms: number;
  policy_budget_ms: number;
  bank_after: number;
}

/** `worked_example` of the #678 model: every decision uses its full `submit_within`. */
function workedExample(schedule: readonly ScheduleRow[], latencyMs = 250): Row[] {
  let timer = startShadowTimer(VGC);
  return schedule.map(([turn, own_alive, opp_alive, own_bench]) => {
    const requested = shadowRequest(timer, turn);
    const s = requested.snapshot;
    const remaining = estimateRemainingDecisions({ own_alive, opp_alive, own_bench });
    const budget = turn === 0 ? previewBudget(s) : battleBudget(s, remaining);
    const elapsed = budget.submit_within_ms + latencyMs;
    expect(elapsed).toBeLessThan(s.turn_seconds_left * 1000);
    timer = shadowCharge(requested.timer, elapsed);
    return {
      turn_left: s.turn_seconds_left,
      bank_before: s.bank_seconds_left,
      remaining,
      ...budget,
      bank_after: Math.min(timer.total, VGC.starting),
    };
  });
}

/**
 * The #678 Python model's output (`timer_budget.py` at a01116c,
 * `worked_example(long_battle_schedule(40))`), in seconds as Python printed
 * them with `repr`: `[#, turn, T, bank before, R̂, submit_within, policy_budget, bank after]`.
 */
type PythonRow = readonly [number, number, number, number, number, number, number, number];

const PYTHON_ROWS: readonly PythonRow[] = [
  [0, 0, 90, 420, 14, 60.0, 59.0, 420],
  [1, 1, 55, 420, 14, 20.0, 19.0, 400],
  [2, 2, 55, 400, 14, 20.0, 19.0, 380],
  [3, 3, 55, 380, 14, 20.0, 19.0, 360],
  [4, 4, 55, 360, 14, 20.0, 19.0, 340],
  [5, 5, 55, 340, 14, 20.0, 19.0, 320],
  [6, 5, 55, 320, 12, 20.0, 19.0, 300],
  [7, 6, 55, 300, 12, 20.0, 19.0, 280],
  [8, 7, 55, 280, 12, 20.0, 19.0, 260],
  [9, 8, 55, 260, 12, 20.0, 19.0, 240],
  [10, 9, 55, 240, 12, 18.75, 17.75, 225],
  [11, 10, 55, 225, 12, 17.5, 16.5, 210],
  [12, 11, 55, 210, 10, 19.5, 18.5, 195],
  [13, 12, 55, 195, 10, 18.0, 17.0, 180],
  [14, 13, 55, 180, 10, 16.5, 15.5, 165],
  [15, 14, 55, 165, 10, 15.0, 14.0, 150],
  [16, 15, 55, 150, 10, 13.5, 12.5, 140],
  [17, 15, 55, 140, 8, 15.625, 14.625, 125],
  [18, 16, 55, 125, 8, 13.75, 12.75, 115],
  [19, 17, 55, 115, 8, 12.5, 11.5, 105],
  [20, 18, 55, 105, 8, 11.25, 10.25, 95],
  [21, 19, 55, 95, 8, 10.0, 9.0, 85],
  [22, 20, 55, 85, 8, 8.75, 7.75, 80],
  [23, 21, 55, 80, 6, 10.833333333333334, 9.833333333333334, 70],
  [24, 22, 55, 70, 6, 9.166666666666666, 8.166666666666666, 65],
  [25, 23, 55, 65, 6, 8.333333333333334, 7.333333333333334, 60],
  [26, 24, 55, 60, 6, 7.5, 6.5, 55],
  [27, 25, 55, 55, 6, 6.666666666666667, 5.666666666666667, 50],
  [28, 26, 50, 50, 6, 5.833333333333333, 4.833333333333333, 45],
  [29, 27, 45, 45, 6, 5.0, 4.0, 40],
  [30, 28, 40, 40, 6, 4.166666666666667, 3.166666666666667, 40],
  [31, 29, 40, 40, 6, 4.166666666666667, 3.166666666666667, 40],
  [32, 30, 40, 40, 6, 4.166666666666667, 3.166666666666667, 40],
  [33, 31, 40, 40, 5, 5.0, 4.0, 35],
  [34, 32, 35, 35, 5, 4.0, 3.0, 35],
  [35, 33, 35, 35, 5, 4.0, 3.0, 35],
  [36, 34, 35, 35, 5, 4.0, 3.0, 35],
  [37, 35, 35, 35, 5, 4.0, 3.0, 35],
  [38, 36, 35, 35, 5, 4.0, 3.0, 35],
  [39, 37, 35, 35, 5, 4.0, 3.0, 35],
  [40, 38, 35, 35, 5, 4.0, 3.0, 35],
  [41, 39, 35, 35, 5, 4.0, 3.0, 35],
  [42, 40, 35, 35, 5, 4.0, 3.0, 35],
];

/** A budget in whole ms is the Python model's seconds floored: never above it, less than 1 ms below. */
function expectFlooredMs(ms: number, pythonSeconds: number): void {
  expect(Number.isSafeInteger(ms)).toBe(true);
  expect(ms).toBeLessThanOrEqual(pythonSeconds * 1000);
  expect(pythonSeconds * 1000 - ms).toBeLessThan(1);
}

describe("worked example: the 40-turn battle of the research note", () => {
  const schedule = longBattleSchedule(40);
  const rows = workedExample(schedule);
  const rowAt = (i: number): Row => {
    const row = rows[i];
    if (row === undefined) throw new Error(`no decision ${i}`);
    return row;
  };

  it("plays the same schedule as the Python model", () => {
    expect(schedule.map(([turn]) => turn)).toEqual(PYTHON_ROWS.map((r) => r[1]));
    expect(rows).toHaveLength(PYTHON_ROWS.length);
  });

  it.each(PYTHON_ROWS.map((r) => [r[0], r] as const))("decision %i matches the Python model", (index, expected) => {
    const [, , turnLeft, bankBefore, remaining, submit, policy, bankAfter] = expected;
    const row = rowAt(index);
    expect(row).toMatchObject({
      turn_left: turnLeft,
      bank_before: bankBefore,
      remaining,
      bank_after: bankAfter,
    });
    expectFlooredMs(row.submit_within_ms, submit);
    expectFlooredMs(row.policy_budget_ms, policy);
  });

  it("gives the exact millisecond budgets of the research note's table", () => {
    const pick = (i: number) => [rowAt(i).submit_within_ms, rowAt(i).policy_budget_ms];
    expect(pick(0)).toEqual([60_000, 59_000]);
    expect(pick(10)).toEqual([18_750, 17_750]);
    expect(pick(17)).toEqual([15_625, 14_625]);
    expect(pick(23)).toEqual([10_833, 9_833]);
    expect(pick(30)).toEqual([4_166, 3_166]);
    expect(pick(42)).toEqual([4_000, 3_000]);
  });

  it("keeps the bank reserve and settles at 35 s", () => {
    expect(Math.min(...rows.map((r) => r.bank_after))).toBeGreaterThanOrEqual(A2_BUDGET_CONFIG.bank_reserve_ms / 1000);
    expect(rowAt(rows.length - 1).bank_after).toBe(35);
  });
});

describe("shadow timer", () => {
  it("reads the VGC Timer from the pinned engine", () => {
    expect(VGC).toEqual({ starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 });
  });

  it("matches the server at preview and clips grace on turn 1", () => {
    let { timer, snapshot } = shadowRequest(startShadowTimer(VGC), 0);
    expect(snapshot).toEqual(shadow(90, 420, 90));
    // 85 s of preview comes out of grace: the turn-1 clip restores 420.
    ({ timer, snapshot } = shadowRequest(shadowCharge(timer, 85_000), 1));
    expect(snapshot).toEqual(shadow(55, 420, 0));
    // A forced switch on the same turn: the 55 s cap, no refill.
    ({ snapshot } = shadowRequest(shadowCharge(timer, 12_000), 1));
    expect(snapshot).toEqual(shadow(55, 410, 0));
  });

  it("charges whole ticks only", () => {
    const timer = startShadowTimer(VGC);
    expect(shadowCharge(timer, 4_990).total).toBe(510);
    expect(shadowCharge(timer, 5_000).total).toBe(505);
    expect(shadowCharge(timer, 14_900).total).toBe(500);
  });

  it("caps the turn at the bank once the bank is low", () => {
    let { timer } = shadowRequest(startShadowTimer(VGC), 0);
    ({ timer } = shadowRequest(timer, 1));
    const { snapshot } = shadowRequest(shadowCharge(timer, 400_000), 2);
    expect(snapshot).toEqual(shadow(20, 20, 0));
  });

  it("gives the first request the first-turn cap when it is a battle turn", () => {
    const { timer, snapshot } = shadowRequest(startShadowTimer(VGC), 3);
    expect(snapshot).toEqual(shadow(90, 420, 90));
    expect(shadowRequest(timer, 3).snapshot).toEqual(shadow(55, 420, 90));
  });

  it("adds add_per_turn on a new turn, never above starting", () => {
    const rules = { ...VGC, add_per_turn: 10 };
    let { timer } = shadowRequest(startShadowTimer(rules), 1);
    ({ timer } = shadowRequest(shadowCharge(timer, 100_000), 2));
    expect(timer.total).toBe(420);
    ({ timer } = shadowRequest(shadowCharge(timer, 100_000), 3));
    expect(timer.total).toBe(330);
  });

  it("stops the total at 0", () => {
    const { snapshot } = shadowRequest(shadowCharge(startShadowTimer(VGC), 600_000), 0);
    expect(snapshot).toEqual(shadow(0, 0, 0));
  });

  it("returns new timers and leaves the old one unchanged", () => {
    const start = startShadowTimer(VGC);
    const after = shadowCharge(shadowRequest(start, 0).timer, 10_000);
    expect(start).toEqual<ShadowTimer>({ rules: VGC, total: 510, turn: null, first_request: true });
    expect(after.total).toBe(500);
    expect(Object.isFrozen(start.rules)).toBe(true);
  });

  it.each([
    [{ ...VGC, tick: 0 }, /tick must be an integer >= 1/],
    [{ ...VGC, starting: -1 }, /starting must be an integer >= 0/],
    [{ ...VGC, max_per_turn: 5.5 }, /max_per_turn/],
  ])("refuses timer rules %j", (rules, message) => {
    expect(() => startShadowTimer(rules)).toThrow(message);
  });

  it("refuses a negative battle turn and a bad elapsed time", () => {
    const timer = startShadowTimer(VGC);
    expect(() => shadowRequest(timer, -1)).toThrow(/battle_turn/);
    expect(() => shadowCharge(timer, -1)).toThrow(RangeError);
    expect(() => shadowCharge(timer, Number.NaN)).toThrow(/finite/);
  });

  it("confirms the premise: a flat 20 s per decision exhausts the bank by turn 21", () => {
    let { timer } = shadowRequest(startShadowTimer(VGC), 0);
    let turn = 0;
    for (;;) {
      turn += 1;
      const requested = shadowRequest(timer, turn);
      if (requested.snapshot.turn_seconds_left <= 20) break;
      timer = shadowCharge(requested.timer, 20_250);
    }
    expect(turn).toBe(21);
  });
});

describe("previewBudget", () => {
  it("is capped at 60 s", () => {
    expect(previewBudget(snap(90, 420, 90))).toEqual({ submit_within_ms: 60_000, policy_budget_ms: 59_000 });
  });

  it("respects a short first request", () => {
    expect(previewBudget(snap(30, 420, 90)).submit_within_ms).toBe(27_000);
  });

  it("does not depend on the bank", () => {
    expect(previewBudget(snap(90, 5, 90))).toEqual(previewBudget(snap(90, 420, 90)));
  });
});

describe("battleBudget", () => {
  it("is capped at 20 s early", () => {
    expect(battleBudget(snap(55, 420), 14)).toEqual({ submit_within_ms: 20_000, policy_budget_ms: 19_000 });
  });

  it("shares the bank over the remaining decisions", () => {
    expect(battleBudget(snap(55, 115), 10).submit_within_ms).toBe(10_000);
  });

  it("skips the policy when only the reserve is left", () => {
    expect(battleBudget(snap(15, 15), 6)).toEqual({
      submit_within_ms: A2_BUDGET_CONFIG.fallback_reserve_ms,
      policy_budget_ms: 0,
    });
  });

  it("never exceeds the turn cap minus the margin", () => {
    expect(battleBudget(snap(5, 5), 1).submit_within_ms).toBe(1_000);
    expect(battleBudget(snap(2, 2), 1)).toEqual({ submit_within_ms: 0, policy_budget_ms: 0 });
  });

  it("takes the safety margin from the configuration", () => {
    const config: BudgetConfig = { ...A2_BUDGET_CONFIG, safety_margin_ms: 40_000 };
    expect(battleBudget(snap(55, 420), 14, config).submit_within_ms).toBe(15_000);
    expect(previewBudget(snap(90, 420, 90), config).submit_within_ms).toBe(50_000);
  });

  it.each([0, -1, 1.5, Number.NaN])("refuses %d remaining decisions", (remaining) => {
    expect(() => battleBudget(snap(55, 420), remaining)).toThrow(/remaining_decisions must be an integer >= 1/);
  });

  it.each([
    [snap(-1, 420), /turn_seconds_left/],
    [snap(55, 4.5), /bank_seconds_left/],
    [snap(55, 420, Number.POSITIVE_INFINITY), /grace_seconds_left/],
  ])("refuses the snapshot %j", (s, message) => {
    expect(() => battleBudget(s, 4)).toThrow(message);
    expect(() => previewBudget(s)).toThrow(message);
  });

  it.each([
    [{ safety_margin_ms: -1 }, /safety_margin_ms/],
    [{ battle_cap_ms: 0.5 }, /battle_cap_ms/],
    [{ min_remaining_decisions: 0 }, /min_remaining_decisions/],
    [{ turns_per_pokemon: Number.NaN }, /turns_per_pokemon/],
    [{ turns_per_pokemon: -1 }, /turns_per_pokemon/],
  ])("refuses the configuration %j", (override, message) => {
    const config = { ...A2_BUDGET_CONFIG, ...override };
    expect(() => battleBudget(snap(55, 420), 4, config)).toThrow(message);
    expect(() => previewBudget(snap(90, 420, 90), config)).toThrow(message);
    expect(() => estimateRemainingDecisions({ own_alive: 1, opp_alive: 1, own_bench: 0 }, config)).toThrow(message);
  });
});

describe("estimateRemainingDecisions", () => {
  it("matches the Python model", () => {
    expect(estimateRemainingDecisions({ own_alive: 4, opp_alive: 4, own_bench: 2 })).toBe(14);
    expect(estimateRemainingDecisions({ own_alive: 1, opp_alive: 1, own_bench: 0 })).toBe(
      A2_BUDGET_CONFIG.min_remaining_decisions,
    );
  });

  it.each(["own_alive", "opp_alive", "own_bench"] as const)("refuses a bad %s", (key) => {
    const counts = { own_alive: 2, opp_alive: 2, own_bench: 0, [key]: -1 };
    expect(() => estimateRemainingDecisions(counts)).toThrow(key);
  });

  it("is frozen as a default", () => {
    expect(Object.isFrozen(A2_BUDGET_CONFIG)).toBe(true);
  });
});
