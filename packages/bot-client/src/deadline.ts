/**
 * Deadline budget for one decision under the VGC Timer (ADR-005 §3,
 * docs/research/timer-deadline-budget.md). Pure functions: the timer state in,
 * `submit_within_ms` and `policy_budget_ms` out, both measured from local
 * receipt of the request.
 *
 * This is the TypeScript port of the #678 model (`timer_budget.py` at
 * a01116c) with the same semantics. Durations are whole milliseconds, as in
 * the `DecisionContext` contract. The bank allocation `(B − B_res) / R̂` is
 * floored to a whole millisecond, so a budget is never above the Python
 * model's and at most 1 ms below it.
 *
 * The timer settings are an input: they come from the format's rule table in
 * the pinned engine (`timerstarting`, `timergrace`, ...) plus the server's
 * 5 s tick, never from constants here.
 */

import type { DecisionContext, TimerRules } from "@areneon/schema/types/v1/decision-context.ts";

export type { TimerRules };

const MS_PER_SECOND = 1000;

/** The player's own timer state when a request arrives (research note §1.5). */
export type TimerSnapshot = Pick<
  DecisionContext,
  "timer_source" | "turn_seconds_left" | "bank_seconds_left" | "grace_seconds_left"
>;

/**
 * The two allowances of one decision (research note §2). The final choice,
 * from the policy or the fallback, must be sent within `submit_within_ms`;
 * the policy must answer within `policy_budget_ms`, and 0 means skip it and
 * send the fallback at once.
 */
export type Budget = Pick<DecisionContext, "submit_within_ms" | "policy_budget_ms">;

/** Areneon's budget policy (research note §2.1). The caps are upper bounds, never targets. */
export interface BudgetConfig {
  /** ADR-005 upper bound for Team Preview. */
  preview_cap_ms: number;
  /** ADR-005 upper bound for a battle decision. */
  battle_cap_ms: number;
  /** `m`: kept clear of the server's turn timeout for latency and stalls. */
  safety_margin_ms: number;
  /** `f`: time to send the precomputed fallback. */
  fallback_reserve_ms: number;
  /** `B_res`: bank that planned spending never touches. */
  bank_reserve_ms: number;
  /** Floor for `R̂`. */
  min_remaining_decisions: number;
  /** Turns per remaining Pokémon in the `R̂` estimate (deliberately pessimistic). */
  turns_per_pokemon: number;
}

/** The A2 (local server) configuration: a 3 s safety margin. */
export const A2_BUDGET_CONFIG: Readonly<BudgetConfig> = Object.freeze({
  preview_cap_ms: 60_000,
  battle_cap_ms: 20_000,
  safety_margin_ms: 3_000,
  fallback_reserve_ms: 1_000,
  bank_reserve_ms: 15_000,
  min_remaining_decisions: 4,
  turns_per_pokemon: 1.5,
});

/** Player-perspective counts behind `R̂` (research note §2.4). */
export interface RemainingCounts {
  /** Own Pokémon not fainted, from the player's own request. */
  own_alive: number;
  /** Opponent's brought count minus public `|faint|` events. */
  opp_alive: number;
  /** Own brought Pokémon not active and not fainted. */
  own_bench: number;
}

function assertCount(name: string, value: number, min: number): void {
  if (!Number.isSafeInteger(value) || value < min) {
    throw new RangeError(`${name} must be an integer >= ${min}, got ${value}`);
  }
}

function assertConfig(config: BudgetConfig): void {
  for (const key of [
    "preview_cap_ms",
    "battle_cap_ms",
    "safety_margin_ms",
    "fallback_reserve_ms",
    "bank_reserve_ms",
  ] as const) {
    assertCount(key, config[key], 0);
  }
  assertCount("min_remaining_decisions", config.min_remaining_decisions, 1);
  if (!Number.isFinite(config.turns_per_pokemon) || config.turns_per_pokemon < 0) {
    throw new RangeError(`turns_per_pokemon must be a finite number >= 0, got ${config.turns_per_pokemon}`);
  }
}

function assertSnapshot(snapshot: TimerSnapshot): void {
  assertCount("turn_seconds_left", snapshot.turn_seconds_left, 0);
  assertCount("bank_seconds_left", snapshot.bank_seconds_left, 0);
  assertCount("grace_seconds_left", snapshot.grace_seconds_left, 0);
}

/**
 * `R̂ = max(min_remaining_decisions, ⌈turns_per_pokemon × (own_alive + opp_alive)⌉ + own_bench)`.
 * Each own bench Pokémon can force at most one extra switch request. Safety
 * does not depend on its accuracy, only how evenly the bank is spent.
 */
export function estimateRemainingDecisions(counts: RemainingCounts, config: BudgetConfig = A2_BUDGET_CONFIG): number {
  assertConfig(config);
  assertCount("own_alive", counts.own_alive, 0);
  assertCount("opp_alive", counts.opp_alive, 0);
  assertCount("own_bench", counts.own_bench, 0);
  const turns = Math.ceil(config.turns_per_pokemon * (counts.own_alive + counts.opp_alive));
  return Math.max(config.min_remaining_decisions, turns + counts.own_bench);
}

/** `submit_within = min(max(0, T − m), max(f, alloc))`; `policy_budget = max(0, submit_within − f)`. */
function budget(turnSecondsLeft: number, allocationMs: number, config: BudgetConfig): Budget {
  const hard = Math.max(0, turnSecondsLeft * MS_PER_SECOND - config.safety_margin_ms);
  const submit = Math.min(hard, Math.max(config.fallback_reserve_ms, allocationMs));
  return { submit_within_ms: submit, policy_budget_ms: Math.max(0, submit - config.fallback_reserve_ms) };
}

/**
 * Team Preview: bounded by the preview cap and the turn cap only. Preview
 * spends grace, and the server clips the total back to `starting` on battle
 * turn 1, so preview never touches the battle bank.
 */
export function previewBudget(snapshot: TimerSnapshot, config: BudgetConfig = A2_BUDGET_CONFIG): Budget {
  assertConfig(config);
  assertSnapshot(snapshot);
  return budget(snapshot.turn_seconds_left, config.preview_cap_ms, config);
}

/**
 * A battle request: `alloc = min(battle_cap, max(0, B − B_res) / R̂)`. With
 * `R̂ >= 4` the spendable bank shrinks at most geometrically, so planned
 * spending never reaches the reserve however long the battle runs.
 */
export function battleBudget(
  snapshot: TimerSnapshot,
  remainingDecisions: number,
  config: BudgetConfig = A2_BUDGET_CONFIG,
): Budget {
  assertConfig(config);
  assertSnapshot(snapshot);
  assertCount("remaining_decisions", remainingDecisions, 1);
  const spendable = Math.max(0, snapshot.bank_seconds_left * MS_PER_SECOND - config.bank_reserve_ms);
  const allocation = Math.min(config.battle_cap_ms, Math.floor(spendable / remainingDecisions));
  return budget(snapshot.turn_seconds_left, allocation, config);
}

/**
 * Areneon's model of the server timer (`RoomBattleTimer` in the pinned
 * `server/room-battle.ts`), for A1, which has no timer, and for A2 while the
 * server timer is off. `total` includes grace, like the server's `secondsLeft`.
 */
export interface ShadowTimer {
  readonly rules: Readonly<TimerRules>;
  readonly total: number;
  /** Battle turn of the last request; null before the first. */
  readonly turn: number | null;
  readonly first_request: boolean;
}

/** A fresh timer: `starting + grace` seconds, the first request still to come. */
export function startShadowTimer(rules: TimerRules): ShadowTimer {
  for (const key of ["starting", "grace", "add_per_turn", "max_per_turn", "max_first_turn"] as const) {
    assertCount(key, rules[key], 0);
  }
  assertCount("tick", rules.tick, 1);
  return { rules: Object.freeze({ ...rules }), total: rules.starting + rules.grace, turn: null, first_request: true };
}

/**
 * A new (non-update, non-wait) request at `battleTurn` (0 is Team Preview).
 * A new battle turn adds `add_per_turn` and clips the total to `starting`,
 * which discards unused grace. The first request is capped at
 * `max_first_turn`, every later one at `max_per_turn`, and neither cap
 * exceeds the total left.
 */
export function shadowRequest(
  timer: ShadowTimer,
  battleTurn: number,
): { timer: ShadowTimer; snapshot: TimerSnapshot } {
  assertCount("battle_turn", battleTurn, 0);
  const { rules } = timer;
  let next: ShadowTimer;
  if (timer.turn === null) {
    next = { ...timer, turn: battleTurn };
  } else if (battleTurn > timer.turn) {
    next = {
      ...timer,
      turn: battleTurn,
      first_request: false,
      total: Math.min(timer.total + rules.add_per_turn, rules.starting),
    };
  } else {
    next = { ...timer, first_request: false };
  }
  const cap = next.first_request ? rules.max_first_turn : rules.max_per_turn;
  const grace = Math.max(0, next.total - rules.starting);
  return {
    timer: next,
    snapshot: {
      timer_source: "shadow",
      turn_seconds_left: Math.min(next.total, cap),
      bank_seconds_left: next.total - grace,
      grace_seconds_left: grace,
    },
  };
}

/**
 * Charge `serverElapsedMs` in whole ticks, as the server's `nextTick` does:
 * an answer within `5k + 4.99` s costs `5k` s. The total stops at 0, where
 * the server forfeits the player.
 */
export function shadowCharge(timer: ShadowTimer, serverElapsedMs: number): ShadowTimer {
  if (!Number.isFinite(serverElapsedMs) || serverElapsedMs < 0) {
    throw new RangeError(`server elapsed time must be a finite number >= 0, got ${serverElapsedMs}`);
  }
  const ticks = Math.floor(serverElapsedMs / (timer.rules.tick * MS_PER_SECOND));
  return { ...timer, total: Math.max(0, timer.total - ticks * timer.rules.tick) };
}
