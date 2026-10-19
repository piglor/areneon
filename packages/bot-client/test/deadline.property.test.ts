import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  A2_BUDGET_CONFIG,
  battleBudget,
  type BudgetConfig,
  estimateRemainingDecisions,
  previewBudget,
  shadowCharge,
  shadowRequest,
  startShadowTimer,
  type TimerSnapshot,
} from "../src/index.ts";
import { vgcTimer } from "./deadline-timer.ts";

const VGC = vgcTimer();

/** Server-side latency below the margin, as fractional ms (the Python property's 0–1.5 s). */
const latencyMs = fc.double({ min: 0, max: 1_500, noNaN: true });

/** Any estimator output, even a wrong one (the Python property's `R̂ ∈ [1, 40]`). */
const remaining = fc.integer({ min: 1, max: 40 });

const seconds = fc.integer({ min: 0, max: 600 });

const snapshot: fc.Arbitrary<TimerSnapshot> = fc.record({
  timer_source: fc.constantFrom("server" as const, "shadow" as const),
  turn_seconds_left: seconds,
  bank_seconds_left: seconds,
  grace_seconds_left: seconds,
});

const ms = fc.integer({ min: 0, max: 120_000 });

const config: fc.Arbitrary<BudgetConfig> = fc.record({
  preview_cap_ms: ms,
  battle_cap_ms: ms,
  safety_margin_ms: fc.integer({ min: 0, max: 10_000 }),
  fallback_reserve_ms: fc.integer({ min: 0, max: 5_000 }),
  bank_reserve_ms: fc.integer({ min: 0, max: 60_000 }),
  min_remaining_decisions: fc.integer({ min: 1, max: 10 }),
  turns_per_pokemon: fc.double({ min: 0, max: 4, noNaN: true }),
});

describe("deadline properties", () => {
  it("never exhausts the bank before the last decision, and auto-choose never fires", () => {
    // A whole battle of up to 300 turns under the VGC Timer, every decision
    // spending its full submit_within plus latency: the Python property
    // test_bank_never_exhausted_and_autochoose_never_fires, plus the reserve.
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 300 }),
        latencyMs,
        fc.array(remaining, { minLength: 300, maxLength: 300 }),
        (turns, latency, estimates) => {
          let requested = shadowRequest(startShadowTimer(VGC), 0);
          const preview = previewBudget(requested.snapshot);
          expect(preview.submit_within_ms + latency).toBeLessThan(requested.snapshot.turn_seconds_left * 1000);
          let timer = shadowCharge(requested.timer, preview.submit_within_ms + latency);
          for (const [i, estimate] of estimates.slice(0, turns).entries()) {
            requested = shadowRequest(timer, i + 1);
            const s = requested.snapshot;
            expect(s.bank_seconds_left).toBeGreaterThan(0);
            expect(s.bank_seconds_left * 1000).toBeGreaterThanOrEqual(A2_BUDGET_CONFIG.bank_reserve_ms);
            const elapsed = battleBudget(s, estimate).submit_within_ms + latency;
            expect(elapsed).toBeLessThan(s.turn_seconds_left * 1000);
            timer = shadowCharge(requested.timer, elapsed);
          }
          expect(Math.min(timer.total, VGC.starting) * 1000).toBeGreaterThanOrEqual(A2_BUDGET_CONFIG.bank_reserve_ms);
        },
      ),
      { numRuns: 1000 },
    );
  });

  it("forced switches in the same turn never exhaust the bank either", () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), remaining), { minLength: 1, maxLength: 300 }),
        latencyMs,
        (decisions, latency) => {
          let timer = shadowCharge(shadowRequest(startShadowTimer(VGC), 0).timer, 60_000 + latency);
          let turn = 0;
          for (const [sameTurn, estimate] of decisions) {
            if (!sameTurn || turn === 0) turn += 1;
            const requested = shadowRequest(timer, turn);
            const s = requested.snapshot;
            expect(s.bank_seconds_left * 1000).toBeGreaterThanOrEqual(A2_BUDGET_CONFIG.bank_reserve_ms);
            const elapsed = battleBudget(s, estimate).submit_within_ms + latency;
            expect(elapsed).toBeLessThan(s.turn_seconds_left * 1000);
            timer = shadowCharge(requested.timer, elapsed);
          }
        },
      ),
      { numRuns: 1000 },
    );
  });

  it("keeps the safety margin when the bank would allow more than the turn", () => {
    // Low T, a large bank and no bank reserve: the allocation exceeds T − m,
    // so only the margin keeps the choice ahead of the server's turn timeout.
    const noReserve: BudgetConfig = { ...A2_BUDGET_CONFIG, bank_reserve_ms: 0, battle_cap_ms: 120_000 };
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 11 }).map((k) => k * VGC.tick),
        fc.integer({ min: 220, max: 420 }),
        fc.integer({ min: 1, max: 4 }),
        fc.double({ min: 0, max: A2_BUDGET_CONFIG.safety_margin_ms - 1, noNaN: true }),
        (turn, bank, r, latency) => {
          const s: TimerSnapshot = {
            timer_source: "server",
            turn_seconds_left: turn,
            bank_seconds_left: bank,
            grace_seconds_left: 0,
          };
          const { submit_within_ms } = battleBudget(s, r, noReserve);
          expect(submit_within_ms).toBe(Math.max(0, turn * 1000 - noReserve.safety_margin_ms));
          expect(submit_within_ms + latency).toBeLessThan(turn * 1000);
        },
      ),
    );
  });

  it("preview never touches the battle bank", () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: VGC.grace * 1000 + VGC.tick * 1000 - 1 }), (elapsed) => {
        const preview = shadowRequest(startShadowTimer(VGC), 0);
        const turn1 = shadowRequest(shadowCharge(preview.timer, elapsed), 1);
        expect(turn1.snapshot.bank_seconds_left).toBe(VGC.starting);
        expect(turn1.snapshot.grace_seconds_left).toBe(0);
      }),
    );
  });

  it("budgets are whole ms, never negative, and bounded by the caps and the turn", () => {
    fc.assert(
      fc.property(snapshot, remaining, config, (s, r, c) => {
        const hard = Math.max(0, s.turn_seconds_left * 1000 - c.safety_margin_ms);
        for (const [b, cap] of [
          [battleBudget(s, r, c), c.battle_cap_ms],
          [previewBudget(s, c), c.preview_cap_ms],
        ] as const) {
          expect(Number.isSafeInteger(b.submit_within_ms)).toBe(true);
          expect(Number.isSafeInteger(b.policy_budget_ms)).toBe(true);
          expect(b.policy_budget_ms).toBeGreaterThanOrEqual(0);
          expect(b.submit_within_ms).toBeGreaterThanOrEqual(0);
          expect(b.submit_within_ms).toBeLessThanOrEqual(hard);
          expect(b.submit_within_ms).toBeLessThanOrEqual(Math.max(cap, c.fallback_reserve_ms));
          expect(b.policy_budget_ms).toBe(Math.max(0, b.submit_within_ms - c.fallback_reserve_ms));
        }
      }),
    );
  });

  it("planned spending never reaches the bank reserve", () => {
    fc.assert(
      fc.property(snapshot, remaining, config, (s, r, c) => {
        const spendable = Math.max(0, s.bank_seconds_left * 1000 - c.bank_reserve_ms);
        const { submit_within_ms } = battleBudget(s, r, c);
        expect(submit_within_ms).toBeLessThanOrEqual(Math.max(c.fallback_reserve_ms, spendable / r));
      }),
    );
  });

  it("is monotonic in the remaining bank", () => {
    fc.assert(
      fc.property(snapshot, seconds, remaining, config, (s, more, r, c) => {
        const richer = { ...s, bank_seconds_left: s.bank_seconds_left + more };
        const low = battleBudget(s, r, c);
        const high = battleBudget(richer, r, c);
        expect(low.submit_within_ms).toBeLessThanOrEqual(high.submit_within_ms);
        expect(low.policy_budget_ms).toBeLessThanOrEqual(high.policy_budget_ms);
      }),
    );
  });

  it("is monotonic in the turn time left", () => {
    fc.assert(
      fc.property(snapshot, seconds, remaining, config, (s, more, r, c) => {
        const longer = { ...s, turn_seconds_left: s.turn_seconds_left + more };
        expect(battleBudget(s, r, c).submit_within_ms).toBeLessThanOrEqual(battleBudget(longer, r, c).submit_within_ms);
        expect(previewBudget(s, c).submit_within_ms).toBeLessThanOrEqual(previewBudget(longer, c).submit_within_ms);
      }),
    );
  });

  it("never grows with more remaining decisions", () => {
    fc.assert(
      fc.property(snapshot, remaining, remaining, config, (s, r, more, c) => {
        expect(battleBudget(s, r + more, c).submit_within_ms).toBeLessThanOrEqual(battleBudget(s, r, c).submit_within_ms);
      }),
    );
  });

  it("the estimate is at least the floor and grows with every count", () => {
    const count = fc.integer({ min: 0, max: 6 });
    fc.assert(
      fc.property(count, count, count, fc.constantFrom("own_alive", "opp_alive", "own_bench"), config, (a, o, b, key, c) => {
        const counts = { own_alive: a, opp_alive: o, own_bench: b };
        const estimate = estimateRemainingDecisions(counts, c);
        expect(estimate).toBeGreaterThanOrEqual(c.min_remaining_decisions);
        expect(estimateRemainingDecisions({ ...counts, [key]: counts[key] + 1 }, c)).toBeGreaterThanOrEqual(estimate);
      }),
    );
  });

  it("charges whole ticks, never more than the elapsed time", () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 600_000, noNaN: true }), (elapsed) => {
        const start = startShadowTimer(VGC);
        const charged = start.total - shadowCharge(start, elapsed).total;
        expect(charged % VGC.tick).toBe(0);
        expect(charged * 1000).toBeLessThanOrEqual(elapsed);
        expect((charged + VGC.tick) * 1000).toBeGreaterThan(Math.min(elapsed, start.total * 1000));
      }),
    );
  });
});
