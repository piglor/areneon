// The two holes the review of PR #413 found (B1, B2): each test builds the bad backend's evidence and requires that it does
// NOT admit. Both failed on the code before the fix (evidence on #1461).
import { describe, expect, it } from "vitest";
import { admissionRouter, buildLedger } from "../src/admission.ts";
import { runCorpus } from "../src/harness.ts";
import { mechanicsExercised } from "../src/mechanics-level.ts";
import { ShadowDivergence, shadowCheck } from "../src/shadow.ts";
import { BACKEND, backendOf, FRESH, oracle, PIN, type ToyIn, tallyOf, toyLevel, truth } from "./support.ts";

const ledgerOf = (...tallies: Parameters<typeof buildLedger>[0]["tallies"]) => buildLedger({ level: "mechanics-query", backend: BACKEND, showdown_commit: PIN, tallies });
const BIG = 200_000;

describe("B1: a backend that returns an extra trailing result", () => {
  it("is never admitted", () => {
    // Every item holds move:a and the kind; one batch gets one result too many, every answer otherwise right.
    const inputs: ToyIn[] = Array.from({ length: BIG }, (_, n) => ({ n, entities: ["kind:damage", "move:a"], reads: ["kind:damage", "move:a"] }));
    let first = true;
    const extra = backendOf("extra", (batch) => {
      const answers = batch.map(truth);
      if (first) {
        first = false;
        answers.push({ ok: true, result: { for: -1, value: 0 } });
      }
      return answers;
    });
    const tally = runCorpus(toyLevel, oracle, extra, FRESH, inputs, 4096);
    const ledger = ledgerOf(tally);
    expect(ledger.entities.filter((e) => e.admitted)).toEqual([]);
  }, 120_000);
});

describe("B2: evidence that does not support admission", () => {
  it("a fresh corpus below the minimum admits nothing", () => {
    const ledger = ledgerOf(tallyOf(FRESH, [{ count: 40, entities: ["kind:damage", "move:a"] }], false));
    expect(ledger.entities.filter((e) => e.admitted)).toEqual([]);
  });

  it("a query kind never seen with an entity routes to the oracle", () => {
    // move:a is admitted from damage queries; kind:move_outcome is admitted from queries without move:a.
    const ledger = ledgerOf(
      tallyOf(FRESH, [
        { count: BIG / 2, entities: ["kind:damage", "move:a"] },
        { count: BIG / 2, entities: ["kind:move_outcome", "move:b"] },
      ]),
    );
    const router = admissionRouter(ledger);
    expect(router.route(["kind:damage", "move:a"])).toBeNull();
    expect(router.route(["kind:move_outcome", "move:a"])).not.toBeNull();
  }, 120_000);
});

describe("should-fix 2 and 4 of the review", () => {
  it("an unsupported answer whose reason the move does not decide exercises the query kind alone", () => {
    const d = { kind: "speed_order" as const, mons: [], used_moves: ["quickattack"], weather: null, terrain: null, trick_room: false, side_conditions: [], slot_conditions: [] };
    expect(mechanicsExercised(d, { ok: true, result: { kind: "unsupported", reason: "random_effect" } as never })).toEqual(["kind:speed_order"]);
  });
  it("a shadow sample with a count mismatch revokes every sampled item's entities", () => {
    const inputs: ToyIn[] = Array.from({ length: 10 }, (_, n) => ({ n, entities: [`move:m${n}`], reads: [] }));
    const recorded = [...inputs.map(truth), { ok: true as const, result: { for: -1, value: 0 } }];
    let caught: unknown;
    try {
      shadowCheck(toyLevel, oracle, inputs, recorded, (i) => i.n, { share_ppm: 0, min: 10 });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ShadowDivergence);
    expect((caught as ShadowDivergence).whole_backend).toBe(true);
    expect((caught as ShadowDivergence).entities).toEqual(inputs.map((i) => `move:m${i.n}`).sort());
  });
});

describe("B3: repeated items are not fresh-scale evidence", () => {
  it("200,000 copies of 100 queries admit nothing", () => {
    const inputs: ToyIn[] = Array.from({ length: BIG }, (_, i) => ({ n: i % 100, entities: ["kind:damage", `move:m${i % 100}`], reads: ["kind:damage", `move:m${i % 100}`] }));
    let ledger: ReturnType<typeof ledgerOf> | null = null;
    try {
      ledger = ledgerOf(runCorpus(toyLevel, oracle, oracle, FRESH, inputs, 4096));
    } catch (error) {
      expect((error as Error).message).toContain("repeated");
    }
    expect(ledger?.entities.filter((e) => e.admitted) ?? []).toEqual([]);
  }, 120_000);
});
