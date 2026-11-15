import { describe, expect, it } from "vitest";
import type { LikelyRow } from "../src/likely-threats.ts";
import { type LikelyCounts, type LikelyInput, type Phase, phaseOf, tallyLikelyDecision } from "../src/situational-likely.ts";
import { type Counts, situationalSetup, tallyGame, type StoredGame } from "../src/situational-move-use.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { engine } from "./damage-aware-support.ts";
import { MAX_TURNS, play, storedGame } from "./situational-support.ts";
import type { SituationRows } from "../src/situation.ts";

// The same doubles request as the #1274 tests: Protect at move 1, a plain move at 2, two benched Pokemon at positions 2 and 3.
const body = {
  side: { pokemon: [{ condition: "100/100" }, { condition: "100/100" }, { condition: "100/100" }, { condition: "0 fnt" }] },
  active: [{ moves: [{ id: "protect" }, { id: "tackle" }] }, { moves: [{ id: "protect" }, { id: "tackle" }] }],
};
const candidates = ["move 1, move 2 1", "move 2 1, move 2 1", "move 2 1, move 1", "switch 3, move 2 1"].map((choice) => ({ choice, pending: false }));

type Reveal = { own: number; foe: number; move: string; ohko: number | null };
const base = (revealed: Reveal[], speeds: Record<string, "own_first" | "foe_first" | "tie">): SituationRows =>
  ({
    threats: revealed.map((t) => ({ own: t.own, mega: false, foe: t.foe, move: t.move, kind: "computed", mean_ppm: 0, max_ppm: 0, min_ppm: 0, ohko_rolls: t.ohko })),
    speeds: Object.entries(speeds).map(([k, relation]) => ({ own: Number(k.split(":")[0]), mega: false, foe: Number(k.split(":")[1]), kind: "computed", relation })),
    foe_priorities: [],
  }) as unknown as SituationRows;
const row = (own: number, foe: number, move: string, inclusion_ppm: number, ohko: number | null, max = 0, mega = false): LikelyRow => ({ own, mega, foe, move, inclusion_ppm, kind: "computed", mean_ppm: 0, max_ppm: max, min_ppm: 0, ohko_rolls: ohko }) as LikelyRow;
const run = (chosen: string, input: Partial<LikelyInput> & Pick<LikelyInput, "base" | "likely">, phase: Phase = "turn1"): LikelyCounts => {
  const counts: LikelyCounts = {};
  tallyLikelyDecision(counts, { body, candidates, chosen, phase, stall: new Set(["protect"]), ...input });
  return counts;
};

describe("Likely-Threatened classes from hand-built rows", () => {
  it("counts a slot that revealed moves leave Safe but a likely row reaches from a faster foe, with the inclusion, and the clean comparator", () => {
    const input = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 67_000, 2), row(0, 2, "y", 30_000, 1)] } };
    // foe 2 has no speed row, so it never acts first by the reading; only foe 1's row counts
    expect(run("move 1, move 2 1", input)).toEqual({
      "likely.decisions": 1,
      "likely.slots.likely_threatened.turn1": 1,
      "likely.likely_threatened.n": 1,
      "likely.likely_threatened.inclusion_max_ppm": 67_000,
      "likely.likely_threatened.inclusion_sum_ppm": 67_000,
      "likely.protect.likely_threatened.turn1.offered": 1,
      "likely.protect.likely_threatened.turn1.chosen": 1,
      "likely.switch.likely_threatened.turn1.offered": 1,
      "likely.slots.safe_unlikely.turn1": 1,
      "likely.protect.safe_unlikely.turn1.offered": 1,
    });
    expect(run("switch 3, move 2 1", input, "fresh")).toMatchObject({ "likely.switch.likely_threatened.fresh.chosen": 1, "likely.switch.likely_threatened.fresh.offered": 1 });
    expect(run("switch 3, move 2 1", input, "later")["likely.protect.likely_threatened.later.chosen"]).toBeUndefined();
  });
  it("sums the inclusions of the first-acting reaching rows, capped at a million, and takes the largest as the slot's maximum", () => {
    const two = { base: base([], { "0:1": "foe_first", "0:2": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 700_000, 1), row(0, 2, "y", 600_000, 3), row(0, 1, "z", 10_000, 0)] } };
    expect(run("move 2 1, move 2 1", two)).toMatchObject({ "likely.likely_threatened.inclusion_max_ppm": 700_000, "likely.likely_threatened.inclusion_sum_ppm": 1_000_000 });
  });
  it("is not Likely-Threatened when the reaching foe acts after the slot, the row is unsupported, null KO counts fall short, or the move does not reach", () => {
    const slower = { base: base([], { "0:1": "own_first" }), likely: { belief_threats: [row(0, 1, "x", 50_000, 4)] } };
    expect(run("move 2 1, move 2 1", slower)).toMatchObject({ "likely.slots.other.turn1": 1, "likely.slots.safe_unlikely.turn1": 1 }); // slot 0 other (reaches, acts after), slot 1 untouched
    const tie = { base: base([], { "0:1": "tie" }), likely: { belief_threats: [row(0, 1, "x", 50_000, 4)] } };
    expect(run("move 2 1, move 2 1", tie)["likely.slots.other.turn1"]).toBe(1);
    const unsupported = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [{ own: 0, mega: false, foe: 1, move: "x", inclusion_ppm: 9, kind: "unsupported", reason: "query_kind" } as unknown as LikelyRow] } };
    expect(run("move 2 1, move 2 1", unsupported)).toMatchObject({ "likely.slots.safe_unlikely.turn1": 2 });
    const short = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 50_000, null, 999_999)] } };
    expect(run("move 2 1, move 2 1", short)).toMatchObject({ "likely.slots.safe_unlikely.turn1": 2 });
    const lethal = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 50_000, null, 1_000_000)] } };
    expect(run("move 2 1, move 2 1", lethal)).toMatchObject({ "likely.slots.likely_threatened.turn1": 1 });
    const mega = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 50_000, 4, 0, true)] } };
    expect(run("move 2 1, move 2 1", mega)).toMatchObject({ "likely.slots.safe_unlikely.turn1": 2 });
  });
  it("a slot the revealed moves already Threaten is counted apart and never in the rates; Exposed slower with no likely row is other", () => {
    const threatened = { base: base([{ own: 0, foe: 1, move: "x", ohko: 3 }], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "y", 90_000, 5)] } };
    const counts = run("move 1, move 2 1", threatened);
    expect(counts).toMatchObject({ "likely.slots.revealed_threatened.turn1": 1, "likely.slots.safe_unlikely.turn1": 1 });
    expect(counts["likely.likely_threatened.n"]).toBeUndefined();
    expect(Object.keys(counts).some((k) => k.includes("revealed_threatened") && /protect|switch/.test(k))).toBe(false);
    const exposed = { base: base([{ own: 0, foe: 1, move: "x", ohko: 3 }], { "0:1": "own_first" }), likely: { belief_threats: [] } };
    expect(run("move 2 1, move 2 1", exposed)).toMatchObject({ "likely.slots.other.turn1": 1 });
  });
  it("a refused likely build classifies nothing and is counted; a no_facts decision counts its slots but no rates; a fainted slot is skipped", () => {
    const refused = run("move 1, move 2 1", { base: base([], {}), likely: { belief_threats: [], belief_threats_refused: "species_not_in_prior" } });
    expect(refused).toEqual({ "likely.decisions": 1, "likely.refused.species_not_in_prior": 1 });
    const input = { base: base([], { "0:1": "foe_first" }), likely: { belief_threats: [row(0, 1, "x", 5, 1)] } };
    expect(run("move 1, move 2 1", input, "no_facts")).toEqual({ "likely.decisions": 1, "likely.slots.likely_threatened.no_facts": 1, "likely.slots.safe_unlikely.no_facts": 1 });
    const fainted = { ...body, side: { pokemon: [{ condition: "0 fnt" }, { condition: "50/100" }] } };
    const counts: LikelyCounts = {};
    tallyLikelyDecision(counts, { body: fainted, candidates, chosen: "move 2 1, move 2 1", phase: "later", stall: new Set(["protect"]), ...input });
    expect(counts["likely.slots.safe_unlikely.later"]).toBe(1);
    expect(() => run("move 1", input)).toThrow("different number of parts");
  });
  it("phases: turn 1, a fresh foe switch-in (active_turns 1), later, unknown facts", () => {
    expect(phaseOf(0, null)).toBe("turn1");
    expect(phaseOf(1, [3])).toBe("turn1");
    expect(phaseOf(2, [1, 4])).toBe("fresh");
    expect(phaseOf(5, [2, null])).toBe("later");
    expect(phaseOf(5, null)).toBe("no_facts");
  });
});

describe("a re-simulated stored game with the likely tally", () => {
  it("leaves every #1274 count and the history-free output byte-identical, and adds a likely block", async () => {
    const { record, teams } = await play(0);
    const game = storedGame(record, teams, 0) as unknown as StoredGame;
    const plain = await tallyGame(situationalSetup(engine, GOLDEN_FORMAT, MAX_TURNS, goldenWorldFactsData()), game);
    const withLikely = await tallyGame(situationalSetup(engine, GOLDEN_FORMAT, MAX_TURNS, goldenWorldFactsData(), undefined, false, true), game);
    for (const seat of ["p1", "p2"] as const) {
      expect(JSON.stringify(withLikely?.seats[seat].counts)).toBe(JSON.stringify(plain?.seats[seat].counts));
      expect(plain?.seats[seat].likely).toBeUndefined();
    }
    const likely = withLikely?.seats.p1.likely as Counts;
    expect(likely["likely.decisions"] ?? 0).toBeGreaterThan(0);
    const slots = Object.entries(likely).filter(([k]) => k.startsWith("likely.slots.")).reduce((n, [, v]) => n + v, 0);
    expect(slots).toBeGreaterThan(0);
    expect(JSON.stringify(plain)).not.toContain("likely");
  }, 120_000);
  it("counts a failed likely build and keeps the #1274 counts", async () => {
    const { record, teams } = await play(0);
    const game = storedGame(record, teams, 0) as unknown as StoredGame;
    const setup = situationalSetup(engine, GOLDEN_FORMAT, MAX_TURNS, goldenWorldFactsData(), undefined, false, true);
    const failing = { ...setup, likely: { service: { run: () => ({ outcome: { kind: "error" as const, code: "reader_gap" as const }, ms: 0, counts: { queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 } }) } } };
    const tally = await tallyGame(failing, game);
    expect(tally?.seats.p1.likely?.["likely.feature_error.reader_gap"]).toBeGreaterThan(0);
    expect(tally?.seats.p1.likely?.["likely.decisions"]).toBeUndefined();
    expect(tally?.seats.p1.counts.decisions).toBeGreaterThan(0);
  }, 120_000);
});
