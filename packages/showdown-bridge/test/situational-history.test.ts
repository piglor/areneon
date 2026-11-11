import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Player, runBattle } from "../src/runner.ts";
import { type HistoryCounts, type HistoryInput, type SlotFact, tallyHistoryDecision } from "../src/situational-history.ts";
import { situationalSetup, type StoredGame, tallyGame } from "../src/situational-move-use.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { corpus, engine } from "./damage-aware-support.ts";

// Doubles request: two actives (Protect at move 1, a plain move at 2, a damaging move at 3).
const body = {
  side: { pokemon: [{ condition: "100/100" }, { condition: "100/100" }, { condition: "0 fnt" }] },
  active: [{ moves: [{ id: "protect" }, { id: "tackle" }, { id: "fakeout" }] }, { moves: [{ id: "protect" }, { id: "tackle" }, { id: "fakeout" }] }],
};
const candidates = ["move 1, move 2 1", "move 2 1, move 2 1", "move 3 1, move 2 1", "move 2 1, move 3 1", "move 2 1, move 1"].map((choice) => ({ choice, pending: false }));

type Out = { slot: number; move: number; target: number; pre: "passes" | "fails" | "chance"; ppm?: number; success?: number; unsupported?: boolean };
const outcomes = (list: Out[]): HistoryInput["outcome"]["outcomes"] =>
  list.map((o) => ({ slot: o.slot, move: o.move, target: o.target, mega: false, action: "switch", foe_move: null, assumed: false, ...(o.unsupported === true ? { kind: "unsupported", reason: "query_kind" } : { kind: "computed", precondition: o.pre, precondition_ppm: o.ppm ?? (o.pre === "fails" ? 0 : 1_000_000), hit_ppm: 1_000_000, success_ppm: o.success ?? (o.pre === "fails" ? 0 : 1_000_000) }) })) as never;
const rows = (list: { slot: number; move: number; target: number; mean?: number }[]): never => list.map((r) => ({ slot: r.slot, move: r.move, target: r.target, mega: false, kind: r.mean === undefined ? "unsupported" : "computed", ...(r.mean === undefined ? { reason: "query_kind" } : { mean_ppm: r.mean, max_ppm: r.mean, min_ppm: r.mean, ohko_rolls: 0 }) })) as never;
const fact = (move_actions: number, stall_counter: number | null = null, item_history = false): SlotFact => ({ move_actions, stall_counter, item_history });

const run = (chosen: string, over: Partial<HistoryInput>): HistoryCounts => {
  const counts: HistoryCounts = {};
  tallyHistoryDecision(counts, { body, candidates, chosen, base: { rows: [] }, outcome: { rows: [], outcomes: [] }, facts: [fact(0), fact(0)], stall: new Set(["protect"]), ...over });
  return counts;
};

describe("history groups from hand-built decisions", () => {
  it("splits stall moves by the counter and keeps the mean success chance of the chosen one", () => {
    const o = outcomes([
      { slot: 0, move: 1, target: 0, pre: "chance", ppm: 333_333, success: 333_333 },
      { slot: 0, move: 1, target: 0, pre: "chance", ppm: 111_111, success: 111_111 },
      { slot: 1, move: 1, target: 0, pre: "passes" },
    ]);
    const counts = run("move 1, move 2 1", { outcome: { rows: [], outcomes: o }, facts: [fact(2, 9), fact(2, null)] });
    expect(counts).toMatchObject({ "history.decisions": 1, "history.stall.repeat.offered": 1, "history.stall.repeat.chosen": 1, "history.stall.repeat.chosen_success_ppm": 222_222, "history.stall.repeat.chosen_success_n": 1, "history.stall.fresh.offered": 1 });
    expect(counts["history.stall.fresh.chosen"]).toBeUndefined();
    // a counter of exactly 1 is fresh, and a slot without facts is excluded
    expect(run("move 2 1, move 2 1", { facts: [fact(0, 1), null] })).toMatchObject({ "history.stall.fresh.offered": 1, "history.stall.no_facts.offered": 1 });
  });

  it("groups history-scaled power by the declared rows against the default rows", () => {
    const base = rows([{ slot: 0, move: 2, target: 1, mean: 400_000 }, { slot: 0, move: 3, target: 1, mean: 400_000 }, { slot: 1, move: 2, target: 1, mean: 500_000 }, { slot: 1, move: 3, target: 1, mean: 300_000 }]);
    const declared = rows([{ slot: 0, move: 2, target: 1, mean: 500_000 }, { slot: 0, move: 3, target: 1, mean: 100_000 }, { slot: 1, move: 2, target: 1, mean: 500_000 }, { slot: 1, move: 3, target: 1 }]);
    const counts = run("move 2 1, move 2 1", { base: { rows: base }, outcome: { rows: declared, outcomes: [] } });
    expect(counts).toMatchObject({ "history.power.up.offered": 1, "history.power.up.chosen": 1, "history.power.down.offered": 1, "history.power.unmatched": 1 });
    expect(counts["history.power.down.chosen"]).toBeUndefined();
    // a part with rows only in one set cannot be compared
    expect(run("move 2 1, move 2 1", { base: { rows: rows([{ slot: 0, move: 2, target: 1, mean: 1 }]) } })["history.power.unmatched"]).toBe(1);
  });

  it("counts parts that fail in every branch, split by the actor's facts, and the conditional ones with the share that pass", () => {
    const o = outcomes([
      { slot: 0, move: 3, target: 1, pre: "fails" },
      { slot: 0, move: 3, target: 1, pre: "fails" },
      { slot: 1, move: 3, target: 1, pre: "fails" },
      { slot: 0, move: 2, target: 1, pre: "passes" },
      { slot: 0, move: 2, target: 1, pre: "fails" },
      { slot: 0, move: 2, target: 1, pre: "chance", ppm: 500_000, success: 500_000 },
    ]);
    const input = { outcome: { rows: [], outcomes: o }, facts: [fact(3, null, true), fact(0)] };
    const counts = run("move 3 1, move 3 1", input);
    expect(counts).toMatchObject({ "history.fails.all.offered": 2, "history.fails.all.chosen": 2, "history.fails.acted.offered": 1, "history.fails.acted.chosen": 1, "history.fails.fresh.offered": 1, "history.fails.fresh.chosen": 1, "history.fails.item.offered": 1, "history.fails.item.chosen": 1, "history.conditional.offered": 1 });
    expect(counts["history.conditional.chosen"]).toBeUndefined();
    const picked = run("move 2 1, move 2 1", input);
    expect(picked).toMatchObject({ "history.conditional.chosen": 1, "history.conditional.chosen_branches": 3, "history.conditional.chosen_passing": 2 });
    expect(picked["history.fails.all.chosen"]).toBeUndefined();
  });

  it("counts an unsupported outcome apart and never classifies it; skips a fainted slot; refuses a choice of the wrong width", () => {
    const o = outcomes([{ slot: 0, move: 3, target: 1, pre: "fails" }, { slot: 0, move: 3, target: 1, pre: "fails", unsupported: true }]);
    expect(run("move 3 1, move 2 1", { outcome: { rows: [], outcomes: o } })).toMatchObject({ "history.outcome_unsupported": 1 });
    const fainted = { ...body, side: { pokemon: [{ condition: "0 fnt" }, { condition: "50/100" }] } };
    const counts: HistoryCounts = {};
    tallyHistoryDecision(counts, { body: fainted, candidates, chosen: "move 2 1, move 2 1", base: { rows: [] }, outcome: { rows: [], outcomes: outcomes([{ slot: 0, move: 3, target: 1, pre: "fails" }, { slot: 1, move: 3, target: 1, pre: "fails" }]) }, facts: [null, null], stall: new Set(["protect"]) });
    expect(counts["history.fails.all.offered"]).toBe(1);
    expect(() => run("move 1", {})).toThrow("different number of parts");
  });

  it("reads no Protect, switch or no-move part as a history group", () => {
    const counts = run("move 2 1, move 2 1", { candidates: [{ choice: "switch 3, move 2 1", pending: false }, { choice: "move 2 1, move 2 1", pending: false }] as never });
    expect(Object.keys(counts)).toEqual(["history.decisions"]);
  });
});

describe("a re-simulated stored game with the history rows", () => {
  const battle = corpus[0] as never as { seed: string; teams: Record<Player, never[]>; actor_seeds: Record<Player, number> };
  it("adds a history block per seat, leaves the existing counts as they were, and is deterministic", async () => {
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams as never, actors: { p1: randomLegalActor(battle.actor_seeds.p1), p2: randomLegalActor(battle.actor_seeds.p2) }, maxTurns: 3 });
    const game: StoredGame = {
      data_class: "synthetic",
      match_id: "m",
      game: 0,
      pair: 0,
      seed: battle.seed,
      policies: { p1: "policy:aa", p2: "actor-rule@v1" },
      teams: { p1: { team: battle.teams.p1 }, p2: { team: battle.teams.p2 } },
      decisions: record.choices.map((c) => ({ player: c.player, request_seq: c.request_seq, chosen: c.choice, candidates: [...(record.requests[c.player][c.request_seq - 1]?.legal?.choices ?? []).map((choice) => ({ choice, pending: false }))] })),
    };
    const plain = await tallyGame(situationalSetup(engine, GOLDEN_FORMAT, 3, goldenWorldFactsData()), game);
    const make = (): ReturnType<typeof situationalSetup> => situationalSetup(engine, GOLDEN_FORMAT, 3, goldenWorldFactsData(), undefined, true);
    const a = await tallyGame(make(), game);
    const b = await tallyGame(make(), game);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    for (const seat of ["p1", "p2"] as const) {
      expect(a?.seats[seat].counts).toEqual(plain?.seats[seat].counts);
      expect(plain?.seats[seat].history).toBeUndefined();
      expect(a?.seats[seat].history?.["history.decisions"]).toBe(a?.seats[seat].counts.decisions);
    }
    // a feature error of the outcome builder is counted and skipped
    const setup = make();
    const failing = { ...setup, history: { ...(setup.history as NonNullable<typeof setup.history>), service: { run: () => ({ outcome: { kind: "error" as const, code: "reader_gap" as const }, ms: 0, counts: { queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 } }) } } };
    const refused = await tallyGame(failing, game);
    expect(refused?.seats.p1.history?.["history.feature_error.reader_gap"]).toBeGreaterThan(0);
  }, 240_000);
});
