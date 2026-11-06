import { describe, expect, it } from "vitest";
import type { CandidateFeatures2 } from "../src/candidate-features-2.ts";
import { type Counts, type DecisionInput, SITUATIONAL_LABEL, situationalSetup, sortedCounts, type StoredGame, tallyDecision, tallyGame } from "../src/situational-move-use.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Player, runBattle } from "../src/runner.ts";
import type { PokemonSet } from "../src/engine.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { progressGameStreams } from "../src/progress-seed.ts";
import { corpus, engine } from "./damage-aware-support.ts";
import { MAX_TURNS, play, storedGame, writeRoot } from "./situational-support.ts";
import { ProgressInputError, readVerifiedGeneration } from "../src/situational-input.ts";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { BattleAbortedError } from "../src/runner.ts";

// Doubles request: two actives (Protect at move 1, a plain move at 2), two benched Pokemon at positions 2 and 3.
const body = {
  side: { pokemon: [{ condition: "100/100" }, { condition: "100/100" }, { condition: "100/100" }, { condition: "0 fnt" }] },
  active: [{ moves: [{ id: "protect" }, { id: "tackle" }] }, { moves: [{ id: "protect" }, { id: "tackle" }] }],
};
const candidates = ["move 1, move 2 1", "move 2 1, move 2 1", "move 2 1, move 1", "switch 3, move 2 1"].map((choice) => ({ choice, pending: false }));
type Threat = { own: number; foe: number; move: string; ohko: number | null; mega?: boolean; max?: number };
const features = (threats: Threat[], speeds: Record<string, "own_first" | "foe_first" | "tie"> = {}, priorities: Record<string, number> = {}): CandidateFeatures2 =>
  ({
    feature_set: "probe-damage/2",
    feature_set_hash: "h",
    rows: [],
    threats: threats.map((t) => ({ own: t.own, mega: t.mega ?? false, foe: t.foe, move: t.move, kind: "computed", mean_ppm: 0, max_ppm: t.max ?? 0, min_ppm: 0, ohko_rolls: t.ohko })),
    speeds: Object.entries(speeds).map(([k, relation]) => ({ own: Number(k.split(":")[0]), mega: false, foe: Number(k.split(":")[1]), kind: "computed", relation })),
    own_priorities: [],
    foe_priorities: Object.entries(priorities).map(([k, priority]) => ({ foe: Number(k.split(":")[0]), move: k.split(":")[1] as string, kind: "computed", priority })),
  }) as CandidateFeatures2;
const run = (chosen: string, f: CandidateFeatures2, streaks: (number | null)[] = [0, 0]): Counts => {
  const counts: Counts = {};
  const input: DecisionInput = { body, candidates, chosen, features: f, streaks, stall: new Set(["protect"]) };
  tallyDecision(counts, input);
  return counts;
};

describe("situations from hand-built features", () => {
  it("Threatened with a safe bench is a bad matchup; Protect and switch are offered and the choice is counted", () => {
    const f = features([{ own: 0, foe: 1, move: "x", ohko: 3 }, { own: 2, foe: 1, move: "x", ohko: 0 }], { "0:1": "foe_first" });
    expect(run("move 1, move 2 1", f)).toEqual({ decisions: 1, "slots.threatened": 1, "slots.safe": 1, "protect.threatened.fresh.offered": 1, "protect.threatened.fresh.chosen": 1, "protect.safe.fresh.offered": 1, "switch.bad_matchup.offered": 1, unknown_foe_priority: 1 });
    expect(run("switch 3, move 2 1", f)).toMatchObject({ "switch.bad_matchup.chosen": 1, "switch.bad_matchup.offered": 1 });
    expect(run("switch 3, move 2 1", f)["protect.threatened.fresh.chosen"]).toBeUndefined();
  });
  it("a bench Pokemon the foes also reach leaves no safe bench", () => {
    const f = features([{ own: 0, foe: 1, move: "x", ohko: 1 }, { own: 2, foe: 1, move: "x", ohko: 5 }], { "0:1": "foe_first" });
    expect(run("move 2 1, move 2 1", f)).toMatchObject({ "switch.threatened_no_safe_bench.offered": 1 });
  });
  it("reachable but slower is neither threatened nor safe; a higher priority makes it threatened again", () => {
    const slower = features([{ own: 0, foe: 1, move: "x", ohko: 1 }], { "0:1": "own_first" });
    expect(run("move 2 1, move 2 1", slower)).toMatchObject({ "slots.exposed_slower": 1, "switch.exposed_slower.offered": 1 });
    expect(run("move 2 1, move 2 1", features([{ own: 0, foe: 1, move: "x", ohko: 1 }], { "0:1": "own_first" }, { "1:x": 1 }))).toMatchObject({ "slots.threatened": 1 });
    expect(run("move 2 1, move 2 1", features([{ own: 0, foe: 1, move: "x", ohko: 1 }], { "0:1": "foe_first" }, { "1:x": -1 }))).toMatchObject({ "slots.exposed_slower": 1 });
    expect(run("move 2 1, move 2 1", features([{ own: 0, foe: 1, move: "x", ohko: 1 }], { "0:1": "tie" }))).toMatchObject({ "slots.exposed_slower": 1 });
  });
  it("counts a repeat Protect apart, and Protect with unavailable facts is excluded from the rates", () => {
    const f = features([{ own: 0, foe: 1, move: "x", ohko: 1 }], { "0:1": "foe_first" });
    expect(run("move 1, move 1", f, [1, null])).toMatchObject({ "protect.threatened.repeat.offered": 1, "protect.threatened.repeat.chosen": 1, "protect.no_facts.offered": 1 });
  });
  it("never threatens on an unsupported or unknown row, and counts them", () => {
    const f = features([{ own: 0, foe: 1, move: "x", ohko: null }]);
    f.threats.push({ own: 1, mega: false, foe: 1, move: "y", kind: "unsupported", reason: "query_kind" });
    const counts = run("move 2 1, move 2 1", f);
    expect(counts).toMatchObject({ "slots.safe": 2, ko_fallback_rows: 1 });
    const lethal = features([{ own: 0, foe: 1, move: "x", ohko: null, max: 1_000_000 }], { "0:1": "foe_first" });
    expect(run("move 2 1, move 2 1", lethal)).toMatchObject({ "slots.threatened": 1, threatened_by_max_fallback: 1, ko_fallback_rows: 1 });
    const short = features([{ own: 0, foe: 1, move: "x", ohko: null, max: 999_999 }], { "0:1": "foe_first" });
    expect(run("move 2 1, move 2 1", short)).toMatchObject({ "slots.safe": 2 });
    const mixed = features([{ own: 0, foe: 1, move: "x", ohko: null, max: 1_000_000 }, { own: 0, foe: 2, move: "y", ohko: 2 }], { "0:1": "foe_first", "0:2": "foe_first" });
    expect(run("move 2 1, move 2 1", mixed)["threatened_by_max_fallback"]).toBeUndefined();
    const g = features([{ own: 0, foe: 1, move: "x", ohko: 1 }]);
    expect(run("move 2 1, move 2 1", g)).toMatchObject({ unknown_foe_priority: 1, unknown_speed_relation: 1, "slots.exposed_slower": 1 });
  });
  it("skips a fainted slot and refuses a choice of the wrong width", () => {
    const fainted = { ...body, side: { pokemon: [{ condition: "0 fnt" }, { condition: "50/100" }] } };
    const counts: Counts = {};
    tallyDecision(counts, { body: fainted, candidates, chosen: "move 2 1, move 2 1", features: features([]), streaks: [0, 0], stall: new Set(["protect"]) });
    expect(counts["slots.safe"]).toBe(1);
    expect(() => run("move 1", features([]))).toThrow("different number of parts");
  });
  it("sorts keys and carries the label", () => {
    expect(Object.keys(sortedCounts({ b: 1, a: 2 }))).toEqual(["a", "b"]);
    expect(SITUATIONAL_LABEL).toBe("situational move-use diagnostic on assumed opponent sets: descriptive, not a strength measure");
  });
});

describe("a re-simulated stored game", () => {
  const battle = corpus[0] as never as { seed: string; teams: Record<Player, never[]>; actor_seeds: Record<Player, number> };
  it("tallies each seat from its own channel, deterministically, and refuses non-synthetic data", async () => {
    const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams as never, actors: { p1: randomLegalActor(battle.actor_seeds.p1), p2: randomLegalActor(battle.actor_seeds.p2) }, maxTurns: 4 });
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
    const setup = situationalSetup(engine, GOLDEN_FORMAT, 4, goldenWorldFactsData());
    const a = await tallyGame(setup, game);
    const b = await tallyGame(situationalSetup(engine, GOLDEN_FORMAT, 4, goldenWorldFactsData()), game);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(a?.seats.p1.policy).toBe("policy:aa");
    expect(a?.seats.p1.counts.decisions).toBeGreaterThan(0);
    expect(a?.seats.p1.counts.not_move_request).toBe(1); // Team Preview
    const slots = Object.entries(a?.seats.p1.counts ?? {}).filter(([k]) => k.startsWith("slots.")).reduce((n, [, v]) => n + v, 0);
    expect(slots).toBeGreaterThanOrEqual(a?.seats.p1.counts.decisions ?? 0);
    expect(await tallyGame(setup, { ...game, decisions: [] })).toBeNull();
    // A feature error is counted and skipped; a choice for a request the battle never asked is refused by the re-simulation.
    const failing = { ...setup, service: { run: () => ({ outcome: { kind: "error" as const, code: "reader_gap" as const }, ms: 0, counts: { queries: 0, probe_runs: 0, unsupported_queries: 0, probe_refusals: 0 } }) } };
    const refused = await tallyGame(failing, game);
    expect(refused?.seats.p1.counts["feature_error.reader_gap"]).toBeGreaterThan(0);
    expect(refused?.seats.p1.counts.decisions).toBeUndefined();
    expect(await tallyGame(setup, { ...game, decisions: [...game.decisions, { player: "p1", request_seq: 999, chosen: "x", candidates: [] }] })).toBeNull();
    await expect(tallyGame(setup, { ...game, data_class: "private" })).rejects.toThrow("synthetic");
  }, 120_000);
});

describe("a verified progress root", () => {
  it("verifies the manifest and the records, tallies twice to the same bytes, and refuses tampering", async () => {
    const { record, teams } = await play(0);
    const games = [storedGame(record, teams, 0), storedGame(record, teams, 1)];
    const root = writeRoot(games);
    const dir = join(root, "g000001");
    const read = readVerifiedGeneration(dir, 1);
    expect(read.games.map((g) => g.game)).toEqual([0, 1]);
    expect(readVerifiedGeneration(dir, 1, ["no-such-match"]).games).toEqual([]);

    const run = async (): Promise<string> => {
      const setup = situationalSetup(engine, GOLDEN_FORMAT, read.record.plan.max_turns, goldenWorldFactsData());
      const out = [];
      for (const g of read.games) out.push(await tallyGame(setup, g));
      return JSON.stringify({ format: "situational-move-use/1", generation: 1, run_id: read.record.run_id, evaluation_key: null, pin: engine.commit, unreplayable_games: 0, games: out });
    };
    const first = await run();
    expect(await run()).toBe(first);
    expect(JSON.parse(first).games[0].seats.p1.counts.decisions).toBeGreaterThan(0);
    writeFileSync(join(root, "raw.json"), first);

    expect(() => readVerifiedGeneration(dir, 2)).toThrow(ProgressInputError);
    const bytes = readFileSync(join(dir, "games", "actor-rule-argmax-0.json"), "utf8");
    writeFileSync(join(dir, "games", "actor-rule-argmax-0.json"), bytes.replace('"turns":', '"turns": '));
    expect(() => readVerifiedGeneration(dir, 1)).toThrow("differ from the manifest");
    writeFileSync(join(dir, "games", "actor-rule-argmax-0.json"), bytes);
    mkdirSync(join(dir, "extra"));
    writeFileSync(join(dir, "extra", "x.json"), "{}");
    expect(() => readVerifiedGeneration(dir, 1)).toThrow("differ from the manifest");
  }, 120_000);

  it("refuses a game of another run, generation or match, and reads a version 1 record", async () => {
    const { record, teams } = await play(0);
    const game = storedGame(record, teams, 0);
    const refuse = (change: object, message: string): void => {
      const root = writeRoot([{ ...game, ...change } as never]);
      expect(() => readVerifiedGeneration(join(root, "g000001"), 1)).toThrow(message);
    };
    refuse({ run_id: "other-run", seed: progressGameStreams("other-run", 0).seed }, "not of this run and generation");
    refuse({ generation: 2 }, "not of this run and generation");
    refuse({ match_id: "random-legal-sample" }, "names a match the record lacks");
    const root = writeRoot([game], "progress-evaluation-record/valid/baseline-match");
    expect(readVerifiedGeneration(join(root, "g000001"), 1).games).toHaveLength(1);
    const wrong = writeRoot([game], "progress-evaluation-record-2/valid/baseline-match", (r) => void (r.generation = 7));
    expect(() => readVerifiedGeneration(join(wrong, "g000001"), 1)).toThrow();
  }, 120_000);

  it("rethrows a defect and returns null only for the re-simulation's refusals", async () => {
    const { record, teams } = await play(0);
    const game = storedGame(record, teams, 0) as unknown as StoredGame;
    const setup = situationalSetup(engine, GOLDEN_FORMAT, MAX_TURNS, goldenWorldFactsData());
    expect(await tallyGame(setup, { ...game, decisions: [] })).toBeNull(); // BattleAbortedError
    expect(BattleAbortedError).toBeDefined();
    const broken = { ...game, teams: { ...game.teams, p2: { team: [] } } } as unknown as StoredGame; // an invalid team is a defect, not "unreplayable"
    await expect(tallyGame(setup, broken)).rejects.toThrow();
  }, 120_000);
});

describe("the counts do not depend on the opponent's unrevealed fields", () => {
  it("gives equal counts for p1 when only p2's never-used moves 2 to 4 differ and p1's channel is equal", async () => {
    const reverse = (team: PokemonSet[]): PokemonSet[] => team.map((set) => ({ ...set, moves: [set.moves[0] as string, ...set.moves.slice(1).reverse()] }));
    const a = await play(0);
    const b = await play(0, reverse);
    expect(b.teams.p2).not.toEqual(a.teams.p2);
    expect(b.record.channels.p1).toEqual(a.record.channels.p1); // the guard: the same player view
    const setup = situationalSetup(engine, GOLDEN_FORMAT, MAX_TURNS, goldenWorldFactsData());
    const ta = await tallyGame(setup, storedGame(a.record, a.teams, 0) as unknown as StoredGame);
    const tb = await tallyGame(setup, storedGame(b.record, b.teams, 0) as unknown as StoredGame);
    expect(ta?.seats.p1.counts).toEqual(tb?.seats.p1.counts);
    expect(ta?.seats.p1.counts.decisions).toBeGreaterThan(0);
  }, 120_000);
});
