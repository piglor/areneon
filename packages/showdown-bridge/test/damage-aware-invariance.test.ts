import { describe, expect, it } from "vitest";
import type { PublicMon } from "@areneon/public-state";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import { type DamageAwareActor, type DamageAwareReport } from "../src/damage-aware-actor.ts";
import type { PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { runBattle } from "../src/runner.ts";
import { exportPaste } from "../src/teams.ts";
import type { RandomBattleFile } from "../src/random-corpus.ts";
import { corpus, engine, newActor, toId } from "./damage-aware-support.ts";
import { comparable, mutated, withGenders } from "./damage-aware-invariance-support.ts";

// ADR-018 section 3 and ticket #944. The baseline plays p1 with the REAL probe; p2 plays the first
// legal candidate (a fixed rule that never reads its own moves beyond the first), so the two battles
// of a pair differ only in what the baseline's opponent holds and in nothing the baseline is shown.

const MAX_TURNS = 6;
async function play(battle: RandomBattleFile, p2: PokemonSet[], make: () => DamageAwareActor, seed = battle.seed, p1: PokemonSet[] = battle.teams.p1) {
  const actor = make();
  const record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed, teams: { p1, p2 }, actors: { p1: actor, p2: firstLegal }, maxTurns: MAX_TURNS });
  return { record, report: actor.report() };
}

/** An assumed set read from the opponent's REAL team: what the baseline must never do (the negative control). */
const cheating = (real: () => PokemonSet[]) => (mon: PublicMon): AssumedSet => {
  const set = structuredClone(real().find((candidate) => [mon.species, mon.base_species].includes(toId(candidate.species)))) as PokemonSet;
  return {
    set,
    paste: exportPaste(engine, [structuredClone(set)]),
    revealed_moves: set.moves.map(toId),
    filler_moves: [],
    unplaceable_moves: [],
    shown_hp_percent: mon.fainted ? 0 : mon.hp_num,
    mega: mon.mega,
  };
};

const battles = corpus.slice(0, 3);

describe("hidden-information invariance (ADR-018 section 3)", () => {
  it("makes the same choices at every request whose player channel is the same, whatever the opponent holds unrevealed", async () => {
    let compared = 0;
    let moves = 0;
    for (const battle of battles) {
      const other = mutated(battle.teams.p2);
      // the pair really differs in what the opponent holds
      expect(JSON.stringify(other)).not.toBe(JSON.stringify(battle.teams.p2));
      expect(other.map((set) => set.species)).toEqual(battle.teams.p2.map((set) => set.species));
      const a = await play(battle, battle.teams.p2, () => newActor());
      const b = await play(battle, other, () => newActor());
      const pairs = comparable(a.record, b.record);
      for (const { seq, first, second } of pairs) expect(second, `${battle.id} request ${seq}`).toBe(first);
      compared += pairs.length;
      moves += pairs.filter((pair) => !pair.first.startsWith("team ")).length;
      // the baseline really scored these requests (not a vacuous equal)
      expect(a.report.counts.scored_decisions).toBeGreaterThan(0);
    }
    expect(compared).toBeGreaterThanOrEqual(6);
    // not only Team Preview: scored battle requests with an equal channel
    expect(moves).toBeGreaterThanOrEqual(3);
  }, 180_000);

  it("has teeth: a baseline given the opponent's real team makes different choices at such a request", async () => {
    let differing = 0;
    let compared = 0;
    for (const battle of battles) {
      const other = mutated(battle.teams.p2);
      const a = await play(battle, battle.teams.p2, () => newActor({ assume: cheating(() => battle.teams.p2) }));
      const b = await play(battle, other, () => newActor({ assume: cheating(() => other) }));
      for (const { first, second } of comparable(a.record, b.record)) {
        compared++;
        if (first !== second) differing++;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(3);
    expect(differing).toBeGreaterThan(0);
  }, 180_000);

  it("does not change when the battle seed changes, at the requests that are the same", async () => {
    let compared = 0;
    for (const battle of battles) {
      const [mine, theirs] = [withGenders(battle.teams.p1), withGenders(battle.teams.p2)];
      const a = await play(battle, theirs, () => newActor(), battle.seed, mine);
      const b = await play(battle, theirs, () => newActor(), "sodium,ffffffffffffffffffffffffffffffff", mine);
      expect(a.record.seed).not.toBe(b.record.seed);
      const pairs = comparable(a.record, b.record);
      for (const { seq, first, second } of pairs) expect(second, `${battle.id} request ${seq}`).toBe(first);
      compared += pairs.length;
    }
    expect(compared).toBeGreaterThanOrEqual(3);
  }, 180_000);

  it("does not change with the actor's own stream seed while epsilon_ppm is 0", async () => {
    const battle = battles[0] as RandomBattleFile;
    const a = await play(battle, battle.teams.p2, () => newActor({ seed: 1 }));
    const b = await play(battle, battle.teams.p2, () => newActor({ seed: 0xdeadbeef }));
    expect(b.record.choices).toEqual(a.record.choices);
    expect(b.report).toEqual(a.report);
  }, 120_000);
});

describe("determinism", () => {
  it("gives the same choices, counts and identity from two runs", async () => {
    for (const battle of battles.slice(0, 2)) {
      const a = await play(battle, battle.teams.p2, () => newActor());
      const b = await play(battle, battle.teams.p2, () => newActor());
      expect(b.record.choices).toEqual(a.record.choices);
      expect(b.record.omniscient_log).toEqual(a.record.omniscient_log);
      const strip = (report: DamageAwareReport) => ({ ...report, counts: { ...report.counts, probe_runs: 0 } });
      expect(jcs(strip(b.report) as unknown as Json)).toBe(jcs(strip(a.report) as unknown as Json));
      expect(a.report.counts.decisions).toBe(a.record.choices.filter((c) => c.player === "p1").length);
    }
  }, 120_000);
});
