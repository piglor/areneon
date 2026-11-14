import { describe, expect, it } from "vitest";
import type { PublicMon, PublicState } from "@areneon/public-state";
import { candidateFeaturesLikely, type CandidateFeaturesLikely, FEATURE_SET_LIKELY } from "../src/candidate-features-likely.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { createFeatureMemo, FeatureBuildError, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import { foeTargetKind } from "../src/damage-aware-view.ts";
import type { PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { type LikelySetup, loadLikelySetup, type MoveList } from "../src/likely-threats-config.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { RandomBattleFile } from "../src/random-corpus.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { exportPaste } from "../src/teams.ts";
import { config, corpus, engine, mechanics, snapshot, stateOf, toId, vocabulary } from "./damage-aware-support.ts";

// ADR-022 erratum E-22.7.6 (privacy test, required): the `probe-outcome-likely/1` features are a function of the player's own channel, own
// team and candidates. Two battles whose player channels agree up to a request, but whose opponents hold different unrevealed items,
// natures, Stat Points, abilities and moves 2 to 4, give byte-identical features there, `belief_threats` included: the likelihood is a
// committed table of the foe's SPECIES and the revealed moves, and the damage is on ADR-018's assumed set. The REAL probe is used.
// The negative control hands the real team in through the `assume` seam (the rows then differ); the canary uses a table of decoy moves
// and shows no real unrevealed move can appear unless the table proposes it.

const MAX_TURNS = 10;
const base = loadLikelySetup(GOLDEN_FORMAT, "probe-outcome-likely.v1.json");
const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics: withMoveOutcome(mechanics), outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT) }, likely: base, snapshot, data_class: "synthetic", ...change });
const dex = engine.sim.Dex.mod("champions") as unknown as {
  moves: { get(id: string): { exists: boolean; id: string; name: string; category: string; target: string; priority: number } };
  abilities: { get(id: string): { exists: boolean; name: string; onModifyPriority?: unknown } };
  items: { get(id: string): { exists: boolean; name: string } };
};
const validator = () => engine.sim.TeamValidator.get(GOLDEN_FORMAT);
const legal = (team: PokemonSet[]): boolean => (validator().validateTeam(structuredClone(team)) ?? []).length === 0;
const rosterOf = (set: PokemonSet) => vocabulary.roster.find((entry) => entry.id === toId(set.species)) ?? vocabulary.roster.find((entry) => entry.id === toId(set.species.split("-")[0] ?? ""));

interface Point {
  seq: number;
  forced: boolean;
  channel: string[];
  features: (change?: Partial<FeatureConfigs>) => CandidateFeaturesLikely;
  /** The features with a game memo of the caller's (a warm cache). */
  features2: (memo: ReturnType<typeof createFeatureMemo>) => CandidateFeaturesLikely;
  state: PublicState;
}

/** p1's move and forced-switch requests of one battle, each with a closure that builds the `/2` features the way the policy side would. */
async function points(battle: RandomBattleFile, p2: PokemonSet[]): Promise<Point[]> {
  const out: Point[] = [];
  const p1: Actor = (request, context) => {
    if ((request.phase === "move" || request.phase === "forceSwitch") && request.legal !== undefined && context !== undefined && context.team !== undefined) {
      const legalSet = request.legal;
      const candidates = [...legalSet.choices, ...legalSet.pending].map((choice) => ({ choice, pending: legalSet.pending.includes(choice) }));
      const lines = [...context.channel];
      const team = context.team;
      out.push({
        seq: request.request_seq,
        forced: request.phase === "forceSwitch",
        channel: lines,
        features: (change = {}) => candidateFeaturesLikely(engine, configs(change), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }), team, candidates),
        features2: (memo) => candidateFeaturesLikely(engine, configs(), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }), team, candidates, memo),
        state: stateOf({ context: { ...context, channel: lines } } as never),
      });
    }
    return firstLegal(request, context as never);
  };
  await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: { p1: battle.teams.p1, p2 }, actors: { p1, p2: firstLegal }, maxTurns: MAX_TURNS });
  return out;
}

// ---------------------------------------------------------------- the opponent variants

/** Apply `change` to set `index` of `team` and keep it only while the whole team stays legal. */
function tryChange(team: PokemonSet[], index: number, change: (set: PokemonSet) => void): boolean {
  const copy = structuredClone(team[index] as PokemonSet);
  change(copy);
  const next = team.map((set, i) => (i === index ? copy : set));
  if (!legal(next)) return false;
  team[index] = copy;
  return true;
}

/** Give set `index` the item `name`, the holder (if any) taking the item set `index` held (so the Item Clause holds); kept only while legal. */
function giveItem(team: PokemonSet[], index: number, name: string): boolean {
  const holder = team.findIndex((set) => toId(set.item) === toId(name));
  if (holder === index) return true;
  const next = team.map((set) => structuredClone(set));
  const before = (next[index] as PokemonSet).item;
  (next[index] as PokemonSet).item = name;
  if (holder >= 0) (next[holder] as PokemonSet).item = before;
  if (!legal(next)) return false;
  next.forEach((set, i) => void (team[i] = set));
  return true;
}

const ITEMS = { scarf: "Choice Scarf", claw: "Quick Claw" };
/** An unrevealed Choice Scarf and Quick Claw on the first two sets, a Speed-raising nature and the Stat Points moved to Speed on the odd ones, a Speed-lowering nature and the Stat Points moved off Speed on the even ones. */
function speedKit(team: PokemonSet[]): PokemonSet[] {
  const out = team.map((set) => structuredClone(set));
  expect(dex.items.get(toId(ITEMS.scarf)).exists && dex.items.get(toId(ITEMS.claw)).exists).toBe(true);
  giveItem(out, 0, ITEMS.scarf);
  giveItem(out, 1, ITEMS.claw);
  out.forEach((_, index) =>
    tryChange(out, index, (set) => {
      const total = Object.values(set.evs).reduce((a, b) => a + b, 0);
      const fast = index % 2 === 1;
      set.nature = fast ? "Jolly" : "Brave";
      set.evs = fast ? { hp: 0, atk: 0, def: 0, spa: 0, spd: Math.max(0, total - 32), spe: Math.min(32, total) } : { ...set.evs, spe: 0, hp: set.evs.hp + set.evs.spe };
    }),
  );
  return out;
}

/** Every set that can holds the roster's other ability, a priority-changing one first (the engine's data says which: `onModifyPriority`). */
function abilityKit(team: PokemonSet[]): PokemonSet[] {
  const out = team.map((set) => structuredClone(set));
  out.forEach((_, index) =>
    tryChange(out, index, (set) => {
      const options = (rosterOf(set)?.abilities ?? []).filter((id) => id !== toId(set.ability));
      const pick = options.find((id) => dex.abilities.get(id).onModifyPriority !== undefined) ?? options[0];
      if (pick !== undefined) set.ability = dex.abilities.get(pick).name;
    }),
  );
  return out;
}

/** Moves 2 to 4 of each set (move 1 is the one `firstLegal` plays, so it is revealed): `want` picks from the roster pool, the rest stays. */
function moveKit(team: PokemonSet[], want: (move: ReturnType<typeof dex.moves.get>) => boolean): PokemonSet[] {
  const out = team.map((set) => structuredClone(set));
  out.forEach((set, index) => {
    const pool = rosterOf(set)?.moves ?? [];
    for (let slot = 1; slot < set.moves.length; slot++) {
      const taken = (out[index] as PokemonSet).moves.map(toId);
      const choice = pool.find((id) => !taken.includes(id) && want(dex.moves.get(id)));
      if (choice === undefined) continue;
      tryChange(out, index, (copy) => void (copy.moves[slot] = dex.moves.get(choice).name));
    }
  });
  return out;
}
const priorityMoves = (team: PokemonSet[]) => moveKit(team, (move) => move.priority > 0 && move.category !== "Status");
const damagingMoves = (team: PokemonSet[]) => moveKit(team, (move) => move.category !== "Status" && move.priority <= 0);
const compose = (...kits: ((team: PokemonSet[]) => PokemonSet[])[]) => (team: PokemonSet[]) => kits.reduce((acc, kit) => kit(acc), team);

const lead = (...names: string[]) => (team: PokemonSet[]): PokemonSet[] => {
  const out = team.map((set) => structuredClone(set));
  names.forEach((name, index) => void tryChange(out, index, (set) => void (set.moves = [name, ...set.moves.filter((m) => toId(m) !== toId(name)).slice(0, 3)].slice(0, set.moves.length))));
  return out;
};
const PAIRS: { id: string; prepare?: (team: PokemonSet[]) => PokemonSet[]; variant: (team: PokemonSet[]) => PokemonSet[] }[] = [
  { id: "b06", variant: compose(speedKit, abilityKit, priorityMoves) },
  { id: "b12", variant: compose(speedKit, damagingMoves) },
  { id: "b02", variant: compose(speedKit, abilityKit) },
  { id: "b27", variant: damagingMoves },
  { id: "b00", prepare: lead("Tailwind", "Trick Room"), variant: compose(priorityMoves, damagingMoves) },
];

const bytes = (f: CandidateFeaturesLikely): string => jcs(f as unknown as Json);
const byId = (id: string): RandomBattleFile => corpus.find((b) => b.id === id) as RandomBattleFile;

/** Request points of a pair whose p1 channels are equal (the prefix up to and including the request). */
function equalPoints(a: Point[], b: Point[]): { a: Point; b: Point }[] {
  const out: { a: Point; b: Point }[] = [];
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const [left, right] = [a[i] as Point, b[i] as Point];
    if (left.seq !== right.seq || jcs(left.channel as unknown as Json) !== jcs(right.channel as unknown as Json)) break;
    out.push({ a: left, b: right });
  }
  return out;
}

/** What the real opponent holds, as a builder handed the real team would build the assumed set (the negative control). */
const cheating = (real: PokemonSet[]) => (mon: PublicMon): AssumedSet => {
  const set = structuredClone(real.find((s) => [mon.species, mon.base_species].includes(toId(s.species)))) as PokemonSet;
  return { set, paste: exportPaste(engine, [structuredClone(set)]), revealed_moves: set.moves.map(toId), filler_moves: [], unplaceable_moves: [], shown_hp_percent: mon.fainted ? 0 : mon.hp_num, mega: mon.mega };
};

const cache: { id: string; base: PokemonSet[]; other: PokemonSet[]; equal: { a: Point; b: Point }[] }[] = [];
async function allPairs(): Promise<typeof cache> {
  if (cache.length > 0) return cache;
  for (const { id, prepare, variant } of PAIRS) {
    const battle = byId(id);
    const teamBase = prepare === undefined ? battle.teams.p2 : prepare(battle.teams.p2);
    const other = variant(teamBase);
    expect(JSON.stringify(other), `${id} differs`).not.toBe(JSON.stringify(teamBase));
    cache.push({ id, base: teamBase, other, equal: equalPoints(await points(battle, teamBase), await points(battle, other)) });
  }
  return cache;
}

function tryFeatures(point: Point, change?: Partial<FeatureConfigs>): CandidateFeaturesLikely | null {
  try {
    return point.features(change);
  } catch (error) {
    if (error instanceof FeatureBuildError) return null;
    throw error;
  }
}

describe("hidden-information invariance of the probe-outcome-likely/1 features (ADR-022 E-22.7.6)", () => {
  it("gives byte-identical features, belief_threats and feature_set_hash at every request with an equal player channel", async () => {
    const seen = { refused: 0, total: 0, pairs: 0, likely: 0, afterMoveReveal: 0, forced: 0, speedControl: 0 };
    for (const { id, equal } of await allPairs()) {
      if (equal.length > 0) seen.pairs += 1;
      for (const { a, b } of equal) {
        const [left, right] = [tryFeatures(a), tryFeatures(b)];
        expect(right === null, `${id} request ${a.seq}: a refusal is the same in both battles`).toBe(left === null);
        if (left === null || right === null) {
          seen.refused += 1;
          continue;
        }
        expect(left.feature_set).toBe(FEATURE_SET_LIKELY);
        expect(bytes(right), `${id} request ${a.seq}`).toBe(bytes(left));
        seen.total += 1;
        if (left.belief_threats.length > 0) seen.likely += 1;
        if (Object.values(a.state.opponent.roster).some((mon) => mon.moves_revealed.length > 0)) seen.afterMoveReveal += 1;
        if (a.forced) seen.forced += 1;
        if (a.state.field_conditions.includes("trickroom")) seen.speedControl += 1;
      }
    }
    expect(seen.pairs).toBeGreaterThanOrEqual(2);
    expect(seen.refused, JSON.stringify(seen)).toBeLessThanOrEqual(seen.total / 4);
    expect(seen.total).toBeGreaterThanOrEqual(12);
    // not vacuous: likely rows are really there, before and after a move reveal
    expect(seen.likely, JSON.stringify(seen)).toBeGreaterThanOrEqual(10);
    expect(seen.afterMoveReveal).toBeGreaterThanOrEqual(5);
  }, 900_000);

  it("is a function of the foe's species and revealed moves: a likely row is never a revealed move, and a foe's rows exist even though it has revealed nothing", async () => {
    let silent = 0;
    for (const { id, equal } of await allPairs()) {
      for (const { a } of equal) {
        const features = tryFeatures(a);
        if (features === null) continue;
        for (const row of features.belief_threats) {
          const letter = Object.keys(a.state.opponent.slots).sort()[row.foe - 1] as string;
          const foe = a.state.opponent.roster[a.state.opponent.slots[letter] as number] as PublicMon;
          expect(foe.moves_revealed.map(toId), `${id} request ${a.seq}`).not.toContain(row.move);
          if (foe.moves_revealed.length === 0) silent += 1;
        }
      }
    }
    expect(silent).toBeGreaterThan(0);
  }, 900_000);

  it("NEGATIVE CONTROL: a builder handed the real opponent team through the assume seam gives different likely rows", async () => {
    let differing = 0;
    let compared = 0;
    for (const { id, other, equal } of await allPairs()) {
      for (const { b } of equal.slice(0, 6)) {
        const honest = tryFeatures(b);
        const cheat = tryFeatures(b, { assume: cheating(other) });
        if (honest === null || cheat === null || honest.belief_threats.length === 0) continue;
        compared += 1;
        if (jcs(honest.belief_threats as unknown as Json) !== jcs(cheat.belief_threats as unknown as Json)) differing += 1;
        void id;
      }
    }
    expect(compared).toBeGreaterThan(3);
    expect(differing, `${differing} of ${compared}`).toBeGreaterThan(0);
  }, 900_000);

  it("CANARY: no real unrevealed move of the opponent appears in the features unless the table proposes it", async () => {
    const real = (team: PokemonSet[]): Set<string> => new Set(team.flatMap((set) => set.moves.map(toId)));
    let checked = 0;
    for (const { id, base: p2base, other, equal } of await allPairs()) {
      const realMoves = new Set([...real(p2base), ...real(other)]);
      // a table of decoys: for each species the roster holds, two damaging pool moves that no real set of either battle holds
      const table = new Map<string, MoveList>();
      for (const entry of vocabulary.roster) {
        const decoys = entry.moves.filter((m) => !realMoves.has(m) && dex.moves.get(m).exists && dex.moves.get(m).category !== "Status" && foeTargetKind(dex.moves.get(m).target) !== "none").slice(0, 2);
        table.set(entry.id, { slots: 4, total: 1_000_000, masses: new Map(decoys.map((m) => [m, 100_000])) });
      }
      const decoyOnly: LikelySetup = { config: base.config, table };
      for (const { a } of equal.slice(0, 5)) {
        const features = tryFeatures(a, { likely: decoyOnly });
        if (features === null) continue;
        const text = jcs(features.belief_threats as unknown as Json);
        const revealed = new Set(Object.values(a.state.opponent.roster).flatMap((mon) => mon.moves_revealed.map(toId)));
        for (const move of realMoves) {
          if (revealed.has(move)) continue;
          expect(text.includes(`"${move}"`), `${id} request ${a.seq}: the real unrevealed move ${move}`).toBe(false);
        }
        checked += 1;
      }
      // negative control: a table that proposes the real moves (as one built from the real team would) is flagged by the same scan
      const proposing: LikelySetup = { config: base.config, table: new Map(vocabulary.roster.map((entry) => [entry.id, { slots: 4, total: 1_000_000, masses: new Map([...realMoves].filter((m) => entry.moves.includes(m)).map((m) => [m, 100_000])) }])) };
      let flagged = false;
      for (const { a } of equal.slice(0, 5)) {
        const features = tryFeatures(a, { likely: proposing });
        if (features === null) continue;
        const text = jcs(features.belief_threats as unknown as Json);
        const revealed = new Set(Object.values(a.state.opponent.roster).flatMap((mon) => mon.moves_revealed.map(toId)));
        if ([...realMoves].some((m) => !revealed.has(m) && text.includes(`"${m}"`))) flagged = true;
      }
      if (equal.length > 0 && id === "b06") expect(flagged).toBe(true);
    }
    expect(checked).toBeGreaterThanOrEqual(5);
  }, 900_000);

  it("a warm game memo gives the bytes of a cold one at every request of a battle", async () => {
    let hits = 0;
    let compared = 0;
    for (const { equal } of await allPairs()) {
      const memo = createFeatureMemo();
      for (const { a } of equal) {
        const cold = tryFeatures(a);
        let warm: CandidateFeaturesLikely | null = null;
        try {
          warm = a.features2(memo);
        } catch (error) {
          if (!(error instanceof FeatureBuildError)) throw error;
        }
        expect(warm === null).toBe(cold === null);
        if (warm !== null && cold !== null) {
          expect(bytes(warm)).toBe(bytes(cold));
          compared += 1;
        }
      }
      hits += (memo.likely?.counts.full_hits ?? 0) + (memo.likely?.counts.base_hits ?? 0) + (memo.likely?.counts.placement_hits ?? 0);
    }
    expect(compared).toBeGreaterThan(5);
    expect(hits).toBeGreaterThan(0);
  }, 900_000);
});

// keep the unused-import checker honest
void ({} as PublicState);
