import { describe, expect, it } from "vitest";
import type { PublicMon, PublicState } from "@areneon/public-state";
import { type CandidateFeatures2, candidateFeatures2, FEATURE_SET_2 } from "../src/candidate-features-2.ts";
import { FeatureBuildError, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { type AssumedSet } from "../src/damage-aware-assumed-set.ts";
import { foeTargetKind } from "../src/damage-aware-view.ts";
import type { PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_FORMAT } from "../src/golden.ts";
import { type BattleRecord, type HiddenVerdicts, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { probeMechanics as engineProbe } from "../src/mechanics-probe.ts";
import { playerChannel } from "../src/player-channel.ts";
import type { RandomBattleFile } from "../src/random-corpus.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { exportPaste } from "../src/teams.ts";
import { config, corpus, engine, mechanics, snapshot, stateOf, toId, vocabulary } from "./damage-aware-support.ts";

// ADR-022 section 3 (privacy test, required), R1 and erratum E-22.3.5, ticket #1173: the `probe-damage/2` arrays (`rows`,
// `threats`, `speeds`, `own_priorities`, `foe_priorities`) and `feature_set_hash` are a function of the player's own channel,
// own team and candidates. Two battles whose player channels agree up to a request, but whose opponents hold different
// unrevealed fields where turn order and incoming damage would leak most (a Choice Scarf or Quick Claw, the Speed nature and
// Stat Points, a priority-changing ability, unrevealed priority and damaging moves), give byte-identical arrays there. The foe
// side of those arrays rests on the assumed set (Hardy, no item, the first ability, revealed facts as revealed), so this is
// where a leak of the real team would show. The REAL probe is used (a stand-in would not see a stat, an item or an ability).
// p1 and p2 both play `firstLegal`, so the pair differs only in what p2 holds.

const MAX_TURNS = 10;
const configs = (change: Partial<FeatureConfigs> = {}): FeatureConfigs => ({ vocabulary, config, mechanics, snapshot, data_class: "synthetic", ...change });
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
  features: (change?: Partial<FeatureConfigs>) => CandidateFeatures2;
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
        features: (change = {}) => candidateFeatures2(engine, configs(change), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines }), team, candidates),
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
const priorityStatus = (team: PokemonSet[]) => moveKit(team, (move) => move.priority > 0);
const compose = (...kits: ((team: PokemonSet[]) => PokemonSet[])[]) => (team: PokemonSet[]) => kits.reduce((acc, kit) => kit(acc), team);

// Pairs differ in (a) the Speed kit alone, (b) abilities (a priority-changing one where a roster allows), (c) unrevealed priority
// and damaging moves, and combinations. A change of the item, nature or Stat Points can reorder turn 1 and so end the equal prefix early;
// the move-only pairs keep the channels equal for longer, across the reveal of move 1.
/** The first move of the first sets replaced by `names` (what `firstLegal` plays at once, so it is revealed and shapes the field in both battles of a pair); kept only while legal. */
const lead = (...names: string[]) => (team: PokemonSet[]): PokemonSet[] => {
  const out = team.map((set) => structuredClone(set));
  names.forEach((name, index) => void tryChange(out, index, (set) => void (set.moves = [name, ...set.moves.filter((m) => toId(m) !== toId(name)).slice(0, 3)].slice(0, set.moves.length))));
  return out;
};
const PAIRS: { id: string; prepare?: (team: PokemonSet[]) => PokemonSet[]; variant: (team: PokemonSet[]) => PokemonSet[] }[] = [
  { id: "b06", variant: compose(speedKit, abilityKit, priorityMoves) },
  { id: "b12", variant: compose(speedKit, damagingMoves) },
  { id: "b01", variant: compose(abilityKit, priorityStatus) },
  { id: "b02", variant: compose(speedKit, abilityKit) },
  { id: "b00", variant: priorityMoves },
  { id: "b03", variant: damagingMoves },
  { id: "b04", variant: priorityStatus },
  { id: "b05", variant: compose(priorityMoves, damagingMoves) },
  { id: "b14", variant: priorityMoves }, // reveal: Kingambit's Pressure at request 2
  { id: "b27", variant: damagingMoves }, // reveal: Krookodile's Intimidate at request 2
  { id: "b20", variant: priorityStatus }, // reveal: Gourgeist's Shuca Berry at request 2
  { id: "b18", variant: compose(priorityMoves, damagingMoves) },
  { id: "b34", variant: priorityStatus },
  { id: "b28", variant: compose(damagingMoves, priorityMoves) },
  // Trick Room and Tailwind go up on turn 1 in both battles of the pair (both leads play them), so later requests have speed control up
  { id: "b00", prepare: lead("Tailwind", "Trick Room"), variant: compose(priorityMoves, damagingMoves) },
  { id: "b24", prepare: lead("Tailwind", "Tailwind"), variant: priorityStatus },
  // Sableye, Klefki and Whimsicott have Prankster as their other ability: the priority-changing ability is really swapped in
  { id: "b04", variant: compose(abilityKit, priorityStatus) },
  { id: "b13", variant: compose(abilityKit, damagingMoves) },
  { id: "b31", prepare: lead("Tailwind", "Trick Room"), variant: compose(abilityKit, damagingMoves) },
];

const bytes = (f: CandidateFeatures2): string => jcs(f as unknown as Json);
const part = (f: CandidateFeatures2, key: keyof CandidateFeatures2): string => jcs(f[key] as unknown as Json);
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

/** What the real opponent holds: the negative control's assumed set, as a builder that was handed the real team would build it. */
const cheating = (real: PokemonSet[]) => (mon: PublicMon): AssumedSet => {
  const set = structuredClone(real.find((s) => [mon.species, mon.base_species].includes(toId(s.species)))) as PokemonSet;
  return { set, paste: exportPaste(engine, [structuredClone(set)]), revealed_moves: set.moves.map(toId), filler_moves: [], unplaceable_moves: [], shown_hp_percent: mon.fainted ? 0 : mon.hp_num, mega: mon.mega };
};

/** Every pair, built twice, with its equal request points (one real-probe run of each battle per variant, shared by the tests). */
const cache: { id: string; base: PokemonSet[]; other: PokemonSet[]; equal: { a: Point; b: Point }[] }[] = [];
async function allPairs(): Promise<typeof cache> {
  if (cache.length > 0) return cache;
  for (const { id, prepare, variant } of PAIRS) {
    const battle = byId(id);
    const base = prepare === undefined ? battle.teams.p2 : prepare(battle.teams.p2);
    const other = variant(base);
    expect(JSON.stringify(other), `${id} differs`).not.toBe(JSON.stringify(base));
    cache.push({ id, base, other, equal: equalPoints(await points(battle, base), await points(battle, other)) });
  }
  return cache;
}

/** The features, or null when the builder refuses the request (`FeatureBuildError`, e.g. a line the shared reader cannot take): a refusal is a function of the player's view too, so it must be the same in both battles of a pair. */
function tryFeatures(point: Point, change?: Partial<FeatureConfigs>): CandidateFeatures2 | null {
  try {
    return point.features(change);
  } catch (error) {
    if (error instanceof FeatureBuildError) return null;
    throw error;
  }
}

/** The revealed moves of the opposing active in the foe location `loc` (the engine's choice location: slot letter order, 1 first). */
function foeAt(state: PublicState, loc: number): PublicMon | undefined {
  const letter = Object.keys(state.opponent.slots).sort()[loc - 1];
  const index = letter === undefined ? undefined : state.opponent.slots[letter];
  return index === null || index === undefined ? undefined : state.opponent.roster[index];
}
const reaches = (id: string): boolean => {
  const move = dex.moves.get(id);
  return move.exists && move.category !== "Status" && foeTargetKind(move.target) !== "none";
};

describe("hidden-information invariance of the probe-damage/2 arrays (ADR-022 section 3, R1, E-22.3.5)", () => {
  it("really changes what the pairs say they change (a legality fallback cannot silently drop a kind of difference)", async () => {
    const held = (team: PokemonSet[]) => team.map((set) => toId(set.item));
    const moveIds = (team: PokemonSet[]) => team.flatMap((set) => set.moves.map(toId));
    const seen = { scarf: 0, claw: 0, nature: 0, speed: 0, priorityAbility: 0, priorityMove: 0, damagingMove: 0, statusPriority: 0 };
    for (const { base, other } of await allPairs()) {
      if (held(other).includes(toId(ITEMS.scarf)) && !held(base).includes(toId(ITEMS.scarf))) seen.scarf += 1;
      if (held(other).includes(toId(ITEMS.claw)) && !held(base).includes(toId(ITEMS.claw))) seen.claw += 1;
      other.forEach((set, i) => {
        const before = base[i] as PokemonSet;
        if (set.nature !== before.nature) seen.nature += 1;
        if (set.evs.spe !== before.evs.spe) seen.speed += 1;
        if (set.ability !== before.ability && dex.abilities.get(toId(set.ability)).onModifyPriority !== undefined) seen.priorityAbility += 1;
      });
      const added = moveIds(other).filter((id) => !moveIds(base).includes(id)).map((id) => dex.moves.get(id));
      seen.priorityMove += added.filter((m) => m.priority > 0 && m.category !== "Status").length;
      seen.statusPriority += added.filter((m) => m.priority > 0 && m.category === "Status").length;
      seen.damagingMove += added.filter((m) => m.category !== "Status").length;
    }
    for (const [kind, count] of Object.entries(seen)) expect(count, `${kind} ${JSON.stringify(seen)}`).toBeGreaterThanOrEqual(1);
  }, 600_000);

  // The real probe on 20+ request points of seeded battle pairs, each built twice: tens of seconds, minutes on a loaded machine.
  it("gives byte-identical rows, threats, speeds, priorities and feature_set_hash at every request with an equal player channel", async () => {
    const seen = { refused: 0, total: 0, pairs: 0, threats: 0, foePriorities: 0, speeds: 0, ownPriorities: 0, afterMoveReveal: 0, afterReveal: 0, forced: 0, speedControl: 0 };
    for (const { id, equal } of await allPairs()) {
      if (equal.length > 0) seen.pairs += 1;
      for (const { a, b } of equal) {
        const [left, right] = [tryFeatures(a), tryFeatures(b)];
        expect(right === null, `${id} request ${a.seq}: a refusal is the same in both battles`).toBe(left === null);
        if (left === null || right === null) {
          seen.refused += 1;
          continue;
        }
        expect(left.feature_set).toBe(FEATURE_SET_2);
        expect(bytes(right), `${id} request ${a.seq}`).toBe(bytes(left));
        for (const key of ["rows", "threats", "speeds", "own_priorities", "foe_priorities"] as const) expect(part(right, key), `${id} request ${a.seq} ${key}`).toBe(part(left, key));
        expect(right.feature_set_hash).toBe(left.feature_set_hash);
        seen.total += 1;
        if (left.threats.length > 0) seen.threats += 1;
        if (left.speeds.length > 0) seen.speeds += 1;
        if (left.own_priorities.length > 0) seen.ownPriorities += 1;
        if (left.foe_priorities.length > 0) seen.foePriorities += 1;
        if (a.forced) seen.forced += 1;
        const foes = Object.values(a.state.opponent.roster);
        if (foes.some((mon) => mon.moves_revealed.length > 0)) seen.afterMoveReveal += 1;
        if (foes.some((mon) => mon.ability !== null || mon.item !== null)) seen.afterReveal += 1;
        if (a.state.field_conditions.includes("trickroom") || Object.keys(a.state.own.conditions).concat(Object.keys(a.state.opponent.conditions)).some((c) => c.includes("tailwind"))) seen.speedControl += 1;
      }
    }
    expect(seen.pairs).toBeGreaterThanOrEqual(2);
    expect(seen.refused, JSON.stringify(seen)).toBeLessThanOrEqual(seen.total / 4);
    expect(seen.total).toBeGreaterThanOrEqual(20);
    // not vacuous: every array is really there, after a move reveal and after an item or ability reveal
    expect(seen.threats).toBeGreaterThanOrEqual(5);
    expect(seen.speeds).toBeGreaterThanOrEqual(15);
    expect(seen.ownPriorities).toBeGreaterThanOrEqual(15);
    expect(seen.foePriorities).toBeGreaterThanOrEqual(5);
    expect(seen.afterMoveReveal).toBeGreaterThanOrEqual(5);
    expect(seen.afterReveal).toBeGreaterThanOrEqual(3);
    expect(seen.forced, JSON.stringify(seen)).toBeGreaterThanOrEqual(1);
    expect(seen.speedControl, JSON.stringify(seen)).toBeGreaterThanOrEqual(1);
  }, 600_000);

  it("uses what is revealed as revealed and nothing else: a threat or priority row exists only for a revealed move, and a foe with none has no row", async () => {
    let foeChecks = 0;
    let noDamagingFoe = 0;
    let noMoveFoe = 0;
    let revealedUsed = 0;
    for (const { id, equal } of await allPairs()) {
      for (const { a } of equal) {
        const features = tryFeatures(a);
        if (features === null) continue;
        for (const row of features.threats) {
          const foe = foeAt(a.state, row.foe);
          expect(foe, `${id} request ${a.seq}: a threat row names a foe in play`).toBeDefined();
          expect(foe?.moves_revealed ?? [], `${id} request ${a.seq}`).toContain(row.move);
          expect(reaches(row.move)).toBe(true);
          revealedUsed += 1;
        }
        for (const row of features.foe_priorities) {
          expect(foeAt(a.state, row.foe)?.moves_revealed ?? [], `${id} request ${a.seq}`).toContain(row.move);
          revealedUsed += 1;
        }
        for (const loc of [1, 2]) {
          const foe = foeAt(a.state, loc);
          if (foe === undefined) continue;
          foeChecks += 1;
          if (!foe.moves_revealed.some(reaches)) {
            noDamagingFoe += 1;
            expect(features.threats.filter((row) => row.foe === loc), `${id} request ${a.seq}: foe ${loc} has no revealed damaging move`).toEqual([]);
          }
          if (foe.moves_revealed.length === 0) {
            noMoveFoe += 1;
            expect(features.foe_priorities.filter((row) => row.foe === loc), `${id} request ${a.seq}: foe ${loc} has no revealed move`).toEqual([]);
          }
        }
      }
    }
    expect(foeChecks).toBeGreaterThanOrEqual(20);
    expect(noDamagingFoe).toBeGreaterThanOrEqual(5);
    expect(noMoveFoe).toBeGreaterThanOrEqual(3);
    expect(revealedUsed).toBeGreaterThanOrEqual(5);
  }, 600_000);

  it("declares a revealed ability, item and move to the probe as revealed, never the vocabulary default", async () => {
    let abilities = 0;
    let notDefault = 0;
    let items = 0;
    let moves = 0;
    for (const id of ["b14", "b27", "b20"]) {
      const battle = byId(id);
      const variant = PAIRS.find((pair) => pair.id === id)?.variant ?? ((team) => team);
      for (const point of await points(battle, variant(battle.teams.p2))) {
        const declared: string[] = [];
        const attackers: { paste: string; move: string }[] = [];
        const probe: NonNullable<FeatureConfigs["probe"]> = (context, query) => {
          const q = query.query;
          if (q.kind === "damage") {
            // a foe attacker is the one whose set is not on p1's side: threats name a foe move as the attacker's
            declared.push(q.attacker.set, q.defender.set);
            attackers.push({ paste: q.attacker.set, move: q.move });
          }
          if (q.kind === "speed_order") for (const side of q.sides) for (const combatant of side) declared.push(combatant.set);
          return engineProbe(context, query);
        };
        tryFeatures(point, { probe });
        const ownSpecies = battle.teams.p1.map((set) => toId(set.species));
        for (const mon of Object.values(point.state.opponent.roster)) {
          if (ownSpecies.some((species) => species.includes(toId(mon.base_species)))) continue; // b20: both sides hold a Gourgeist, so a paste cannot be told apart by species
          const mine = declared.filter((paste) => toId((paste.split("\n")[0] as string).split("@")[0] as string).includes(toId(mon.base_species)));
          // b20 is the item case only: there the reader attributes Gourgeist's Frisk to the Pokemon whose item it shows, which the assumed set ignores (Gardevoir has no Frisk)
          if (mon.ability !== null && id !== "b20") {
            for (const paste of mine) expect(toId(/^Ability: (.+)$/m.exec(paste)?.[1] ?? ""), `${id} request ${point.seq}`).toBe(mon.ability);
            abilities += mine.length;
            const first = vocabulary.roster.find((r) => r.id === mon.base_species)?.abilities[0];
            if (mine.length > 0 && first !== mon.ability) notDefault += 1;
          }
          if (mon.item !== null) {
            for (const paste of mine) expect(toId((paste.split("\n")[0] as string).split("@")[1] ?? ""), `${id} request ${point.seq}`).toBe(mon.item);
            items += mine.length;
          }
          // a foe attacker uses the move it was seen to use
          for (const attacker of attackers.filter((x) => toId((x.paste.split("\n")[0] as string).split("@")[0] as string).includes(toId(mon.base_species)))) {
            if (mon.moves_revealed.includes(attacker.move)) moves += 1;
          }
        }
      }
    }
    expect(abilities).toBeGreaterThanOrEqual(2);
    expect(notDefault).toBeGreaterThanOrEqual(1);
    expect(items).toBeGreaterThanOrEqual(1);
    expect(moves).toBeGreaterThanOrEqual(1);
  }, 600_000);

  // Negative control: the invariance comparison, pair for pair, with a builder handed the opponent's REAL team must differ on
  // each of threats, speeds and foe_priorities, so the test can see a leak in each.
  it("has teeth: a builder handed the real opponent team gives different threats, speeds and foe priorities at such a request", async () => {
    const differing = { threats: 0, speeds: 0, foe_priorities: 0, any: 0 };
    let compared = 0;
    for (const { base, other, equal } of await allPairs()) {
      for (const pair of equal) {
        compared += 1;
        const [left, right] = [tryFeatures(pair.a, { assume: cheating(base) }), tryFeatures(pair.b, { assume: cheating(other) })];
        if (left === null || right === null) continue;
        if (bytes(left) !== bytes(right)) differing.any += 1;
        for (const key of ["threats", "speeds", "foe_priorities"] as const) {
          if (part(left, key) !== part(right, key)) {
            differing[key] += 1;
          }
        }
      }
    }
    expect(compared).toBeGreaterThanOrEqual(20);
    expect(differing.threats).toBeGreaterThan(0);
    expect(differing.speeds).toBeGreaterThan(0);
    expect(differing.foe_priorities).toBeGreaterThan(0);
  }, 600_000);
});

describe("canary (ADR-022 section 3, E-22.3.5)", () => {
  /** The moves, items and natures of the unrevealed fields of `team`, as lower-case ids and names that the channel never shows. */
  const markers = (team: PokemonSet[], channel: string): { ids: string[]; names: string[] } => {
    const shown = channel.toLowerCase();
    const names = team.flatMap((set) => [set.item, set.nature, ...set.moves.slice(1)]).filter((name) => name !== "" && !shown.includes(name.toLowerCase())).map((name) => name.toLowerCase());
    return { ids: [...new Set(names.map(toId))], names: [...new Set(names)] };
  };
  const tokens = (value: unknown, into: string[] = []): string[] => {
    if (typeof value === "string") into.push(value);
    else if (Array.isArray(value)) value.forEach((v) => tokens(v, into));
    else if (value !== null && typeof value === "object") Object.entries(value).forEach(([k, v]) => (into.push(k), tokens(v, into)));
    return into;
  };

  it("shows no unrevealed move as a foe move id and no never-public field anywhere in the JCS bytes of any array", async () => {
    let scanned = 0;
    let withMoves = 0;
    for (const { id, other, base, equal } of await allPairs()) {
      for (const [team, pick] of [[base, "a"], [other, "b"]] as const) {
        for (const pair of equal) {
          const point = pair[pick];
          const features = tryFeatures(point);
          if (features === null) continue;
          const { ids, names } = markers(team, point.channel.join("\n"));
          expect(ids.length).toBeGreaterThan(0);
          const strings = tokens(features).map(toId);
          const text = bytes(features).toLowerCase();
          for (const marker of ids) expect(strings.includes(marker), `${id} request ${point.seq}: ${marker} is a token`).toBe(false);
          for (const marker of names) expect(text.includes(marker), `${id} request ${point.seq}: ${marker}`).toBe(false);
          for (const row of [...features.threats, ...features.foe_priorities]) expect(ids.includes(row.move), `${id} request ${point.seq}: foe move ${row.move}`).toBe(false);
          if (features.threats.length + features.foe_priorities.length > 0) withMoves += 1;
          scanned += 1;
        }
      }
    }
    expect(scanned).toBeGreaterThanOrEqual(40);
    expect(withMoves).toBeGreaterThanOrEqual(5);
  }, 600_000);

  it("is a scan that sees a marker when one is there (negative controls)", async () => {
    // the scan flags a leaking row set, and the real-team builder really does put unrevealed moves into foe_priorities
    const leaking = { threats: [{ move: "extremespeed" }], note: "Choice Scarf" } as unknown as Json;
    expect(tokens(leaking).map(toId)).toContain("extremespeed");
    expect(jcs(leaking).toLowerCase().includes("choice scarf")).toBe(true);
    let leaked = 0;
    for (const { base, equal } of await allPairs()) {
      for (const pair of equal.slice(0, 2)) {
        const { ids } = markers(base, pair.a.channel.join("\n"));
        const features = tryFeatures(pair.a, { assume: cheating(base) });
        if (features === null) continue;
        if ([...features.threats, ...features.foe_priorities].some((row) => ids.includes(row.move))) leaked += 1;
      }
    }
    expect(leaked).toBeGreaterThan(0);
  }, 600_000);
});

describe("the type rule (ADR-022 section 3)", () => {
  it("takes no opponent team, omniscient log, hidden verdicts, battle record or seed", () => {
    // Compile-time checks: `pnpm typecheck` fails if any of these compiles.
    const neverRun = (log: OmniscientLog, verdicts: HiddenVerdicts[], record: BattleRecord, seed: string, raw: { player: "p1"; lines: string[] }, channel: PlayerChannel): void => {
      const team = [] as never;
      // @ts-expect-error: the omniscient log is not a PlayerChannel
      void candidateFeatures2(engine, configs(), log, team, []);
      // @ts-expect-error: nor hidden verdicts
      void candidateFeatures2(engine, configs(), verdicts, team, []);
      // @ts-expect-error: nor a battle record
      void candidateFeatures2(engine, configs(), record, team, []);
      // @ts-expect-error: nor a seed
      void candidateFeatures2(engine, configs(), seed, team, []);
      // @ts-expect-error: nor an unchecked log
      void candidateFeatures2(engine, configs(), raw, team, []);
      // @ts-expect-error: the configs hold no opponent team
      void candidateFeatures2(engine, { ...configs(), opponent_team: [] }, channel, team, []);
      // @ts-expect-error: the team argument is the player's own team, not a record
      void candidateFeatures2(engine, configs(), channel, record, []);
    };
    expect(neverRun).toBeTypeOf("function");
  });
});
