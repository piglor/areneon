import { beforeAll, describe, expect, it } from "vitest";
import { interpretChannel, type PublicMon, type PublicState } from "@areneon/public-state";
import { type AssumedSet, AssumedSetError, assumedHp, buildAssumedSet } from "../src/damage-aware-assumed-set.ts";
import { type DamageAwareConfig, loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { type RandomBattleFile, loadRandomCorpus } from "../src/random-corpus.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

let engine: Engine;
let vocabulary: TeamVocabulary;
let config: DamageAwareConfig;
let corpus: RandomBattleFile[];
beforeAll(() => {
  engine = loadEngine();
  vocabulary = goldenTeamVocabulary();
  config = loadDamageAwareConfig("v1", vocabulary);
  corpus = loadRandomCorpus();
});

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const context = () => ({ engine, vocabulary, config });
const venusaur = (): TeamVocabulary["roster"][number] => {
  const entry = vocabulary.roster.find((r) => r.id === "venusaur");
  if (entry === undefined) throw new Error("venusaur is not in the roster");
  return entry;
};
const mon = (change: (m: PublicMon) => void = () => {}): PublicMon => {
  const base: PublicMon = {
    own: false, name: "Foe", base_species: "venusaur", species: "venusaur", temp_species: null, level: 50, gender: "M", shiny: false,
    hp_num: 100, hp_den: 100, status: null, fainted: false, brought: true, active_slot: "a",
    boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0 }, volatiles: [], moves_revealed: [], item: null,
    item_known: false, item_lost: false, ability: null, ability_suppressed: false, mega: false, mega_stone: null, stats: null,
    moves_own: null, pp: {}, can_mega_evo: false,
  };
  change(base);
  return base;
};

/** Every distinct opponent Pokemon the player's view shows at a request, with the real opponent set it is (for the canary only). */
interface Case { id: string; mon: PublicMon; real: PokemonSet | undefined; state: PublicState }
function* cases(): Generator<Case> {
  for (const battle of corpus) {
    for (const player of ["p1", "p2"] as const) {
      const foe = player === "p1" ? "p2" : "p1";
      const run = interpretChannel(player, battle.channels[player].lines);
      let requests = 0;
      for (const frame of run.frames) {
        if (frame.request === requests) continue;
        requests = frame.request;
        for (const m of frame.state.opponent.roster) {
          if (m.species === null) continue;
          const real = (battle.teams[foe] as PokemonSet[]).find((s) => toId(s.species) === m.base_species || toId(s.species) === m.species);
          yield { id: `${battle.id}/${player}/${frame.index}/${m.species}`, mon: m, real, state: frame.state };
        }
      }
    }
  }
}

/** The ADR-018 section 3 canary: nothing in an assumed set but revealed facts and the record's rules. */
function problems(assumed: AssumedSet, m: PublicMon): string[] {
  const out: string[] = [];
  const rule = config.assumed_set;
  const natureName = vocabulary.natures.find((n) => n.id === rule.nature)?.name;
  if (assumed.set.nature !== natureName) out.push("nature is not the record's");
  if (JSON.stringify(assumed.set.evs) !== JSON.stringify(rule.stat_points)) out.push("Stat Points are not the record's");
  const ruleItem = rule.item.kind === "fixed" ? rule.item.item : "";
  const item = m.mega ? m.mega_stone : m.item_known ? m.item : ruleItem;
  if (toId(assumed.set.item) !== (item ?? "")) out.push("item is neither revealed nor the record's");
  const abilities = vocabulary.roster.find((r) => r.id === toId(assumed.set.species))?.abilities ?? [];
  const ruleAbility = rule.ability === "first_in_vocabulary_order" ? abilities[0] : abilities[abilities.length - 1];
  if (![m.ability, ruleAbility].includes(toId(assumed.set.ability))) out.push("ability is neither revealed nor the record's");
  const held = assumed.set.moves.map(toId);
  if (held.some((move) => !m.moves_revealed.includes(move) && !assumed.filler_moves.includes(move))) out.push("a move was invented");
  if (assumed.revealed_moves.some((move) => assumed.filler_moves.includes(move))) out.push("a filler move is also a threat");
  if (assumed.filler_moves.length > 0 && assumed.revealed_moves.length > 0) out.push("a filler move beside a revealed one");
  return out;
}

describe("assumed opponent sets over the committed random-corpus channels", () => {
  it("builds a legal set for every opponent Pokemon at every request, with no unrevealed field of the real team in it", () => {
    const seen = new Set<string>();
    let built = 0;
    let differing = 0;
    const checker = engine.sim.TeamValidator.get(GOLDEN_FORMAT);
    for (const c of cases()) {
      const key = JSON.stringify(c.mon);
      if (seen.has(key)) continue;
      seen.add(key);
      const assumed = buildAssumedSet(context(), c.mon);
      built++;
      expect(checker.validateSet(structuredClone(assumed.set), {}) ?? [], c.id).toEqual([]);
      expect(problems(assumed, c.mon), c.id).toEqual([]);
      // revealed facts are used as revealed
      const inRoster = c.mon.species !== null && vocabulary.roster.some((r) => r.id === c.mon.species);
      expect(toId(assumed.set.species)).toBe(c.mon.mega || !inRoster ? c.mon.base_species : c.mon.species);
      for (const move of assumed.revealed_moves) expect(c.mon.moves_revealed).toContain(move);
      expect(assumed.shown_hp_percent).toBe(c.mon.fainted ? 0 : c.mon.hp_num);
      if (c.real !== undefined && (c.real.nature !== assumed.set.nature || JSON.stringify(c.real.evs) !== JSON.stringify(assumed.set.evs))) differing++;
    }
    expect(built).toBeGreaterThan(200);
    // the check is not vacuous: the real teams hold natures and spreads other than the assumed ones
    expect(differing).toBeGreaterThan(built / 2);
  });

  it("the canary has teeth: an assumed set that takes the real team's nature, spread, item or a hidden move is flagged", () => {
    let flagged = 0;
    for (const c of cases()) {
      if (c.real === undefined || c.mon.moves_revealed.length === 0) continue;
      const real = c.real;
      const honest = buildAssumedSet(context(), c.mon);
      expect(problems(honest, c.mon)).toEqual([]);
      const leaks: Array<(a: AssumedSet) => void> = [
        (a) => (a.set.nature = real.nature === a.set.nature ? "Adamant" : real.nature),
        (a) => (a.set.evs = { ...a.set.evs, hp: a.set.evs.hp + 1 }),
        (a) => (a.set.item = "Leftovers"),
        (a) => a.set.moves.push("hiddenmove"),
      ];
      for (const leak of leaks) {
        const copy = structuredClone(honest);
        leak(copy);
        expect(problems(copy, c.mon).length, c.id).toBeGreaterThan(0);
        flagged++;
      }
      if (flagged >= 40) break;
    }
    expect(flagged).toBeGreaterThanOrEqual(40);
  });

  it("refuses a record whose nature the vocabulary lacks", () => {
    const odd = { ...config, assumed_set: { ...config.assumed_set, nature: "grumpy" } };
    expect(() => buildAssumedSet({ ...context(), config: odd }, mon())).toThrow("nature");
  });
});

describe("buildAssumedSet", () => {
  it("puts a revealed move first and never adds a filler beside it", () => {
    const assumed = buildAssumedSet(context(), mon((m) => (m.moves_revealed = ["protect"])));
    expect(assumed.revealed_moves).toEqual(["protect"]);
    expect(assumed.filler_moves).toEqual([]);
    expect(assumed.paste).toContain("- Protect");
  });

  it("marks a filler move when nothing is revealed, taken from the species' pool by the record's rule", () => {
    const entry = venusaur();
    const first = buildAssumedSet(context(), mon());
    expect(first.revealed_moves).toEqual([]);
    expect(first.filler_moves).toEqual([entry.moves[0]]);
    const last = buildAssumedSet({ ...context(), config: { ...config, assumed_set: { ...config.assumed_set, filler_move: "last_in_vocabulary_order", ability: "last_in_vocabulary_order" } } }, mon());
    expect(last.filler_moves).toEqual([entry.moves.at(-1)]);
    expect(toId(last.set.ability)).toBe(entry.abilities.at(-1));
  });

  it("uses a revealed item, ability, level and gender as revealed, and a fixed item for an unrevealed one", () => {
    const revealed = buildAssumedSet(context(), mon((m) => { m.item = "leftovers"; m.item_known = true; m.ability = "overgrow"; m.level = 50; m.gender = "F"; }));
    expect(revealed.set.item).toBe("Leftovers");
    expect(revealed.set.ability).toBe("Overgrow");
    expect(revealed.set.gender).toBe("F");
    const none = buildAssumedSet(context(), mon((m) => { m.item_known = true; m.item = null; m.item_lost = true; }));
    expect(none.set.item).toBe("");
    const item = "leftovers";
    expect(vocabulary.items).toContain(item);
    const fixed = buildAssumedSet({ ...context(), config: { ...config, assumed_set: { ...config.assumed_set, item: { kind: "fixed", item } } } }, mon());
    expect(fixed.set.item).toBe("Leftovers");
  });

  it("ignores a revealed ability the species' vocabulary does not list, and falls back to the level of the rules", () => {
    const assumed = buildAssumedSet(context(), mon((m) => { m.ability = "intimidate"; m.level = null; m.gender = null; }));
    expect(toId(assumed.set.ability)).toBe(venusaur().abilities[0]);
    expect(assumed.set.level).toBe(vocabulary.rules.adjust_level);
    expect(assumed.set.gender).toBe("");
  });

  it("sets a Mega Evolved Pokemon as its base forme holding the revealed stone, flagged mega", () => {
    const assumed = buildAssumedSet(context(), mon((m) => { m.mega = true; m.mega_stone = "venusaurite"; m.species = "venusaurmega"; }));
    expect(assumed.set.species).toBe("Venusaur");
    expect(assumed.set.item).toBe("Venusaurite");
    expect(assumed.mega).toBe(true);
  });

  it("lists a revealed move no legal set can hold as unplaceable, and keeps the set legal", () => {
    const assumed = buildAssumedSet(context(), mon((m) => (m.moves_revealed = ["protect", "struggle"])));
    expect(assumed.revealed_moves).toEqual(["protect"]);
    expect(assumed.unplaceable_moves).toEqual(["struggle"]);
  });

  it("shows 0 for a fainted Pokemon and null before one was seen", () => {
    expect(buildAssumedSet(context(), mon((m) => { m.fainted = true; m.hp_num = 0; })).shown_hp_percent).toBe(0);
    expect(buildAssumedSet(context(), mon((m) => { m.hp_num = null; m.hp_den = null; })).shown_hp_percent).toBeNull();
  });

  it("refuses what it cannot build or what is not an opponent", () => {
    expect(() => buildAssumedSet(context(), mon((m) => (m.own = true)))).toThrow(AssumedSetError);
    expect(() => buildAssumedSet(context(), mon((m) => { m.species = "notamon"; m.base_species = "notamon"; }))).toThrow("roster");
    expect(() => buildAssumedSet(context(), mon((m) => (m.hp_den = 160)))).toThrow("out of 100");
    const five = venusaur().moves.slice(0, 5);
    expect(() => buildAssumedSet(context(), mon((m) => (m.moves_revealed = five)))).toThrow("no legal set");
    expect(() => buildAssumedSet(context(), mon((m) => (m.item = "notanitem", m.item_known = true)))).toThrow("no legal set");
  });

  it("refuses a species with no ability or move in the vocabulary", () => {
    const empty: TeamVocabulary = { ...vocabulary, roster: vocabulary.roster.map((r) => (r.id === "venusaur" ? { ...r, abilities: [] as never, moves: [] as never } : r)) };
    expect(() => buildAssumedSet({ ...context(), vocabulary: empty }, mon())).toThrow("no ability");
    const noMoves: TeamVocabulary = { ...vocabulary, roster: vocabulary.roster.map((r) => (r.id === "venusaur" ? { ...r, moves: [] as never } : r)) };
    expect(() => buildAssumedSet({ ...context(), vocabulary: noMoves }, mon())).toThrow("no move");
  });
});

describe("assumedHp", () => {
  it.each([
    ["nearest", 33, 200, 66],
    ["nearest", 1, 150, 2],
    ["nearest", 50, 151, 76],
    ["floor", 33, 151, 49],
    ["ceil", 33, 151, 50],
    ["floor", 1, 50, 1],
    ["ceil", 100, 151, 151],
    ["nearest", 100, 151, 151],
  ] as const)("%s of %i%% of %i is %i", (rounding, percent, max, expected) => {
    expect(assumedHp(percent, max, rounding)).toBe(expected);
  });

  it("is 0 only for 0%, and stays within 1..max otherwise", () => {
    expect(assumedHp(0, 150, "ceil")).toBe(0);
    for (const rounding of ["floor", "ceil", "nearest"] as const) {
      for (let percent = 1; percent <= 100; percent++) {
        const hp = assumedHp(percent, 7, rounding);
        expect(hp).toBeGreaterThanOrEqual(1);
        expect(hp).toBeLessThanOrEqual(7);
      }
    }
  });

  it("refuses a value that is not a percentage or a positive maximum", () => {
    expect(() => assumedHp(101, 100, "ceil")).toThrow(AssumedSetError);
    expect(() => assumedHp(50.5, 100, "ceil")).toThrow("percentage");
    expect(() => assumedHp(50, 0, "ceil")).toThrow("maximum hit points");
  });
});
