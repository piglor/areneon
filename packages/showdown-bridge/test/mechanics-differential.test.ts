import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { calculate, toID, Field, Generations, Move, Pokemon } from "@smogon/calc";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, PokemonSet, SimModdedDex, StatTable } from "../src/engine.ts";
import { GOLDEN_FORMAT, syntheticTeam, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { type MechanicsContext, probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-016 section 4 item 3 (#935): an independent-tool comparison. Every damage
// and stats fixture of slices 1 and 3 runs through the probe and through
// @smogon/calc (a devDependency; the calc is an oracle only, never a runtime
// path, a fallback or a tie-breaker). Calc generation 0 is Champions (the
// calc's own data tables list CHAMPIONS first, and its stat formula for
// generation 0 is base + Stat Points + 20, base + Stat Points + 75 for HP).
// The known differences are listed in calc-divergences.json, each with a
// reason and the exact two values; the list is exact: an unlisted difference
// fails, and a listed entry that no longer differs (or differs otherwise)
// fails. Triage follows ADR-016 section 4: the in-game or hand-worked value
// decides, never the calc. The probe's damage already equals the hand-worked
// fixtures (mechanics-damage.test.ts) and its stats the game's displayed values
// (mechanics-stats.test.ts); a divergence is therefore the calc's, with the
// reason it gives.

const CALC_GEN = Generations.get(0);
const dex = (): SimModdedDex => engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod);
const FIXTURES = new URL("../../../fixtures/mechanics/gen9championsvgc2026regmc/", import.meta.url);

interface Divergence {
  case: string;
  reason: string;
  probe: unknown;
  calc: unknown;
}

const STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"] as const;
type Boosts = Record<"atk" | "def" | "spa" | "spd" | "spe", number>;

let engine: Engine;
let config: MechanicsConfig;
let context: MechanicsContext;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  context = { engine, config };
});

/** Every difference found, by case id. */
const found = new Map<string, { probe: unknown; calc: unknown }>();
const note = (id: string, probe: unknown, calc: unknown): void => {
  if (JSON.stringify(probe) !== JSON.stringify(calc)) found.set(id, { probe, calc });
};

/**
 * The name the calc's Champions table knows a species by: the engine's own name;
 * else the engine's base forme spelled out (the engine's Aegislash is the calc's
 * Aegislash-Shield); else, for a cosmetic forme the engine flags (Vivillon,
 * Alcremie: same stats and types), its base species. Anything else the calc
 * lacks stays as it is and is listed as a divergence.
 */
function calcSpeciesName(name: string): string {
  const known = (candidate: string): boolean => CALC_GEN.species.get(toID(candidate)) !== undefined;
  if (known(name)) return name;
  const species = dex().species.get(name) as { baseSpecies: string; baseForme?: string; isCosmeticForme?: boolean };
  const spelled = `${name}-${species.baseForme ?? ""}`;
  if (species.baseForme !== undefined && species.baseForme !== "" && known(spelled)) return spelled;
  return species.isCosmeticForme === true ? species.baseSpecies : name;
}

/** A calc Pokemon from a set as the engine reads it: Stat Points go in the calc's `evs`, a Mega forme by its own name. */
function calcPokemon(set: PokemonSet, options: { mega?: boolean; boosts?: Boosts; status?: string | null; hp?: number | null } = {}): Pokemon {
  let species = set.species;
  let ability = set.ability;
  if (options.mega === true) {
    const stone = dex().items.get(set.item);
    const forme = stone.megaStone === undefined ? "" : Object.values(stone.megaStone as Record<string, string>)[0];
    if (forme === undefined || forme === "") throw new Error("a Mega set needs a Mega Stone");
    species = forme;
    ability = ""; // the calc gives its Mega species the species' own ability
  }
  species = calcSpeciesName(species);
  return new Pokemon(CALC_GEN, species, {
    level: set.level,
    nature: set.nature,
    evs: { ...set.evs },
    ivs: { ...set.ivs },
    ...(ability === "" ? {} : { ability }),
    ...(set.item === "" ? {} : { item: set.item }),
    ...(options.boosts === undefined ? {} : { boosts: options.boosts }),
    ...(options.status == null ? {} : { status: options.status as never }),
    ...(options.hp == null ? {} : { curHP: options.hp }),
  });
}

const setOf = (paste: string): PokemonSet => {
  const sets = importPaste(engine, paste);
  if (sets.length !== 1) throw new Error("one set expected");
  return sets[0] as PokemonSet;
};

interface DamageCase {
  id: string;
  query: { query: { attacker: Declared; defender: Declared; move: string; state: State } };
  expected: { no_crit: number[]; crit: number[] };
}
interface Declared { set: string; boosts: Boosts; status: string | null; hp: number | null; mega: boolean }
interface State {
  weather: string | null;
  terrain: string | null;
  trick_room: boolean;
  sides: { aurora_veil: boolean; light_screen: boolean; reflect: boolean; tailwind: boolean }[];
}
const damageFixture = JSON.parse(readFileSync(new URL("damage/cases.json", FIXTURES), "utf8")) as { cases: DamageCase[] };

const WEATHER: Record<string, string> = { raindance: "Rain", sunnyday: "Sun", sandstorm: "Sand", snowscape: "Snow" };
const TERRAIN: Record<string, string> = { electricterrain: "Electric", grassyterrain: "Grassy", mistyterrain: "Misty", psychicterrain: "Psychic" };

/** The calc's sixteen rolls, ascending (it returns a number, a list, or a list of lists for a multi-hit move). */
function rolls(damage: unknown): number[] {
  if (typeof damage === "number") return [damage];
  if (Array.isArray(damage) && damage.every((n) => typeof n === "number")) return [...(damage as number[])].sort((a, b) => a - b);
  throw new Error("calc: a damage shape this differential does not map");
}

function calcDamage(c: DamageCase, crit: boolean): number[] {
  const q = c.query.query;
  const attacker = calcPokemon(setOf(q.attacker.set), q.attacker);
  const defender = calcPokemon(setOf(q.defender.set), q.defender);
  const name = dex().moves.get(q.move).name;
  const move = new Move(CALC_GEN, name, { isCrit: crit });
  const [own, foe] = q.state.sides;
  if (own === undefined || foe === undefined) throw new Error("two sides expected");
  // The attacker's side is index 0: its screens would protect the attacker, so only the defender's side (index 1) matters to the damage.
  const field = new Field({
    gameType: "Doubles",
    ...(q.state.weather === null ? {} : { weather: WEATHER[q.state.weather] as never }),
    ...(q.state.terrain === null ? {} : { terrain: TERRAIN[q.state.terrain] as never }),
    defenderSide: { isReflect: foe.reflect, isLightScreen: foe.light_screen, isAuroraVeil: foe.aurora_veil, isTailwind: foe.tailwind } as never,
    attackerSide: { isTailwind: own.tailwind } as never,
  });
  return rolls(calculate(CALC_GEN, attacker, defender, move, field).damage);
}

describe("damage fixtures: the probe's rolls against @smogon/calc generation 0", () => {
  it.each(damageFixture.cases.map((c) => [c.id, c] as const))("%s", (_id, c) => {
    const ascending = (list: number[]): number[] => [...list].sort((a, b) => a - b);
    note(`damage:${c.id}:no_crit`, ascending(c.expected.no_crit), calcDamage(c, false));
    note(`damage:${c.id}:crit`, ascending(c.expected.crit), calcDamage(c, true));
  });
});

/** The probe's stats of a set and the calc's. */
function statsPair(label: string, set: PokemonSet): void {
  const { result } = probeMechanics(context, readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "stats", set: exportPaste(engine, [set]) } }));
  if (result.kind !== "computed" || result.answer.kind !== "stats") throw new Error(`not computed: ${JSON.stringify(result)}`);
  const probe: StatTable = result.answer.stats;
  let calc: StatTable | null = null;
  try {
    calc = calcPokemon(set).rawStats as StatTable;
  } catch {
    // The calc's Champions table has no such species: one case per set, with the probe's stats.
  }
  if (calc === null) note(`stats:${label}:species`, set.species, null);
  else for (const key of STAT_KEYS) note(`stats:${label}:${key}`, probe[key], calc[key]);
}

const ZERO: StatTable = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const jsonl = (file: string): { paste: string }[] => readFileSync(file, "utf8").trim().split("\n").map((line) => JSON.parse(line) as { paste: string });

describe("stats fixtures: the probe's six stats against @smogon/calc generation 0", () => {
  it("covers the damage sets, the five reference cases and every committed fixture team", () => {
    let sets = 0;
    const seen = new Set<string>();
    const add = (label: string, set: PokemonSet): void => {
      const key = JSON.stringify(set);
      if (seen.has(key)) return;
      seen.add(key);
      sets += 1;
      statsPair(label, set);
    };
    for (const c of damageFixture.cases) {
      for (const side of ["attacker", "defender"] as const) {
        const set = setOf(c.query.query[side].set);
        add(`damage-${set.species.toLowerCase()}-${set.nature.toLowerCase()}`, set);
      }
    }
    const IVS: StatTable = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
    const reference: [string, string, string, Partial<StatTable>][] = [
      ["Empoleon", "Torrent", "Modest", { hp: 32, spa: 32, spe: 2 }],
      ["Whimsicott", "Prankster", "Bold", { hp: 32, def: 32, spe: 2 }],
      ["Pikachu", "Static", "Jolly", { atk: 32, def: 2, spe: 32 }],
      ["Gardevoir", "Synchronize", "Timid", { spa: 32, spd: 2, spe: 32 }],
      ["Arcanine", "Intimidate", "Relaxed", { atk: 2, def: 32, spa: 32 }],
    ];
    for (const [species, ability, nature, sp] of reference) {
      add(`reference-${species.toLowerCase()}`, { name: species, species, item: "", ability, moves: ["protect"], nature, gender: "", evs: { ...ZERO, ...sp }, ivs: IVS, level: 50 });
    }
    syntheticTeam().forEach((set, index) => add(`golden-team-${index}`, set));
    for (const file of ["legal-synthetic.txt", "legal-opponent.txt"]) {
      importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8")).forEach((set, index) => add(`${file}-${index}`, set));
    }
    for (const file of ["teams-dev.jsonl", "teams-validation.jsonl"]) {
      jsonl(join(TEAMS_DIR, "datasets", "fixtures-v1", file)).forEach((record, team) =>
        importPaste(engine, record.paste).forEach((set, index) => add(`${file}-${team}-${index}`, set)));
    }
    expect(sets).toBeGreaterThan(200);
  });
});

describe("the divergence list is exact", () => {
  const path = new URL("calc-divergences.json", FIXTURES);
  const listed = (): Divergence[] => {
    if (!existsSync(path)) return [];
    const json = JSON.parse(readFileSync(path, "utf8")) as { data_class: string; divergences: Divergence[] };
    expect(json.data_class).toBe("synthetic");
    return json.divergences;
  };

  it("lists every difference found, with its reason and the two values, and nothing else", () => {
    const entries = listed();
    expect(new Set(entries.map((e) => e.case)).size).toBe(entries.length);
    for (const entry of entries) expect(entry.reason.trim().length, entry.case).toBeGreaterThan(20);
    const unlisted = [...found.keys()].filter((id) => !entries.some((e) => e.case === id));
    if (process.env.ARENEON_PRINT_DIVERGENCES === "1") console.log(JSON.stringify([...found].map(([id, v]) => ({ case: id, ...v })), null, 1));
    expect(unlisted, "differences with no entry in calc-divergences.json").toEqual([]);
    const stale = entries.filter((e) => !found.has(e.case)).map((e) => e.case);
    expect(stale, "entries that no longer differ").toEqual([]);
    for (const entry of entries) expect({ probe: entry.probe, calc: entry.calc }, entry.case).toEqual(found.get(entry.case));
    expect(entries.map((e) => e.case)).toEqual([...entries.map((e) => e.case)].sort());
  });
});
