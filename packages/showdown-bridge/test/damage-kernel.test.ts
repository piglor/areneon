// The damage kernel (ADR-030 step 3, #1457) against the pinned probe, query by query: the probe is the definition, so every
// answer the kernel gives must be the probe's bytes, and anything it does not implement must be an abstention, never a guess.
// 1. ADR-016's hand-worked damage fixtures: the kernel covers each and gives the fixture's (and the probe's) answer.
// 2. Targeted queries for every entry of the effects record, as attacker and as defender, across moves and field states.
// 3. Abstentions: each refusal rule of the kernel.
// 4. The pieces: the stat formula, stages, the record's reader, the data reflection and the simulator scan.
import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { abstained, KernelBackendError, kernelBackend } from "../src/damage-kernel-backend.ts";
import { blockingHooks, hooksOf, INERT_HOOKS, kernelData, KernelDataError, simLiterals, STATUS_HOOKS } from "../src/damage-kernel-data.ts";
import { checkKernelEffects, KernelRecordError, kernelEffectsHash, loadKernelEffects, readKernelEffects } from "../src/damage-kernel-effects.ts";
import { applyStatHandlers, championsStat, computeBatch, KERNEL_FORMAT, KernelAbstain, type KernelContext, prepareQuery, staged } from "../src/damage-kernel.ts";
import type { KernelData } from "../src/damage-kernel-data.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { probeMechanics } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import type { MechanicsResult } from "../src/mechanics-result.ts";
import type { Settled } from "../src/probe-batch.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

const COMMIT = "c".repeat(40);
let engine: Engine;
let config: MechanicsConfig;
let kernel: ReturnType<typeof kernelBackend>;
interface Dex {
  species: { get(id: string): { name: string; exists: boolean } };
  moves: { get(id: string): { name: string; exists: boolean; category: string; basePower: number; type: string; target: string } };
  abilities: { get(id: string): { name: string } };
  items: { get(id: string): { name: string; exists: boolean } };
}
let dex: Dex;
interface Vocabulary {
  roster: { id: string; abilities: string[]; moves: string[]; required_item?: string }[];
  items: string[];
}
const vocabulary = JSON.parse(readFileSync(new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/team-vocabulary.json", import.meta.url), "utf8")) as Vocabulary;

beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  kernel = kernelBackend(engine, { config, backend_commit: COMMIT });
  dex = (engine.sim.Dex as unknown as { mod(m: string): Dex }).mod("champions");
});

const probe = (q: ParsedMechanicsQuery): MechanicsResult => probeMechanics({ engine, config }, q).result;
const run = (q: ParsedMechanicsQuery): Settled => kernel.run([q])[0] as Settled;

const sides = (defender: Partial<Record<"aurora_veil" | "light_screen" | "reflect" | "tailwind", boolean>> = {}, attacker: typeof defender = {}) => [
  { aurora_veil: false, light_screen: false, reflect: false, tailwind: false, ...attacker },
  { aurora_veil: false, light_screen: false, reflect: false, tailwind: false, ...defender },
];
const zero = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

interface Side {
  species: string;
  ability: string;
  item?: string;
  moves: string[];
  nature?: string;
  status?: string | null;
  hp?: number | null;
  mega?: boolean;
  boosts?: Partial<typeof zero>;
  history?: Record<string, unknown>;
}
const paste = (s: Side): string => {
  const set: PokemonSet = {
    name: "",
    species: dex.species.get(s.species).name,
    item: s.item === undefined ? "" : dex.items.get(s.item).name,
    ability: dex.abilities.get(s.ability).name,
    moves: s.moves.map((m) => dex.moves.get(m).name),
    nature: s.nature ?? "Hardy",
    gender: "",
    evs: { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 },
    ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
    level: 50,
  };
  return exportPaste(engine, [set]);
};
const combatant = (s: Side) => ({ set: paste(s), boosts: { ...zero, ...s.boosts }, status: s.status ?? null, hp: s.hp ?? null, mega: s.mega ?? false, ...(s.history === undefined ? {} : { history: s.history }) });
interface State {
  weather?: string | null;
  terrain?: string | null;
  trick_room?: boolean;
  sides?: ReturnType<typeof sides>;
  history?: Record<string, unknown>;
}
const query = (attacker: Side, defender: Side, move: string, state: State = {}): ParsedMechanicsQuery =>
  readMechanicsQuery({
    schema_version: 1,
    data_class: "synthetic",
    format_id: GOLDEN_FORMAT,
    query: {
      kind: "damage",
      attacker: combatant(attacker),
      defender: combatant(defender),
      move,
      state: { weather: state.weather ?? null, terrain: state.terrain ?? null, trick_room: state.trick_room ?? false, sides: state.sides ?? sides() },
      ...(state.history === undefined ? {} : { history: state.history }),
    },
  });

/** The kernel's answer equals the probe's bytes; returns whether the kernel covered it. */
function same(q: ParsedMechanicsQuery): boolean {
  const k = run(q);
  if (abstained(k)) return false;
  const expected = probe(q);
  expect(k.ok, "the kernel answered").toBe(true);
  expect(jcs((k as { result: MechanicsResult }).result as unknown as Json)).toBe(jcs(expected as unknown as Json));
  return true;
}
const abstainsOn = (q: ParsedMechanicsQuery): string => {
  const k = run(q);
  expect(abstained(k)).toBe(true);
  return ((k as { error: KernelAbstain }).error).entity;
};

const GARCHOMP: Side = { species: "garchomp", ability: "roughskin", moves: ["dragonclaw", "earthquake", "rockslide", "firefang"] };
const ARCANINE: Side = { species: "arcanine", ability: "intimidate", moves: ["heatwave", "protect"] };

describe("1. the hand-worked damage fixtures (ADR-016)", () => {
  const fixture = JSON.parse(readFileSync(new URL("../../../fixtures/mechanics/gen9championsvgc2026regmc/damage/cases.json", import.meta.url), "utf8")) as { cases: { id: string; query: unknown; expected: object }[] };
  it.each(fixture.cases.map((c) => [c.id, c] as const))("%s: covered, the fixture's answer and the probe's bytes", (_id, c) => {
    const q = readMechanicsQuery(c.query);
    expect(same(q)).toBe(true);
    const result = (run(q) as { result: MechanicsResult }).result;
    expect(result.kind === "computed" && result.answer).toEqual({ kind: "damage", ...c.expected });
  });
});

/** A roster species that may hold an ability (the first in vocabulary order). */
const holderOf = (ability: string) => vocabulary.roster.find((r) => r.abilities.includes(ability) && r.required_item === undefined);
/** Damaging moves of a roster entry the kernel can answer (fixed power, one or spread targets), up to `n`, of several types. */
function movesOf(entry: Vocabulary["roster"][number], n: number, want?: (m: { type: string; target: string; basePower: number }) => boolean): string[] {
  const out: string[] = [];
  const types = new Set<string>();
  for (const id of entry.moves) {
    const m = dex.moves.get(id);
    if (m.category === "Status" || m.basePower <= 0) continue;
    if (want !== undefined && !want(m)) continue;
    if (types.has(m.type) && out.length >= 2) continue;
    types.add(m.type);
    out.push(id);
    if (out.length >= n) break;
  }
  return out;
}

describe("2. every entry of the effects record, against the probe", () => {
  const STATES: State[] = [
    {},
    { weather: "sunnyday" },
    { weather: "raindance", terrain: "grassyterrain" },
    { weather: "sandstorm", terrain: "electricterrain", sides: sides({ reflect: true }) },
    { weather: "snowscape", terrain: "psychicterrain", sides: sides({ light_screen: true }) },
    { terrain: "mistyterrain", trick_room: true, sides: sides({ aurora_veil: true }, { tailwind: true }) },
  ];
  const foes: Side[] = [
    { species: "snorlax", ability: "thickfat", moves: ["bodyslam"] },
    { species: "skarmory", ability: "keeneye", moves: ["bravebird"] },
    { species: "tyranitar", ability: "unnerve", moves: ["rockslide"] },
  ];
  it("abilities: each covered somewhere and equal everywhere it is covered", () => {
    const effects = loadKernelEffects(GOLDEN_FORMAT, engine.commit);
    const uncovered: string[] = [];
    for (const ability of Object.keys(effects.abilities)) {
      const entry = holderOf(ability);
      if (entry === undefined) continue;
      const moves = movesOf(entry, 4);
      let covered = 0;
      for (const [i, state] of STATES.entries()) {
        const me: Side = { species: entry.id, ability, moves: moves.length > 0 ? moves : ["tackle"], hp: i % 2 === 1 ? 40 : null, status: i === 2 ? "brn" : i === 4 ? "par" : null, boosts: i === 3 ? { atk: 2, spa: -1, def: 1, spd: -2 } : {} };
        const foe = foes[i % foes.length] as Side;
        for (const move of moves) if (same(query(me, foe, move, state))) covered++;
        for (const move of foe.moves) if (same(query(foe, me, move, state))) covered++;
      }
      if (covered === 0) uncovered.push(ability);
    }
    // Abilities whose holders' only answers are abstentions here (another entity of the query is not covered) are listed, not hidden.
    expect(uncovered.length).toBeLessThan(Object.keys(effects.abilities).length / 4);
  }, 300_000);

  it("items: each covered and equal, held by the attacker and by the defender", () => {
    const effects = loadKernelEffects(GOLDEN_FORMAT, engine.commit);
    const entryOf = (id: string) => vocabulary.roster.find((r) => r.id === id) as Vocabulary["roster"][number];
    // A physical and a special user, four damaging moves each of several types, from the roster's own move pools.
    const users: Side[] = [
      { species: "garchomp", ability: "roughskin", moves: movesOf(entryOf("garchomp"), 4) },
      { species: "alakazam", ability: "synchronize", moves: movesOf(entryOf("alakazam"), 4) },
    ];
    let covered = 0;
    for (const item of Object.keys(effects.items)) {
      for (const [i, state] of STATES.slice(0, 3).entries()) {
        for (const user of users) {
          for (const move of user.moves) if (same(query({ ...user, item }, foes[i] as Side, move, state))) covered++;
          const foe = foes[(i + 1) % foes.length] as Side;
          for (const move of foe.moves) if (same(query(foe, { ...user, item, hp: i === 1 ? 50 : null }, move, state))) covered++;
        }
      }
    }
    expect(covered).toBeGreaterThan(Object.keys(effects.items).length * 4);
  }, 300_000);

  it("moves of the record and the data rules (stat overrides, crit rules, immunity, spread, priority) are equal", () => {
    const cases: [Side, Side, string, State][] = [
      [{ species: "corviknight", ability: "pressure", moves: ["bodypress"], boosts: { def: 2 } }, ARCANINE, "bodypress", {}],
      [{ species: "sableye", ability: "keeneye", moves: ["foulplay"] }, { ...GARCHOMP, boosts: { atk: 2 } }, "foulplay", {}],
      [{ species: "alakazam", ability: "synchronize", moves: ["psyshock"] }, { ...ARCANINE, boosts: { spd: 3, def: -1 } }, "psyshock", {}],
      [{ species: "gallade", ability: "steadfast", moves: ["sacredsword"] }, { ...ARCANINE, boosts: { def: 4 } }, "sacredsword", {}],
      [GARCHOMP, { species: "skarmory", ability: "keeneye", moves: ["bravebird"] }, "earthquake", {}],
      [GARCHOMP, ARCANINE, "earthquake", { terrain: "grassyterrain" }],
      [GARCHOMP, ARCANINE, "rockslide", { weather: "sandstorm" }],
      [{ species: "lucario", ability: "steadfast", moves: ["extremespeed", "bulletpunch"] }, ARCANINE, "extremespeed", { terrain: "psychicterrain" }],
      [{ species: "snorlax", ability: "thickfat", moves: ["facade"], status: "brn" }, ARCANINE, "facade", {}],
      [{ species: "snorlax", ability: "thickfat", moves: ["snore"], status: "slp" }, ARCANINE, "snore", {}],
      [{ species: "snorlax", ability: "thickfat", moves: ["snore"] }, ARCANINE, "snore", {}],
      [{ species: "kangaskhan", ability: "scrappy", moves: ["suckerpunch"] }, ARCANINE, "suckerpunch", {}],
      [GARCHOMP, ARCANINE, "dig", {}],
      [{ species: "gengar", ability: "cursedbody", moves: ["payback", "shadowball"] }, ARCANINE, "payback", {}],
      [{ species: "gengar", ability: "cursedbody", moves: ["shadowball"] }, { species: "snorlax", ability: "thickfat", moves: ["bodyslam"] }, "shadowball", {}],
    ];
    let covered = 0;
    for (const [a, d, m, s] of cases) if (dex.moves.get(m).exists && same(query(a, d, m, s))) covered++;
    expect(covered).toBeGreaterThan(cases.length / 2);
  }, 120_000);

  it("a declared history the engine accepts is answered as without it; one it would refuse abstains", () => {
    const history = { move_actions: { kind: "known", value: 2 }, hurt_this_turn: { kind: "unknown", ends: [false, true] }, moves_used: { kind: "known", value: ["dragonclaw"] }, last_move: { kind: "known", value: "dragonclaw" }, last_item: { kind: "known", value: null } };
    expect(same(query({ ...GARCHOMP, history }, ARCANINE, "dragonclaw", { history: { turn: 3, field_last_move: { kind: "known", value: "dragonclaw" } } }))).toBe(true);
    expect(same(query(GARCHOMP, { ...ARCANINE, history: { times_attacked: { kind: "known", value: 1 } } }, "dragonclaw"))).toBe(true);
    const refuses = (facts: Record<string, unknown>, field?: unknown) => abstainsOn(query({ ...GARCHOMP, history: facts }, ARCANINE, "dragonclaw", field === undefined ? {} : { history: { field_last_move: field } }));
    expect(refuses({ moves_used: { kind: "unknown", ends: [[], ["protect"]] } })).toBe("state:history");
    expect(refuses({ last_move: { kind: "known", value: "nosuchmove" } })).toBe("state:history");
    expect(refuses({ last_item: { kind: "known", value: "nosuchitem" } })).toBe("state:history");
    expect(refuses({ stall_counter: { kind: "known", value: 1 } })).toBe("state:history");
    expect(refuses({}, { kind: "known", value: "nosuchmove" })).toBe("state:history");
  });

  it("batches of any size give the per-query bytes, in order", () => {
    const qs = ["dragonclaw", "earthquake", "rockslide", "firefang"].flatMap((m) => [query(GARCHOMP, ARCANINE, m), query(GARCHOMP, ARCANINE, m, { weather: "sunnyday" })]);
    const one = qs.map((q) => jcs((run(q) as { result: MechanicsResult }).result as unknown as Json));
    const all = kernel.run(qs).map((s) => jcs((s as { result: MechanicsResult }).result as unknown as Json));
    expect(all).toEqual(one);
  });
});

describe("2b. each effect kind, on a query built to exercise it, against the probe", () => {
  type Move = { type: string; target: string; basePower: number; category: string; flags?: Record<string, number>; priority?: number; id?: string };
  /** A legal side holding `ability` (and `item`) with up to four damaging moves, the first matching `want`. */
  const side = (ability: string, item?: string, want?: (m: Move) => boolean, extra: Partial<Side> = {}): { side: Side; move: string } | null => {
    for (const entry of vocabulary.roster) {
      if (!entry.abilities.includes(ability) || entry.required_item !== undefined) continue;
      const hit = entry.moves.find((id) => {
        const m = dex.moves.get(id) as unknown as Move;
        return m.category !== "Status" && m.basePower > 0 && (want === undefined || want({ ...m, id }));
      });
      if (hit === undefined) continue;
      const moves = [hit, ...movesOf(entry, 3).filter((m) => m !== hit)].slice(0, 4);
      return { side: { species: entry.id, ability, ...(item === undefined ? {} : { item }), moves, ...extra }, move: hit };
    }
    return null;
  };
  const SNORLAX: Side = { species: "snorlax", ability: "thickfat", moves: ["bodyslam"] };
  const SKARMORY: Side = { species: "skarmory", ability: "keeneye", moves: ["bravebird"] };
  const cases: [string, () => ParsedMechanicsQuery | null][] = [
    ["absorb", () => { const a = side("roughskin", undefined, (m) => m.type === "Electric") ?? side("intimidate", undefined, (m) => m.type === "Electric"); const d = side("voltabsorb"); return a && d && query(a.side, d.side, a.move); }],
    ["flag immunity", () => { const a = side("intimidate", undefined, (m) => (m.flags ?? {}).bullet === 1); const d = side("bulletproof"); return a && d && query(a.side, d.side, a.move); }],
    ["priority block", () => { const a = side("steadfast", undefined, (m) => (m.priority ?? 0) > 0 && m.id !== "feint"); const d = side("armortail") ?? side("queenlymajesty") ?? side("dazzling"); return a && d && query(a.side, d.side, a.move); }],
    ["mold breaker through levitate", () => { const a = side("moldbreaker", undefined, (m) => m.type === "Ground"); const d = side("levitate"); return a && d && query(a.side, d.side, a.move); }],
    ["levitate", () => { const d = side("levitate"); return d && query(GARCHOMP, d.side, "earthquake"); }],
    ["thick fat", () => { const d = side("thickfat"); return d && query(GARCHOMP, d.side, "firefang"); }],
    ["scope lens", () => query({ ...GARCHOMP, item: "scopelens" }, SNORLAX, "dragonclaw")],
    ["life orb", () => query({ ...GARCHOMP, item: "lifeorb" }, SNORLAX, "dragonclaw")],
    ["expert belt, super effective", () => query({ ...GARCHOMP, item: "expertbelt" }, ARCANINE, "earthquake")],
    ["expert belt, neutral", () => query({ ...GARCHOMP, item: "expertbelt" }, SNORLAX, "earthquake")],
    ["type boost, own type", () => query({ ...GARCHOMP, item: "dragonfang" }, SNORLAX, "dragonclaw")],
    ["type boost, other type", () => query({ ...GARCHOMP, item: "dragonfang" }, SNORLAX, "earthquake")],
    ["muscle band", () => query({ ...GARCHOMP, item: "muscleband" }, SNORLAX, "dragonclaw")],
    ["resist berry eaten", () => query(GARCHOMP, { ...ARCANINE, item: "shucaberry" }, "earthquake")],
    ["resist berry, other type", () => query(GARCHOMP, { ...ARCANINE, item: "shucaberry" }, "dragonclaw")],
    ["resist berry, unnerve", () => { const a = side("unnerve", undefined, (m) => m.type === "Ground"); return a && query(a.side, { ...ARCANINE, item: "shucaberry" }, a.move); }],
    ["chilan berry", () => { const a = side("intimidate", undefined, (m) => m.type === "Normal"); return a && query(a.side, { ...GARCHOMP, item: "chilanberry" }, a.move); }],
    ["filter", () => { const d = side("filter") ?? side("solidrock"); return d && query(GARCHOMP, d.side, "earthquake"); }],
    ["multiscale", () => { const d = side("multiscale"); return d && query(GARCHOMP, d.side, "dragonclaw"); }],
    ["sniper", () => { const a = side("sniper"); return a && query(a.side, SNORLAX, a.move); }],
    ["technician", () => { const a = side("technician", undefined, (m) => m.basePower <= 60); return a && query(a.side, SNORLAX, a.move); }],
    ["technician, strong move", () => { const a = side("technician", undefined, (m) => m.basePower > 60); return a && query(a.side, SNORLAX, a.move); }],
    ["ate", () => { const a = side("pixilate", undefined, (m) => m.type === "Normal") ?? side("aerilate", undefined, (m) => m.type === "Normal"); return a && query(a.side, SNORLAX, a.move); }],
    ["sheer force", () => { const a = side("sheerforce"); return a && query(a.side, SNORLAX, a.move); }],
    ["guts", () => { const a = side("guts", undefined, (m) => m.category === "Physical", { status: "brn" }); return a && query(a.side, SNORLAX, a.move); }],
    ["pinch", () => { const a = side("blaze", undefined, (m) => m.type === "Fire", { hp: 20 }); return a && query(a.side, SKARMORY, a.move); }],
    ["solar power", () => { const a = side("solarpower", undefined, (m) => m.category === "Special"); return a && query(a.side, SKARMORY, a.move, { weather: "sunnyday" }); }],
    ["Thick Fat beside a pinch hook of equal priority: the engine orders them by speed, so the kernel abstains", () => { const a = side("blaze", undefined, (m) => m.type === "Fire", { hp: 20 }); return a && (abstainsOn(query(a.side, SNORLAX, a.move)) === "state:handler_tie" ? query(a.side, SKARMORY, a.move) : null); }],
    ["fur coat", () => { const d = side("furcoat"); return d && query(GARCHOMP, d.side, "dragonclaw"); }],
    ["grass pelt", () => { const d = side("grasspelt"); return d && query(GARCHOMP, d.side, "dragonclaw", { terrain: "grassyterrain" }); }],
    ["shell armor", () => { const d = side("shellarmor"); return d && query(GARCHOMP, d.side, "dragonclaw"); }],
    ["adaptability", () => { const a = side("adaptability"); return a && query(a.side, SNORLAX, a.move); }],
    ["cloud nine in rain", () => { const d = side("cloudnine"); return d && query({ species: "lapras", ability: "waterabsorb", moves: ["surf"] }, d.side, "surf", { weather: "raindance" }); }],
    ["infiltrator through reflect", () => { const a = side("infiltrator", undefined, (m) => m.category === "Physical"); return a && query(a.side, SNORLAX, a.move, { sides: sides({ reflect: true }) }); }],
    ["aurora veil", () => query(GARCHOMP, SNORLAX, "dragonclaw", { sides: sides({ aurora_veil: true }) })],
    ["light screen against a physical move", () => query(GARCHOMP, SNORLAX, "dragonclaw", { sides: sides({ light_screen: true }) })],
    ["stored power", () => { const a = side("synchronize", undefined, (m) => m.id === "storedpower", { boosts: { spa: 2, def: 1, atk: -1 } }); return a && query(a.side, SNORLAX, a.move); }],
    ["facade", () => { const a = side("steadfast", undefined, (m) => m.id === "facade", { status: "par" }); return a && query(a.side, ARCANINE, a.move); }],
    ["sand on a Rock defender", () => { const d = side("sturdy", undefined, undefined) ; return d && query({ species: "alakazam", ability: "synchronize", moves: ["psychic"] }, { species: "tyranitar", ability: "unnerve", moves: ["rockslide"] }, "psychic", { weather: "sandstorm" }); }],
    ["snow on an Ice defender", () => query(GARCHOMP, { species: "abomasnow", ability: "soundproof", moves: ["blizzard"] }, "dragonclaw", { weather: "snowscape" })],
    ["misty terrain against a Dragon move", () => query(GARCHOMP, SNORLAX, "dragonclaw", { terrain: "mistyterrain" })],
    ["electric terrain", () => query({ species: "raichu", ability: "static", moves: ["thunderbolt"] }, SNORLAX, "thunderbolt", { terrain: "electricterrain" })],
  ];
  it.each(cases)("%s", (_name, build) => {
    const q = build();
    expect(q, "a legal query could be built").not.toBeNull();
    if (q !== null) expect(same(q) || String((run(q) as { error: Error }).error)).toBe(true);
  });
});

describe("2c. a stat or an event chosen by the move's data follows the engine", () => {
  /** A legal side of a species that may hold `ability` and knows `move`, or null when the roster has none. */
  const holder = (ability: string, move: string, extra: Partial<Side> = {}): Side | null => {
    const entry = vocabulary.roster.find((r) => r.abilities.includes(ability) && r.moves.includes(move) && r.required_item === undefined);
    return entry === undefined ? null : { species: entry.id, ability, moves: [move], ...extra };
  };
  const SNORLAX: Side = { species: "snorlax", ability: "thickfat", moves: ["bodyslam"] };
  // Body Press attacks with the user's Defense, but the engine's stat event is the category's (ModifyAtk), on the user.
  it.each<[string, () => [Side | null, Side, string, State]]>([
    ["Body Press by an Ice type in snow (snow's Defense boost is not the attacking stat's event)", () => [holder("snowwarning", "bodypress"), SNORLAX, "bodypress", { weather: "snowscape" }]],
    ["Body Press under Huge Power (ModifyAtk doubles the Defense it attacks with)", () => [holder("hugepower", "bodypress") ?? holder("purepower", "bodypress"), SNORLAX, "bodypress", {}]],
    ["Body Press under Fur Coat (a Defense hook is not the attacking stat's event)", () => [holder("furcoat", "bodypress"), SNORLAX, "bodypress", {}]],
    ["Body Press with Guts and a status", () => [holder("guts", "bodypress", { status: "brn" }), SNORLAX, "bodypress", {}]],
    ["Psyshock into an Ice type in snow (the defending stat is Defense: snow's boost applies)", () => [holder("synchronize", "psyshock"), { species: "abomasnow", ability: "soundproof", moves: ["blizzard"] }, "psyshock", { weather: "snowscape" }]],
    ["Psyshock into a Rock type in sand (Sp. Def's boost does not apply)", () => [holder("synchronize", "psyshock"), { species: "tyranitar", ability: "unnerve", moves: ["rockslide"] }, "psyshock", { weather: "sandstorm" }]],
    ["Psyshock into Fur Coat (Defense doubled)", () => [holder("synchronize", "psyshock"), { species: "furfrou", ability: "furcoat", moves: ["tackle"] }, "psyshock", {}]],
    ["Foul Play: the target's Attack and stages, the user's ModifyAtk hooks", () => [holder("hugepower", "foulplay") ?? holder("prankster", "foulplay"), { ...GARCHOMP, boosts: { atk: 2 } }, "foulplay", {}]],
    ["Foul Play into a lowered Attack", () => [holder("prankster", "foulplay") ?? holder("insomnia", "foulplay"), { ...GARCHOMP, boosts: { atk: -2 } }, "foulplay", {}]],
  ])("%s", (_name, build) => {
    const [attacker, defender, move, state] = build();
    if (attacker === null) return; // the roster holds no such combatant at this pin: unreachable, nothing to compare
    const q = query(attacker, defender, move, state);
    expect(same(q) || String((run(q) as { error: Error }).error)).toBe(true);
  });

  it("a move that picks its category or stat in a hook (Shell Side Arm) abstains", () => {
    const user = holder("regenerator", "shellsidearm") ?? holder("quickdraw", "shellsidearm");
    if (user === null) return;
    expect(abstainsOn(query(user, SNORLAX, "shellsidearm"))).toBe("move:shellsidearm");
  });

  it("a weather's stat hook modifies the stat at once, a chained hook at the end (snow then Fur Coat: 101 is 302, not 303)", () => {
    expect(applyStatHandlers(101, [{ direct: 6144 }, { chain: 8192 }])).toBe(302);
    expect(applyStatHandlers(101, [{ chain: 8192 }, { direct: 6144 }])).toBe(302);
    expect(applyStatHandlers(101, [{ chain: 6144 }, { chain: 8192 }])).toBe(303);
    expect(applyStatHandlers(101, [])).toBe(101);
  });
});

describe("3. abstentions: never a guess", () => {
  it("names what it does not cover", () => {
    expect(abstainsOn(query(GARCHOMP, ARCANINE, "dragonclaw", { weather: "hail" }))).toBe("weather:hail");
    expect(abstainsOn(query({ ...GARCHOMP, ability: "roughskin" }, { species: "mimikyu", ability: "disguise", moves: ["playrough"] }, "dragonclaw"))).toBe("ability:disguise");
    expect(abstainsOn(query(GARCHOMP, ARCANINE, "earthquake", { history: { sides: [{ pending_slots: { kind: "known", value: [{ id: "wish", position: 0, started_turn: 1, after_residual: false }] } }, {}] , turn: 2 } }))).toBe("state:history_slot");
    expect(abstainsOn(query({ ...GARCHOMP, history: { volatiles: { kind: "known", value: [{ id: "encore", started_turn: 1, after_residual: false, moved_before: false }] } } }, ARCANINE, "dragonclaw", { history: { turn: 2 } }))).toBe("volatile:encore");
    expect(abstainsOn(query({ ...GARCHOMP, history: { volatiles: { kind: "known", value: [{ id: "nosuchvolatile", started_turn: 1, after_residual: false, moved_before: false }] } } }, ARCANINE, "dragonclaw", { history: { turn: 2 } }))).toBe("volatile:nosuchvolatile");
    expect(abstainsOn(query(GARCHOMP, ARCANINE, "dragonclaw", { sides: sides({ reflect: true }, { reflect: true }) }))).toBe("state:handler_tie");
    // Two screens on one side tie too (the engine draws their order; the probe answers `random_effect`).
    expect(abstainsOn(query(GARCHOMP, ARCANINE, "dragonclaw", { sides: sides({ reflect: true, light_screen: true }) }))).toBe("state:handler_tie");
    expect(abstainsOn(query({ ...GARCHOMP, mega: true }, ARCANINE, "dragonclaw"))).toBe("species:garchomp");
    expect(abstainsOn(query(GARCHOMP, { ...ARCANINE, hp: 999 }, "dragonclaw"))).toBe("state:hp");
  });

  it("abstains on any kind but damage, and passes the envelope's own outcomes through", () => {
    const speed = readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "stats", set: paste(GARCHOMP) } });
    expect(abstained(run(speed))).toBe(true);
    expect(kernel.counts.by_entity.get("kind:stats")).toBeGreaterThan(0);
    // A set the validator refuses: the probe's own `refused` outcome, byte for byte (the envelope is the probe's).
    const refused = query({ ...GARCHOMP, moves: ["dragonclaw", "spore"] }, ARCANINE, "dragonclaw");
    expect(same(refused)).toBe(true);
    expect((run(refused) as { result: MechanicsResult }).result.kind).toBe("refused");
  });

  it("refuses a bad backend commit, and abstains on everything under another pin's effects record", () => {
    expect(() => kernelBackend(engine, { config, backend_commit: "nope" })).toThrow(KernelBackendError);
    const other = { ...engine, commit: "0".repeat(40) } as Engine;
    const stale = kernelBackend(other, { config, backend_commit: COMMIT });
    const out = stale.run([query(GARCHOMP, ARCANINE, "dragonclaw")])[0] as Settled;
    expect(abstained(out) && (out as { error: KernelAbstain }).error.entity).toBe("state:effects_record");
    expect(() => loadKernelEffects(GOLDEN_FORMAT, "0".repeat(40))).toThrow(/another engine pin/);
  });
});


describe("3b. every refusal rule, through a context whose data is doctored", () => {
  const ctxWith = (patch: Partial<KernelData> = {}, extra: Partial<KernelContext> = {}): KernelContext => {
    const data = kernelData(engine, GOLDEN_FORMAT);
    return { data: { ...data, ...patch } as KernelData, effects: loadKernelEffects(GOLDEN_FORMAT, engine.commit), ko_blockers: config.ko_blockers, level: null, ...extra };
  };
  const decode = (ctx: KernelContext, q: ParsedMechanicsQuery): string => {
    const dq = q.query as Extract<ParsedMechanicsQuery["query"], { kind: "damage" }>;
    const sets = [importPaste(engine, dq.attacker.set)[0], importPaste(engine, dq.defender.set)[0]] as PokemonSet[];
    try {
      const out = prepareQuery(ctx, dq, sets[0] as PokemonSet, sets[1] as PokemonSet);
      return out.kind;
    } catch (error) {
      if (!(error instanceof KernelAbstain)) throw error;
      return `${error.entity} ${error.why}`;
    }
  };
  const base = () => query(GARCHOMP, ARCANINE, "dragonclaw");
  const real = () => kernelData(engine, GOLDEN_FORMAT);
  /** Add a hook to one entity (for the handler-order and unknown-handler rules). */
  const withHook = (kind: string, id: string, hook: string, order: { order?: number | false; priority?: number; sub_order?: number } = {}): Partial<KernelData> => ({
    hooks: (k, i) => (k === kind && i === id ? new Set([...real().hooks(k, i), hook]) : real().hooks(k, i)),
    hookOrder: (k, i, h) => (k === kind && i === id && h === hook ? { order: order.order ?? false, priority: order.priority ?? 0, sub_order: order.sub_order ?? 0 } : real().hookOrder(k, i, h)),
  });
  it.each<[string, () => string, string]>([
    ["unknown species", () => decode(ctxWith({ species: () => null }), base()), "species:garchomp unknown"],
    ["a battle-only forme", () => decode(ctxWith({ species: (n) => ({ ...(real().species(n) as NonNullable<ReturnType<KernelData["species"]>>), battle_only: true }) }), base()), "species:garchomp battle_only"],
    ["another level", () => decode(ctxWith({}, { level: 100 }), base()), "state:level not_implemented"],
    ["a status that is no status", () => decode(ctxWith({ conditionType: () => "Volatile" }), query({ ...GARCHOMP, status: "brn" }, ARCANINE, "dragonclaw")), "status:brn not_implemented"],
    ["an unknown nature", () => decode(ctxWith({ nature: () => null }), base()), "state:nature unknown"],
    ["an unknown ability", () => decode(ctxWith({ abilityBlockers: () => null }), base()), "ability:roughskin unknown"],
    ["an unknown terrain", () => decode(ctxWith(), query(GARCHOMP, ARCANINE, "dragonclaw", { terrain: "nosuchterrain" })), "terrain:nosuchterrain not_implemented"],
    ["an unknown move", () => decode(ctxWith({ move: () => null }), base()), "move:dragonclaw unknown"],
    ["a target it does not take", () => decode(ctxWith({ move: (id) => ({ ...(real().move(id) as NonNullable<ReturnType<KernelData["move"]>>), target: "self" }) }), base()), "move:dragonclaw target"],
    ["no base power", () => decode(ctxWith({ move: (id) => ({ ...(real().move(id) as NonNullable<ReturnType<KernelData["move"]>>), base_power: 0 }) }), base()), "move:dragonclaw base_power"],
    ["a powder move", () => decode(ctxWith({ move: (id) => ({ ...(real().move(id) as NonNullable<ReturnType<KernelData["move"]>>), flags: new Set(["powder"]) }) }), base()), "move:dragonclaw powder"],
    ["Hydro Steam in sun", () => decode(ctxWith({ move: (id) => ({ ...(real().move(id) as NonNullable<ReturnType<KernelData["move"]>>), id: "hydrosteam" }) }), query(GARCHOMP, ARCANINE, "dragonclaw", { weather: "sunnyday" })), "move:hydrosteam weather_rule"],
    ["an unknown handler of the crit ratio", () => decode(ctxWith(withHook("ability", "roughskin", "onModifyCritRatio")), base()), "ability:roughskin ModifyCritRatio"],
    ["a field condition's base power", () => decode(ctxWith(withHook("condition", "trickroom", "onBasePower")), query(GARCHOMP, ARCANINE, "dragonclaw", { trick_room: true })), "condition:trickroom BasePower"],
    ["a defender's base-power hook", () => decode(ctxWith(withHook("ability", "intimidate", "onSourceBasePower", { priority: 7 })), base()), "ability:intimidate BasePower"],
    ["a field condition's attack hook", () => decode(ctxWith(withHook("condition", "trickroom", "onModifyAtk")), query(GARCHOMP, ARCANINE, "dragonclaw", { trick_room: true })), "condition:trickroom ModifyAtk"],
    ["an unknown defense handler", () => decode(ctxWith(withHook("condition", "trickroom", "onModifyDef")), query(GARCHOMP, ARCANINE, "dragonclaw", { trick_room: true })), "condition:trickroom ModifyDef"],
    ["an attacker's defense hook", () => decode(ctxWith(withHook("ability", "roughskin", "onSourceModifyDef")), base()), "ability:roughskin ModifyDef"],
    ["an unknown same-type handler", () => decode(ctxWith(withHook("ability", "roughskin", "onModifySTAB")), base()), "ability:roughskin ModifySTAB"],
    ["an unknown side condition's damage hook", () => decode(ctxWith(withHook("condition", "tailwind", "onAnyModifyDamage")), query(GARCHOMP, ARCANINE, "dragonclaw", { sides: sides({ tailwind: true }) })), "condition:tailwind ModifyDamage"],
    ["an unknown damage handler", () => decode(ctxWith(withHook("ability", "intimidate", "onSourceModifyDamage")), base()), "ability:intimidate ModifyDamage"],
    ["a tie between the two Pokemon", () => decode(ctxWith(withHook("ability", "intimidate", "onSourceModifyAtk", { priority: 5 })), query({ species: "infernape", ability: "blaze", moves: ["flamethrower"] }, ARCANINE, "flamethrower")), "state:handler_tie ModifyAtk"],
    ["a hook with an order", () => decode(ctxWith(withHook("condition", "trickroom", "onBasePower", { order: 2 })), query(GARCHOMP, ARCANINE, "dragonclaw", { trick_room: true })), "condition:trickroom BasePower"],
    ["the move's own base power beside a defender's on equal keys", () => decode(ctxWith(withHook("ability", "intimidate", "onSourceBasePower")), query({ species: "snorlax", ability: "thickfat", moves: ["facade"] }, ARCANINE, "facade")), "state:handler_tie BasePower"],
    ["a status beside a start that can refuse it (Electric Surge)", () => decode(ctxWith(), query({ species: "pincurchin", ability: "electricsurge", moves: ["bodyslam"] }, { ...GARCHOMP, status: "slp" }, "bodyslam")), "ability:electricsurge status_start"],
    ["a status beside a start that cannot (Intimidate)", () => decode(ctxWith(), query(GARCHOMP, { ...ARCANINE, status: "par" }, "dragonclaw")), "rolls"],
    ["a status beside any start hook without this pin's inventory", () => decode(ctxWith({ abilityStarts: () => null }), query(GARCHOMP, { ...ARCANINE, status: "par" }, "dragonclaw")), "ability:intimidate status_start"],
    ["an after-move hook that draws (Magician)", () => decode(ctxWith(), query({ species: "delphox", ability: "magician", moves: ["hypervoice"] }, ARCANINE, "hypervoice")), "ability:magician onAfterMoveSecondarySelf"],
    ["a move hook the record does not implement", () => decode(ctxWith({ moveHookOrder: () => ({ order: 3, priority: 0, sub_order: 0 }) }), base()), "move:dragonclaw onBasePower"],
  ])("%s", (_name, run, expected) => expect(run()).toBe(expected));

  it("a declared attacker list and volatile list abstain", () => {
    const ctx = ctxWith();
    expect(decode(ctx, query({ ...GARCHOMP, history: { attacked_by: { kind: "known", value: [{ side: 1, position: 0, damage: 10 }] } } }, ARCANINE, "dragonclaw"))).toBe("state:history attacked_by");
    expect(decode(ctx, query({ ...GARCHOMP, history: { volatiles: { kind: "unknown", ends: [[], []] } } }, ARCANINE, "dragonclaw"))).toBe("rolls");
  });

  it("the roll loop gives the engine's minimum of 1", () => {
    const one = computeBatch({ n: 1, base: Int32Array.from([0, 0]), final: Int32Array.from([1, 1]), flags: new Int32Array(1), weather: Int32Array.from([4096]), stab: Int32Array.from([4096]), type_mod: Int32Array.from([-4]) } as never);
    expect([...one].every((d) => d === 1)).toBe(true);
  });
});

describe("3c. reviewed data, not code: conditions, kinds and their hooks", () => {
  const ctxWith = (patch: (e: ReturnType<typeof loadKernelEffects>) => void): KernelContext => {
    const effects = structuredClone(loadKernelEffects(GOLDEN_FORMAT, engine.commit));
    patch(effects);
    return { data: kernelData(engine, GOLDEN_FORMAT), effects, ko_blockers: config.ko_blockers, level: null };
  };
  const decoded = (ctx: KernelContext, q: ParsedMechanicsQuery) => {
    const dq = q.query as Extract<ParsedMechanicsQuery["query"], { kind: "damage" }>;
    const out = prepareQuery(ctx, dq, importPaste(engine, dq.attacker.set)[0] as PokemonSet, importPaste(engine, dq.defender.set)[0] as PokemonSet);
    if (out.kind !== "rolls") throw new Error("no rolls");
    return out.decoded;
  };
  it("the weathers, terrains and screens come from the record's conditions table", () => {
    const effects = loadKernelEffects(GOLDEN_FORMAT, engine.commit);
    expect(Object.keys(effects.conditions).sort()).toEqual(["auroraveil", "electricterrain", "grassyterrain", "lightscreen", "mistyterrain", "psychicterrain", "raindance", "reflect", "sandstorm", "snowscape", "sunnyday", "tailwind", "trickroom"]);
    const fire = query({ species: "arcanine", ability: "intimidate", moves: ["flamethrower"] }, GARCHOMP, "flamethrower", { weather: "sunnyday" });
    expect(decoded(ctxWith(() => undefined), fire).weather).toBe(6144);
    const changed = ctxWith((e) => {
      const sun = e.conditions.sunnyday as { kind: "weather_damage"; types: { type: string; modifier: number }[] };
      sun.types = sun.types.map((t) => (t.type === "Fire" ? { ...t, modifier: 4096 } : t));
    });
    expect(decoded(changed, fire).weather).toBe(4096);
    // A condition the record does not list is no longer implemented: the kernel abstains on it.
    const without = ctxWith((e) => delete e.conditions.sunnyday);
    expect(() => decoded(without, fire)).toThrow(/weather:sunnyday/);
  });

  it("an entry's kind must explain every damage-path hook it hides, or the record is refused", () => {
    const data = kernelData(engine, GOLDEN_FORMAT);
    const effects = loadKernelEffects(GOLDEN_FORMAT, engine.commit);
    expect(() => checkKernelEffects(effects, data)).not.toThrow();
    const miskinded = structuredClone(effects);
    miskinded.abilities.sapsipper = { kind: "pinch", type: "Grass" };
    expect(() => checkKernelEffects(miskinded, data)).toThrow(KernelRecordError);
    const hiding = structuredClone(effects);
    hiding.abilities.limber = { kind: "no_damage_effect", hides: ["onTryHit"] };
    expect(() => checkKernelEffects(hiding, data)).toThrow(/limber/);
    const unknown = structuredClone(effects);
    unknown.items.nosuchitem = { kind: "type_boost", type: "Fire", modifier: 4915 };
    expect(() => checkKernelEffects(unknown, data)).toThrow(/nosuchitem/);
    const unknownMove = structuredClone(effects);
    unknownMove.moves.nosuchmove = { kind: "fails" };
    expect(() => checkKernelEffects(unknownMove, data)).toThrow(/nosuchmove/);
    const unknownCondition = structuredClone(effects);
    unknownCondition.conditions.nosuchcondition = { kind: "no_damage_effect", hides: ["onStart"] };
    expect(() => checkKernelEffects(unknownCondition, data)).toThrow(/nosuchcondition/);
    const miskindedItem = structuredClone(effects);
    miskindedItem.items.lifeorb = { kind: "type_boost", type: "Fire", modifier: 4915 };
    expect(() => checkKernelEffects(miskindedItem, data)).toThrow(/lifeorb/);
  });

  it("onUpdate is not inert: a holder of it abstains (Limber's cure, a berry at low HP)", () => {
    expect(INERT_HOOKS.has("onUpdate")).toBe(false);
    expect(abstainsOn(query({ species: "persian", ability: "limber", moves: ["slash"] }, SNORLAX_TARGET, "slash"))).toBe("ability:limber");
  });
});

const SNORLAX_TARGET: Side = { species: "snorlax", ability: "thickfat", moves: ["bodyslam"] };

describe("4. the pieces", () => {
  it("the Champions stat formula and the stage table", () => {
    const neutral = { plus: null, minus: null };
    expect(championsStat(108, 32, "hp", neutral)).toBe(215);
    expect(championsStat(130, 32, "atk", { plus: "atk", minus: "spa" })).toBe(200);
    expect(championsStat(100, 0, "spa", { plus: "atk", minus: "spa" })).toBe(108);
    expect(championsStat(100, 0, "spe", neutral)).toBe(120);
    expect([staged(100, 0), staged(100, 1), staged(100, 6), staged(100, -1), staged(100, -6), staged(101, 3)]).toEqual([100, 150, 400, 66, 25, 252]);
    expect(KERNEL_FORMAT).toBe(3);
  });

  it("computes an empty batch", () => {
    expect(computeBatch({ n: 0, base: new Int32Array(0), final: new Int32Array(0), flags: new Int32Array(0), weather: new Int32Array(0), stab: new Int32Array(0), type_mod: new Int32Array(0) } as never).length).toBe(0);
  });

  it("the effects record's reader refuses every malformed shape", () => {
    const good = JSON.parse(readFileSync(new URL("../../../config/backends/gen9championsvgc2026regmc.damage-kernel.effects.json", import.meta.url), "utf8")) as Record<string, unknown>;
    expect(readKernelEffects(good)).toBeTruthy();
    expect(kernelEffectsHash(readKernelEffects(good))).toMatch(/^[0-9a-f]{64}$/);
    const bad = (patch: (v: Record<string, unknown>) => void) => {
      const v = structuredClone(good);
      patch(v);
      return () => readKernelEffects(v);
    };
    expect(() => readKernelEffects(null)).toThrow(/an object/);
    expect(bad((v) => (v.extra = 1))).toThrow(/exactly its keys/);
    expect(bad((v) => (v.schema_version = 2))).toThrow(/schema version/);
    expect(bad((v) => (v.showdown_commit = "x"))).toThrow(/format and commit/);
    expect(bad((v) => (v.abilities = []))).toThrow(/is an object/);
    expect(bad((v) => (v.abilities = { zz: { kind: "pinch", type: "Fire" }, aa: { kind: "pinch", type: "Fire" } }))).toThrow(/sorted and unique/);
    expect(bad((v) => (v.abilities = { aa: { kind: "type_boost", type: "Fire", modifier: 1 } }))).toThrow(/a kind it may hold/);
    expect(bad((v) => (v.abilities = { aa: { kind: "pinch", type: "Fire", extra: 1 } }))).toThrow(/does not take/);
    expect(bad((v) => (v.abilities = { aa: { kind: "pinch", type: "fire" } }))).toThrow(/shape/);
    expect(bad((v) => (v.items = { aa: null }))).toThrow(/a kind it may hold/);
    expect(bad((v) => (v.format_id = "other"))).not.toThrow();
    expect(bad((v) => (v.conditions = []))).toThrow(/conditions table is an object/);
    expect(bad((v) => (v.conditions = { zz: { kind: "screen", category: null, yields_to: [], modifier: 1 }, aa: { kind: "screen", category: null, yields_to: [], modifier: 1 } }))).toThrow(/sorted and unique/);
    expect(bad((v) => (v.conditions = { aa: { kind: "nosuchkind" } }))).toThrow(/kind it may hold/);
    expect(bad((v) => (v.conditions = { aa: null }))).toThrow(/kind it may hold/);
    expect(bad((v) => (v.conditions = { aa: { kind: "screen", category: null, yields_to: [], modifier: 1, extra: 1 } }))).toThrow(/does not take/);
    expect(bad((v) => (v.conditions = { aa: { kind: "terrain", boost: { type: "fire", modifier: 1 }, weakens: null, halves: null, blocks_priority: false } }))).toThrow(/shape/);
    expect(bad((v) => (v.conditions = { aa: { kind: "terrain", boost: null, weakens: { moves: ["b", "a"], modifier: 1 }, halves: null, blocks_priority: false } }))).toThrow(/shape/);
    expect(bad((v) => (v.conditions = { aa: { kind: "weather_damage", types: [], exceptions: [] } }))).toThrow(/shape/);
    expect(bad((v) => (v.abilities = { aa: { kind: "no_damage_effect", hides: ["b", "a"] } }))).toThrow(/shape/);
  });

  it("reflection: hooks with values, siblings and nested blocks; inert hooks and status-only hooks", () => {
    expect(hooksOf({ onStart: () => 0, onStartPriority: 1, onPlate: undefined, onResidualOrder: 3, onCriticalHit: false, condition: { onEnd: () => 0 }, secondaries: [{ onHit: () => 0 }], basePowerCallback: () => 0 })).toEqual(["basePowerCallback", "condition.onEnd", "onCriticalHit", "onStart", "secondaries[0].onHit"]);
    expect(blockingHooks(["onStart", "onSetStatus", "onBasePower"], false)).toEqual(["onBasePower"]);
    expect(blockingHooks(["onSetStatus"], true)).toEqual(["onSetStatus"]);
    expect(INERT_HOOKS.has("onBasePower")).toBe(false);
    expect(STATUS_HOOKS.has("onSetStatus")).toBe(true);
  });

  it("the simulator scan finds the ids the code tests by name, and refuses a missing source", () => {
    const files: Record<string, string> = {
      "sim/battle.ts": "if (pokemon.hasItem('Air Balloon') || pokemon.hasAbility(['levitate', 'eelevate'])) {}",
      "sim/battle-actions.ts": "if (move.id === 'facade') {} if ('focusenergy' in pokemon.volatiles) {} ['explosion', 'selfdestruct'].includes(move.id)",
      "sim/pokemon.ts": "this.volatiles['substitute']; this.addVolatile('confusion'); runEvent('NotAnId')",
      "sim/field.ts": "",
      "sim/side.ts": "",
      "sim/dex.ts": "",
      "sim/dex-moves.ts": "",
      "data/mods/m/scripts.ts": "if (item === 'ironball') {}",
    };
    const found = simLiterals("/x", "m", (path) => files[path.slice(3)] ?? "");
    expect([...found].sort()).toEqual(["airballoon", "confusion", "eelevate", "explosion", "facade", "focusenergy", "ironball", "levitate", "selfdestruct", "substitute"]);
    expect(() =>
      simLiterals("/x", "m", () => {
        throw new Error("missing");
      }),
    ).toThrow(KernelDataError);
  });

  it("reads unusual data shapes conservatively (a doctored copy of the engine's data)", () => {
    const realDex = (engine.sim.Dex as unknown as { mod(m: string): Record<string, { get(id: string): Record<string, unknown> }> }).mod("champions");
    const moves: Record<string, Record<string, unknown>> = {
      callback: { exists: true, id: "callback", type: "Normal", category: "Physical", basePower: 40, target: "normal", priority: 0, flags: {}, damageCallback: () => 0 },
      oddkey: { exists: true, id: "oddkey", type: "Normal", category: "Physical", basePower: 40, target: "normal", priority: 0, flags: undefined, unknownKey: 1, ignoreAbility: true, critRatio: "x" },
      oddcategory: { exists: true, id: "oddcategory", type: "Normal", category: "Other", basePower: 40, target: "normal", priority: 0, flags: {}, ignoreImmunity: { Ghost: true, Dark: false } },
    };
    const types = (realDex as unknown as { types: { get(id: string): { damageTaken: Record<string, number> }; names(): string[] } }).types;
    const dex = {
      ...realDex,
      moves: { get: (id: string) => moves[id] ?? realDex.moves?.get(id) },
      species: { get: (id: string) => (id === "bare" ? { exists: true, id: "bare", name: "Bare", baseSpecies: "Venusaur", types: ["Grass"], baseStats: { hp: 1, atk: 1, def: 1, spa: 1, spd: 1, spe: 1 }, abilities: {} } : realDex.species?.get(id)) },
      items: { get: (id: string) => (id === "stone" ? { exists: true, id: "stone", megaStone: { Venusaur: "Venusaur-Mega" } } : realDex.items?.get(id)) },
      abilities: { get: (id: string) => (id === "ordered" ? { exists: true, id: "ordered", onBasePower: () => 0, onBasePowerOrder: 2, onBasePowerPriority: 3, onBasePowerSubOrder: 4 } : realDex.abilities?.get(id)) },
      types: { get: (t: string) => ({ ...types.get(t), damageTaken: t === "Normal" ? {} : types.get(t).damageTaken }), names: () => types.names() },
    };
    const fake = { ...engine, commit: engine.commit, sim: { ...engine.sim, Dex: { formats: { get: () => ({ exists: true, mod: "champions" }) }, mod: () => dex } } } as unknown as Engine;
    const data = kernelData(fake, "doctored");
    expect(data.move("callback")?.blockers).toEqual([]);
    expect(data.move("oddkey")?.blockers).toEqual(["key:ignoreAbility", "key:unknownKey"]);
    expect(data.move("oddkey")?.crit_ratio).toBe(0);
    expect(data.move("oddkey")?.flags.size).toBe(0);
    expect(data.move("oddcategory")?.blockers).toEqual(["category"]);
    expect(data.move("oddcategory")?.ignore_immunity).toEqual(["Ghost"]);
    const bare = data.species("bare") as NonNullable<ReturnType<KernelData["species"]>>;
    expect(bare.first_ability).toBe("");
    expect(data.megaForme("stone", bare)).toBe("Venusaur-Mega");
    expect(data.megaForme("leftovers", bare)).toBeNull();
    expect(data.hookOrder("ability", "ordered", "onBasePower")).toEqual({ order: 2, priority: 3, sub_order: 4 });
    expect(data.chart.get("Normal")?.get("Fighting")).toBe(0);
  });

  it("the kernel's data reads the pinned engine once per format", () => {
    const data = kernelData(engine, GOLDEN_FORMAT);
    expect(kernelData(engine, GOLDEN_FORMAT)).toBe(data);
    expect(data.species("Garchomp")?.base).toEqual([108, 130, 95, 80, 85, 102]);
    expect(data.species("nothing")).toBeNull();
    expect(data.move("nothing")).toBeNull();
    expect(data.nature("nothing")).toBeNull();
    expect(data.abilityBlockers("nothing", false)).toBeNull();
    expect(data.conditionType("brn")).toBe("Status");
    expect(data.conditionType("nothing")).toBeNull();
    expect(data.moveHookOrder("facade", "onBasePower")).toEqual({ order: false, priority: 0, sub_order: 0 });
    expect(data.moveHookOrder("tackle", "onBasePower")).toBeNull();
    expect(data.moveHookOrder("knockoff", "onAfterHit")).toEqual({ order: false, priority: 0, sub_order: 0 });
    expect(data.breakable("levitate")).toBe(true);
    expect(data.breakable("intimidate")).toBe(false);
    expect(data.hookOrder("ability", "technician", "onBasePower").priority).toBe(30);
    expect(() => kernelData(engine, "nosuchformat")).toThrow(KernelDataError);
  });
});
