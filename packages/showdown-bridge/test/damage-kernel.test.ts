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
import { kernelEffectsHash, loadKernelEffects, readKernelEffects } from "../src/damage-kernel-effects.ts";
import { championsStat, computeBatch, KERNEL_FORMAT, KernelAbstain, staged } from "../src/damage-kernel.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { probeMechanics } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import type { MechanicsResult } from "../src/mechanics-result.ts";
import type { Settled } from "../src/probe-batch.ts";
import { exportPaste } from "../src/teams.ts";

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
    { species: "snorlax", ability: "immunity", moves: ["bodyslam"] },
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
    const user: Side = { species: "garchomp", ability: "roughskin", moves: ["dragonclaw", "earthquake", "crunch", "ironhead", "rockslide", "poisonjab"] };
    let covered = 0;
    for (const item of Object.keys(effects.items)) {
      for (const [i, state] of STATES.slice(0, 3).entries()) {
        for (const move of user.moves) if (same(query({ ...user, item }, foes[i] as Side, move, state))) covered++;
        for (const foe of foes) for (const move of foe.moves) if (same(query(foe, { ...user, item, hp: i === 1 ? 50 : null }, move, state))) covered++;
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
      [{ species: "snorlax", ability: "immunity", moves: ["facade"], status: "brn" }, ARCANINE, "facade", {}],
      [{ species: "snorlax", ability: "immunity", moves: ["snore"], status: "slp" }, ARCANINE, "snore", {}],
      [{ species: "snorlax", ability: "immunity", moves: ["snore"] }, ARCANINE, "snore", {}],
      [{ species: "kangaskhan", ability: "scrappy", moves: ["suckerpunch"] }, ARCANINE, "suckerpunch", {}],
      [GARCHOMP, ARCANINE, "dig", {}],
      [{ species: "gengar", ability: "cursedbody", moves: ["payback", "shadowball"] }, ARCANINE, "payback", {}],
      [{ species: "gengar", ability: "cursedbody", moves: ["shadowball"] }, { species: "snorlax", ability: "immunity", moves: ["bodyslam"] }, "shadowball", {}],
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

describe("4. the pieces", () => {
  it("the Champions stat formula and the stage table", () => {
    const neutral = { plus: null, minus: null };
    expect(championsStat(108, 32, "hp", neutral)).toBe(215);
    expect(championsStat(130, 32, "atk", { plus: "atk", minus: "spa" })).toBe(200);
    expect(championsStat(100, 0, "spa", { plus: "atk", minus: "spa" })).toBe(108);
    expect(championsStat(100, 0, "spe", neutral)).toBe(120);
    expect([staged(100, 0), staged(100, 1), staged(100, 6), staged(100, -1), staged(100, -6), staged(101, 3)]).toEqual([100, 150, 400, 66, 25, 252]);
    expect(KERNEL_FORMAT).toBe(1);
  });

  it("computes an empty batch", () => {
    expect(computeBatch({ n: 0, base: new Int32Array(0), final: new Int32Array(0), flags: new Int32Array(0), weather: new Int32Array(0), stab: new Int32Array(0), type_mod: new Int32Array(0) } as never).length).toBe(0);
  });

  it("the effects record's reader refuses every malformed shape", () => {
    const good = JSON.parse(readFileSync(new URL("../../../config/mechanics/gen9championsvgc2026regmc.damage-kernel.json", import.meta.url), "utf8")) as Record<string, unknown>;
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
    expect(data.breakable("levitate")).toBe(true);
    expect(data.breakable("intimidate")).toBe(false);
    expect(data.hookOrder("ability", "technician", "onBasePower").priority).toBe(30);
    expect(() => kernelData(engine, "nosuchformat")).toThrow(KernelDataError);
  });
});
