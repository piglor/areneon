/* eslint-disable @typescript-eslint/no-explicit-any */
// The battle-state tensor schema (ADR-030 step 5 groundwork): the committed record against the pinned engine and
// every committed corpus. Each battle is replayed with a spy on the engine's BattleStream (as the fork oracle and
// `world-writers-live.test.ts` do) and `Battle.toJSON()` is read at every request of both seats. The gate: every
// key of the engine's state is a field or an omitted key with a reason; every field's source resolves on real
// states; vocabulary values resolve in the engine-data table; every field another seat may see (public, or
// revealed-only where its mask is set) equals what that seat's own shared reader or world facts show (an
// independent oracle, with the Illusion and roster-gate rules); and a canary with planted markers.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel, ownChannel, type PublicMon, type PublicState, toId, type WorldFacts, worldFacts } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { type BattleStateBinding, type BattleStateSpec, BattleStateSpecError, BOOST_IDS, battleStateSpecHash, bindingFor, coveredKeys, ENGINE_LEVELS, type Field, GROUPS, loadBattleStateSpec, pathKeys, readBattleStateSpec, resolvePath, seatView, vocabularyIds } from "../src/battle-state-spec.ts";
import { type Engine, loadEngine, STAT_IDS } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { loadRandomCorpus, pendingFirstActor, RANDOM_MAX_TURNS, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { loadWorldWriters, STATE_ITEM_IDS } from "../src/world-writers.ts";
import { readWorldFactsData } from "./world-facts-data.ts";

// Replaying every committed corpus (about 250 battles, 8,500 requests) takes about a minute on an idle machine.
const CORPUS_TIMEOUT_MS = 400_000;
const SPEC_PATH = new URL(`../../../config/engine-state/${GOLDEN_FORMAT}/battle-state-spec.v1.json`, import.meta.url);
const committedSpec = (): any => JSON.parse(readFileSync(SPEC_PATH, "utf8"));
const SPEC_HASH = "044a8a0863f86c9795740e0f02097cdcee548e121ec99cbc551db5a2fcd574ec";
// Record v2 (#1464): v1 plus what the projection needs (the own Pokemon that were not brought, the own base species over a Mega
// Evolution, raw numbered volatile ids, a fifth revealed move, the slot occupant's stages); v1 stays as it was.
const SPEC_V2_PATH = new URL(`../../../config/engine-state/${GOLDEN_FORMAT}/battle-state-spec.v2.json`, import.meta.url);
const SPEC_V2_HASH = "69fb5894cce780c458646648ca1ad92bf6906b15423e458bb2e8e18d2e3303c1";

let engine: Engine;
let spec: BattleStateSpec;
let specV2: BattleStateSpec;
let binding: BattleStateBinding;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  binding = bindingFor(engine, GOLDEN_FORMAT);
  spec = loadBattleStateSpec(engine, GOLDEN_FORMAT);
  specV2 = loadBattleStateSpec(engine, GOLDEN_FORMAT, SPEC_V2_PATH);
});

type Player = "p1" | "p2";
interface Point {
  id: string;
  player: Player;
  json: any;
  channel: string[];
}

/** Every committed battle: the random-play corpus, the pending-first corpus, the special battles and the targeted fidelity corpus. */
function corpora(): { file: RandomBattleFile; actor: (player: Player) => (seed: number) => Actor }[] {
  const plain = () => randomLegalActor;
  const out: { file: RandomBattleFile; actor: (player: Player) => (seed: number) => Actor }[] = [];
  for (const file of loadRandomCorpus()) out.push({ file, actor: plain });
  for (const file of loadRandomCorpus(join(GOLDEN_DIR, "unavailable"))) out.push({ file, actor: () => pendingFirstActor });
  for (const dir of ["illusion", "baton-pass", "baton-pass-volatiles", "fling", "side-guard", "revival"]) {
    out.push({ file: JSON.parse(readFileSync(join(GOLDEN_DIR, dir, "battle.json"), "utf8")) as RandomBattleFile, actor: plain });
  }
  for (const file of loadRandomCorpus(join(GOLDEN_DIR, "search-fidelity"))) {
    const kinds = (file as unknown as { actors: Record<Player, string> }).actors;
    out.push({ file, actor: (player) => (kinds[player] === "pending-first" ? pendingFirstActor : randomLegalActor) });
  }
  return out;
}

/** Replay every committed battle and keep `toJSON()` and the seat's channel at every request of both seats. */
async function points(visit: (point: Point) => void): Promise<number> {
  let captured: any = null;
  const capture = (stream: unknown): void => {
    captured = stream;
  };
  const Base = engine.sim.BattleStream as unknown as new () => { battle: unknown };
  class Spy extends Base {
    constructor() {
      super();
      capture(this);
    }
  }
  const spy: Engine = { ...engine, sim: { ...engine.sim, BattleStream: Spy as never } };
  let count = 0;
  for (const { file, actor } of corpora()) {
    const seat = (player: Player): Actor => {
      const inner = actor(player)(file.actor_seeds[player]);
      return (request, ctx) => {
        if (ctx !== undefined && captured?.battle != null) {
          count++;
          visit({ id: file.id, player, json: JSON.parse(JSON.stringify(captured.battle.toJSON())), channel: [...ctx.channel] });
        }
        return inner(request, ctx);
      };
    };
    await runBattle({ engine: spy, formatId: GOLDEN_FORMAT, seed: file.seed, teams: file.teams, actors: { p1: seat("p1"), p2: seat("p2") }, maxTurns: RANDOM_MAX_TURNS });
  }
  return count;
}

const isMap = (v: unknown): v is Record<string, any> => typeof v === "object" && v !== null && !Array.isArray(v);
const engineId = (value: unknown): string => {
  const text = String(value ?? "");
  const ref = /^\[[A-Za-z]+:(.*)\]$/.exec(text);
  return toId(ref === null ? text : (ref[1] as string));
};
const nameOf = (mon: any): string => (mon.set.name || mon.set.species) as string;
const levelOf = (details: string): number => Number(/, L(\d+)/.exec(details)?.[1] ?? 100);

/** The keys of every classified level of one engine state, as `level.key`. */
function stateKeys(json: any, into: Set<string>): void {
  const add = (level: string, value: unknown): void => {
    if (isMap(value)) for (const key of Object.keys(value)) into.add(`${level}.${key}`);
  };
  add("battle", json);
  add("field", json.field);
  add("effect", json.field.weatherState);
  add("effect", json.field.terrainState);
  for (const v of Object.values(json.field.pseudoWeather)) add("effect", v);
  for (const side of json.sides) {
    add("side", side);
    for (const v of Object.values(side.sideConditions)) add("effect", v);
    for (const slot of side.slotConditions) for (const v of Object.values(slot)) add("effect", v);
    for (const mon of side.pokemon) {
      add("pokemon", mon);
      for (const key of ["statusState", "itemState", "abilityState", "speciesState"]) add("effect", mon[key]);
      for (const v of Object.values(mon.volatiles)) add("effect", v);
      for (const slot of [...mon.moveSlots, ...(mon.baseMoveSlots ?? [])]) add("move_slot", slot);
    }
  }
}

/** The ids a vocabulary-valued engine field holds in one state. */
function engineIds(field: Field, json: any): string[] {
  if (field.source.kind !== "engine" || field.source.read !== "id") return [];
  const path = field.source.path;
  const sides = json.sides as any[];
  const mons = sides.flatMap((s) => s.pokemon);
  const actives = sides.flatMap((s) => s.pokemon.filter((m: any) => m.isActive === true));
  const keys = path.split(".").slice(1);
  const at = (value: any): unknown => keys.reduce((v: any, k) => (isMap(v) ? v[k] : undefined), value);
  const values: unknown[] = path.startsWith("pokemon.moveSlots")
    ? mons.flatMap((m) => m.moveSlots.map((s: any) => s.id))
    : path.startsWith("pokemon.baseMoveSlots")
      ? mons.flatMap((m) => (m.baseMoveSlots ?? []).map((s: any) => s.id))
      : path.startsWith("pokemon.")
        ? mons.map(at)
        : path.startsWith("active.")
          ? actives.map(at)
          : path.startsWith("battle.")
            ? [at(json)]
            : [at(json.field)];
  // A fainted Pokemon's engine status is the marker fnt: roster.fainted holds it (the field's description).
  return values.filter((v) => v !== null && v !== undefined && v !== "" && !(field.id === "roster.status" && v === "fnt")).map(engineId);
}

// ---- the oracle: what another seat may see equals what its own reader shows ----

/** One opponent entry as the viewing seat reads it, with the engine Pokemon packed at it (the Illusion rule). */
interface Packed {
  entry: PublicMon;
  /** The Pokemon whose battle state is packed at the entry (a disguised Pokemon in play is packed at its disguise's entry). */
  state: any;
  /** The Pokemon whose identity the entry shows (the disguise's target while one is in play). */
  identity: any;
}

/** The engine Pokemon packed at each opponent entry the seat has seen (the roster gate), by the Illusion rule. */
function packOpponent(foe: any, state: PublicState): Packed[] {
  const byName = new Map<string, { state: any; identity: any }>();
  for (const mon of foe.pokemon) if (!byName.has(nameOf(mon))) byName.set(nameOf(mon), { state: mon, identity: mon });
  // A disguised Pokemon in play is packed at its disguise's entry; its own entry holds what the seat last saw of
  // it, which the engine's current state does not: the engine packs nothing there until the |replace|.
  const hidden = new Set<string>();
  for (const mon of foe.pokemon) {
    if (mon.illusion == null || mon.isActive !== true) continue;
    const letter = /\[Pokemon:p\d([a-z])\]/.exec(String(mon.illusion))?.[1] as string;
    const target = foe.pokemon[letter.charCodeAt(0) - 97];
    byName.set(nameOf(target), { state: mon, identity: target });
    hidden.add(nameOf(mon));
  }
  const out: Packed[] = [];
  for (const entry of state.opponent.roster) {
    if (entry.phantom === true || entry.hp_num === null) continue; // the roster gate: an unseen entry packs as unknown
    if (hidden.has(entry.name as string)) continue;
    const packed = byName.get(entry.name as string);
    if (packed === undefined) throw new Error(`seen entry ${entry.name} has no engine Pokemon`);
    out.push({ entry, ...packed });
  }
  return out;
}

interface Context {
  json: any;
  foe: any;
  state: PublicState;
  facts: WorldFacts | null;
  packed: Packed[];
  vocab: (symbol: string) => Set<string>;
  /** The ids a -start shows that are no engine volatile (the world writers' slot_conditions and illusion_transform rows). */
  slotIds: Set<string>;
}

type Compare = (ctx: Context) => string[];
const one = (miss: string | null): string[] => (miss === null ? [] : [miss]);
const each =
  (check: (p: Packed) => string | null): Compare =>
  (ctx) =>
    ctx.packed.flatMap((p) => {
      const miss = check(p);
      return miss === null ? [] : [`${miss} (${p.entry.name}, ${p.entry.species})`];
    });
const same = (what: string, a: unknown, b: unknown): string | null => (JSON.stringify(a) === JSON.stringify(b) ? null : `${what}: engine ${JSON.stringify(a)}, seat ${JSON.stringify(b)}`);
const GENDER: Record<string, string | null> = { M: "M", F: "F", N: null, "": null };

/** How each field another seat may see is shown to that seat. A field classed public or revealed-only with no entry here fails the gate. */
const COMPARE: Record<string, Compare> = {
  "battle.turn": (c) => one(same("turn", c.json.turn, c.state.turn)),
  "battle.ended": (c) => one(same("ended", c.json.ended, c.state.winner !== null)),
  "field.weather": (c) => one(same("weather", toId(c.json.field.weather), toId(c.state.weather ?? ""))),
  "field.terrain": (c) => one(same("terrain", toId(c.json.field.terrain), toId(c.state.terrain ?? ""))),
  "field.pseudo_present": (c) => one(same("field conditions", Object.keys(c.json.field.pseudoWeather).sort(), c.state.field_conditions.map(toId).sort())),
  "side.condition_present": (c) => one(same("side conditions", Object.keys(c.foe.sideConditions).sort(), Object.keys(c.state.opponent.conditions).map(toId).sort())),
  "side.condition_layers": (c) =>
    Object.entries(c.foe.sideConditions as Record<string, any>).flatMap(([id, v]) => one(same(`layers ${id}`, v.layers ?? 1, Object.entries(c.state.opponent.conditions).find(([k]) => toId(k) === id)?.[1]))),
  // The faint lines the seat saw (a revived Pokemon fainted once and counts).
  "side.total_fainted": (c) => (c.facts === null ? [] : one(same("total fainted", c.foe.totalFainted, c.facts.opponent.total_fainted))),
  // A temporary forme (Castform's weather forme) is the reader's temp_species.
  "roster.species": each((p) => (p.entry.species === null ? null : same("species", engineId(p.identity.species), toId(p.entry.temp_species ?? p.entry.species)))),
  "roster.level": each((p) => same("level", levelOf(p.identity.details), p.entry.level)),
  "roster.gender": each((p) => same("gender", GENDER[p.identity.gender as string] ?? null, p.entry.gender)),
  "roster.shiny": each((p) => same("shiny", String(p.identity.details).includes(", shiny"), p.entry.shiny)),
  "roster.boosts": each((p) => same("boosts", p.state.boosts, p.entry.boosts)),
  "roster.fainted": each((p) => same("fainted", p.state.fainted, p.entry.fainted)),
  "roster.is_active": each((p) => same("in play", p.state.isActive === true, p.entry.active_slot !== null)),
  "roster.transformed": each((p) => same("transformed", p.state.transformed, p.entry.volatiles.includes("transform"))),
  "active.volatile_present": (c) => {
    const known = c.vocab("VOLATILE");
    const out: string[] = [];
    for (const p of c.packed) {
      if (p.state.isActive !== true) continue;
      for (const shown of p.entry.volatiles) {
        const id = [toId(shown), toId(shown).replace(/\d+$/, ""), toId(shown).replace(/(atk|def|spa|spd|spe)$/, "")].find((x) => known.has(x) && !c.slotIds.has(x));
        if (id !== undefined && !(id in p.state.volatiles)) out.push(`volatile ${id} shown but not in the engine`);
      }
    }
    return out;
  },
  // v2: the stages of the Pokemon in play at an entry (a disguised one at its disguise's entry) are what that entry shows.
  "active.boosts": each((p) => (p.state.isActive === true ? same("slot stages", p.state.boosts, p.entry.boosts) : null)),
  "active.active_turns": (c) => activeFacts(c, (fact, mon) => same("active turns", mon.activeTurns, fact.active_turns)),
  // Only a deliberately misclassified record reaches these (the committed record gives them class own-exact): they
  // compare like for like, so a record that calls them public fails on a value. Hit points as a share of the
  // maximum (the seat sees points of 100, the last ones it was shown), the status the seat was last shown (the
  // engine's fnt marker is the reader's fainted flag) and the held item.
  "roster.hp": each((p) => (p.entry.hp_num === null || p.entry.hp_den === null ? null : same("hp share", p.state.hp * p.entry.hp_den, p.entry.hp_num * p.state.maxhp))),
  "roster.status": each((p) => same("status", p.state.status === "fnt" ? "" : toId(p.state.status), toId(p.entry.status ?? ""))),
  "roster.item": each((p) => (p.entry.item_known ? same("item", engineId(p.state.item), p.entry.item_lost ? "" : (p.entry.item ?? "")) : null)),
};

function activeFacts(c: Context, check: (fact: { active_turns: number; move_actions: number }, mon: any) => string | null): string[] {
  if (c.facts === null) return [];
  const out: string[] = [];
  for (const [letter, fact] of Object.entries(c.facts.opponent.active)) {
    const mon = c.foe.pokemon.find((m: any) => m.isActive === true && m.position === letter.charCodeAt(0) - 97);
    if (fact === null || mon === undefined) continue;
    out.push(...one(check(fact, mon)));
  }
  return out;
}

/** Every field another seat may see, compared at one point; a field with no comparison fails. */
function oracle(record: BattleStateSpec, ctx: Context): string[] {
  const out: string[] = [];
  for (const field of record.fields) {
    if (field.visibility !== "public" && field.visibility !== "revealed-only") continue;
    const compare = COMPARE[field.id];
    if (compare === undefined) out.push(`${field.id}: no derivation of another seat shows it`);
    else for (const miss of compare(ctx)) out.push(`${field.id}: ${miss}`);
  }
  return out;
}

function factsOf(player: Player, channel: string[], data: ReturnType<typeof readWorldFactsData>): WorldFacts | null {
  try {
    return worldFacts({ channel: ownChannel(player, channel), data });
  } catch {
    return null;
  }
}

const SAMPLES = new URL("../../../fixtures/contracts/v1/battle-state-spec/", import.meta.url);
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(new URL(`${kind}/${name}.json`, SAMPLES), "utf8"));

describe("the committed battle-state spec", () => {
  it("reads against the pinned engine, and its hash is pinned (a change is a new record)", () => {
    expect(spec.format_id).toBe(GOLDEN_FORMAT);
    expect(battleStateSpecHash(spec)).toBe(SPEC_HASH);
    expect(spec.groups.map((g) => g.id)).toEqual([...GROUPS]);
    expect(spec.world_items.map((w) => w.id)).toEqual(STATE_ITEM_IDS);
    expect(binding.counts.players).toBe(2);
  });

  it("binds the vocabulary sizes to the engine-data table: a doctored table or count is refused", () => {
    const table = { ...binding.engine_data, species: [] };
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, engine_data: table })).toThrow("engine_data_hash is not the table's");
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, counts: { ...binding.counts, max_move_count: 6 } })).toThrow("MOVE has size 4, the max_move_count is 6");
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, vocabulary_hash: "0".repeat(64) })).toThrow("vocabulary_hash");
    expect(vocabularyIds(binding.engine_data, ["abilities", "conditions", "items", "moves"]).length).toBeGreaterThan(1000);
    expect(() => vocabularyIds(binding.engine_data, ["nosuchlist"])).toThrow("no list nosuchlist");
  });

  it("refuses a vocabulary field whose indices the table makes too many for its dtype", () => {
    const doctored = committedSpec();
    doctored.fields.find((f: any) => f.id === "roster.base_species").dtype = "int8";
    expect(() => readBattleStateSpec(doctored, binding)).toThrow("roster.base_species: 401 indices do not fit int8");
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, engine_data: { ...binding.engine_data, types: [] } })).toThrow();
  });

  it("refuses a record of another format", () => {
    expect(() => loadBattleStateSpec(engine, "gen9ou", SPEC_PATH)).toThrow(BattleStateSpecError);
    // The open-sheet variant has the same counts and table, so only the record's own format refuses it.
    expect(() => loadBattleStateSpec(engine, `${GOLDEN_FORMAT}bo3`, SPEC_PATH)).toThrow(`the record is for ${GOLDEN_FORMAT}, not ${GOLDEN_FORMAT}bo3`);
  });
});

describe("the committed battle-state spec, record v2 (#1464)", () => {
  it("reads against the pinned engine, its hash is pinned, and v1 is unchanged beside it", () => {
    expect(battleStateSpecHash(specV2)).toBe(SPEC_V2_HASH);
    expect(battleStateSpecHash(spec)).toBe(SPEC_HASH);
    expect(specV2.format_id).toBe(GOLDEN_FORMAT);
  });

  it("holds every field of v1 as it was, plus the fields the projection reads", () => {
    const v2 = new Map(specV2.fields.map((f) => [f.id, f]));
    for (const f of spec.fields) {
      const kept = v2.get(f.id) as Field;
      expect(kept, f.id).toBeDefined();
      // Only descriptions may say more; class, source, shape and value are v1's.
      // The one widened range: v1's 0..127 cannot hold the stall counter's 729 (see "engine-side counter ranges").
      const widened = f.id === "active.volatile_counter" ? { dtype: "int16", value: { kind: "range", min: 0, max: 1023 } } : {};
      expect({ ...kept, description: "" }, f.id).toEqual({ ...f, ...widened, description: "" });
    }
    const added = specV2.fields.filter((f) => !spec.fields.some((g) => g.id === f.id)).map((f) => f.id);
    expect(added).toEqual([
      "active.boosts",
      "seat_active.own_moves_used",
      "seat_active.protocol_derived",
      "seat_active.protocol_volatiles",
      "seat_active.volatile_numbered",
      "seat_active.volatile_oldest_started",
      "seat_active.volatile_started_count",
      "seat_active.volatile_stat_raw",
      "seat_roster.ability_suppressed",
      "seat_roster.hp_den_shown",
      "seat_roster.own_can_mega_evo",
      "seat_roster.own_moves",
      "seat_roster.own_pp",
      "seat_roster.own_stats",
      "seat_roster.revealed_extra",
      "seat_roster.revealed_extra_at",
    ]);
    // Every added field but the slot occupant's public stages is a seat field: the seat's own derivation.
    for (const id of added) expect((v2.get(id) as Field).visibility, id).toBe(id === "active.boosts" ? "public" : "seat");
  });
});

describe("engine-side counter ranges (#1464)", () => {
  it("the engine-side volatile counter holds the largest counter the pinned engine's conditions reach (the stall condition's 729)", () => {
    const sim = engine.sim as any;
    const dex = sim.Dex.mod(sim.Dex.formats.get(GOLDEN_FORMAT).mod);
    const maxima = Object.keys(dex.data.Conditions)
      .map((id) => dex.conditions.get(id).counterMax)
      .filter((n): n is number => typeof n === "number");
    expect(maxima).toContain(729);
    const largest = Math.max(...maxima);
    for (const record of [spec, specV2]) {
      const field = record.fields.find((f) => f.id === "active.volatile_counter") as any;
      expect(field.value.kind).toBe("range");
      if (record === specV2) expect(field.value.max, "record v2").toBeGreaterThanOrEqual(largest);
    }
  });
});

describe("the reader", () => {
  it.each([
    ["encoder-reads-hidden", "battle.prng is hidden"],
    ["encoder-unknown-field", "battle.turns is not a field"],
    ["gate-not-seat", "its gate must be a 0/1 seat field"],
    ["group-mismatch", "battle.turn: the id does not start with its group"],
    ["groups-out-of-order", "groups: every group exactly once"],
    ["mask-of-another-field", "masks active.volatile_present, not this field"],
    ["masks-on-engine-field", "only a seat field masks another"],
    ["omitted-but-read", "battle.turn is read by a field"],
    ["opponent-reads-exact-hp", "roster.hp is not projectable to another seat"],
    ["opponent-reads-unprojectable", "roster.stats is not projectable to another seat"],
    ["projection-on-public", "a projection exactly for an own-exact field"],
    ["range-overflows-dtype", "the range does not fit int8"],
    ["range-reversed", "min exceeds max"],
    ["reveal-missing", "a reveal mask exactly for a revealed-only field"],
    ["reveal-wrong-mask", "its mask must be a 0/1 seat field"],
    ["seat-class-on-engine-field", "the class seat is exactly for a seat field"],
    ["seat-engine-source", "a seat field is a derivation"],
    ["seat-not-seat-class", "the class seat is exactly for a seat field"],
    ["unknown-dim", "MOVES is not a dimension"],
    ["unsorted-fields", "is repeated or out of order"],
    ["unsorted-out-of-scope", "out_of_scope"],
    ["unsorted-reducer", "reducer:"],
    ["unused-dim", "ZZZ is not used"],
    ["world-item-missing", "every state item of the world writers exactly once"],
    ["world-item-unknown-field", "roster.nothing is not a field"],
    ["wrong-engine-root", "cannot read the engine root battle"],
  ])("refuses the reader-invalid sample %s", (name, message) => {
    expect(() => readBattleStateSpec(sample("reader-invalid", name))).toThrow(BattleStateSpecError);
    expect(() => readBattleStateSpec(sample("reader-invalid", name))).toThrow(message);
  });

  it("refuses every schema-invalid sample and accepts the valid one", () => {
    for (const name of ["bad-engine-path", "extra-field", "fractional-size", "missing-world-items", "next-version", "unknown-list", "unknown-read", "unknown-visibility"]) {
      expect(() => readBattleStateSpec(sample("invalid", name)), name).toThrow("invalid BattleStateSpec");
    }
    expect(readBattleStateSpec(sample("valid", "v1"), binding)).toEqual(spec);
  });

  it("refuses an unsorted vocabulary list of a field and of a dimension, and a reveal mask that is no field", () => {
    const lists = committedSpec();
    lists.fields.find((f: any) => f.id === "roster.item").value.lists = ["items", "abilities"];
    expect(() => readBattleStateSpec(lists)).toThrow("roster.item: lists");
    const dims = committedSpec();
    dims.dims.find((d: any) => d.symbol === "VOLATILE").lists.reverse();
    expect(() => readBattleStateSpec(dims)).toThrow("dims: VOLATILE");
    const mask = committedSpec();
    mask.fields.find((f: any) => f.id === "roster.species").reveal = "seat_roster.nothing";
    expect(() => readBattleStateSpec(mask)).toThrow("its mask seat_roster.nothing is not a field");
    const unsorted = committedSpec();
    unsorted.world_items.find((w: any) => w.kind === "held" && w.fields.length > 1).fields.reverse();
    expect(() => readBattleStateSpec(unsorted)).toThrow("world_items:");
  });
});

describe("engine paths", () => {
  it("name the level of each key they read: an active Pokemon is a Pokemon, after a move-slot list a move slot, after an effect state an effect", () => {
    expect(pathKeys("active.volatiles{}.duration")).toEqual([["pokemon", "volatiles"], ["effect", "duration"]]);
    expect(pathKeys("pokemon.moveSlots[].pp")).toEqual([["pokemon", "moveSlots"], ["move_slot", "pp"]]);
    expect(pathKeys("side.slotConditions[]{}.duration")).toEqual([["side", "slotConditions"], ["effect", "duration"]]);
    expect(pathKeys("pokemon.statusState.stage")).toEqual([["pokemon", "statusState"], ["effect", "stage"]]);
    expect(pathKeys("battle.lastMove.move")).toEqual([["battle", "lastMove"]]);
  });

  it("resolve on a state: the parents of the last step and how many hold it", () => {
    const state = { turn: 3, field: { pseudoWeather: {} }, sides: [{ active: ["[Pokemon:p1a]"], pokemon: [{ isActive: true, hp: 5, moveSlots: [{ pp: 1 }], volatiles: { taunt: { duration: 2 } } }, { isActive: false, moveSlots: [] }] }] };
    expect(resolvePath(state, "battle.turn")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "pokemon.hp")).toEqual({ parents: 2, found: 1 });
    expect(resolvePath(state, "active.volatiles{}.duration")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "active.volatiles{}")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "pokemon.moveSlots[]")).toEqual({ parents: 2, found: 2 });
    expect(resolvePath(state, "pokemon.moveSlots[].pp")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "field.pseudoWeather{}")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "side.active[]")).toEqual({ parents: 1, found: 1 });
    expect(resolvePath(state, "pokemon.nothing.deeper")).toEqual({ parents: 0, found: 0 });
  });
});

describe("the projection rules (privacy by construction)", () => {
  const field = (visibility: Field["visibility"], projection?: Field["projection"]): Field => ({ id: "roster.x", group: "roster", extra: [], dtype: "int8", value: { kind: "range", min: 0, max: 1 }, source: { kind: "derived", from: "reducer", name: "x" }, visibility, ...(projection === undefined ? {} : { projection }), description: "x" }) as Field;
  const views = (f: Field): string[] => [true, false].flatMap((owner) => [true, false].map((seat) => seatView(f, { owner, seat })));
  it("gives each class its view, for the owner and another seat, at the reader's own seat index and another's", () => {
    expect(views(field("public"))).toEqual(["exact", "exact", "exact", "exact"]);
    expect(views(field("hidden"))).toEqual(["none", "none", "none", "none"]);
    expect(views(field("revealed-only"))).toEqual(["masked", "masked", "masked", "masked"]);
    expect(views(field("own-exact", "none"))).toEqual(["exact", "exact", "none", "none"]);
    // A seat field is exact only at the reader's own seat index: seat 1 never reads seat 0's hp_shown or request.
    expect(views(field("seat"))).toEqual(["exact", "none", "exact", "none"]);
  });

  it("no mapping reads a hidden field, and no opponent mapping a field another seat cannot see", () => {
    const fields = new Map(spec.fields.map((f) => [f.id, f]));
    for (const row of [...spec.encoder, ...spec.reducer]) {
      for (const ref of [...row.own, ...row.opponent, ...row.shared]) expect((fields.get(ref) as Field).visibility, ref).not.toBe("hidden");
      for (const ref of [...row.opponent, ...row.shared]) expect(seatView(fields.get(ref) as Field, { owner: false, seat: true }), ref).not.toBe("none");
    }
    const leak = committedSpec();
    leak.encoder.find((c: any) => c.column === "stat_*").opponent = ["roster.stats"];
    expect(() => readBattleStateSpec(leak)).toThrow("roster.stats is not projectable to another seat");
    const rng = committedSpec();
    rng.reducer.find((r: any) => r.key === "turn").shared = ["battle.prng", "battle.turn"];
    expect(() => readBattleStateSpec(rng)).toThrow("battle.prng is hidden");
  });
});

describe("the spec against every committed corpus (the drift gate and the independent oracle)", () => {
  const keys = new Set<string>();
  const resolved = new Map<string, { parents: number; found: number }>();
  const vocab = new Map<string, Set<string>>();
  const dimIds = new Map<string, Set<string>>();
  const unknownIds: string[] = [];
  const failures: string[] = [];
  const failuresV2: string[] = [];
  const doctored: Record<string, string> = {};
  let statusMisclassified = 0;
  let statusPublic: BattleStateSpec;
  const compared = { points: 0, skipped: 0, entries: 0, disguised: 0, unseen: 0, sources: new Set<string>() };
  const canary = { states: 0, leaks: [] as string[], control: 0 };
  let requests = 0;
  const data = readWorldFactsData();
  let slotIds = new Set<string>();

  /** A doctored record: one field moved to class public, which the oracle must refuse. */
  const misclassified = (id: string): BattleStateSpec => {
    const copy = structuredClone(spec);
    const f = copy.fields.find((x) => x.id === id) as Field;
    f.visibility = "public";
    delete f.projection;
    return copy;
  };
  // Each with the reason it must fail for: a value the seat's reader shows otherwise, or no derivation of the seat at all.
  const DOCTORED: Record<string, RegExp> = {
    "roster.hp": /^roster\.hp: hp share: engine \d+, seat \d+/,
    "roster.item": /^roster\.item: item: engine "[a-z0-9]*", seat "[a-z0-9]*"/,
    "roster.move_pp": /^roster\.move_pp: no derivation of another seat shows it$/,
  };

  beforeAll(async () => {
    for (const f of spec.fields) if (f.value.kind === "vocabulary") vocab.set(f.id, new Set(vocabularyIds(binding.engine_data, f.value.lists)));
    for (const d of spec.dims) if (d.kind === "vocabulary") dimIds.set(d.symbol, new Set(vocabularyIds(binding.engine_data, d.lists)));
    // The world writers' rows name the ids a -start shows that are no engine volatile: a delayed move's is a slot
    // condition, Transform's is roster.transformed.
    const rows = loadWorldWriters(engine, GOLDEN_FORMAT).items;
    slotIds = new Set(rows.filter((i) => i.id === "slot_conditions" || i.id === "illusion_transform").flatMap((i) => i.ids ?? []));
    statusPublic = misclassified("roster.status");
    const variants = Object.keys(DOCTORED).map((id) => [id, misclassified(id)] as const);
    requests = await points(({ id: battle, player, json, channel }) => {
      stateKeys(json, keys);
      for (const f of spec.fields) {
        if (f.source.kind !== "engine") continue;
        const r = resolvePath(json, f.source.path);
        const sum = resolved.get(f.id) ?? { parents: 0, found: 0 };
        resolved.set(f.id, { parents: sum.parents + r.parents, found: sum.found + r.found });
        const known = vocab.get(f.id);
        if (known !== undefined) for (const id of engineIds(f, json)) if (!known.has(id)) unknownIds.push(`${f.id}=${id}`);
      }
      const check = (symbol: string, ids: string[]): void => {
        for (const id of ids) if (!(dimIds.get(symbol) as Set<string>).has(id)) unknownIds.push(`${symbol}=${id}`);
      };
      check("PSEUDO_WEATHER", Object.keys(json.field.pseudoWeather));
      for (const side of json.sides) {
        check("SIDE_CONDITION", Object.keys(side.sideConditions));
        for (const slot of side.slotConditions) check("SLOT_CONDITION", Object.keys(slot));
        for (const mon of side.pokemon) {
          check("VOLATILE", Object.keys(mon.volatiles));
          // The engine's typeless marker (Burn Up) is no type: it sets no flag (roster.types' description).
          check("TYPE", (mon.types as string[]).filter((t) => t !== "???").map(engineId));
          expect(Object.keys(mon.boosts)).toEqual([...BOOST_IDS]);
          expect(Object.keys(mon.storedStats)).toEqual(STAT_IDS.slice(1));
          expect(mon.moveSlots.length).toBeLessThanOrEqual(binding.counts.max_move_count as number);
        }
        expect(side.active.length).toBe(binding.counts.active_per_side);
      }
      expect(json.sides.length).toBe(binding.counts.players);
      expect(String(json.prng).split(",")[1]?.length).toBe((binding.counts.prng_words as number) * 8);

      // The oracle: every field another seat may see, against that seat's own shared reader and world facts.
      const run = interpretChannel(player, channel, { terrains: [...data.terrains] });
      const state = run.frames.at(-1)?.state;
      const foe = json.sides.find((s: any) => s.id !== player);
      if (state === undefined || run.flags.length > 0 || state.request_kind === "preview") {
        compared.skipped++; // a flagged channel (the reader's Illusion twin) or Team Preview, before any entry is seen
      } else {
        // The world facts refuse an Illusion |replace|: the facts' fields are not compared there.
        const facts = factsOf(player, channel, data);
        const packed = packOpponent(foe, state);
        compared.points++;
        compared.entries += packed.length;
        compared.disguised += packed.filter((p) => p.state !== p.identity).length;
        compared.unseen += state.opponent.roster.filter((m) => m.hp_num === null && m.phantom !== true).length;
        const ctx: Context = { json, foe, state, facts, packed, vocab: (s) => dimIds.get(s) as Set<string>, slotIds };
        for (const miss of oracle(spec, ctx)) failures.push(`${miss} @ ${battle} ${player} turn ${json.turn}`);
        for (const miss of oracle(specV2, ctx)) failuresV2.push(`${miss} @ ${battle} ${player} turn ${json.turn}`);
        statusMisclassified += oracle(statusPublic, ctx).filter((m) => m.startsWith("roster.status:")).length;
        for (const [id, record] of variants) {
          const miss = oracle(record, ctx).find((m) => m.startsWith(`${id}:`));
          if (miss !== undefined) doctored[id] ??= miss;
        }
        for (const f of spec.fields) if (f.source.kind === "derived" && f.source.from === "shared_reader") resolveReader(f.source.name, state, compared.sources);

      }

      // Canary: plant markers in every value another seat may not see, then project the state for this seat.
      const planted = structuredClone(json);
      planted.prng = "sodium,7777000077770000777700007777000077770000777700007777000077770000";
      planted.quickClawRoll = 77771;
      if (isMap(planted.field.weatherState)) planted.field.weatherState.duration = 77779;
      if (isMap(planted.field.terrainState)) planted.field.terrainState.duration = 77786;
      for (const v of Object.values(planted.field.pseudoWeather) as any[]) v.duration = 77785;
      const target = planted.sides.find((s: any) => s.id !== player);
      for (const mon of target.pokemon) {
        mon.hp = 77770;
        mon.maxhp = 77772;
        mon.storedStats = { atk: 77773, def: 77773, spa: 77773, spd: 77773, spe: 77773 };
        for (const slot of mon.moveSlots) Object.assign(slot, { pp: 77774, maxpp: 77774, id: "m77787", disabled: 77782, used: 77783 });
        for (const v of Object.values(mon.volatiles) as any[]) Object.assign(v, { duration: 77775, counter: 77775, hp: 77775, layers: 77775 });
        if (isMap(mon.statusState)) Object.assign(mon.statusState, { time: 77776, startTime: 77776, stage: 77776 });
        Object.assign(mon, { position: 77777, illusion: "[Pokemon:77780]", canMegaEvo: "77781", status: "s77788", lastItem: "i77789", timesAttacked: 77790, item: "i77791", ability: "a77792", baseAbility: "a77793", activeMoveActions: 77794 });
      }
      for (const v of Object.values(target.sideConditions) as any[]) v.duration = 77778;
      for (const slot of target.slotConditions) for (const v of Object.values(slot) as any[]) v.duration = 77784;
      const projected = JSON.stringify(project(spec, planted, player, seatView));
      canary.states++;
      const found = projected.match(/7777\d/g);
      if (found !== null) canary.leaks.push([...new Set(found)].join());
      if (/7777\d/.test(JSON.stringify(project(spec, planted, player, () => "exact")))) canary.control++;
    });
  }, CORPUS_TIMEOUT_MS);

  it("read every request of every corpus", () => {
    expect(requests).toBeGreaterThan(8000);
  });

  it("every key of the engine's state at a classified level is a field's or an omitted key with a reason", () => {
    const covered = coveredKeys(spec);
    const omitted = new Set(spec.omitted.map((o) => `${o.level}.${o.key}`));
    expect([...keys].filter((k) => !covered.has(k) && !omitted.has(k)).sort(), "add a field or an omitted row with a reason").toEqual([]);
    expect([...omitted].filter((k) => !keys.has(k)).sort()).toEqual([]);
    for (const level of new Set([...keys].map((k) => k.split(".")[0]))) expect(ENGINE_LEVELS).toContain(level);
  });

  it("every field's engine source resolves on real states (everywhere, when it is one key below the battle, a side or the field)", () => {
    for (const f of spec.fields) {
      if (f.source.kind !== "engine") continue;
      const r = resolved.get(f.id) as { parents: number; found: number };
      expect(r.found, `${f.id} never resolves`).toBeGreaterThan(0);
      // A Pokemon's keys such as lastMoveTargetLoc are set when first used, and a deeper path may cross a null.
      if (/^(battle|side|field)\.[A-Za-z]+$/.test(f.source.path)) expect(r.found, `${f.id} is missing on some state`).toBe(r.parents);
    }
  });

  it("every vocabulary value on real states is an id of the field's engine-data lists", () => {
    expect(unknownIds.slice(0, 20)).toEqual([]);
  });

  it("every field another seat may see equals that seat's own reader and world facts, the Illusion users included (independent oracle)", () => {
    const byField: Record<string, string> = {};
    for (const miss of failures) {
      const id = miss.split(":")[0] as string;
      byField[id] ??= `${failures.filter((m) => m.startsWith(`${id}:`)).length}x ${miss}`;
    }
    expect(byField).toEqual({});
    expect(compared.points).toBeGreaterThan(7000);
    expect(compared.entries).toBeGreaterThan(20000);
    expect(compared.disguised, "the Illusion battle's disguises are compared at their shown entry").toBeGreaterThan(0);
    expect(compared.unseen, "unseen entries (brought or not) exist and are gated").toBeGreaterThan(1000);
  });

  it("record v2: every field another seat may see equals that seat's own reader and world facts too", () => {
    const byField: Record<string, string> = {};
    for (const miss of failuresV2) {
      const id = miss.split(":")[0] as string;
      byField[id] ??= `${failuresV2.filter((m) => m.startsWith(`${id}:`)).length}x ${miss}`;
    }
    expect(byField).toEqual({});
  });

  it("a deliberately misclassified field fails the oracle, for the reason it must (negative control)", () => {
    for (const [id, reason] of Object.entries(DOCTORED)) expect(doctored[id] ?? "never failed", id).toMatch(reason);
  });

  it("status: the engine's status equals the reader's on every seen entry at this pin, so its own-exact class is a margin, not a measured leak", () => {
    // Finding: the Champions mod prints every Natural Cure cure, so no committed request shows another seat a status
    // other than the engine's. The class stays own-exact (a pin whose Natural Cure hides the cure would leak), and the
    // comparison itself has teeth: a status the seat was not shown fails it.
    expect(statusMisclassified).toBe(0);
    const entry = { name: "X", species: "x", status: "brn" } as unknown as PublicMon;
    const ctx = { packed: [{ entry, state: { status: "par" }, identity: {} }] } as unknown as Context;
    expect((COMPARE["roster.status"] as Compare)(ctx)).toEqual(['status: engine "par", seat "brn" (X, x)']);
  });

  it("every shared-reader derivation a seat field names exists in the seat's reader state", () => {
    const named = new Set(spec.fields.flatMap((f) => (f.source.kind === "derived" && f.source.from === "shared_reader" ? [f.source.name] : [])));
    expect([...named].filter((n) => !compared.sources.has(n)).sort()).toEqual([]);
  });

  it("projects no hidden or unprojectable value to another seat (canary), and the scan has teeth (negative control)", () => {
    expect(canary.states).toBe(requests);
    expect(canary.leaks.slice(0, 10)).toEqual([]);
    expect(canary.control).toBe(canary.states);
  });
});

/** Record which shared-reader names a seat field derives from exist in the seat's state (on a Pokemon, a side or the state). */
function resolveReader(name: string, state: PublicState, into: Set<string>): void {
  const mon = state.opponent.roster[0] as unknown as Record<string, unknown> | undefined;
  if (name in state || name in state.opponent || (mon !== undefined && name in mon)) into.add(name);
}

/**
 * The canary's projection of the engine-sourced fields for one seat, by `view` alone: every field the class lets the
 * seat read exactly, on every entry of both sides. It reads no reveal mask (a revealed-only field is never read here)
 * and applies no roster gate (every entry, seen or not, is projected), so it is the most a class could let through:
 * if no marker survives it, none survives the gated, masked packing either.
 */
function project(record: BattleStateSpec, json: any, seat: Player, view: (field: Field, viewer: { owner: boolean; seat: boolean }) => string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const field of record.fields) {
    if (field.source.kind !== "engine") continue;
    const path = field.source.path;
    const how = field.source.read;
    const read = (part: any): unknown[] => {
      const r: unknown[] = [];
      walk(part, path.match(/\.[A-Za-z]+|\[\]|\{\}/g) as string[], r);
      return r.map((v) => (how === "present" ? 1 : how === "bool" ? (v ? 1 : 0) : how === "length" ? (v as unknown[]).length : v));
    };
    if (field.group === "battle" || field.group === "field") {
      if (view(field, { owner: false, seat: true }) === "exact") out[field.id] = read(field.group === "battle" ? json : json.field);
      continue;
    }
    for (const side of json.sides) {
      const shown = view(field, { owner: side.id === seat, seat: true });
      if (shown === "none" || shown === "masked") continue;
      const parts = path.startsWith("side") ? [side] : path.startsWith("active") ? side.pokemon.filter((m: any) => m.isActive === true) : side.pokemon;
      out[`${field.id}:${side.id}`] = parts.map(read);
    }
  }
  return out;
}

function walk(value: any, segments: string[], into: unknown[]): void {
  if (segments.length === 0) {
    into.push(value);
    return;
  }
  const [head, ...rest] = segments as [string, ...string[]];
  if (head === "[]") for (const v of Array.isArray(value) ? value : []) walk(v, rest, into);
  else if (head === "{}") for (const v of isMap(value) ? Object.values(value) : []) walk(v, rest, into);
  else if (isMap(value) && head.slice(1) in value) walk(value[head.slice(1)], rest, into);
}
