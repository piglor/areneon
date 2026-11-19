/* eslint-disable @typescript-eslint/no-explicit-any */
// The battle-state tensor schema (ADR-030 step 5 groundwork): the committed record against the pinned engine and
// every committed corpus. Each battle is replayed with a spy on the engine's BattleStream (as the fork oracle and
// `world-writers-live.test.ts` do) and `Battle.toJSON()` is read at every request of both seats. The gate: every
// key of the engine's state is a field or an omitted key with a reason; every field's source resolves on real
// states; vocabulary values resolve in the engine-data table; the display rule is what the shared reader shows;
// and the projection rules let no hidden or unprojectable value reach another seat (a canary with a negative control).

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import { beforeAll, describe, expect, it } from "vitest";
import { type BattleStateBinding, type BattleStateSpec, BattleStateSpecError, BOOST_IDS, battleStateSpecHash, bindingFor, coveredKeys, ENGINE_LEVELS, type Field, GROUPS, loadBattleStateSpec, pathKeys, readBattleStateSpec, resolvePath, seatView, vocabularyIds } from "../src/battle-state-spec.ts";
import { type Engine, loadEngine, STAT_IDS } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { shownPercent } from "../src/hp-display.ts";
import { loadRandomCorpus, pendingFirstActor, RANDOM_MAX_TURNS, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { STATE_ITEM_IDS } from "../src/world-writers.ts";

// Replaying every committed corpus (about 250 battles, 8,500 requests) takes about 40 s on an idle machine.
const CORPUS_TIMEOUT_MS = 300_000;
const SPEC_PATH = new URL(`../../../config/engine-state/${GOLDEN_FORMAT}/battle-state-spec.v1.json`, import.meta.url);
const committedSpec = (): any => JSON.parse(readFileSync(SPEC_PATH, "utf8"));

let engine: Engine;
let spec: BattleStateSpec;
let binding: BattleStateBinding;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  binding = bindingFor(engine, GOLDEN_FORMAT);
  spec = loadBattleStateSpec(engine, GOLDEN_FORMAT);
});

type Player = "p1" | "p2";
interface Point {
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
          visit({ player, json: JSON.parse(JSON.stringify(captured.battle.toJSON())), channel: [...ctx.channel] });
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
  return (ref === null ? text : (ref[1] as string)).toLowerCase().replace(/[^a-z0-9]/g, "");
};

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

/** The ids a vocabulary-valued engine field holds in one state (for a map, its keys). */
function engineIds(field: Field, json: any): string[] {
  if (field.source.kind !== "engine") return [];
  const path = field.source.path;
  const read = field.source.read;
  const out: string[] = [];
  const sides = json.sides as any[];
  const mons = sides.flatMap((s) => s.pokemon);
  const actives = sides.flatMap((s) => s.pokemon.filter((m: any) => m.isActive === true));
  const pick = (values: unknown[]): void => {
    // A fainted Pokemon's engine status is the marker fnt: roster.fainted holds it, so the status reads as none (the field's description).
    for (const v of values) if (v !== null && v !== undefined && v !== "" && !(field.id === "roster.status" && v === "fnt")) out.push(engineId(v));
  };
  const keys = path.split(".").slice(1);
  const at = (value: any): unknown => keys.reduce((v: any, k) => (isMap(v) ? v[k] : undefined), value);
  if (read === "id" && path.startsWith("pokemon.moveSlots") ) pick(mons.flatMap((m) => m.moveSlots.map((s: any) => s.id)));
  else if (read === "id" && path.startsWith("pokemon.baseMoveSlots")) pick(mons.flatMap((m) => (m.baseMoveSlots ?? []).map((s: any) => s.id)));
  else if (read === "id" && path.startsWith("pokemon.")) pick(mons.map(at));
  else if (read === "id" && path.startsWith("active.")) pick(actives.map(at));
  else if (read === "id" && path.startsWith("battle.")) pick([at(json)]);
  else if (read === "id" && path.startsWith("field.")) pick([at(json.field)]);
  return out;
}

describe("the committed battle-state spec", () => {
  it("reads against the pinned engine, and its hash is pinned (a change is a new record)", () => {
    expect(spec.format_id).toBe(GOLDEN_FORMAT);
    expect(battleStateSpecHash(spec)).toBe("22735303afd5a636ff68ad5941749da2a7dbf938c991a24329603d77e4720d41");
    expect(spec.groups.map((g) => g.id)).toEqual([...GROUPS]);
    expect(spec.world_items.map((w) => w.id)).toEqual(STATE_ITEM_IDS);
  });

  it("binds the vocabulary sizes to the engine-data table: a doctored table or count is refused", () => {
    const table = { ...binding.engine_data, species: [] };
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, engine_data: table })).toThrow("engine_data_hash is not the table's");
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, counts: { ...binding.counts, max_move_count: 6 } })).toThrow("MOVE has size 4, the max_move_count is 6");
    expect(() => readBattleStateSpec(committedSpec(), { ...binding, vocabulary_hash: "0".repeat(64) })).toThrow("vocabulary_hash");
    // Sizes are the table's, never the record's: the union of the volatile lists is large and fits int16.
    expect(vocabularyIds(binding.engine_data, ["abilities", "conditions", "items", "moves"]).length).toBeGreaterThan(1000);
    expect(() => vocabularyIds(binding.engine_data, ["nosuchlist"])).toThrow("no list nosuchlist");
  });

  it("refuses a vocabulary field whose indices the table makes too many for its dtype, and an empty vocabulary dimension", () => {
    const doctored = committedSpec();
    const species = doctored.fields.find((f: any) => f.id === "roster.base_species");
    species.dtype = "int8";
    expect(() => readBattleStateSpec(doctored, binding)).toThrow("roster.base_species: 401 indices do not fit int8");
    const empty = readBattleStateSpec(committedSpec());
    expect(() => readBattleStateSpec(empty, { ...binding, engine_data: { ...binding.engine_data, types: [] } })).toThrow();
  });

  it("refuses a record of another format", () => {
    expect(() => loadBattleStateSpec(engine, "gen9ou", SPEC_PATH)).toThrow(BattleStateSpecError);
    // The open-sheet variant has the same counts and table, so only the record's own format refuses it.
    expect(() => loadBattleStateSpec(engine, `${GOLDEN_FORMAT}bo3`, SPEC_PATH)).toThrow(`the record is for ${GOLDEN_FORMAT}, not ${GOLDEN_FORMAT}bo3`);
  });
});

const SAMPLES = new URL("../../../fixtures/contracts/v1/battle-state-spec/", import.meta.url);
const sample = (kind: string, name: string): unknown => JSON.parse(readFileSync(new URL(`${kind}/${name}.json`, SAMPLES), "utf8"));

describe("the reader", () => {
  it.each([
    ["encoder-reads-hidden", "battle.prng is hidden"],
    ["encoder-unknown-field", "battle.turns is not a field"],
    ["group-mismatch", "battle.turn: the id does not start with its group"],
    ["groups-out-of-order", "groups: every group exactly once"],
    ["omitted-but-read", "battle.turn is read by a field"],
    ["opponent-reads-unprojectable", "roster.stats is not projectable to another seat"],
    ["projection-on-public", "a projection exactly for an own-exact field"],
    ["range-overflows-dtype", "the range does not fit int8"],
    ["range-reversed", "min exceeds max"],
    ["reveal-missing", "a reveal mask exactly for a revealed-only field"],
    ["reveal-wrong-mask", "its mask must be a 0/1 seat field"],
    ["seat-engine-source", "a seat field is a derivation"],
    ["seat-not-public", "its class is public"],
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
    lists.fields.find((f: any) => f.id === "battle.last_move").value.lists = ["moves", "abilities"];
    expect(() => readBattleStateSpec(lists)).toThrow("battle.last_move: lists");
    const dims = committedSpec();
    dims.dims.find((d: any) => d.symbol === "VOLATILE").lists.reverse();
    expect(() => readBattleStateSpec(dims)).toThrow("dims: VOLATILE");
    const mask = committedSpec();
    mask.fields.find((f: any) => f.id === "roster.item").reveal = "seat_roster.nothing";
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
  it("gives each class its view, for the owner and for another seat", () => {
    expect([seatView(field("public"), true), seatView(field("public"), false)]).toEqual(["exact", "exact"]);
    expect([seatView(field("hidden"), true), seatView(field("hidden"), false)]).toEqual(["none", "none"]);
    expect([seatView(field("revealed-only"), true), seatView(field("revealed-only"), false)]).toEqual(["masked", "masked"]);
    expect([seatView(field("own-exact", "hp_percent"), true), seatView(field("own-exact", "hp_percent"), false)]).toEqual(["exact", "percent"]);
    expect([seatView(field("own-exact", "none"), true), seatView(field("own-exact", "none"), false)]).toEqual(["exact", "none"]);
  });

  it("no encoder or reducer mapping reads a hidden field, and no opponent mapping reads a field another seat cannot see", () => {
    const fields = new Map(spec.fields.map((f) => [f.id, f]));
    for (const row of [...spec.encoder, ...spec.reducer]) {
      for (const ref of [...row.own, ...row.opponent, ...row.shared]) expect((fields.get(ref) as Field).visibility, ref).not.toBe("hidden");
      for (const ref of [...row.opponent, ...row.shared]) expect(seatView(fields.get(ref) as Field, false), ref).not.toBe("none");
    }
    // Negative controls: the reader refuses an opponent column that reads the exact stats, and any mapping of the PRNG.
    const leak = committedSpec();
    leak.encoder.find((c: any) => c.column === "stat_*").opponent = ["roster.stats"];
    expect(() => readBattleStateSpec(leak)).toThrow("roster.stats is not projectable to another seat");
    const rng = committedSpec();
    rng.reducer.find((r: any) => r.key === "turn").shared = ["battle.prng", "battle.turn"];
    expect(() => readBattleStateSpec(rng)).toThrow("battle.prng is hidden");
  });
});

describe("the spec against every committed corpus (the drift gate)", () => {
  const keys = new Set<string>();
  const resolved = new Map<string, { parents: number; found: number }>();
  const vocab = new Map<string, Set<string>>();
  const unknownIds: string[] = [];
  const hp = { compared: 0, own: 0, mismatches: [] as string[] };
  const canary = { states: 0, leaks: [] as string[], control: 0 };
  let requests = 0;

  beforeAll(async () => {
    for (const f of spec.fields) if (f.value.kind === "vocabulary") vocab.set(f.id, new Set(vocabularyIds(binding.engine_data, f.value.lists)));
    const dimIds = new Map(spec.dims.filter((d) => d.kind === "vocabulary").map((d) => [d.symbol, new Set(vocabularyIds(binding.engine_data, (d as { lists: string[] }).lists))]));
    requests = await points(({ player, json, channel }) => {
      stateKeys(json, keys);
      for (const f of spec.fields) {
        if (f.source.kind !== "engine") continue;
        const r = resolvePath(json, f.source.path);
        const sum = resolved.get(f.id) ?? { parents: 0, found: 0 };
        resolved.set(f.id, { parents: sum.parents + r.parents, found: sum.found + r.found });
        const known = vocab.get(f.id);
        if (known !== undefined) for (const id of engineIds(f, json)) if (!known.has(id)) unknownIds.push(`${f.id}=${id}`);
      }
      // map-keyed vocabulary dimensions: volatiles, side, field and slot conditions
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
      expect(String(json.prng).split(",")[1]?.length).toBe((binding.counts.prng_words as number) * 8);

      // The display rule: what the seat's shared reader shows of another seat's hit points is the projection of the exact value.
      const run = interpretChannel(player, channel);
      const own = json.sides.find((s: any) => s.id === player);
      const other = json.sides.find((s: any) => s.id !== player);
      if (run.flags.length === 0 && run.frames.length > 0) {
        const state = run.frames.at(-1)?.state;
        const byName = (side: any, name: string | null): any[] => side.pokemon.filter((m: any) => (m.set.name || m.set.species) === name && m.illusion == null);
        for (const mon of state?.opponent.roster ?? []) {
          const real = byName(other, mon.name);
          // In play only: a benched Pokemon's shown value is the last display, and the engine changes some silently (seat_roster.hp_shown).
          if (real.length !== 1 || real[0].isActive !== true || mon.hp_num === null || mon.hp_den !== 100 || other.pokemon.some((m: any) => m.illusion != null)) continue;
          hp.compared++;
          const shown = shownPercent(real[0].hp, real[0].maxhp, "floor");
          if (shown !== mon.hp_num) hp.mismatches.push(`${mon.name}: shown ${mon.hp_num}, rule ${shown}`);
        }
        for (const mon of state?.own.roster ?? []) {
          const real = byName(own, mon.name);
          if (real.length !== 1 || mon.hp_num === null) continue;
          hp.own++;
          if (mon.hp_num !== real[0].hp) hp.mismatches.push(`own ${mon.name}: shown ${mon.hp_num}, exact ${real[0].hp}`);
        }
      }

      // Canary: plant markers in every value another seat may not see, then project the state for this seat.
      const planted = structuredClone(json);
      planted.prng = "sodium,7777000077770000777700007777000077770000777700007777000077770000";
      planted.quickClawRoll = 77771;
      const foe = planted.sides.find((s: any) => s.id !== player);
      for (const mon of foe.pokemon) {
        mon.maxhp = Math.max(mon.maxhp, mon.hp) + 77772;
        mon.storedStats = { atk: 77773, def: 77773, spa: 77773, spd: 77773, spe: 77773 };
        for (const slot of mon.moveSlots) Object.assign(slot, { pp: 77774, maxpp: 77774 });
        for (const v of Object.values(mon.volatiles) as any[]) Object.assign(v, { duration: 77775, counter: 77775, hp: 77775 });
        if (isMap(mon.statusState)) Object.assign(mon.statusState, { time: 77776, startTime: 77776 });
        mon.position = 77777;
      }
      for (const v of Object.values(foe.sideConditions) as any[]) v.duration = 77778;
      if (isMap(planted.field.weatherState)) planted.field.weatherState.duration = 77779;
      const projected = JSON.stringify(project(planted, player, seatView));
      canary.states++;
      const found = projected.match(/7777\d/g);
      if (found !== null) canary.leaks.push([...new Set(found)].join());
      if (/7777\d/.test(JSON.stringify(project(planted, player, () => "exact")))) canary.control++;
    });
  }, CORPUS_TIMEOUT_MS);

  it("read every request of every corpus", () => {
    expect(requests).toBeGreaterThan(8000);
  });

  it("every key of the engine's state at a classified level is a field's or an omitted key with a reason", () => {
    const covered = coveredKeys(spec);
    const omitted = new Set(spec.omitted.map((o) => `${o.level}.${o.key}`));
    const unknown = [...keys].filter((k) => !covered.has(k) && !omitted.has(k)).sort();
    expect(unknown, "add a field or an omitted row with a reason").toEqual([]);
    // No stale row: every omitted key occurs on some request (a pin bump shows what it removes as well as what it adds).
    expect([...omitted].filter((k) => !keys.has(k)).sort()).toEqual([]);
    for (const level of new Set([...keys].map((k) => k.split(".")[0]))) expect(ENGINE_LEVELS).toContain(level);
  });

  it("every field's engine source resolves on real states (everywhere, when it is one key below the battle, a side or the field)", () => {
    for (const f of spec.fields) {
      if (f.source.kind !== "engine") continue;
      const r = resolved.get(f.id) as { parents: number; found: number };
      expect(r.found, `${f.id} never resolves`).toBeGreaterThan(0);
      // One key below the battle, a side or the field: the engine serialises it on every state (a Pokemon's keys such
      // as lastMoveTargetLoc are set when first used, and a deeper path may cross a null).
      const plain = /^(battle|side|field)\.[A-Za-z]+$/.test(f.source.path);
      if (plain) expect(r.found, `${f.id} is missing on some state`).toBe(r.parents);
    }
  });

  it("every vocabulary value on real states is an id of the field's engine-data lists", () => {
    expect(unknownIds.slice(0, 20)).toEqual([]);
  });

  it("the display projection is what the shared reader shows: floor of the share, exact for the owner", () => {
    expect(hp.mismatches.slice(0, 10)).toEqual([]);
    expect(hp.compared).toBeGreaterThan(5000);
    expect(hp.own).toBeGreaterThan(5000);
  });

  it("projects no hidden or unprojectable value to another seat (canary), and the scan has teeth (negative control)", () => {
    expect(canary.states).toBe(requests);
    expect(canary.leaks.slice(0, 10)).toEqual([]);
    expect(canary.control).toBe(canary.states);
  });
});

/**
 * A reference projection of the engine-sourced fields of the side-bound groups for one seat: what the rules let the
 * seat read (a masked value stays unknown: this test has no reveal masks, so it is never read). The test's view, so
 * the canary checks the rules, not an implementation.
 */
function project(json: any, seat: Player, view: (field: Field, owner: boolean) => string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const field of spec.fields) {
    if (field.source.kind !== "engine") continue;
    const path = field.source.path;
    const how = field.source.read;
    const read = (part: any): unknown[] => {
      const r: unknown[] = [];
      walk(part, path.match(/\.[A-Za-z]+|\[\]|\{\}/g) as string[], r);
      // The value the field holds, not the engine object it is read from (a map entry is only "present").
      return r.map((v) => (how === "present" ? 1 : how === "bool" ? (v ? 1 : 0) : how === "length" ? (v as unknown[]).length : v));
    };
    if (field.group === "battle" || field.group === "field") {
      if (view(field, false) === "exact") out[field.id] = read(field.group === "battle" ? json : json.field);
      continue;
    }
    for (const side of json.sides) {
      const owner = side.id === seat;
      const shown = view(field, owner);
      if (shown === "none" || shown === "masked") continue;
      const parts = path.startsWith("side") ? [side] : path.startsWith("active") ? side.pokemon.filter((m: any) => m.isActive === true) : side.pokemon;
      out[`${field.id}:${side.id}`] = parts.map((part: any) => {
        if (shown === "percent") return shownPercent(Math.min(part.hp, part.maxhp), part.maxhp, "floor");
        return read(part);
      });
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
