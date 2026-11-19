/* eslint-disable @typescript-eslint/no-explicit-any */
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { exportEngineDataV2 } from "../src/engine-data.ts";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { KO_HOOKS, koCandidates } from "../src/mechanics-ko-candidates.ts";
import {
  DAMAGE_HANDLER_NAMES,
  exportMechanicsInventory,
  INVENTORY_FILE,
  INVENTORY_SUMMARY,
  inventoryBytes,
  inventorySummary,
  type MechanicsInventory,
  MechanicsInventoryError,
  readMechanicsInventory,
  scanFiles,
  scanRefs,
} from "../src/mechanics-inventory.ts";
import { readMoveClassTable, TARGET_KINDS } from "../src/move-classes.ts";

// #1465 (ADR-030 groundwork G5): the inventory is read from the pinned engine by
// reflection and held byte for byte against the golden file.

let engine: Engine;
let v2: ReturnType<typeof exportEngineDataV2>;
const vocabulary = goldenTeamVocabulary();
const read = (name: string): string => readFileSync(join(GOLDEN_DIR, name), "utf8");
let golden: MechanicsInventory;
beforeAll(() => {
  engine = loadEngine();
  v2 = exportEngineDataV2(engine, GOLDEN_FORMAT);
  golden = readMechanicsInventory(JSON.parse(read(INVENTORY_FILE)), vocabulary, v2);
});
const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("the golden inventory", () => {
  it("re-exports to the same bytes and the same summary at the pin", () => {
    const fresh = exportMechanicsInventory(engine, GOLDEN_FORMAT, vocabulary, v2);
    expect(read(INVENTORY_FILE)).toBe(inventoryBytes(fresh));
    expect(read(INVENTORY_SUMMARY)).toBe(`${JSON.stringify(inventorySummary(fresh), null, 2)}\n`);
  });

  it("has exactly the roster's moves, abilities and items and the engine-data conditions", () => {
    expect(golden.moves.map((e) => e.id)).toEqual([...new Set(vocabulary.roster.flatMap((s) => s.moves))].sort());
    expect(golden.abilities.map((e) => e.id)).toEqual([...new Set(vocabulary.roster.flatMap((s) => s.abilities))].sort());
    expect(golden.items.map((e) => e.id)).toEqual([...vocabulary.items].sort());
    const conditions = new Set(golden.conditions.map((e) => e.id));
    for (const id of [...v2.conditions, ...v2.statuses, ...v2.weathers, ...v2.terrains].map((e) => e.id)) expect(conditions.has(id), id).toBe(true);
    expect(golden.species.map((s) => s.id)).toEqual(vocabulary.roster.map((s) => s.id));
  });

  it("reproduces ADR-030's count: the reflective count over the whole mod is the grep's, give or take the mod overrides", () => {
    const total = Object.values(golden.unfiltered).reduce((n, v) => n + v, 0);
    expect(total).toBeGreaterThanOrEqual(375);
    expect(total).toBeLessThanOrEqual(385);
    // the roster reaches fewer definitions than the mod holds
    const reached = [golden.moves, golden.abilities, golden.items, golden.conditions].flatMap((l) => l.filter((e) => e.defined_in === undefined || e.defined_in === "table")).reduce((n, e) => n + e.damage_handlers.length, 0);
    expect(reached).toBeGreaterThan(0);
    expect(reached).toBeLessThan(total);
  });

  it("states handlers, starts and simulation references the engine has", () => {
    const by = (list: MechanicsInventory["moves"], id: string) => list.find((e) => e.id === id) as MechanicsInventory["moves"][number];
    expect(by(golden.items, "choicescarf").handlers).toContain("onModifySpe");
    expect(by(golden.moves, "attract").orders).toContain("condition.onBeforeMovePriority=2");
    expect(by(golden.moves, "auroraveil").damage_handlers).toEqual(["condition.onAnyModifyDamage"]);
    expect(by(golden.moves, "facade").damage_handlers).toEqual(["onBasePower"]);
    expect(by(golden.moves, "tailwind").starts).toContain("sideCondition:tailwind");
    expect(by(golden.moves, "protect").starts).toContain("condition:protect");
    expect(by(golden.conditions, "tailwind").started_by).toContain("move:tailwind");
    expect(by(golden.conditions, "raindance").started_by).toContain("move:raindance");
    expect(golden.conditions.some((e) => e.defined_in === "table")).toBe(true);
    expect([...golden.moves, ...golden.abilities, ...golden.items, ...golden.conditions].some((e) => e.sim_refs.length > 0)).toBe(true);
    for (const name of DAMAGE_HANDLER_NAMES) {
      expect([...golden.moves, ...golden.abilities, ...golden.items, ...golden.conditions].some((e) => e.damage_handlers.some((h) => h.endsWith(name))), name).toBe(true);
    }
  });
});

describe("the inventory reproduces the narrower exports as filters", () => {
  it("koCandidates (#1166): the KO hooks the roster's abilities and items define at their top level", () => {
    const candidates = koCandidates(engine, GOLDEN_FORMAT, vocabulary);
    for (const [kind, list] of [["abilities", golden.abilities], ["items", golden.items]] as const) {
      const filtered: Record<string, string[]> = {};
      for (const e of list) {
        const hooks = KO_HOOKS.filter((hook) => e.handlers.includes(hook));
        if (hooks.length > 0) filtered[e.id] = hooks;
      }
      expect(filtered, kind).toEqual(candidates[kind]);
    }
  });

  it("the move-class table (#1164): every class its rules give", () => {
    const classes = readMoveClassTable(JSON.parse(read("move-classes.json"))).classes;
    const derived: Record<string, string[]> = { status: [], lowers_target_stat: [], raises_own_stat: [], side_condition: [], field_condition: [], protection: [] };
    for (const move of golden.moves) {
      const kind = TARGET_KINDS[move.target as string];
      const flags = new Set(move.data_flags);
      const boosts = (prefix: string, sign: "-" | "+"): boolean => move.data_flags.some((f) => f.startsWith(`${prefix}boosts.`) && f.includes("=") && (sign === "-" ? f.split("=")[1]?.startsWith("-") : !f.split("=")[1]?.startsWith("-")));
      if (move.category === "Status") derived.status?.push(move.id);
      if (move.starts.some((s) => /^sideCondition:/.test(s))) derived.side_condition?.push(move.id);
      if (move.starts.some((s) => /^(?:weather|terrain|pseudoWeather):/.test(s))) derived.field_condition?.push(move.id);
      if (flags.has("stallingMove=true")) derived.protection?.push(move.id);
      if (kind === "foe" && boosts("", "-")) derived.lowers_target_stat?.push(move.id);
      if ((kind === "friendly" && boosts("", "+")) || boosts("self.", "+") || boosts("selfBoost.", "+")) derived.raises_own_stat?.push(move.id);
    }
    // The class table covers every engine-data move, the inventory the roster's: compare on the roster's moves. The
    // chance-100 secondary rules need a secondary's chance, which the inventory keeps only as a key, so the two
    // boost classes are checked as a subset: everything the plain rules derive is in the class.
    const roster = new Set(golden.moves.map((m) => m.id));
    for (const name of ["status", "side_condition", "field_condition", "protection"] as const) expect(derived[name], name).toEqual(classes[name].filter((id) => roster.has(id)));
    for (const name of ["lowers_target_stat", "raises_own_stat"] as const) for (const id of derived[name] as string[]) expect(classes[name], `${name} ${id}`).toContain(id);
  });
});

describe("handler reflection and the id scan on a doctored copy", () => {
  it("lists an added handler name and an added hasAbility literal with no code change", () => {
    const real = (engine.sim as any).Dex;
    const doctoredDex = (name: string) => {
      const dex = real.mod(name);
      return new Proxy(dex, {
        get(target, prop) {
          if (prop === "moves") {
            return { get: (id: string) => (id === "protect" ? { ...target.moves.get(id), onTotallyNewHandler() {}, onTotallyNewHandlerPriority: 7 } : target.moves.get(id)), all: () => target.moves.all() };
          }
          return target[prop];
        },
      });
    };
    const fake = { commit: engine.commit, sim: { Dex: { formats: real.formats, mod: doctoredDex } } } as unknown as Engine;
    const dir = mkdtempSync(join(tmpdir(), "inventory-doctored-"));
    dirs.push(dir);
    mkdirSync(join(dir, "sim"));
    writeFileSync(join(dir, "sim", "doctored.ts"), "if (pokemon.hasAbility('intimidate') || move.id === 'protect') { this.addVolatile('flinch'); }\n");
    const inv = exportMechanicsInventory(fake, GOLDEN_FORMAT, vocabulary, v2, dir);
    const protect = inv.moves.find((e) => e.id === "protect") as MechanicsInventory["moves"][number];
    expect(protect.handlers).toContain("onTotallyNewHandler");
    expect(protect.orders).toContain("onTotallyNewHandlerPriority=7");
    expect(protect.sim_refs).toEqual(["sim/doctored.ts idIs"]);
    expect(inv.abilities.find((e) => e.id === "intimidate")?.sim_refs).toEqual(["sim/doctored.ts hasAbility"]);
    expect(inv.sources.map((s) => s.path)).toEqual(["sim/doctored.ts"]);
    // and the real checkout's inventory has neither
    expect(golden.moves.find((e) => e.id === "protect")?.handlers).not.toContain("onTotallyNewHandler");
  });

  it("scanFiles reads sim and the mod chain's scripts and conditions", () => {
    const files = scanFiles(engine.sim === undefined ? "" : join(GOLDEN_DIR, "..", "..", "..", "vendor", "pokemon-showdown"), (name) => (engine.sim as any).Dex.mod(name), "champions");
    expect(files).toContain("sim/battle.ts");
    expect(files).toContain("data/mods/champions/scripts.ts");
    expect(files).toContain("data/mods/champions/conditions.ts");
  });
});

describe("scanRefs", () => {
  it("finds each call, comparison and switch pattern", () => {
    const text = `
      pokemon.hasAbility('Intimidate'); pokemon.hasAbility(['a b', "c"]); this.hasItem('leftovers'); source.hasMove('protect');
      p.hasVolatile('taunt'); p.getVolatile("flinch"); this.field.isWeather('raindance'); this.field.isTerrain(['grassyterrain']);
      p.addVolatile('confusion'); p.setStatus('brn'); p.trySetStatus('par'); this.field.setWeather('sunnyday'); this.field.setTerrain('mistyterrain');
      side.addSideCondition('tailwind'); this.field.addPseudoWeather('trickroom'); side.addSlotCondition(p, 'wish');
      if (p.ability === 'sturdy' || p.baseAbility !== 'levitate') {} if (p.item == 'ironball' || p.lastItem === 'sitrusberry') {}
      if (p.status === 'tox' && p.volatiles['substitute']) {} if (effect.id === 'x' || move.id !== 'y') {}
      switch (effect.id) { case 'one': case "two": { if (true) { x(); } break; } default: break; }
      switch (kind) { case 'ignored': break; }
    `;
    const found = scanRefs(text).map((r) => `${r.via}:${r.kind}:${r.id}${r.start ? "!" : ""}`);
    for (const expected of [
      "hasAbility:ability:intimidate", "hasAbility:ability:ab", "hasAbility:ability:c", "hasItem:item:leftovers", "hasMove:move:protect", "hasVolatile:condition:taunt", "getVolatile:condition:flinch",
      "isWeather:condition:raindance", "isTerrain:condition:grassyterrain", "addVolatile:condition:confusion!", "setStatus:condition:brn!", "trySetStatus:condition:par!", "setWeather:condition:sunnyday!",
      "setTerrain:condition:mistyterrain!", "addSideCondition:condition:tailwind!", "addPseudoWeather:condition:trickroom!",
      "abilityIs:ability:sturdy", "abilityIs:ability:levitate", "itemIs:item:ironball", "itemIs:item:sitrusberry", "statusIs:condition:tox", "volatileKey:condition:substitute", "idIs:any:x", "idIs:any:y",
      "switchId:any:one", "switchId:any:two",
    ]) expect(found, expected).toContain(expected);
    expect(found).not.toContain("switchId:any:ignored");
    expect(scanRefs("nothing to see")).toEqual([]);
  });
});

describe("readMechanicsInventory", () => {
  const edit = (change: (copy: any) => void): unknown => {
    const copy = structuredClone(golden) as any;
    change(copy);
    return copy;
  };
  it("accepts the golden file with and without the roster and table", () => {
    expect(readMechanicsInventory(JSON.parse(read(INVENTORY_FILE))).moves.length).toBe(golden.moves.length);
  });
  it.each([
    ["another format", (c: any) => void (c.inventory_format = "x"), /not a mechanics inventory/],
    ["unsorted moves", (c: any) => void c.moves.reverse(), /out of order/],
    ["an unsorted entity list", (c: any) => void c.moves[0].handlers.push("b", "a"), /out of order/],
    ["a damage handler not among the handlers", (c: any) => void (c.moves.find((e: any) => e.damage_handlers.length > 0).damage_handlers = ["onModifyAtk"]), /is not a damage handler/],
    ["a condition with no starters list", (c: any) => void delete c.conditions[0].started_by, /no started_by/],
    ["a list that is not a list", (c: any) => void (c.items = 3), /not a list/],
    ["strings that are not strings", (c: any) => void (c.moves[0].handlers = [1]), /not a list of strings/],
  ])("refuses %s", (_name, change, message) => {
    expect(() => readMechanicsInventory(edit(change as (c: any) => void), vocabulary, v2)).toThrow(message);
  });
  it("refuses another vocabulary, a missing id and an extra one", () => {
    expect(() => readMechanicsInventory(edit((c: any) => void (c.team_vocabulary_hash = "0".repeat(64))), vocabulary, v2)).toThrow(/another vocabulary/);
    expect(() => readMechanicsInventory(edit((c: any) => void c.moves.pop()), vocabulary, v2)).toThrow(/is missing/);
    const extra = edit((c: any) => void c.moves.push({ ...c.moves[0], id: "zzzz" }));
    expect(() => readMechanicsInventory(extra, vocabulary, v2)).toThrow(/not exactly the vocabulary's/);
    expect(() => readMechanicsInventory(edit((c: any) => void c.conditions.shift()), vocabulary, v2)).toThrow(/is missing/);
  });
  it("is a MechanicsInventoryError", () => {
    expect(() => readMechanicsInventory(1)).toThrow(MechanicsInventoryError);
    expect(() => readMechanicsInventory(null)).toThrow(/not a mechanics inventory/);
  });
});

describe("the exporter's own refusals", () => {
  it("refuses an unknown format, a vocabulary or table of another pin or format", () => {
    expect(() => exportMechanicsInventory(engine, "gen9nonesuch", vocabulary, v2)).toThrow(/unknown format/);
    expect(() => exportMechanicsInventory(engine, GOLDEN_FORMAT, { ...vocabulary, showdown_commit: "0".repeat(40) }, v2)).toThrow(/another format or pin/);
    expect(() => exportMechanicsInventory(engine, GOLDEN_FORMAT, vocabulary, { ...v2, format_id: "x" })).toThrow(/another format or pin/);
    expect(() => exportMechanicsInventory(engine, GOLDEN_FORMAT, { ...vocabulary, roster: [{ id: "x", abilities: [], moves: ["nonesuchmove"] }] as any }, v2)).toThrow(/no such entry/);
  });
  it("loads the format's mechanics config as a sanity check of the fixtures", () => {
    expect(loadMechanicsConfig(GOLDEN_FORMAT).format_id).toBe(GOLDEN_FORMAT);
  });
});
