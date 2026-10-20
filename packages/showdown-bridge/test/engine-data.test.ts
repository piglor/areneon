import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import {
  ENGINE_DATA_FILE,
  ENGINE_DATA_SUMMARY,
  type EngineData,
  EngineDataError,
  engineDataHash,
  engineDataSummary,
  exportEngineData,
  readEngineData,
} from "../src/engine-data.ts";
import { type Engine, EngineError, loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";

const engine = loadEngine();
const golden = (file: string): unknown => JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8"));
const fresh = exportEngineData(engine, GOLDEN_FORMAT);

type Raw = Record<string, unknown>;
type Lists = Record<"species" | "moves" | "items" | "abilities" | "types", Raw[]>;

/** The real engine with its entry lists copied and then changed, to reach the fail-closed paths. */
function patched(change: (lists: Lists, conditions: string[]) => void): Engine {
  const dex = engine.sim.Dex as unknown as {
    formats: unknown;
    mod(name: string): Record<string, { all(): object[] }> & { conditions: unknown; data: { Conditions: Raw } };
  };
  return {
    commit: engine.commit,
    sim: {
      Dex: {
        formats: dex.formats,
        mod(name: string) {
          const real = dex.mod(name);
          const lists = Object.fromEntries(
            (["species", "moves", "items", "abilities", "types"] as const).map((k) => [k, (real[k] as { all(): object[] }).all().map((e) => JSON.parse(JSON.stringify(e)) as Raw)]),
          ) as Lists;
          const conditions = Object.keys(real.data.Conditions);
          change(lists, conditions);
          return {
            ...Object.fromEntries(Object.entries(lists).map(([k, v]) => [k, { all: () => v }])),
            conditions: real.conditions,
            data: { Conditions: Object.fromEntries(conditions.map((c) => [c, {}])) },
          };
        },
      },
    },
  } as unknown as Engine;
}
const byName = (list: Raw[], name: string): Raw => list.find((e) => e.name === name) as Raw;

describe("the golden engine-data export (ADR-010 section 3)", () => {
  it("is reproducible from the pin: a fresh export equals the committed table and summary, byte for byte", () => {
    const bytes = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;
    expect(readFileSync(join(GOLDEN_DIR, ENGINE_DATA_FILE), "utf8")).toBe(bytes(fresh));
    expect(readFileSync(join(GOLDEN_DIR, ENGINE_DATA_SUMMARY), "utf8")).toBe(bytes(engineDataSummary(fresh)));
    expect(readEngineData(golden(ENGINE_DATA_FILE))).toEqual(fresh);
  });

  it("is hashed as the SHA-256 of its JCS bytes, and any change of the table or the pin changes the hash", () => {
    const hash = engineDataHash(fresh);
    expect(hash).toBe(createHash("sha256").update(jcs(fresh as unknown as Json), "utf8").digest("hex"));
    const moved = structuredClone(fresh);
    moved.showdown_commit = "0".repeat(40);
    expect(engineDataHash(moved)).not.toBe(hash);
    const edited = structuredClone(fresh);
    (edited.species[0] as { base_stats: { hp: number } }).base_stats.hp += 1;
    expect(engineDataHash(edited)).not.toBe(hash);
  });

  it("carries the engine's own values: spot-checked against the engine's data, not constants", () => {
    const dex = (engine.sim.Dex as unknown as { mod(name: string): { species: { get(n: string): Raw }; moves: { get(n: string): Raw } } }).mod(fresh.mod);
    for (const s of fresh.species) {
      const raw = dex.species.get(s.name);
      expect(s.base_stats, s.id).toEqual(raw.baseStats);
      expect(s.num, s.id).toBe(raw.num);
    }
    for (const m of fresh.moves) {
      const raw = dex.moves.get(m.name);
      expect([m.id, m.base_power, m.priority, m.pp]).toEqual([raw.id, raw.basePower, raw.priority, raw.pp]);
      expect(m.always_hits).toBe(raw.accuracy === true);
    }
    // The type chart is the engine's: no hand-written effectiveness.
    expect(fresh.types.every((t) => t.damage_taken.length === fresh.types.length)).toBe(true);
    expect(fresh.statuses.map((s) => s.id)).toEqual(["brn", "frz", "par", "psn", "slp", "tox"]);
    expect(fresh.weathers.length).toBeGreaterThan(0);
    expect(fresh.terrains.length).toBeGreaterThan(0);
  });

  it("admits exactly these nonstandard entries, so a pin bump that changes them is visible", () => {
    const dex = (engine.sim.Dex as unknown as { mod(name: string): Record<"species" | "abilities", { get(n: string): Raw }> }).mod(fresh.mod);
    const nonstandard = (kind: "species" | "abilities", entries: { name: string }[]) =>
      entries.filter((e) => dex[kind].get(e.name).isNonstandard !== null).map((e) => e.name);
    expect(nonstandard("species", fresh.species)).toEqual(["Cramorant", "Eiscue", "Floette", "Meloetta", "Minior", "Ogerpon", "Terapagos"]);
    expect(nonstandard("abilities", fresh.abilities)).toEqual(["Aura Guard"]);
  });

  it("is the table of the format's mod", () => {
    expect([fresh.format_id, fresh.mod, fresh.showdown_commit]).toEqual([GOLDEN_FORMAT, "champions", engine.commit]);
  });
});

describe("exportEngineData fails closed", () => {
  it("refuses an unknown format", () => {
    expect(() => exportEngineData(engine, "gen9nosuchformat")).toThrow(EngineError);
  });

  it("drops an entry the engine says does not exist", () => {
    const table = exportEngineData(patched((l) => (byName(l.items, "Leftovers").exists = false)), GOLDEN_FORMAT);
    expect(table.items).toHaveLength(fresh.items.length - 1);
  });

  it.each([
    ["a species type outside the type vocabulary", (l: Lists) => ((byName(l.species, "Garchomp").types as string[])[0] = "Nope"), /type: Nope is not in the vocabulary/],
    ["a species type that is not text", (l: Lists) => ((byName(l.species, "Garchomp").types as unknown[])[0] = 3), /type: not a non-empty string/],
    ["species types that are not a list", (l: Lists) => (byName(l.species, "Garchomp").types = "Dragon"), /species types: not an array/],
    ["a species ability slot it does not know", (l: Lists) => ((byName(l.species, "Garchomp").abilities as Raw).X = "Sand Veil"), /unknown slot X/],
    ["species abilities that are not an object", (l: Lists) => (byName(l.species, "Garchomp").abilities = null), /species abilities: not an object/],
    ["a base species outside the vocabulary", (l: Lists) => (byName(l.species, "Garchomp").baseSpecies = "Nope"), /species: Nope is not in the vocabulary/],
    ["a fractional base stat", (l: Lists) => ((byName(l.species, "Garchomp").baseStats as Raw).hp = 1.5), /hp: not an integer/],
    ["a move accuracy that is neither a number nor true", (l: Lists) => (byName(l.moves, "Earthquake").accuracy = false), /move accuracy: not an integer/],
    ["a move without flags", (l: Lists) => (byName(l.moves, "Earthquake").flags = undefined), /move flags: not an object/],
    ["a move category that is not text", (l: Lists) => (byName(l.moves, "Earthquake").category = 7), /move category: not a non-empty string/],
    ["a move category the contract does not list", (l: Lists) => (byName(l.moves, "Earthquake").category = "Typeless"), /invalid engine data/],
    ["a fling that is not an object", (l: Lists) => (byName(l.items, "Leftovers").fling = 3), /item fling: not an object/],
    ["a type whose damageTaken lacks an attacker", (l: Lists) => delete (byName(l.types, "Fire").damageTaken as Raw).Water, /Fire damageTaken Water: not an integer/],
    ["an item number that is not a number", (l: Lists) => (byName(l.items, "Leftovers").num = "1"), /item num: not an integer/],
  ] as [string, (lists: Lists) => void, RegExp][])("refuses %s", (_label, change, message) => {
    const engineWith = patched((lists) => change(lists));
    expect(() => exportEngineData(engineWith, GOLDEN_FORMAT)).toThrow(message);
    expect(() => exportEngineData(engineWith, GOLDEN_FORMAT)).toThrow(EngineDataError);
  });

  it("refuses a move that sets a terrain the engine has no condition for", () => {
    const engineWith = patched((l) => (byName(l.moves, "Electric Terrain").terrain = "nosuchterrain"));
    expect(() => exportEngineData(engineWith, GOLDEN_FORMAT)).toThrow(/no such condition/);
  });
});

describe("readEngineData", () => {
  const mutated = (change: (t: EngineData) => void): EngineData => {
    const copy = structuredClone(fresh);
    change(copy);
    return copy;
  };

  it.each([
    ["not an object", () => null, /invalid engine data/],
    ["a new field", (t: EngineData) => Object.assign(t, { learnsets: [] }), /invalid engine data/],
    ["the next schema version", (t: EngineData) => Object.assign(t, { schema_version: 2 }), /invalid engine data/],
    ["species out of order", (t: EngineData) => t.species.reverse(), /species: .* repeated or out of order/],
    ["a repeated move", (t: EngineData) => t.moves.splice(1, 0, t.moves[0] as never), /moves: .* repeated or out of order/],
    ["unsorted move targets", (t: EngineData) => t.move_targets.reverse(), /move_targets: .* repeated or out of order/],
    ["unsorted move flags", (t: EngineData) => t.move_flags.reverse(), /move_flags: .* repeated or out of order/],
    ["a short damage row", (t: EngineData) => (t.types[0] as { damage_taken: number[] }).damage_taken.pop(), /damage_taken is not one entry per type/],
    ["a damage code outside 0..3", (t: EngineData) => ((t.types[0] as { damage_taken: number[] }).damage_taken[0] = 4), /invalid engine data/],
    ["a species type outside the vocabulary", (t: EngineData) => ((t.species[0] as { types: string[] }).types = ["nosuchtype"]), /unknown type/],
    ["a species ability outside the vocabulary", (t: EngineData) => ((t.species[0] as { abilities: string[] }).abilities = ["nosuchability"]), /unknown ability/],
    ["a base species outside the vocabulary", (t: EngineData) => ((t.species[0] as { base_species: string }).base_species = "nosuchspecies"), /unknown base species/],
    ["a move type outside the vocabulary", (t: EngineData) => ((t.moves[0] as { type: string }).type = "nosuchtype"), /unknown type/],
    ["a move target outside the vocabulary", (t: EngineData) => ((t.moves[0] as { target: string }).target = "nosuchtarget"), /unknown target/],
    ["unsorted move flags on a move", (t: EngineData) => ((t.moves[0] as { flags: string[] }).flags = ["protect", "contact"]), /flags: .* repeated or out of order/],
    ["a move flag outside the vocabulary", (t: EngineData) => ((t.moves[0] as { flags: string[] }).flags = ["zzz"]), /unknown flag/],
  ] as [string, (t: EngineData) => unknown, RegExp][])("refuses %s", (_label, change, message) => {
    const value = _label === "not an object" ? null : mutated(change as (t: EngineData) => void);
    expect(() => readEngineData(value)).toThrow(message);
    expect(() => readEngineData(value)).toThrow(EngineDataError);
  });
});
