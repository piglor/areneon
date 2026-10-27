import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { exportEngineData } from "../src/engine-data.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import {
  exportMoveClasses,
  MOVE_CLASS_NAMES,
  MOVE_CLASSES_FILE,
  MOVE_CLASSES_SUMMARY,
  MoveClassError,
  moveClassSummary,
  moveClassTableHash,
  readMoveClassTable,
} from "../src/move-classes.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const golden = readMoveClassTable(JSON.parse(readFileSync(join(GOLDEN_DIR, MOVE_CLASSES_FILE), "utf8")), table);
const bytes = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

describe("the golden move-class table (#1164)", () => {
  it("re-exports to the same bytes and the same summary at the pin", () => {
    const fresh = exportMoveClasses(engine, GOLDEN_FORMAT, table);
    expect(readFileSync(join(GOLDEN_DIR, MOVE_CLASSES_FILE), "utf8")).toBe(bytes(fresh));
    expect(readFileSync(join(GOLDEN_DIR, MOVE_CLASSES_SUMMARY), "utf8")).toBe(bytes(moveClassSummary(fresh)));
    expect(moveClassTableHash(fresh)).toBe(moveClassSummary(golden).move_class_table_hash);
  });

  // These pin what each rule derives from the engine's data, so a pin bump or a rule change shows up as a reviewed diff.
  it.each([
    ["protection", ["banefulbunker", "detect", "endure", "kingsshield", "protect", "spikyshield"], ["wideguard", "quickguard"]],
    ["side_condition", ["tailwind", "reflect", "lightscreen", "auroraveil", "stealthrock"], ["protect", "trickroom"]],
    ["field_condition", ["trickroom", "raindance", "electricterrain", "gravity", "snowscape"], ["tailwind", "protect"]],
    // a status move that lowers a foe's stage, a damaging move whose secondary always does, and a drop on the user is not one
    ["lowers_target_stat", ["charm", "screech", "icywind", "snarl", "spicyextract"], ["closecombat", "swordsdance", "psychic", "protect"]],
    // a self boost, an ally boost, a damaging move whose secondary always raises the user's stage, and the user's drop is not one
    ["raises_own_stat", ["swordsdance", "dragondance", "shellsmash", "coaching", "howl", "flamecharge", "scaleshot"], ["closecombat", "charm", "swagger", "protect"]],
    ["status", ["protect", "tailwind", "swordsdance", "charm"], ["icywind", "closecombat", "flamethrower"]],
  ] as const)("%s holds %j and not %j", (name, inside, outside) => {
    for (const id of inside) expect(golden.classes[name]).toContain(id);
    for (const id of outside) expect(golden.classes[name]).not.toContain(id);
  });

  it("names the engine-data table it was read against and nothing outside it", () => {
    expect(golden.engine_data_hash).toBe(JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.summary.json"), "utf8")).engine_data_hash);
    const ids = new Set(table.moves.map((m) => m.id));
    for (const name of MOVE_CLASS_NAMES) for (const id of golden.classes[name]) expect(ids.has(id)).toBe(true);
  });
});

describe("readMoveClassTable", () => {
  const edit = (change: (t: { classes: Record<string, string[]>; [k: string]: unknown }) => void) => {
    const copy = structuredClone(golden) as unknown as { classes: Record<string, string[]>; [k: string]: unknown };
    change(copy);
    return copy;
  };

  it("refuses a table of another engine-data table, format or pin", () => {
    expect(() => readMoveClassTable(edit((t) => (t.engine_data_hash = "f".repeat(64))), table)).toThrow(/another engine-data table/);
    expect(() => readMoveClassTable(edit((t) => (t.mod = "other")), table)).toThrow(/another format, mod or pin/);
  });
  it("refuses unsorted, repeated and foreign ids, and a status class that is not the Status moves", () => {
    expect(() => readMoveClassTable(edit((t) => (t.classes.protection = ["protect", "detect"])))).toThrow(MoveClassError);
    expect(() => readMoveClassTable(edit((t) => (t.classes.protection = ["protect", "protect"])))).toThrow(/repeated or out of order/);
    expect(() => readMoveClassTable(edit((t) => (t.classes.protection = ["protect", "zzzzzz"])), table)).toThrow(/not a move of the engine-data table/);
    expect(() => readMoveClassTable(edit((t) => (t.classes.status as string[]).pop()), table)).toThrow(/exactly the Status moves/);
  });
  it("refuses a value that is not the contract", () => {
    expect(() => readMoveClassTable({})).toThrow(/invalid move class table/);
  });
});

describe("exportMoveClasses", () => {
  it("refuses an unknown format and a target kind in no group", () => {
    expect(() => exportMoveClasses(engine, "gen9nonesuch")).toThrow();
    const odd = structuredClone(table);
    odd.move_targets = [...odd.move_targets, "zzzzz"];
    expect(() => exportMoveClasses(engine, GOLDEN_FORMAT, odd)).toThrow(/not in a group of target kinds/);
  });

  // The rules refuse any engine value of a shape they do not know: a doctored dex per case.
  const doctored = (id: string, change: Record<string, unknown>) => {
    const real = engine.sim as unknown as { Dex: { formats: { get(id: string): unknown }; mod(name: string): { moves: { get(id: string): unknown } } } };
    const stub = {
      ...engine,
      sim: {
        Dex: {
          formats: real.Dex.formats,
          mod: (name: string) => ({
            moves: {
              get: (moveId: string) => {
                const move = real.Dex.mod(name).moves.get(moveId) as Record<string, unknown>;
                return moveId === id ? { ...move, ...change } : move;
              },
            },
          }),
        },
      },
    };
    return () => exportMoveClasses(stub as never, GOLDEN_FORMAT, table);
  };
  it.each([
    ["a stage change that is not an integer", "charm", { boosts: { atk: 1.5 } }, /not a stage change/],
    ["a stage the engine has no name for", "charm", { boosts: { luck: -1 } }, /not a stage change/],
    ["a boosts table that is not an object", "charm", { boosts: 3 }, /not an object/],
    ["secondaries that are not an array", "icywind", { secondaries: {} }, /not an array/],
    ["a side condition that is not a string", "tailwind", { sideCondition: 3 }, /not a string/],
    ["a stallingMove that is not a boolean", "protect", { stallingMove: 1 }, /not a boolean/],
    ["a move the engine lacks", "protect", { exists: false }, /no such move/],
  ])("refuses %s", (_what, id, change, message) => {
    expect(doctored(id, change)).toThrow(MoveClassError);
    expect(doctored(id, change)).toThrow(message);
  });
  it("reads a self-ally boost and a certain secondary only at chance 100", () => {
    const certain = doctored("psychic", { secondaries: [{ chance: 100, boosts: { spd: -1 } }] });
    expect(certain()).toBeDefined();
    expect(certain().classes.lowers_target_stat).toContain("psychic");
    const chancy = doctored("icywind", { secondaries: [{ chance: 10, boosts: { spe: -1 } }] });
    expect(chancy().classes.lowers_target_stat).not.toContain("icywind");
  });
});
