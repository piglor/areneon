/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import worldWritersSchema from "@areneon/schema/contracts/v1/world-writers.schema.json" with { type: "json" };
import { ADMITTED_KINDS, STATE_CLASSES, STATE_ITEM_IDS, STATE_ITEMS, WorldWritersError, loadWorldWriters, readWorldWriters, worldWritersHash } from "../src/index.ts";
import { loadEngine, type Engine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const committed = (file: string): any => JSON.parse(readFileSync(join(SAMPLES_DIR, file), "utf8"));
let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

describe("the closed lists", () => {
  it("are the schema's enums", () => {
    const defs = (worldWritersSchema as any).$defs;
    expect([...defs.Item.properties.id.enum]).toEqual([...STATE_ITEM_IDS]);
    expect(defs.StateClass.enum).toEqual([...STATE_CLASSES]);
    expect(defs.Admitted.properties.kind.enum).toEqual([...ADMITTED_KINDS]);
    // every item can be refused (R) or has the class its row is for; the generic id row cannot be anything else
    for (const id of STATE_ITEM_IDS) expect(STATE_ITEMS[id].classes.length).toBeGreaterThan(0);
  });
});

describe("the committed record (config/search/world-writers.v1.json)", () => {
  it("reads against the pinned engine, with every item classified once and no id admitted yet", () => {
    const record = loadWorldWriters(engine, GOLDEN_FORMAT);
    expect(record.items.map((item) => item.id)).toEqual([...STATE_ITEM_IDS]);
    // until the fidelity oracle admits ids (slice F) every generic id is R: nothing is listed
    expect(record.admitted).toEqual([]);
    const classes = Object.fromEntries(record.items.map((item) => [item.id, item.class]));
    expect(classes).toMatchObject({ toxic_stage: "W", bench_status: "W", sleep_counter: "X", confusion_duration: "X", per_turn_fields: "T", substitute: "R", unadmitted_id: "R" });
  });

  it("has a hash pinned from outside the code (Python's hashlib over the sorted-key bytes)", () => {
    expect(worldWritersHash(loadWorldWriters(engine, GOLDEN_FORMAT))).toBe("89b4bb2e79d0a395332bcc0c4d56cbc01d3d45bce6f4eef31e5dbda7151612b6");
  });

  it("is refused for another format", () => {
    expect(() => loadWorldWriters(engine, "gen9ou")).toThrow("not gen9ou");
  });
});

describe("readWorldWriters", () => {
  it("reads the admitted sample", () => {
    expect(readWorldWriters(committed("v1/world-writers/valid/admitted.json"), engine).admitted).toHaveLength(3);
  });

  it.each([
    ["missing-item", "every state item is classified exactly once"],
    ["unsorted-items", "out of order"],
    ["class-not-offered", "cannot have class X"],
    ["condition-on-wrong-item", "cannot have a condition"],
    ["unknown-condition", "notacondition is not an engine condition"],
    ["unknown-engine-id", "notanid is not an engine condition, move, ability or item"],
    ["unsorted-ids", "repeated or out of order"],
    ["id-under-two-rows", "is listed under"],
    ["bad-line-kind", "not a protocol line kind"],
    ["duplicate-admitted", "repeated or out of order"],
    ["unsorted-admitted", "repeated or out of order"],
    ["admitted-class", "cannot have class E"],
    ["admitted-unknown-id", "not in the engine's data"],
    ["admitted-refusal-id", "also a refusal row of slot_conditions"],
    ["admitted-confusion", "the confusion condition is the item confusion_duration's"],
  ])("refuses %s", (name, message) => {
    const value = committed(`v1/world-writers/reader-invalid/${name}.json`);
    expect(() => readWorldWriters(value, engine)).toThrow(WorldWritersError);
    expect(() => readWorldWriters(value, engine)).toThrow(message);
  });

  it("refuses a format the engine does not have", () => {
    const value = committed("v1/world-writers/valid/v1.json");
    value.format_id = "gen9notaformat";
    expect(() => readWorldWriters(value, engine)).toThrow("not an engine format");
  });

  it("refuses an item that needs ids without them, and a missing condition", () => {
    const noIds = committed("v1/world-writers/valid/v1.json");
    delete noIds.items.find((item: any) => item.id === "substitute").ids;
    expect(() => readWorldWriters(noIds, engine)).toThrow("substitute needs ids");
    const noCondition = committed("v1/world-writers/valid/v1.json");
    delete noCondition.items.find((item: any) => item.id === "choice_lock").condition;
    expect(() => readWorldWriters(noCondition, engine)).toThrow("choice_lock needs a condition");
  });

  it.each(["extra-field", "unknown-item", "unknown-class", "fractional-version", "missing-admitted", "unknown-admitted-kind", "next-version"])("the schema refuses %s", (name) => {
    expect(() => readWorldWriters(committed(`v1/world-writers/invalid/${name}.json`), engine)).toThrow("invalid WorldWriters");
  });
});
