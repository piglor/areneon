import { recordHash, teamVocabularyHash } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { PilotError, readPilot, runPilot } from "../src/pilot.ts";
import { rangeSummary } from "../src/range.ts";
import { Drawer } from "../src/draw.ts";
import { syntheticTeam } from "../../showdown-bridge/src/golden.ts";
import { drawerFor, engine, format, KEY, labelSet, small, vocabulary } from "./support.ts";

const cfg = small();
const identity = { config_hash: recordHash(cfg), label_set_hash: recordHash(labelSet), vocabulary_hash: teamVocabularyHash(vocabulary) };
const pilot = runPilot(drawerFor(cfg), identity, labelSet, 150);

describe("the recorded pilot (ADR-013 section 2)", () => {
  it("counts every pilot draw once: an accepted draw in its cell, a rejected one apart", () => {
    expect(pilot.draws).toBe(150);
    expect(pilot.cell_hits).toHaveLength(64);
    expect(pilot.cell_hits.reduce((a, b) => a + b, 0) + pilot.rejected).toBe(150);
    expect(pilot.rejected).toBeGreaterThan(0);
    expect(pilot).toMatchObject(identity);
  });

  it("is a function of the config and the key alone, not of any plan", () => {
    expect(runPilot(drawerFor(cfg), identity, labelSet, 150)).toEqual(pilot);
    expect(runPilot(drawerFor(cfg, "22".repeat(32)), identity, labelSet, 150)).not.toEqual(pilot);
  });

  it("is read back when it belongs to the config, label set and vocabulary", () => {
    expect(readPilot(JSON.parse(JSON.stringify(pilot)), identity, labelSet)).toEqual(pilot);
  });

  it.each([
    ["not an object", [], /not an object/],
    ["null", null, /not an object/],
    ["an extra field", { ...pilot, note: 1 }, /has the fields/],
    ["a missing field", { ...pilot, rejected: undefined }, /has the fields|rejected/],
    ["another format version", { ...pilot, format_version: 2 }, /format_version is not 1/],
    ["a malformed hash", { ...pilot, config_hash: "xyz" }, /config_hash is not a SHA-256/],
    ["another config", { ...pilot, config_hash: "0".repeat(64) }, /another config/],
    ["another label set", { ...pilot, label_set_hash: "0".repeat(64) }, /another label_set/],
    ["another vocabulary", { ...pilot, vocabulary_hash: "0".repeat(64) }, /another vocabulary/],
    ["zero draws", { ...pilot, draws: 0 }, /draws is not a positive integer/],
    ["a negative rejection count", { ...pilot, rejected: -1 }, /rejected is not a count/],
    ["hits of the wrong length", { ...pilot, cell_hits: [1, 2] }, /not one count per cell/],
    ["a fractional hit", { ...pilot, cell_hits: pilot.cell_hits.map((h, k) => (k === 0 ? 0.5 : h)) }, /not one count per cell/],
    ["counts that do not add up", { ...pilot, rejected: pilot.rejected + 1 }, /do not add up to draws/],
  ])("refuses a pilot with %s", (_name, value, message) => {
    expect(() => readPilot(value, identity, labelSet)).toThrow(PilotError);
    expect(() => readPilot(value, identity, labelSet)).toThrow(message);
  });
});

describe("a range of the stream, compared across platforms (ADR-013 section 2)", () => {
  const stream = { config_hash: identity.config_hash, plan_id: "range-test" };
  const drawer = drawerFor(cfg);

  it("summarises a range by the digest of its per-draw summaries, the same each time", () => {
    const first = rangeSummary(drawer, stream, 10, 40);
    expect(first.digest).toMatch(/^[0-9a-f]{64}$/);
    expect(first.accepted + first.rejected).toBe(40);
    expect(first.accepted).toBeGreaterThan(30);
    expect(rangeSummary(drawerFor(cfg), stream, 10, 40)).toEqual(first);
  });

  it("differs when the range, the plan or the key differs", () => {
    const base = rangeSummary(drawer, stream, 10, 40).digest;
    expect(rangeSummary(drawer, stream, 11, 40).digest).not.toBe(base);
    expect(rangeSummary(drawer, { ...stream, plan_id: "other" }, 10, 40).digest).not.toBe(base);
    expect(rangeSummary(drawerFor(cfg, "22".repeat(32)), stream, 10, 40).digest).not.toBe(base);
  });
});

describe("a range with a draw the paste round trip refuses", () => {
  it("counts it as rejected with the round-trip status", () => {
    const team = syntheticTeam();
    team[0] = { ...(team[0] as (typeof team)[number]), item: "Leftovers\nShiny: Yes" };
    const bad = new Drawer({ engine, format, proposal: { propose: () => team } as never, labelSet, generationKey: KEY });
    const good = new Drawer({ engine, format, proposal: { propose: () => syntheticTeam() } as never, labelSet, generationKey: KEY });
    const stream = { config_hash: identity.config_hash, plan_id: "range-test" };
    expect(rangeSummary(bad, stream, 0, 3)).toMatchObject({ accepted: 0, rejected: 3 });
    expect(rangeSummary(good, stream, 0, 3)).toMatchObject({ accepted: 3, rejected: 0 });
    expect(rangeSummary(bad, stream, 0, 3).digest).not.toBe(rangeSummary(good, stream, 0, 3).digest);
  });
});
