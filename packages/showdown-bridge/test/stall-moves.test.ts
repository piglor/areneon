import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { exportStallMoves, readStallMoves, STALL_MOVES_DIR, StallMovesError, stallMovesFile, stallMovesHash } from "../src/stall-moves.ts";
import { type FactsTable, goldenWorldFactsData } from "../src/world-facts-data.ts";

const engine = loadEngine();
const path = join(STALL_MOVES_DIR, stallMovesFile(GOLDEN_FORMAT));
const committed = JSON.parse(readFileSync(path, "utf8"));
const table = JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as FactsTable;
const { stall_moves_hash: _hash, ...body } = committed;
const seal = (b: Record<string, unknown>) => ({ ...b, stall_moves_hash: stallMovesHash(b as never) });

describe("the stall-moves table of the pinned engine", () => {
  it("equals a fresh export byte for byte", () => {
    const fresh = exportStallMoves(engine, GOLDEN_FORMAT, table);
    expect(`${JSON.stringify(fresh, null, 2)}\n`).toBe(readFileSync(path, "utf8"));
    expect(readStallMoves(committed)).toEqual(fresh);
  });

  it("is the list the world facts take, and holds the Protect family", () => {
    expect(committed.stall_moves).toEqual(goldenWorldFactsData().stall_moves);
    expect(committed.stall_moves).toEqual(expect.arrayContaining(["protect", "detect", "endure"]));
  });

  it("refuses a table with no stalling move", () => {
    expect(() => exportStallMoves(engine, GOLDEN_FORMAT, { ...table, moves: [] })).toThrow(StallMovesError);
  });
});

describe("readStallMoves", () => {
  it("refuses what is not a sealed table", () => {
    expect(() => readStallMoves(null)).toThrow(/not an object/);
    expect(() => readStallMoves({ ...committed, extra: 1 })).toThrow(/key set/);
    expect(() => readStallMoves(seal({ ...body, schema_version: 2 }))).toThrow(/schema_version/);
    expect(() => readStallMoves({ ...committed, format_id: "" })).toThrow(/format_id/);
    expect(() => readStallMoves({ ...committed, stall_moves: "x" })).toThrow(/ids/);
    expect(() => readStallMoves({ ...committed, stall_moves: ["b", "a"] })).toThrow(/sorted/);
    expect(() => readStallMoves({ ...committed, stall_moves: ["protect", "Bad Id"] })).toThrow(/ids/);
    expect(() => readStallMoves({ ...committed, stall_moves: ["protect"] })).toThrow(/hash/);
  });
});
