import { describe, expect, it } from "vitest";
import { encodeCurrent, readCurrentPointer } from "../src/current-pointer.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";

const A = "ab".repeat(32);
const B = "cd".repeat(32);
const enc = (text: string): Uint8Array => new TextEncoder().encode(text);

describe("current.json", () => {
  it("round-trips a pointer with and without a parent", () => {
    for (const parent of [null, B]) {
      const pointer = { schema_version: 1 as const, snapshot_hash: A, parent_snapshot_hash: parent };
      expect(readCurrentPointer(encodeCurrent(pointer))).toEqual(pointer);
    }
    expect(new TextDecoder().decode(encodeCurrent({ schema_version: 1, snapshot_hash: A, parent_snapshot_hash: null }))).toBe(`{"parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"${A}"}`);
  });

  it.each([
    ["not json", "{"],
    ["an array", "[]"],
    ["a string", '"x"'],
    ["a missing key", `{"schema_version":1,"snapshot_hash":"${A}"}`],
    ["an extra key", `{"extra":1,"parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"${A}"}`],
    ["another version", `{"parent_snapshot_hash":null,"schema_version":2,"snapshot_hash":"${A}"}`],
    ["a short hash", `{"parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"ab"}`],
    ["an upper-case hash", `{"parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"${A.toUpperCase()}"}`],
    ["a bad parent", `{"parent_snapshot_hash":"x","schema_version":1,"snapshot_hash":"${A}"}`],
    ["a numeric parent", `{"parent_snapshot_hash":1,"schema_version":1,"snapshot_hash":"${A}"}`],
    ["non-canonical bytes", `{ "parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"${A}"}`],
    ["a duplicate key", `{"parent_snapshot_hash":null,"schema_version":1,"snapshot_hash":"${B}","snapshot_hash":"${A}"}`],
  ])("refuses %s with the fixed code", (_name, text) => {
    expect(() => readCurrentPointer(enc(text))).toThrow(SnapshotIoError);
    try {
      readCurrentPointer(enc(text));
    } catch (error) {
      expect((error as SnapshotIoError).code).toBe("current_invalid");
    }
  });

  it("refuses bytes that are not UTF-8", () => {
    expect(() => readCurrentPointer(new Uint8Array([0xff, 0xfe]))).toThrow(SnapshotIoError);
  });
});
