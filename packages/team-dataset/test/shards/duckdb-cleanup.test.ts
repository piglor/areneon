import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";

// A scratch directory that cannot be removed (the removal is made to fail for
// it): the private path says so in a fixed message, unless the tool failed
// first, and then the tool's failure stands.

const stuck = vi.hoisted(() => ({ on: false }));
vi.mock("node:fs", async (original) => {
  const real = await original<typeof import("node:fs")>();
  return {
    ...real,
    rmSync: (path: string, options?: Parameters<typeof real.rmSync>[1]) => {
      if (stuck.on && String(path).includes("duckdb-home-")) throw new Error(`EBUSY: ${String(path)} CanaryPath7`);
      return real.rmSync(path, options);
    },
  };
});

const { columnsFor } = await import("../../src/shards/columns.ts");
const { CLEANUP_MESSAGE, DuckdbWriter, WITHHELD_MESSAGE } = await import("../../src/shards/duckdb.ts");
const { labelSet } = await import("./support.ts");

const scratch = mkdtempSync(join(tmpdir(), "duckdb-cleanup-"));
afterAll(() => {
  stuck.on = false;
  rmSync(scratch, { recursive: true, force: true });
});
const binary = join(scratch, "duckdb");
writeFileSync(binary, "stand-in\n");

describe("a scratch directory that will not go", () => {
  it("fails a successful private write with the fixed message, and leaves a tool failure as it was", () => {
    const columns = columnsFor(labelSet);
    const rows = join(scratch, "r.jsonl");
    writeFileSync(rows, "");
    stuck.on = true;
    const ok = DuckdbWriter.open(binary, (...args) => (args[1][0] === "--version" ? { status: 0, stdout: "v1.5.6 (x)\n", stderr: "" } : { status: 0, stdout: "", stderr: "" }));
    expect(() => ok.writeShard(rows, join(scratch, "o.parquet"), columns, { isolated: { dir: scratch } })).toThrow(CLEANUP_MESSAGE);
    expect(() => ok.writeShard(rows, join(scratch, "o.parquet"), columns, { isolated: { dir: scratch } })).not.toThrow(/CanaryPath7/);
    const failing = DuckdbWriter.open(binary, (...args) => (args[1][0] === "--version" ? { status: 0, stdout: "v1.5.6 (x)\n", stderr: "" } : { status: 1, stdout: "", stderr: "boom" }));
    expect(() => failing.writeShard(rows, join(scratch, "o.parquet"), columns, { isolated: { dir: scratch } })).toThrow(WITHHELD_MESSAGE);
  });
});
