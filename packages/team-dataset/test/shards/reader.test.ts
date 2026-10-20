import { jcs, recordHash } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { columnsFor, labelColumn, recordOf, ShardError } from "../../src/shards/columns.ts";
import { readShard, type ShardExpectation } from "../../src/shards/reader.ts";
import { expectationOf, fixtureRecords, labelSet, parquetOf, plan, shardOrder } from "./support.ts";

const staged = shardOrder(fixtureRecords("dev"));
const expected = expectationOf(staged, "dev");
const rows = () => staged.map((s) => ({ ...s.row }));
const good = parquetOf(rows());

/** The refusal message of reading `bytes` as the dev shard, or "read" if it was accepted. */
async function outcome(bytes: Uint8Array, expectation: ShardExpectation = expected): Promise<string> {
  try {
    await readShard(bytes, expectation, plan);
    return "read";
  } catch (error) {
    expect(error).toBeInstanceOf(ShardError);
    return (error as Error).message;
  }
}

describe("readShard", () => {
  it("rebuilds the records of a shard exactly, byte for byte in canonical form", async () => {
    const read = await readShard(good, expected, plan);
    expect(read.records.map((r) => jcs(r as never))).toEqual(staged.map((s) => jcs(s.record as never)));
    expect(read.recordHashes).toEqual(staged.map((s) => recordHash(s.record)));
    expect(read.strata).toEqual(staged.map((s) => s.stratum));
  });

  it("reads the same rows from files written with other settings", async () => {
    for (const options of [{ rowGroupSize: 7 }, { codec: "UNCOMPRESSED" as const }, { rowGroupSize: 1000 }]) {
      const bytes = parquetOf(rows(), columnsFor(labelSet), options);
      expect(bytes.byteLength).not.toBe(good.byteLength);
      const read = await readShard(bytes, expected, plan);
      expect(read.recordHashes).toEqual(staged.map((s) => recordHash(s.record)));
    }
  });

  it("refuses a null in a required column", async () => {
    for (const column of ["paste", "team_hash", "origin_index", "labels", "stratum", labelColumn("fake_out")]) {
      const bad = rows();
      (bad[3] as Record<string, unknown>)[column] = null;
      expect(await outcome(parquetOf(bad)), column).toMatch(/row 3: /);
    }
  });

  it("refuses a null inside a label struct", async () => {
    const bad = rows();
    (bad[0] as Record<string, unknown>).labels = [{ label_id: "speed_control", value: null }, ...((bad[0] as Record<string, unknown>).labels as unknown[]).slice(1)];
    expect(await outcome(parquetOf(bad))).toMatch(/row 0: labels\[0\]\.value: expected a string/);
  });

  it("refuses an integer outside +-(2^53-1)", async () => {
    for (const [value, fits] of [[2n ** 53n, false], [-(2n ** 53n), false], [2n ** 62n, false]] as const) {
      const bad = rows();
      (bad[5] as Record<string, unknown>).origin_draw_counter = value;
      expect(await outcome(parquetOf(bad)), String(value)).toMatch(/row 5: origin_draw_counter: .* outside \+-\(2\^53-1\)/);
      expect(fits).toBe(false);
    }
  });

  it("accepts the largest exact integer and refuses a negative index through the record schema", async () => {
    const edge = rows();
    (edge[5] as Record<string, unknown>).origin_draw_counter = 2 ** 53 - 1;
    const read = await readShard(parquetOf(edge), { ...expected, logical_hash: expectationOf(shardOrder(staged.map((s, k) => (k === 5 ? { ...s.record, origin: { ...s.record.origin, draw_counter: 2 ** 53 - 1 } } : s.record))), "dev").logical_hash }, plan);
    expect(read.records[5]?.origin.draw_counter).toBe(2 ** 53 - 1);
    const negative = rows();
    (negative[2] as Record<string, unknown>).origin_index = -1;
    expect(await outcome(parquetOf(negative))).toMatch(/row 2: not a TeamRecord/);
  });

  it("refuses rows out of order and a repeated row", async () => {
    const swapped = rows();
    [swapped[10], swapped[11]] = [swapped[11] as never, swapped[10] as never];
    expect(await outcome(parquetOf(swapped))).toMatch(/not in strictly ascending \(stratum, canonical hash\) order/);
    const twice = rows();
    twice[4] = { ...(twice[3] as Record<string, never>) };
    expect(await outcome(parquetOf(twice))).toMatch(/strictly ascending/);
    const reversed = rows().reverse();
    expect(await outcome(parquetOf(reversed))).toMatch(/strictly ascending/);
  });

  it("refuses a derived column that does not match the labels", async () => {
    const stratum = rows();
    (stratum[1] as Record<string, unknown>).stratum = ((stratum[1] as Record<string, unknown>).stratum as number) + 1;
    expect(await outcome(parquetOf(stratum))).toMatch(/row 1: stratum is .*, the labels give/);
    const label = rows();
    (label[1] as Record<string, unknown>)[labelColumn("fake_out")] = "yes";
    (label[2] as Record<string, unknown>)[labelColumn("fake_out")] = "no";
    const messages = [await outcome(parquetOf(label))];
    expect(messages[0]).toMatch(/column label:fake_out is not the label's value/);
  });

  it("refuses labels in another order, of another count, or with a value the label set does not have", async () => {
    const reordered = rows();
    const labels = [...((reordered[0] as Record<string, unknown>).labels as { label_id: string; value: string }[])];
    reordered[0] = { ...reordered[0], labels: [labels[1], labels[0], ...labels.slice(2)] } as never;
    expect(await outcome(parquetOf(reordered))).toMatch(/row 0: labels\[0\] is/);
    const short = rows();
    (short[0] as Record<string, unknown>).labels = labels.slice(1);
    expect(await outcome(parquetOf(short))).toMatch(/row 0: labels holds 4 entries, the label set has 5/);
    const unknown = rows();
    (unknown[0] as Record<string, unknown>).labels = labels.map((l, k) => (k === 0 ? { ...l, value: "sideways" } : l));
    expect(await outcome(parquetOf(unknown))).toMatch(/row 0: label speed_control: sideways is not one of its values/);
  });

  it("refuses a column that is missing, extra, or of the wrong type", async () => {
    expect(await outcome(parquetOf(rows(), undefined, { without: ["paste"] }))).toMatch(/lacks column paste/);
    expect(await outcome(parquetOf(rows(), undefined, { extra: [{ name: "note", data: rows().map(() => "x") }] }))).toMatch(/unexpected column note/);
    expect(await outcome(parquetOf(rows(), undefined, { extra: [{ name: "paste", data: rows().map(() => "x") }] }))).toMatch(/unexpected column \(repeated\)/);
    const wrongType = rows().map((r) => ({ ...r, origin_index: String(r.origin_index) }));
    expect(await outcome(parquetOf(wrongType, columnsFor(labelSet).map((c) => (c.name === "origin_index" ? { ...c, type: "VARCHAR" as const } : c))))).toMatch(/origin_index: expected an int64/);
  });

  it("refuses a file without the schema version, or with another", async () => {
    expect(await outcome(parquetOf(rows(), undefined, { noVersion: true }))).toMatch(/areneon_parquet_schema_version is missing/);
    expect(await outcome(parquetOf(rows(), undefined, { version: "2" }))).toMatch(/is 2, this reader reads 1/);
  });

  it("refuses what the manifest does not say: row count, split, logical hash, cell ranges", async () => {
    expect(await outcome(good, { ...expected, rows: expected.rows + 1 })).toMatch(/holds 120 rows, the manifest says 121/);
    expect(await outcome(good, { ...expected, split: "validation" })).toMatch(/row 0: its split is dev, the shard's is validation/);
    expect(await outcome(good, { ...expected, logical_hash: "0".repeat(64) })).toMatch(/logical hash/);
    const ranges = expected.cell_ranges.map((r) => ({ ...r }));
    (ranges[0] as { rows: number }).rows += 1;
    expect(await outcome(good, { ...expected, cell_ranges: ranges as ShardExpectation["cell_ranges"] })).toMatch(/cell ranges/);
    expect(await outcome(good, { ...expected, cell_ranges: ranges.slice(1) as ShardExpectation["cell_ranges"] })).toMatch(/cell ranges/);
  });

  it("refuses a row whose hashes are not hex digests (the record schema)", async () => {
    const bad = rows();
    (bad[0] as Record<string, unknown>).team_hash = "xyz";
    expect(await outcome(parquetOf(bad))).toMatch(/row 0: not a TeamRecord/);
  });

  it("refuses bytes that are not Parquet", async () => {
    expect(await outcome(new TextEncoder().encode("not parquet at all"))).toMatch(/not a readable Parquet file/);
    expect(await outcome(good.subarray(0, 40))).toMatch(/not a readable Parquet file/);
  });

  it("checks every row's shard with an assigner, and returns the one index", async () => {
    expect((await readShard(good, expected, plan, () => 3)).index).toBe(3);
    let n = 0;
    await expect(readShard(good, expected, plan, () => (n++ === 5 ? 1 : 0))).rejects.toThrow(/row 5: it belongs to shard 1, the shard's other rows to 0/);
  });

  it("takes an ArrayBuffer or a sliced view as well as a Uint8Array", async () => {
    const padded = new Uint8Array(good.byteLength + 10);
    padded.set(good, 10);
    expect((await readShard(padded.subarray(10), expected, plan)).records).toHaveLength(120);
    expect((await readShard(good.slice().buffer, expected, plan)).records).toHaveLength(120);
  });
});

describe("recordOf", () => {
  it("accepts a plain number for a safe integer and refuses a fractional or non-numeric one", () => {
    const row = { ...(staged[0] as { row: Record<string, unknown> }).row };
    expect(recordOf(row, plan, "r").record.origin.index).toBe((staged[0] as (typeof staged)[number]).record.origin.index);
    expect(() => recordOf({ ...row, origin_index: 1.5 }, plan, "r")).toThrow(/expected an int64/);
    expect(() => recordOf({ ...row, origin_index: undefined }, plan, "r")).toThrow(/got undefined/);
    expect(() => recordOf({ ...row, labels: "x" }, plan, "r")).toThrow(/r: labels: expected a list, got string/);
    expect(() => recordOf({ ...row, labels: null }, plan, "r")).toThrow(/got null/);
    expect(() => recordOf({ ...row, labels: [null, ...(row.labels as unknown[]).slice(1)] }, plan, "r")).toThrow(/labels\[0\] is not a struct/);
  });
});
