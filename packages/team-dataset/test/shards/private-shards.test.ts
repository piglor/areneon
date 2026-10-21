import { describe, expect, it } from "vitest";
import { labelColumn, PARQUET_SCHEMA_VERSION, ShardError } from "../../src/shards/columns.ts";
import { byPrivateOrder, layoutPrivate, privateCellRanges, privateColumnsFor, privateRecordOf, privateRowOf } from "../../src/shards/private-shards.ts";
import { readPrivateShard, type PrivateShardExpectation } from "../../src/shards/private-reader.ts";
import { shardAssigner } from "../../src/shards/layout.ts";
import { recordHash, shardLogicalHash, type PrivateTeamRecord } from "@areneon/showdown-bridge";
import { at, fixtureRecords, labelSet, parquetOf, plan, SHARD_KEY } from "./support.ts";

// Private records built without the engine: the team fields are the fixture's (the shard layer checks shape and
// identity of the file, not the paste; the engine round trip is the snapshot reader's).
const hex32 = (n: number): string => n.toString(16).padStart(32, "0");
const privateRecords = (n: number): PrivateTeamRecord[] =>
  fixtureRecords("dev").slice(0, n).map((t, k) => ({
    schema_version: 1,
    data_class: "private",
    team_id: hex32(k + 1),
    revision: 1,
    supersedes_record_hash: null,
    team_hash: t.team_hash,
    canonical_team_hash: t.canonical_team_hash,
    paste: t.paste,
    status: k % 4 === 3 ? "retired" : "active",
    tags: k % 2 === 0 ? ["alpha", "beta"] : [],
    holdout: k % 3 === 0,
    labels: t.labels.map((l) => ({ label_id: l.label_id, value: l.value })),
  })) as PrivateTeamRecord[];

const staged = (n: number) =>
  privateRecords(n)
    .map((record) => ({ record, stratum: plan.stratumOf(record.labels), canonical_team_hash: record.canonical_team_hash, team_id: record.team_id }))
    .sort(byPrivateOrder);
const rowsOf = (n = 20): Record<string, unknown>[] => staged(n).map((s) => privateRowOf(s.record, s.stratum));
const columns = privateColumnsFor(labelSet);
const expectationOf = (rows: ReturnType<typeof staged>): PrivateShardExpectation => ({
  rows: rows.length,
  logical_hash: shardLogicalHash(rows.map((s) => recordHash(s.record))),
  cell_ranges: privateCellRanges(rows.map((s) => ({ stratum: s.stratum, status: s.record.status }))),
});
const outcome = async (file: Uint8Array, expected = expectationOf(staged(20))): Promise<string> => {
  try {
    await readPrivateShard(file, expected, plan);
    return "read";
  } catch (error) {
    expect(error).toBeInstanceOf(ShardError);
    return (error as Error).message;
  }
};

describe("the private column layout", () => {
  it("has the record's columns, the derived stratum and one label column per label", () => {
    const names = columns.map((c) => c.name);
    expect(names.slice(0, 13)).toEqual(["schema_version", "data_class", "team_id", "revision", "supersedes_record_hash", "team_hash", "canonical_team_hash", "paste", "status", "tags", "holdout", "labels", "stratum"]);
    expect(names.slice(13)).toEqual(labelSet.labels.map((l) => labelColumn(l.label_id)));
    expect(PARQUET_SCHEMA_VERSION).toBe(1);
  });
});

describe("readPrivateShard", () => {
  it("rebuilds the records of a shard it was given, with their hashes, strata and placement", async () => {
    const rows = staged(20);
    const read = await readPrivateShard(parquetOf(rowsOf(20), columns), expectationOf(rows), plan, shardAssigner(SHARD_KEY, 1));
    expect(read.records).toEqual(rows.map((s) => s.record));
    expect(read.recordHashes).toEqual(rows.map((s) => recordHash(s.record)));
    expect(read.strata).toEqual(rows.map((s) => s.stratum));
    expect(read.index).toBe(0);
    expect(expectationOf(rows).cell_ranges.some((r) => r.active_rows < r.rows)).toBe(true);
  });

  it("reads a record with a null predecessor and one with a hash, and empty and full tag lists", async () => {
    const rows = staged(4);
    const withPredecessor = rows.map((s, k) => (k === 0 ? { ...s, record: { ...s.record, revision: 2, supersedes_record_hash: "ab".repeat(32) } as PrivateTeamRecord } : s));
    const file = parquetOf(withPredecessor.map((s) => privateRowOf(s.record, s.stratum)), columns);
    expect((await readPrivateShard(file, expectationOf(withPredecessor), plan)).records.map((r) => r.supersedes_record_hash)).toEqual(withPredecessor.map((s) => s.record.supersedes_record_hash));
  });

  it("refuses by fixed message: a file that is not Parquet, wrong columns, wrong types, a wrong version", async () => {
    expect(await outcome(new TextEncoder().encode("PAR1 nonsense"))).toBe("not a readable Parquet file");
    expect(await outcome(parquetOf(rowsOf(), columns, { noVersion: true }))).toMatch(/schema version is missing/);
    expect(await outcome(parquetOf(rowsOf(), columns, { version: "2" }))).toMatch(/schema version/);
    expect(await outcome(parquetOf(rowsOf(), columns, { without: ["paste"] }))).toBe("the file's columns are not the private layout's");
    expect(await outcome(parquetOf(rowsOf(), columns, { extra: [{ name: "extra", data: rowsOf().map(() => "x") }] }))).toBe("the file's columns are not the private layout's");
    expect(await outcome(parquetOf(rowsOf(), columns, { indexAsDouble: true }))).toBe(await outcome(parquetOf(rowsOf(), columns, { indexAsDouble: true })));
    const asText = columns.map((c) => (c.name === "holdout" ? { ...c, type: "VARCHAR" as const } : c));
    expect(await outcome(parquetOf(rowsOf().map((r) => ({ ...r, holdout: "true" })), asText))).toBe("column holdout does not have the layout's type");
    const tagsText = columns.map((c) => (c.name === "tags" ? { ...c, type: "VARCHAR" as const } : c));
    expect(await outcome(parquetOf(rowsOf().map((r) => ({ ...r, tags: "alpha" })), tagsText))).toBe("column tags does not have the layout's type");
    expect(await outcome(parquetOf(rowsOf(), columns, { labelExtra: true }))).toBe("column labels does not have the layout's type");
    expect(await outcome(parquetOf(rowsOf(5), columns), expectationOf(staged(20)))).toMatch(/row count/);
  });

  it("refuses by fixed message: null or mistyped values, out-of-layout records, wrong strata and labels", async () => {
    const change = async (edit: (rows: Record<string, unknown>[]) => void): Promise<string> => {
      const rows = rowsOf();
      edit(rows);
      return outcome(parquetOf(rows, columns));
    };
    expect(await change((r) => (at(r, 2).paste = null))).toBe("row 2: paste: not a string");
    expect(await change((r) => (at(r, 2).holdout = null))).toBe("row 2: holdout: not a boolean");
    expect(await change((r) => (at(r, 2).supersedes_record_hash = "short"))).toBe("row 2: not a PrivateTeamRecord: /supersedes_record_hash pattern");
    expect(await change((r) => (at(r, 2).tags = [null]))).toBe("row 2: tags[0]: not a string");
    expect(await change((r) => (at(r, 2).team_id = "NOT-HEX"))).toMatch(/^row 2: not a PrivateTeamRecord: \/team_id pattern$/);
    expect(await change((r) => (at(r, 2).stratum = 9999))).toBe("row 2: stratum: not the one the labels give");
    expect(await change((r) => (at(r, 2).revision = 2 ** 60))).toBe("row 2: revision: outside +-(2^53-1)");
    expect(await change((r) => (at(r, 2).labels = []))).toBe("row 2: labels: not one entry per label of the label set");
    expect(await change((r) => (at(r, 2).labels = null))).toBe("row 2: labels: not a list");
    const ids = (r: Record<string, unknown>[], f: (l: { label_id: string; value: string }[]) => unknown) => ((at(r, 2).labels as never) = f(at(r, 2).labels as never) as never);
    expect(await change((r) => ids(r, (l) => [l[1], l[0], ...l.slice(2)]))).toBe("row 2: labels[0]: not the label set's label");
    expect(await change((r) => ids(r, (l) => [{ ...at(l, 0), value: "zzz" }, ...l.slice(1)]))).toBe("row 2: labels[0]: not one of the label's values");
    expect(await change((r) => (at(r, 2)[labelColumn(at(labelSet.labels, 0).label_id)] = "zzz"))).toBe("row 2: labels[0]: its derived column differs");
  });

  it("refuses, for values a Parquet file cannot hold, when the rebuild is given them directly", () => {
    const direct = (edit: (row: Record<string, unknown>) => void): string => {
      const row = { ...at(rowsOf(3), 1) };
      edit(row);
      try {
        privateRecordOf(row, plan, "row 1");
        return "read";
      } catch (error) {
        return (error as Error).message;
      }
    };
    const labels = (row: Record<string, unknown>) => row.labels as { label_id: unknown; value: unknown }[];
    expect(direct(() => undefined)).toBe("read");
    expect(direct((r) => (r.labels = labels(r).map(() => 7)))).toBe("row 1: labels[0]: not a struct");
    expect(direct((r) => (r.labels = labels(r).map((l) => ({ ...l, extra: 1 }))))).toBe("row 1: labels[0]: not exactly label_id and value");
    expect(direct((r) => (r.labels = [{ label_id: 5, value: "x" }, ...labels(r).slice(1)]))).toBe("row 1: labels[0].label_id: not a string");
    expect(direct((r) => (r.supersedes_record_hash = 7))).toBe("row 1: supersedes_record_hash: neither a string nor null");
    expect(direct((r) => (r.tags = "alpha"))).toBe("row 1: tags: not a list");
    expect(direct((r) => (r.revision = 1.5))).toBe("row 1: revision: not an int64 within +-(2^53-1)");
    expect(direct((r) => (r.revision = 2n ** 70n))).toBe("row 1: revision: outside +-(2^53-1)");
    expect(direct((r) => (r.revision = 4n))).toBe("read");
    expect(direct((r) => (r.schema_version = 2.5))).toBe("row 1: schema_version: not an int64 within +-(2^53-1)");
  });

  it("refuses rows out of order or repeated, a stray shard placement, and a shard that is not the manifest's", async () => {
    const swapped = rowsOf();
    [swapped[10], swapped[11]] = [at(swapped, 11), at(swapped, 10)];
    expect(await outcome(parquetOf(swapped, columns))).toMatch(/row 11: rows are not in strictly ascending/);
    const twice = rowsOf();
    twice[4] = { ...at(twice, 3) };
    expect(await outcome(parquetOf(twice, columns))).toMatch(/strictly ascending/);
    await expect(readPrivateShard(parquetOf(rowsOf(), columns), expectationOf(staged(20)), plan, (hash) => hash.charCodeAt(0))).rejects.toThrow(/another shard/);
    const wrongHash = { ...expectationOf(staged(20)), logical_hash: "00".repeat(32) };
    expect(await outcome(parquetOf(rowsOf(), columns), wrongHash)).toBe("the shard's logical hash is not the manifest's");
    const wrongRanges = expectationOf(staged(20));
    wrongRanges.cell_ranges = wrongRanges.cell_ranges.map((r, k) => (k === 0 ? { ...r, active_rows: r.active_rows + 1 } : r)) as never;
    expect(await outcome(parquetOf(rowsOf(), columns), wrongRanges)).toBe("the shard's cell ranges are not the manifest's");
    expect(await outcome(parquetOf(rowsOf(), columns, { rowGroupSize: 3 }))).toBe("read");
  });
});

describe("the private layout", () => {
  it("places rows by the HMAC of the canonical hash and sorts each shard, a retired record sharing its team's hash included", () => {
    const rows = staged(20);
    const same = { ...at(rows, 0), team_id: hex32(99), record: { ...at(rows, 0).record, team_id: hex32(99), status: "retired" as const } };
    const shards = layoutPrivate([...rows, same], SHARD_KEY, 8);
    expect(shards).toHaveLength(3);
    expect(shards.flat()).toHaveLength(21);
    for (const shard of shards) expect([...shard].sort(byPrivateOrder)).toEqual(shard);
    expect(() => layoutPrivate([same], SHARD_KEY, 8)).not.toThrow();
    expect(() => layoutPrivate(rows, SHARD_KEY, 1 << 30).flat()).not.toThrow();
  });

  it("refuses a shard that would be empty", () => {
    const rows = staged(2);
    expect(() => layoutPrivate(rows, SHARD_KEY, 1)).toThrow(/received no row/);
  });

  it("counts active rows per cell range", () => {
    expect(privateCellRanges([{ stratum: 1, status: "active" }, { stratum: 1, status: "retired" }, { stratum: 4, status: "retired" }])).toEqual([
      { stratum: 1, first_row: 0, rows: 2, active_rows: 1 },
      { stratum: 4, first_row: 2, rows: 1, active_rows: 0 },
    ]);
  });

  it("drops a library error's text from a row refusal", () => {
    const row = at(rowsOf(1), 0);
    Object.defineProperty(row, "labels", { get: () => { throw new TypeError("SECRET-TEXT"); } });
    expect(() => privateRecordOf(row, plan, "row 0")).toThrow("row 0: the row cannot be read");
  });
});
