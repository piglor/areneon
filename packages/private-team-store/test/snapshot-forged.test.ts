import { createHash } from "node:crypto";
import { type PrivateSnapshotManifest, type PrivateTeamRecord, privateLogicalSnapshotHash, privateSnapshotHash, recordHash, shardLogicalHash } from "@areneon/showdown-bridge";
import { byPrivateOrder, privateCellRanges, privateColumnsFor, privateRowOf, shardAssigner } from "@areneon/team-dataset";
import { describe, expect, it, vi } from "vitest";
import { parquetOf } from "../../team-dataset/test/shards/support.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { INSPECT_ONLY, readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { at, engine, input, labelSet, plan, recordOf, records, SHARD_KEY } from "./snapshot-support.ts";

// Each case decodes shard files in worker threads; a loaded CI machine needs more than the default 5 s.
vi.setConfig({ testTimeout: 60_000 });

// Snapshots the writer refuses to make but a hostile store could serve: the
// reader must refuse them on its own. `forge` builds a manifest and files for
// any grouping of records into shards, with every hash and count consistent.

interface Forged {
  manifest: PrivateSnapshotManifest;
  files: Uint8Array[];
}

function forge(base: PrivateSnapshotManifest, groups: PrivateTeamRecord[][], change: Record<string, unknown> = {}): Forged {
  const built = groups.map((group) => {
    const rows = group
      .map((record) => ({ record, stratum: plan.stratumOf(record.labels), canonical_team_hash: record.canonical_team_hash, team_id: record.team_id }))
      .sort(byPrivateOrder);
    const file = parquetOf(rows.map((r) => privateRowOf(r.record, r.stratum)), privateColumnsFor(labelSet));
    return {
      file,
      entry: {
        index: 0,
        file_sha256: createHash("sha256").update(file).digest("hex"),
        bytes: file.byteLength,
        logical_hash: shardLogicalHash(rows.map((r) => recordHash(r.record))),
        rows: rows.length,
        split: "owner" as const,
        cell_ranges: privateCellRanges(rows.map((r) => ({ stratum: r.stratum, status: r.record.status }))),
      },
    };
  });
  built.sort((a, b) => (a.entry.logical_hash < b.entry.logical_hash ? -1 : 1));
  const shards = built.map((b, index) => ({ ...b.entry, index }));
  const active = new Map<number, number>();
  for (const shard of shards) for (const range of shard.cell_ranges) if (range.active_rows > 0) active.set(range.stratum, (active.get(range.stratum) ?? 0) + range.active_rows);
  const all = groups.flat();
  const activeCount = all.filter((r) => r.status === "active").length;
  const next = {
    ...base,
    shards,
    cells: [...active.entries()].sort((a, b) => a[0] - b[0]).map(([stratum, rows]) => ({ split: "owner", stratum, rows })),
    records: { active: activeCount, retired: all.length - activeCount },
    ...change,
  } as Record<string, unknown>;
  next.logical_snapshot_hash = privateLogicalSnapshotHash(next as never);
  return { manifest: next as unknown as PrivateSnapshotManifest, files: built.map((b) => b.file) };
}

const refusal = async (forged: Forged, shardRows?: number): Promise<SnapshotIoError> => {
  const error = await readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: forged.manifest, expectedSnapshotHash: privateSnapshotHash(forged.manifest as never), shard: (i) => at(forged.files, i), labelSet, ...(shardRows === undefined ? {} : { shardRows }) }).catch((e: unknown) => e);
  expect(error).toBeInstanceOf(SnapshotIoError);
  return error as SnapshotIoError;
};

describe("readPrivateSnapshot on a forged snapshot", () => {
  it("accepts what forge makes when it is consistent (so the refusals below are the forgeries' own)", async () => {
    const w = await writePrivateSnapshot(input({ records: records(8), shardRows: 4 }));
    const all = records(8);
    const assign = shardAssigner(SHARD_KEY, 2);
    const groups = [all.filter((r) => assign(r.canonical_team_hash) === 0), all.filter((r) => assign(r.canonical_team_hash) === 1)];
    const forged = forge(w.manifest, groups);
    const read = await readPrivateSnapshot({ engine, expected: INSPECT_ONLY, manifest: forged.manifest, expectedSnapshotHash: privateSnapshotHash(forged.manifest as never), shard: (i) => at(forged.files, i), labelSet, shardRows: 4 });
    expect(read.records).toHaveLength(8);
  });

  it("refuses one team in two active records", async () => {
    const w = await writePrivateSnapshot(input({ records: [recordOf(0, 1)] }));
    const forged = forge(w.manifest, [[recordOf(0, 1), recordOf(0, 2)]]);
    expect((await refusal(forged)).code).toBe("record_invalid");
  });

  it("refuses two shards that hold rows of the same layout shard", async () => {
    const w = await writePrivateSnapshot(input({ records: records(8), shardRows: 4 }));
    const assign = shardAssigner(SHARD_KEY, 2);
    const zero = records(20).filter((r) => assign(r.canonical_team_hash) === 0);
    expect(zero.length).toBeGreaterThanOrEqual(2);
    const half = Math.ceil(zero.length / 2);
    const forged = forge(w.manifest, [zero.slice(0, half), zero.slice(half)]);
    const error = await refusal(forged, half);
    expect(error.code).toBe("shard_layout_invalid");
  });

  it("refuses a Parquet layout version this reader does not read", async () => {
    const w = await writePrivateSnapshot(input({ records: records(4) }));
    const forged = forge(w.manifest, [records(4)], { parquet_schema_version: 2 });
    expect((await refusal(forged)).code).toBe("version_mismatch");
  });
});
