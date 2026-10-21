// Reading one private Parquet shard back into PrivateTeamRecords (ADR-015
// sections 2 and 3). The decoder is hyparquet, run in a worker thread (`private-decode.ts`); the
// checks are those of `reader.ts` with the private columns: the footer, the
// rebuild rules of `private-shards.ts`, the row order, the HMAC placement, the
// recomputed `logical_hash` and the cell ranges with their active rows.
//
// It does not call the engine and holds no key: whether each team is legal and
// each hold-out flag is right is the snapshot reader's work. Nothing in a
// message comes from the file (no column name, value or library text).

import { recordHash, shardLogicalHash, type PrivateTeamRecord, type PrivateSnapshotManifest } from "@areneon/showdown-bridge";
import { refuse, type RowPlan } from "./columns.ts";
import { decodeIsolated } from "./private-decode.ts";
import { byPrivateOrder, privateCellRanges, privateColumnsFor, privateRecordOf } from "./private-shards.ts";

/** What the manifest says a private shard holds. */
export type PrivateShardExpectation = Pick<PrivateSnapshotManifest["shards"][number], "rows" | "logical_hash" | "cell_ranges">;

export interface PrivateShardRead {
  /** The records in file order. */
  records: PrivateTeamRecord[];
  /** `record_hash` of each record, in file order. */
  recordHashes: string[];
  /** The stratum of each record, in file order. */
  strata: number[];
  /** The one shard index every row maps to, when an assigner was given. */
  index?: number;
}

/**
 * Decode, rebuild and verify one private shard. Every refusal is a
 * `ShardError` with a fixed message. `assign`, when given, maps a canonical
 * hash to its shard index; every row must give the same index, which is
 * returned.
 */
export async function readPrivateShard(
  file: ArrayBuffer | Uint8Array,
  expected: PrivateShardExpectation,
  plan: RowPlan,
  assign?: (canonicalTeamHash: string) => number,
  timeoutMs?: number,
): Promise<PrivateShardRead> {
  const bytes = file instanceof Uint8Array ? file : new Uint8Array(file);
  const rows = await decodeIsolated(bytes, { columns: privateColumnsFor(plan.labelSet), rows: expected.rows }, timeoutMs);
  const records: PrivateTeamRecord[] = [];
  const recordHashes: string[] = [];
  const strata: number[] = [];
  let index: number | undefined;
  rows.forEach((row, n) => {
      const where = `row ${n}`;
      const { record, stratum } = privateRecordOf(row, plan, where);
      const previous = records[n - 1];
      if (previous !== undefined) {
        const before = { stratum: strata[n - 1] as number, canonical_team_hash: previous.canonical_team_hash, team_id: previous.team_id };
        if (byPrivateOrder(before, { stratum, canonical_team_hash: record.canonical_team_hash, team_id: record.team_id }) >= 0) {
          refuse(`${where}: rows are not in strictly ascending (stratum, canonical hash, team_id) order`);
        }
      }
      if (assign !== undefined) {
        const at = assign(record.canonical_team_hash);
        if (index !== undefined && at !== index) refuse(`${where}: it belongs to another shard than the shard's other rows`);
        index = at;
      }
      records.push(record);
      recordHashes.push(recordHash(record));
      strata.push(stratum);
  });
  if (records.length !== expected.rows) refuse("the row groups do not hold the manifest's row count");
  if (shardLogicalHash(recordHashes) !== expected.logical_hash) refuse("the shard's logical hash is not the manifest's");
  const ranges = privateCellRanges(records.map((r, k) => ({ stratum: strata[k] as number, status: r.status })));
  const same =
    ranges.length === expected.cell_ranges.length &&
    ranges.every((r, k) => {
      const given = expected.cell_ranges[k] as PrivateShardExpectation["cell_ranges"][number];
      return r.stratum === given.stratum && r.first_row === given.first_row && r.rows === given.rows && r.active_rows === given.active_rows;
    });
  if (!same) refuse("the shard's cell ranges are not the manifest's");
  return { records, recordHashes, strata, ...(index === undefined ? {} : { index }) };
}
