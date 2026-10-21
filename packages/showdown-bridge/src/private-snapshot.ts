// The PrivateSnapshotManifest contract's reader and derivations (ADR-015
// section 2). The schema (`packages/schema/contracts/v1/private-snapshot-manifest.schema.json`)
// holds the shape; this module holds what the contract profile cannot say:
// exactly one of `parent_snapshot_hash` and `first`, the shard order and
// indices, the cell ranges tiling each shard, the sums and counts, and the
// recomputed logical hash. It also holds the hold-out derivation.
//
// Nothing here needs a secret. The hold-out key is in the manifest and is not
// one (it only makes the split stable and not hand-picked); `holdoutFlag`
// derives a record's flag from it, and the snapshot reader recomputes each
// flag with it. The manifest holds hashes, counts and id tokens only, so a
// failure message states a rule and never repeats a value or a key name.
//
// Identities (SHA-256 of RFC 8785 bytes, like the public snapshot's):
//   shard logical_hash   `shardLogicalHash` (team-records.ts), the same rule
//   logical_snapshot_hash  `privateLogicalSnapshotHash`
//   snapshot hash        `privateSnapshotHash`, the name of the snapshot in the bucket

import { createHash, createHmac } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import privateSnapshotManifestSchema from "@areneon/schema/contracts/v1/private-snapshot-manifest.schema.json" with { type: "json" };
import type { PrivateSnapshotManifest } from "@areneon/schema/types/v1/private-snapshot-manifest.ts";
import { jcs, type Json } from "./canonical-json.ts";

export type { PrivateSnapshotManifest };

export class PrivateSnapshotError extends Error {
  override name = "PrivateSnapshotError";
}

const bad = (message: string): never => {
  throw new PrivateSnapshotError(message);
};

/** The derivation word the manifest names (`holdout.rule`). */
export const HOLDOUT_RULE = "hmac-sha256-team-id/1";
/** The held-out share of `HOLDOUT_RULE`, in parts per million. */
export const HOLDOUT_PPM = 200_000;

const HEX64 = /^[0-9a-f]{64}$/;
const HEX32 = /^[0-9a-f]{32}$/;
const TWO_POW_64 = 1n << 64n;

/**
 * Whether the team `teamId` is held out under the snapshot key `key`
 * (ADR-015 section 2, rule `hmac-sha256-team-id/1`). The HMAC key is the 32
 * bytes `key` spells in hex, the message is `teamId` as its 32 lower-case hex
 * characters (ASCII), the first 8 bytes of the tag are read big endian as an
 * unsigned integer n, and the team is held out when n / 2^64 < ppm / 10^6,
 * decided in integers as n * 10^6 < ppm * 2^64. It depends on the two ids
 * alone, so a team keeps its flag on every revision and in every snapshot that
 * carries the same key.
 */
export function holdoutFlag(key: string, teamId: string, ppm: number = HOLDOUT_PPM): boolean {
  if (!HEX64.test(key)) bad("the hold-out key is not 64 lower-case hex characters");
  if (!HEX32.test(teamId)) bad("team_id is not 32 lower-case hex characters");
  if (!Number.isInteger(ppm) || ppm < 0 || ppm > 1_000_000) bad("ppm is not an integer from 0 to 1000000");
  const n = createHmac("sha256", Buffer.from(key, "hex")).update(teamId, "ascii").digest().readBigUInt64BE(0);
  return n * 1_000_000n < BigInt(ppm) * TWO_POW_64;
}

const sha256 = (value: unknown): string => createHash("sha256").update(jcs(value as Json), "utf8").digest("hex");
const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * A private snapshot's writer-independent identity: SHA-256 of the JCS bytes
 * of `{shards: the sorted shard logical hashes, cells, label_set_hash,
 * manifest, holdout, and either parent_snapshot_hash or first: true}`.
 */
export function privateLogicalSnapshotHash(snapshot: PrivateSnapshotManifest): string {
  const shards = snapshot.shards.map((s) => s.logical_hash).sort(codeUnits);
  const lineage = snapshot.first === true ? { first: true } : { parent_snapshot_hash: snapshot.parent_snapshot_hash };
  return sha256({ shards, cells: snapshot.cells, label_set_hash: snapshot.label_set_hash, manifest: snapshot.manifest, holdout: snapshot.holdout, ...lineage });
}

/** One physical publication: SHA-256 of the whole manifest's JCS bytes. It names `snapshots/<hash>/`. */
export const privateSnapshotHash = (snapshot: PrivateSnapshotManifest): string => sha256(snapshot);

const validate = ajvFor("contract").compile(privateSnapshotManifestSchema);

/**
 * Parse a PrivateSnapshotManifest: the schema, then exactly one of
 * `parent_snapshot_hash` and `first`, the hold-out block, shards in fixed
 * order (strictly ascending logical hashes) with their own position as index,
 * each shard's cell ranges tiling it with strictly ascending strata and no
 * more active rows than rows, the cells equal to the shards' active sums and
 * listing exactly the cells that hold an active record, the record counts
 * equal to the shards' totals, and `logical_snapshot_hash` recomputed. It
 * needs no key and no engine. A message never repeats an input value.
 */
export function readPrivateSnapshotManifest(value: unknown): PrivateSnapshotManifest {
  if (!validate(value)) {
    // Only the path and keyword of each error: `params` can hold an offending key name.
    const errors = (validate.errors ?? []) as { instancePath: string; keyword: string }[];
    return bad(`invalid private snapshot manifest: ${errors.map((e) => `${e.instancePath || "/"} ${e.keyword}`).join("; ")}`);
  }
  const snapshot = value as unknown as PrivateSnapshotManifest;
  if ((snapshot.first === true) === (snapshot.parent_snapshot_hash !== undefined)) bad("exactly one of parent_snapshot_hash and first must be present");

  let rows = 0;
  const active = new Map<number, number>();
  let previous: string | undefined;
  snapshot.shards.forEach((shard, position) => {
    if (shard.index !== position) bad(`shard ${position}: index is not its position`);
    if (previous !== undefined && !(previous < shard.logical_hash)) bad("shards are repeated or out of logical-hash order");
    previous = shard.logical_hash;
    let next = 0;
    let last = -1;
    for (const range of shard.cell_ranges) {
      if (range.stratum <= last) bad(`shard ${position}: cell ranges are repeated or out of stratum order`);
      if (range.first_row !== next) bad(`shard ${position}: cell ranges do not tile the shard`);
      if (range.active_rows > range.rows) bad(`shard ${position}: a cell range has more active rows than rows`);
      next += range.rows;
      last = range.stratum;
      if (range.active_rows > 0) active.set(range.stratum, (active.get(range.stratum) ?? 0) + range.active_rows);
    }
    if (next !== shard.rows) bad(`shard ${position}: rows is not the sum of its cell ranges`);
    rows += shard.rows;
  });

  let last = -1;
  let activeTotal = 0;
  for (const cell of snapshot.cells) {
    if (cell.stratum <= last) bad("cells are repeated or out of stratum order");
    last = cell.stratum;
    if (active.get(cell.stratum) !== cell.rows) bad("a cell's rows is not the shards' active sum");
    activeTotal += cell.rows;
  }
  if (snapshot.cells.length !== active.size) bad("cells do not list exactly the cells that hold an active record");
  if (snapshot.records.active !== activeTotal) bad("records.active is not the sum of the cells");
  if (snapshot.records.active + snapshot.records.retired !== rows) bad("records.active plus records.retired is not the shards' rows");
  if (snapshot.logical_snapshot_hash !== privateLogicalSnapshotHash(snapshot)) bad("logical_snapshot_hash is not recomputed from the shards, cells, label set, manifest, hold-out block and parent");
  return snapshot;
}
