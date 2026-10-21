// The private team record's and the private snapshot manifest's share of the
// sample corpus (ADR-008 §2, ADR-015 sections 2 and 5). Synthetic only: the
// pastes are the committed synthetic team corpus (`fixtures/teams/`), and a
// manifest holds hashes and counts of nothing real (the same guard shape
// check applies to its marker files, `PRIVATE_MANIFEST_MARKERS`).
//
// A PrivateTeamRecord sample says `data_class: private`, so each such file is
// listed in `PRIVATE_SAMPLE_MARKERS` (scripts/check_data_class.py), which also
// requires its paste to be a corpus paste and its shape to be exactly a
// record. The guard therefore cannot hold an invalid sample that carries the
// private label with a broken field, and an `invalid` sample here says
// `data_class: synthetic` instead (itself a violation of the contract's
// const). A test flips the label back and requires the sample to stay
// invalid, so each one is refused for its own fault.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Engine } from "./engine.ts";
import { TEAMS_DIR } from "./golden.ts";
import { HOLDOUT_PPM, HOLDOUT_RULE, privateLogicalSnapshotHash, type PrivateSnapshotManifest } from "./private-snapshot.ts";
import { canonicalTeamHash, recordHash, shardLogicalHash } from "./team-records.ts";
import { teamHash } from "./team-class.ts";
import { importPaste } from "./teams.ts";

type Cases = Record<string, unknown>;

/** The corpus pastes the record samples carry. */
export const PRIVATE_SAMPLE_PASTES = ["legal-synthetic.txt", "legal-opponent.txt"] as const;

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

const hexOf = (c: string): string => c.repeat(64);
/** The record samples' `valid` and `reader-invalid` cases carry the private label; `invalid` ones the synthetic one. */
export function privateSamples(engine: Engine): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const pastes = PRIVATE_SAMPLE_PASTES.map((file) => readFileSync(join(TEAMS_DIR, file), "utf8")) as [string, string];
  const hashesOf = (paste: string): { team_hash: string; canonical_team_hash: string } => {
    const sets = importPaste(engine, paste);
    return { team_hash: teamHash(sets), canonical_team_hash: canonicalTeamHash(sets) };
  };
  const first = {
    schema_version: 1,
    data_class: "private",
    team_id: "00112233445566778899aabbccddeeff",
    revision: 1,
    supersedes_record_hash: null,
    ...hashesOf(pastes[0]),
    paste: pastes[0],
    status: "active",
    tags: [],
    labels: [],
    holdout: false,
  };
  // A hold-out team (ADR-015 section 2): the flag follows team_id alone, so the same value on every revision.
  const holdout = { ...first, team_id: "ffeeddccbbaa99887766554433221100", ...hashesOf(pastes[1]), paste: pastes[1], holdout: true };
  const edited = {
    ...first,
    revision: 2,
    supersedes_record_hash: recordHash(first),
    ...hashesOf(pastes[1]),
    paste: pastes[1],
    tags: ["synthetic", "vgc"],
    labels: [{ label_id: "archetype", value: "offense" }],
  };
  const retired = { ...edited, revision: 3, supersedes_record_hash: recordHash(edited), status: "retired" };
  const withRecord = (change: (r: Record<string, unknown>) => void): Record<string, unknown> => mutated(edited, change);
  const synthetic = (change: (r: Record<string, unknown>) => void): Record<string, unknown> =>
    mutated(edited, (r) => {
      r.data_class = "synthetic";
      change(r);
    });

  return {
    valid: {
      "private-team-record/first": first,
      "private-team-record/edited": edited,
      "private-team-record/retired": retired,
      "private-team-record/holdout": holdout,
    },
    invalid: {
      "private-team-record/synthetic-class": synthetic(() => undefined),
      "private-team-record/extra-field": synthetic((r) => (r.rating = 1500)),
      "private-team-record/next-version": synthetic((r) => (r.schema_version = 2)),
      "private-team-record/no-team-id": synthetic((r) => delete r.team_id),
      "private-team-record/short-team-id": synthetic((r) => (r.team_id = "0".repeat(31))),
      "private-team-record/uppercase-team-id": synthetic((r) => (r.team_id = "A".repeat(32))),
      "private-team-record/revision-zero": synthetic((r) => (r.revision = 0)),
      "private-team-record/revision-over-max": synthetic((r) => (r.revision = 2 ** 31)),
      "private-team-record/fractional-revision": synthetic((r) => (r.revision = 1.5)),
      "private-team-record/no-supersedes": synthetic((r) => delete r.supersedes_record_hash),
      "private-team-record/short-supersedes": synthetic((r) => (r.supersedes_record_hash = "abc")),
      "private-team-record/uppercase-team-hash": synthetic((r) => (r.team_hash = hexOf("A"))),
      "private-team-record/empty-paste": synthetic((r) => (r.paste = "")),
      "private-team-record/paste-too-long": synthetic((r) => (r.paste = "x".repeat(8193))),
      "private-team-record/no-holdout": synthetic((r) => delete r.holdout),
      "private-team-record/holdout-not-boolean": synthetic((r) => (r.holdout = "yes")),
      "private-team-record/unknown-status": synthetic((r) => (r.status = "archived")),
      "private-team-record/too-many-tags": synthetic((r) => (r.tags = Array.from({ length: 17 }, (_, i) => `tag-${String.fromCharCode(97 + i)}`))),
      "private-team-record/uppercase-tag": synthetic((r) => (r.tags = ["Vgc"])),
      "private-team-record/long-tag": synthetic((r) => (r.tags = ["a".repeat(33)])),
      "private-team-record/too-many-labels": synthetic((r) => (r.labels = Array.from({ length: 65 }, (_, i) => ({ label_id: `l${i}`, value: "yes" })))),
      "private-team-record/label-extra-field": synthetic((r) => (r.labels = [{ label_id: "archetype", value: "offense", weight: 1 }])),
    },
    readerInvalid: {
      "private-team-record/team-hash-mismatch": withRecord((r) => (r.team_hash = hexOf("a"))),
      "private-team-record/canonical-hash-mismatch": withRecord((r) => (r.canonical_team_hash = hexOf("a"))),
      "private-team-record/first-revision-supersedes": mutated(first, (r) => (r.supersedes_record_hash = hexOf("a"))),
      "private-team-record/later-revision-no-predecessor": withRecord((r) => (r.supersedes_record_hash = null)),
      "private-team-record/unsorted-tags": withRecord((r) => (r.tags = ["vgc", "synthetic"])),
      "private-team-record/duplicate-tags": withRecord((r) => (r.tags = ["vgc", "vgc"])),
      "private-team-record/duplicate-label-ids": withRecord((r) => (r.labels = [{ label_id: "archetype", value: "offense" }, { label_id: "archetype", value: "defense" }])),
    },
  };
}

/** The hold-out key of the manifest samples: the bytes 0..31, so the known-answer vector of the tests covers it. */
export const PRIVATE_SAMPLE_HOLDOUT_KEY = Array.from({ length: 32 }, (_, i) => i.toString(16).padStart(2, "0")).join("");

interface ManifestContext {
  manifest: unknown;
  vocabularyHash: string;
  labelSetHash: string;
}

type Shard = { index: number; file_sha256: string; bytes: number; logical_hash: string; rows: number; split: string; cell_ranges: { stratum: number; first_row: number; rows: number; active_rows: number }[] };

/**
 * The PrivateSnapshotManifest samples. `valid` and `readerInvalid` carry the
 * private label and sit in `PRIVATE_MANIFEST_MARKERS`; `invalid` says
 * `data_class: synthetic` and a test puts the private label back to require
 * each to stay invalid for its own fault. Reader-invalid ones are not resealed
 * (their logical hash is stale or right as the fault allows): the reader's
 * structural checks run before the hash check, and the sample for the hash
 * itself is `wrong-logical-hash`.
 */
export function privateSnapshotSamples(context: ManifestContext): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const shard = (seed: string, rows: number, ranges: [number, number, number][]): Shard => {
    let first = 0;
    return {
      index: 0,
      file_sha256: hexOf(seed),
      bytes: 4096 * rows,
      logical_hash: shardLogicalHash([hexOf(seed), hexOf("0")]),
      rows,
      split: "owner",
      cell_ranges: ranges.map(([stratum, count, active]) => {
        const range = { stratum, first_row: first, rows: count, active_rows: active };
        first += count;
        return range;
      }),
    };
  };
  /** Shards in the fixed order (ascending logical hash), each with its position as index. */
  const ordered = (shards: Shard[]): Shard[] => [...shards].sort((a, b) => a.logical_hash.localeCompare(b.logical_hash)).map((s, index) => ({ ...s, index }));
  const seal = (m: Record<string, unknown>): Record<string, unknown> => ({ ...m, logical_snapshot_hash: privateLogicalSnapshotHash(m as unknown as PrivateSnapshotManifest) });
  const base = {
    schema_version: 1,
    data_class: "private",
    snapshot_id: "owner-snapshot-1",
    first: true,
    manifest: context.manifest,
    vocabulary_hash: context.vocabularyHash,
    label_set_hash: context.labelSetHash,
    logical_snapshot_hash: hexOf("0"),
    shard_key: hexOf("5"),
    shards: ordered([shard("6", 5, [[0, 3, 3], [7, 2, 1]])]),
    cells: [{ split: "owner", stratum: 0, rows: 3 }, { split: "owner", stratum: 7, rows: 1 }],
    records: { active: 4, retired: 1 },
    holdout: { rule: HOLDOUT_RULE, key: PRIVATE_SAMPLE_HOLDOUT_KEY, ppm: HOLDOUT_PPM },
    writer: { name: "duckdb", version: "1.5.6", binary_sha256: hexOf("7") },
    parquet_schema_version: 1,
  };
  const first = seal(base);
  const { first: _first, ...withoutFirst } = base;
  const extended = seal({
    ...withoutFirst,
    snapshot_id: "owner-snapshot-2",
    parent_snapshot_hash: hexOf("8"),
    shards: ordered([shard("6", 5, [[0, 3, 3], [7, 2, 1]]), shard("9", 2, [[0, 2, 2]])]),
    cells: [{ split: "owner", stratum: 0, rows: 5 }, { split: "owner", stratum: 7, rows: 1 }],
    records: { active: 6, retired: 1 },
  });
  const allRetired = seal({ ...base, shards: ordered([shard("6", 2, [[3, 2, 0]])]), cells: [], records: { active: 0, retired: 2 } });

  const of = (from: Record<string, unknown>, change: (m: Record<string, unknown>) => void): Record<string, unknown> => mutated(from, change);
  const shardsOf = (m: Record<string, unknown>): Shard[] => m.shards as Shard[];
  const synthetic = (change: (m: Record<string, unknown>) => void): Record<string, unknown> =>
    of(first, (m) => {
      m.data_class = "synthetic";
      change(m);
    });
  const holdout = (m: Record<string, unknown>): Record<string, unknown> => m.holdout as Record<string, unknown>;

  return {
    valid: {
      "private-snapshot-manifest/first": first,
      "private-snapshot-manifest/extended": extended,
      "private-snapshot-manifest/all-retired": allRetired,
    },
    invalid: {
      "private-snapshot-manifest/synthetic-class": synthetic(() => undefined),
      "private-snapshot-manifest/extra-field": synthetic((m) => (m.uploaded_by = "x")),
      "private-snapshot-manifest/dataset-hashes": synthetic((m) => (m.dataset_hashes = [hexOf("1")])),
      "private-snapshot-manifest/next-version": synthetic((m) => (m.schema_version = 2)),
      "private-snapshot-manifest/first-false": synthetic((m) => (m.first = false)),
      "private-snapshot-manifest/short-parent": synthetic((m) => {
        delete m.first;
        m.parent_snapshot_hash = "abc";
      }),
      "private-snapshot-manifest/no-label-set": synthetic((m) => delete m.label_set_hash),
      "private-snapshot-manifest/no-records": synthetic((m) => delete m.records),
      "private-snapshot-manifest/negative-retired": synthetic((m) => (m.records = { active: 4, retired: -1 })),
      "private-snapshot-manifest/no-holdout": synthetic((m) => delete m.holdout),
      "private-snapshot-manifest/holdout-short-hex": synthetic((m) => (holdout(m).key = "abc")),
      "private-snapshot-manifest/holdout-uppercase-hex": synthetic((m) => (holdout(m).key = hexOf("A"))),
      "private-snapshot-manifest/holdout-other-rule": synthetic((m) => (holdout(m).rule = "hmac-sha256-team-id/2")),
      "private-snapshot-manifest/holdout-other-ppm": synthetic((m) => (holdout(m).ppm = 300000)),
      "private-snapshot-manifest/holdout-extra-field": synthetic((m) => (holdout(m).commitment = hexOf("1"))),
      "private-snapshot-manifest/no-shards": synthetic((m) => (m.shards = [])),
      "private-snapshot-manifest/shard-split-not-owner": synthetic((m) => (shardsOf(m)[0] as Shard).split = "dev"),
      "private-snapshot-manifest/cell-split-not-owner": synthetic((m) => ((m.cells as { split: string }[])[0] as { split: string }).split = "dev"),
      "private-snapshot-manifest/shard-object-path": synthetic((m) => ((shardsOf(m)[0] as unknown as Record<string, unknown>).object_key = `snapshots/${hexOf("1")}/shards/0.parquet`)),
      "private-snapshot-manifest/shard-index-over-max": synthetic((m) => (shardsOf(m)[0] as Shard).index = 4096),
      "private-snapshot-manifest/range-no-active-rows": synthetic((m) => delete ((shardsOf(m)[0] as Shard).cell_ranges[0] as Record<string, unknown>).active_rows),
      "private-snapshot-manifest/uppercase-shard-hex": synthetic((m) => (m.shard_key = hexOf("A"))),
      "private-snapshot-manifest/writer-no-version": synthetic((m) => (m.writer = { name: "duckdb", binary_sha256: hexOf("7") })),
      "private-snapshot-manifest/free-text-snapshot-id": synthetic((m) => (m.snapshot_id = "my team notes")),
    },
    readerInvalid: {
      "private-snapshot-manifest/both-parent-and-first": of(first, (m) => (m.parent_snapshot_hash = hexOf("8"))),
      "private-snapshot-manifest/neither-parent-nor-first": of(first, (m) => delete m.first),
      "private-snapshot-manifest/shard-index-not-position": of(first, (m) => ((shardsOf(m)[0] as Shard).index = 1)),
      "private-snapshot-manifest/shards-out-of-order": of(extended, (m) => (m.shards = [...shardsOf(extended)].reverse().map((s, index) => ({ ...s, index })))),
      "private-snapshot-manifest/duplicate-shard": of(extended, (m) => (m.shards = [shardsOf(extended)[0], { ...(shardsOf(extended)[0] as Shard), index: 1 }])),
      "private-snapshot-manifest/ranges-do-not-tile": of(first, (m) => ((shardsOf(m)[0] as Shard).cell_ranges[1] as { first_row: number }).first_row = 4),
      "private-snapshot-manifest/ranges-out-of-order": of(first, (m) => ((shardsOf(m)[0] as Shard).cell_ranges = [{ stratum: 7, first_row: 0, rows: 2, active_rows: 1 }, { stratum: 0, first_row: 2, rows: 3, active_rows: 3 }])),
      "private-snapshot-manifest/active-rows-over-rows": of(first, (m) => ((shardsOf(m)[0] as Shard).cell_ranges[1] as { active_rows: number }).active_rows = 3),
      "private-snapshot-manifest/rows-not-range-sum": of(first, (m) => ((shardsOf(m)[0] as Shard).rows = 6)),
      "private-snapshot-manifest/cell-not-active-sum": of(first, (m) => (m.cells = [{ split: "owner", stratum: 0, rows: 2 }, { split: "owner", stratum: 7, rows: 1 }])),
      "private-snapshot-manifest/cells-out-of-order": of(first, (m) => (m.cells = [{ split: "owner", stratum: 7, rows: 1 }, { split: "owner", stratum: 0, rows: 3 }])),
      "private-snapshot-manifest/unlisted-active-cell": of(first, (m) => (m.cells = [{ split: "owner", stratum: 0, rows: 3 }])),
      "private-snapshot-manifest/records-active-not-cells": of(first, (m) => (m.records = { active: 5, retired: 0 })),
      "private-snapshot-manifest/counts-not-shard-rows": of(first, (m) => (m.records = { active: 4, retired: 2 })),
      "private-snapshot-manifest/wrong-logical-hash": of(first, (m) => (m.logical_snapshot_hash = hexOf("a"))),
    },
  };
}
