// The derived working database (ADR-013 section 7): a local DuckDB file built
// from a verified PUBLIC snapshot, for exploring teams with SQL. It is a cache
// of the snapshot, never a source of truth: it is not committed, not in a
// Release, not in R2 and not in an image (`*.duckdb` is gitignored and the
// binary guard refuses it).
//
// The build verifies first, then loads. Every shard goes through the same
// reader a consumer of the snapshot uses (file size and SHA-256, the layout
// rules, the record rebuild, the logical hash), every record's label set must
// be the one given, and every paste is expanded with the pinned engine's
// `Teams.import` (`importPaste`, which refuses anything but the engine's own
// export). The tables are written as JSON Lines files in a scratch directory
// and loaded by the versioned SQL script `sql/working-db.v1.sql` that the
// user-installed DuckDB command-line tool runs; DuckDB is never imported.
// Private teams are out of scope (ADR-015's workflow has its own tooling).

import { closeSync, existsSync, linkSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { type Engine, importPaste, readTeamSnapshotManifest, recordHash, snapshotHash, type TeamDatasetManifest, type TeamLabelSet } from "@areneon/showdown-bridge";
import { cellOf } from "../labeller.ts";
import { type LabelEntry, refuse, type RowPlan } from "../shards/columns.ts";
import type { DuckdbWriter } from "../shards/duckdb.ts";
import { SHARD_ROWS } from "../shards/layout.ts";
import { datasetPlan, eachSnapshotShard, shardsInDirectory } from "../shards/snapshot.ts";

/** The version of the table layout in `sql/working-db.v1.sql`; it is the `duckdb_schema_version` column of the `snapshot` table. */
export const DUCKDB_SCHEMA_VERSION = 1;
export const SQL_SCRIPT = new URL("../../sql/working-db.v1.sql", import.meta.url);

/** What runs before the script: the tool may not fetch or load extensions (as on the private path). */
export const SCRIPT_PRELUDE = "SET autoinstall_known_extensions = false;\nSET autoload_known_extensions = false;\n";

/** The tables the script loads, one JSON Lines file each. */
export const TABLE_FILES = ["snapshot", "shards", "teams", "labels", "sets", "moves"] as const;

export interface WorkingDatabaseOptions {
  /** `snapshot.json`. */
  snapshotFile: string;
  /** The directory that holds `pub/v1/shards/<sha256>`. */
  shardsDir: string;
  labelSet: TeamLabelSet;
  /** The dataset manifest, for the merged strata; without it every base cell is its own stratum. */
  datasetManifest?: TeamDatasetManifest;
  engine: Engine;
  writer: DuckdbWriter;
  /** The database file to create; an existing file is never replaced. */
  outFile: string;
  /** Rows per shard the layout rule assumes; 2^18 unless a test says otherwise. */
  shardRows?: number;
}

export interface WorkingDatabaseResult {
  database: string;
  teams: number;
  sets: number;
  moves: number;
}

/** One JSON Lines file, written in chunks. */
class Lines {
  readonly #fd: number;
  #pending: string[] = [];
  #closed = false;
  constructor(path: string) {
    this.#fd = openSync(path, "w");
  }
  add(row: unknown): void {
    this.#pending.push(`${JSON.stringify(row)}\n`);
    if (this.#pending.length >= 2048) this.flush();
  }
  flush(): void {
    if (this.#pending.length > 0) writeSync(this.#fd, this.#pending.join(""));
    this.#pending = [];
  }
  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.flush();
    closeSync(this.#fd);
  }
}

const planFor = (labelSet: TeamLabelSet, datasetManifest: TeamDatasetManifest | undefined): RowPlan =>
  datasetManifest === undefined ? { labelSet, stratumOf: (labels: readonly LabelEntry[]) => cellOf(labelSet, labels) } : datasetPlan(datasetManifest, labelSet);

/**
 * Build the working database. The snapshot is verified shard by shard before
 * anything is loaded; a failure leaves no database file behind.
 */
export async function buildWorkingDatabase(options: WorkingDatabaseOptions): Promise<WorkingDatabaseResult> {
  if (existsSync(options.outFile)) return refuse(`${options.outFile} exists; the working database is rebuilt into a new file, never over one`);
  const snapshot = readTeamSnapshotManifest(JSON.parse(readFileSync(options.snapshotFile, "utf8")));
  const labelSetHash = recordHash(options.labelSet);
  const scratch = mkdtempSync(join(tmpdir(), "team-working-db-"));
  const files = Object.fromEntries(TABLE_FILES.map((name) => [name, new Lines(join(scratch, `${name}.jsonl`))])) as Record<(typeof TABLE_FILES)[number], Lines>;
  const counts = { teams: 0, sets: 0, moves: 0 };
  // Absolute, so the tool never reads the path as an option, whatever the name starts with.
  const target = resolve(options.outFile);
  const dir = dirname(target);
  mkdirSync(dir, { recursive: true });
  const partial = join(dir, `.${basename(target)}.partial`);
  for (const stale of [partial, `${partial}.wal`]) rmSync(stale, { force: true });
  try {
    for await (const { entry, read } of eachSnapshotShard(snapshot, shardsInDirectory(options.shardsDir), planFor(options.labelSet, options.datasetManifest), options.shardRows ?? SHARD_ROWS)) {
      files.shards.add({ split: entry.split, object_key: entry.object_key, file_sha256: entry.file_sha256, bytes: entry.bytes, logical_hash: entry.logical_hash, rows: entry.rows });
      read.records.forEach((record, k) => {
        if (record.label_set_hash !== labelSetHash) refuse(`${entry.split}: record ${k} of shard ${entry.logical_hash} was labelled with another label set`);
        const recordHashOfRow = read.recordHashes[k] as string;
        const sets = importPaste(options.engine, record.paste);
        files.teams.add({
          record_hash: recordHashOfRow,
          team_hash: record.team_hash,
          canonical_team_hash: record.canonical_team_hash,
          dataset_id: record.origin.dataset_id,
          split: record.origin.split,
          origin_index: record.origin.index,
          draw_counter: record.origin.draw_counter,
          stratum: read.strata[k],
          shard_logical_hash: entry.logical_hash,
          paste: record.paste,
        });
        for (const label of record.labels) files.labels.add({ record_hash: recordHashOfRow, label_id: label.label_id, value: label.value });
        sets.forEach((set, slot) => {
          files.sets.add({
            record_hash: recordHashOfRow,
            slot,
            nickname: set.name,
            species: set.species,
            item: set.item,
            ability: set.ability,
            nature: set.nature,
            gender: set.gender,
            level: set.level,
            sp_hp: set.evs.hp,
            sp_atk: set.evs.atk,
            sp_def: set.evs.def,
            sp_spa: set.evs.spa,
            sp_spd: set.evs.spd,
            sp_spe: set.evs.spe,
          });
          set.moves.forEach((move, moveSlot) => files.moves.add({ record_hash: recordHashOfRow, slot, move_slot: moveSlot, move }));
          counts.moves += set.moves.length;
        });
        counts.sets += sets.length;
        counts.teams += 1;
      });
    }
    files.snapshot.add({
      snapshot_id: snapshot.snapshot_id,
      snapshot_hash: snapshotHash(snapshot),
      logical_snapshot_hash: snapshot.logical_snapshot_hash,
      parent_snapshot_hash: snapshot.parent_snapshot_hash ?? null,
      vocabulary_hash: snapshot.vocabulary_hash,
      format_id: snapshot.manifest.format_id,
      regulation: snapshot.manifest.regulation,
      showdown_commit: snapshot.manifest.showdown_commit,
      writer_name: snapshot.writer.name,
      writer_version: snapshot.writer.version,
      parquet_schema_version: snapshot.parquet_schema_version,
      teams: counts.teams,
    });
    for (const file of Object.values(files)) file.close();
    options.writer.runScript(SCRIPT_PRELUDE + readFileSync(SQL_SCRIPT, "utf8"), partial, scratch);
    // A hard link fails with EEXIST if a file appeared meanwhile, where a rename would replace it.
    for (const suffix of ["", ".wal"]) {
      if (!existsSync(partial + suffix)) continue;
      try {
        linkSync(partial + suffix, target + suffix);
      } catch (error) {
        if ((error as { code?: string }).code === "EEXIST") return refuse(`${target + suffix} appeared while the database was built; nothing was replaced`);
        throw error;
      }
    }
  } finally {
    for (const file of Object.values(files)) file.close();
    rmSync(scratch, { recursive: true, force: true });
    rmSync(partial, { force: true });
    rmSync(`${partial}.wal`, { force: true });
  }
  return { database: options.outFile, ...counts };
}
