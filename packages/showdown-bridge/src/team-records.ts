// The team records of ADR-013 (section 4) and their identities. The schemas
// (`packages/schema/contracts/v1/team-*.schema.json`) hold the shape; this
// module holds what the contract profile cannot say: sorted and unique lists,
// sums, hashes, and the paste's round trip through the pinned engine. Nothing
// here decides legality: the engine does (`validateTeam`), and the generator
// (a later slice) calls it before a record is made.
//
// Identities, all SHA-256 of RFC 8785 (JCS) bytes, so they do not depend on
// key order or on how a language prints a number:
//   team_hash            the set array in paste order (`teamHash`, team-class.ts)
//   canonical_team_hash  the same array sorted by species id, ties by the set's
//                        JCS bytes: the identity for the no-overlap rules (L1),
//                        the same for every permutation of one team
//   record_hash          the whole TeamRecord
//   logical_hash         a shard: the sorted list of its record hashes
//   logical_snapshot_hash  a snapshot, independent of the Parquet writer
//   snapshot hash        the whole TeamSnapshotManifest (one physical publication)

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import teamDatasetManifestSchema from "@areneon/schema/contracts/v1/team-dataset-manifest.schema.json" with { type: "json" };
import teamGeneratorConfigSchema from "@areneon/schema/contracts/v1/team-generator-config.schema.json" with { type: "json" };
import teamLabelSetSchema from "@areneon/schema/contracts/v1/team-label-set.schema.json" with { type: "json" };
import teamRecordSchema from "@areneon/schema/contracts/v1/team-record.schema.json" with { type: "json" };
import teamSnapshotManifestSchema from "@areneon/schema/contracts/v1/team-snapshot-manifest.schema.json" with { type: "json" };
import type { TeamDatasetManifest } from "@areneon/schema/types/v1/team-dataset-manifest.ts";
import type { TeamGeneratorConfig } from "@areneon/schema/types/v1/team-generator-config.ts";
import type { TeamLabelSet } from "@areneon/schema/types/v1/team-label-set.ts";
import type { TeamRecord } from "@areneon/schema/types/v1/team-record.ts";
import type { TeamSnapshotManifest } from "@areneon/schema/types/v1/team-snapshot-manifest.ts";
import { jcs, type Json } from "./canonical-json.ts";
import type { EngineData } from "./engine-data.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { teamHash } from "./team-class.ts";
import { exportPaste, importPaste, TeamImportError } from "./teams.ts";

export type { TeamDatasetManifest, TeamGeneratorConfig, TeamLabelSet, TeamRecord, TeamSnapshotManifest };

export class TeamRecordError extends Error {
  override name = "TeamRecordError";
}

const bad = (message: string): never => {
  throw new TeamRecordError(message);
};

const sha256 = (value: unknown): string => createHash("sha256").update(jcs(value as Json), "utf8").digest("hex");

/** SHA-256 of a record's JCS bytes: `record_hash`, the label-set hash, the config hash, the dataset hash. */
export const recordHash = (record: unknown): string => sha256(record);

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

/**
 * The team's identity independent of order (ADR-013 section 4): the hash of
 * its sets sorted by species id, ties broken by each set's JCS bytes. Any
 * permutation of the same sets has the same hash; it equals the positional
 * `teamHash` exactly when the sets are already in that order.
 */
export function canonicalTeamHash(sets: readonly PokemonSet[]): string {
  const keyed = sets.map((set) => ({ set, species: toId(set.species), bytes: jcs(set as unknown as Json) }));
  keyed.sort((a, b) => codeUnits(a.species, b.species) || codeUnits(a.bytes, b.bytes));
  return teamHash(keyed.map((k) => k.set));
}

/** `values` must be strictly ascending: sorted by code unit, no repeat. */
function ordered(values: readonly string[], what: string): void {
  values.forEach((v, k) => {
    if (k > 0 && !((values[k - 1] as string) < v)) bad(`${what}: ${v} is repeated or out of order`);
  });
}

/** No value repeats. */
function distinct(values: readonly string[], what: string): void {
  if (new Set(values).size !== values.length) bad(`${what}: a value is repeated`);
}

/** A shard's logical hash: SHA-256 of the JCS bytes of the sorted list of its record hashes. */
export function shardLogicalHash(recordHashes: readonly string[]): string {
  distinct(recordHashes, "shard record hashes");
  return sha256([...recordHashes].sort(codeUnits));
}

/**
 * A snapshot's writer-independent identity: SHA-256 of the JCS bytes of
 * `{shards: the sorted shard logical hashes, cells, dataset_hashes, manifest}`.
 */
export function logicalSnapshotHash(snapshot: TeamSnapshotManifest): string {
  const shards = snapshot.shards.map((s) => s.logical_hash).sort(codeUnits);
  return sha256({ shards, cells: snapshot.cells, dataset_hashes: snapshot.dataset_hashes, manifest: snapshot.manifest });
}

/** One physical publication: SHA-256 of the whole manifest's JCS bytes. */
export const snapshotHash = (snapshot: TeamSnapshotManifest): string => sha256(snapshot);

const validators = {
  record: ajvFor("contract").compile(teamRecordSchema),
  labelSet: ajvFor("contract").compile(teamLabelSetSchema),
  config: ajvFor("contract").compile(teamGeneratorConfigSchema),
  dataset: ajvFor("contract").compile(teamDatasetManifestSchema),
  snapshot: ajvFor("contract").compile(teamSnapshotManifestSchema),
};

function parsed<T>(validate: ((value: unknown) => boolean) & { errors?: unknown }, value: unknown, what: string): T {
  if (!validate(value)) return bad(`invalid ${what}: ${JSON.stringify(validate.errors)}`);
  return value as T;
}

/** The label set a record's labels follow, as its order and values. */
function followsLabelSet(record: TeamRecord, set: TeamLabelSet): void {
  if (record.label_set_hash !== recordHash(set)) bad("label_set_hash is not the hash of the given label set");
  if (record.labels.length !== set.labels.length) bad("labels are not one per label of the label set");
  record.labels.forEach((entry, k) => {
    const label = set.labels[k] as TeamLabelSet["labels"][number];
    if (entry.label_id !== label.label_id) bad(`labels[${k}] is ${entry.label_id}, but the label set's is ${label.label_id}`);
    if (entry.value !== label.default && !label.values.some((v) => v.value === entry.value)) {
      bad(`label ${entry.label_id}: ${entry.value} is not one of its values`);
    }
  });
}

/**
 * Parse a TeamRecord: the schema, then the engine's round trip of the paste
 * (it must be exactly the engine's own export), both team hashes recomputed
 * from the imported sets, and unique label ids. With the label set the labels
 * were computed with, also that they follow its order and values. Whether a
 * label's value is the labeller's output is the labeller's check, not this one.
 */
export function readTeamRecord(engine: Engine, value: unknown, labelSet?: TeamLabelSet): TeamRecord {
  const record = parsed<TeamRecord>(validators.record, value, "team record");
  let sets: PokemonSet[];
  try {
    sets = importPaste(engine, record.paste);
  } catch (error) {
    if (!(error instanceof TeamImportError)) throw error;
    return bad(`the paste is refused by the pinned engine: ${error.message}`);
  }
  if (exportPaste(engine, sets) !== record.paste) bad("the paste is not the engine's own export of its sets");
  if (teamHash(sets) !== record.team_hash) bad("team_hash is not the hash of the paste's sets in paste order");
  if (canonicalTeamHash(sets) !== record.canonical_team_hash) bad("canonical_team_hash is not the hash of the paste's sets in canonical order");
  distinct(record.labels.map((l) => l.label_id), "label ids");
  if (labelSet !== undefined) followsLabelSet(record, labelSet);
  return record;
}

/** Each leaf kind's id list, and the engine-data table those ids must be in. */
const LEAF_IDS = {
  has_species: ["species", "species"],
  has_move: ["moves", "moves"],
  has_item: ["items", "items"],
  has_ability: ["abilities", "abilities"],
} as const;

/**
 * Parse a TeamLabelSet: the schema, then unique label ids, unique values per
 * label and sorted, unique id lists. With the engine-data table of the pinned
 * format, also that every id resolves in it (the labeller code holds no ids;
 * this is where a typo in a record is caught).
 */
export function readTeamLabelSet(value: unknown, table?: EngineData): TeamLabelSet {
  const set = parsed<TeamLabelSet>(validators.labelSet, value, "team label set");
  distinct(set.labels.map((l) => l.label_id), "label ids");
  for (const label of set.labels) {
    distinct(label.values.map((v) => v.value), `label ${label.label_id} values`);
    if (label.values.some((v) => v.value === label.default)) bad(`label ${label.label_id}: the default ${label.default} is also a listed value`);
    for (const { when } of label.values) {
      for (const leaf of when.kind === "all_of" ? when.of : [when]) {
        const [field, vocabulary] = LEAF_IDS[leaf.kind];
        const ids = (leaf as unknown as Record<string, string[]>)[field] as string[];
        ordered(ids, `label ${label.label_id} ${leaf.kind} ids`);
        const known = table === undefined ? undefined : new Set(table[vocabulary].map((e) => e.id));
        if (known !== undefined) for (const id of ids) if (!known.has(id)) bad(`label ${label.label_id}: ${id} is not in the engine-data table`);
      }
    }
  }
  return set;
}

/**
 * Parse a TeamGeneratorConfig: the schema, then item weights sorted by item
 * id and unique. With the engine-data table, also that every item exists.
 */
export function readTeamGeneratorConfig(value: unknown, table?: EngineData): TeamGeneratorConfig {
  const config = parsed<TeamGeneratorConfig>(validators.config, value, "team generator config");
  ordered(config.item_weights.map((w) => w.item), "item_weights");
  if (config.proposal !== undefined) {
    // The profile cannot say: components sorted by kind and unique, and shares that add up to the 1000 draws of the component pick.
    const { components } = config.proposal;
    ordered(components.map((c) => c.kind), "proposal components");
    const total = components.reduce((sum, c) => sum + c.share_permille, 0);
    if (total !== 1000) bad(`proposal components: the shares add up to ${total}, not 1000`);
  }
  if (table !== undefined) {
    const items = new Set(table.items.map((i) => i.id));
    for (const { item } of config.item_weights) if (!items.has(item)) bad(`item_weights: ${item} is not in the engine-data table`);
  }
  return config;
}

const SPLIT_ORDER = { dev: 0, validation: 1, training: 2 } as const;

/** What a dataset manifest is read against, when the reader has it. */
export interface DatasetManifestContext {
  /** The generator config: `d_max`, `pilot_draws` and `config_hash` must agree with it. */
  config?: TeamGeneratorConfig;
  /** The label set: its hash must be `label_set_hash`, and every cell number must exist in its label space. */
  labelSet?: TeamLabelSet;
}

/** The base cells of a label set: one per combination of a label's listed values and its default. */
export const labelCellCount = (set: TeamLabelSet): number => set.labels.reduce((n, l) => n * (l.values.length + 1), 1);

/** The merged strata of a manifest, checked: ascending, disjoint, each numbered by its lowest member. Returns the stratum of each member cell. */
function mergedStrata(manifest: TeamDatasetManifest): Map<number, number> {
  const owner = new Map<number, number>();
  let previous = -1;
  for (const group of manifest.merged_cells ?? []) {
    if (group.stratum <= previous) bad(`merged_cells: stratum ${group.stratum} is repeated or out of order`);
    previous = group.stratum;
    if (group.cells[0] !== group.stratum) bad(`merged_cells: stratum ${group.stratum} is not its lowest member`);
    group.cells.forEach((cell, k) => {
      if (k > 0 && (group.cells[k - 1] as number) >= cell) bad(`merged_cells: stratum ${group.stratum}: cell ${cell} is repeated or out of order`);
      if (owner.has(cell)) bad(`merged_cells: cell ${cell} is in two strata`);
      owner.set(cell, group.stratum);
    });
  }
  return owner;
}

/**
 * Parse a TeamDatasetManifest: the schema, then split and cell order, each
 * split's team count, and the cell merge: the merged strata are well formed
 * and every cell of a split is a merged stratum or an unmerged base cell.
 * With the config it was generated with, also that `d_max`, `pilot_draws`
 * and `config_hash` agree with it; with the label set, that `label_set_hash`
 * is its hash and no cell number lies outside its label space.
 */
export function readTeamDatasetManifest(value: unknown, context: DatasetManifestContext = {}): TeamDatasetManifest {
  const manifest = parsed<TeamDatasetManifest>(validators.dataset, value, "team dataset manifest");
  const splits = manifest.splits.map((s) => s.split);
  if (!splits.every((s, k) => k === 0 || SPLIT_ORDER[splits[k - 1] as keyof typeof SPLIT_ORDER] < SPLIT_ORDER[s])) {
    bad("splits are repeated or not in the order dev, validation, training");
  }
  const owner = mergedStrata(manifest);
  for (const plan of manifest.splits) {
    plan.cells.forEach((cell, k) => {
      if (k > 0 && (plan.cells[k - 1] as { stratum: number }).stratum >= cell.stratum) bad(`split ${plan.split}: cell ${cell.stratum} is repeated or out of order`);
      const merged = owner.get(cell.stratum);
      if (merged !== undefined && merged !== cell.stratum) bad(`split ${plan.split}: cell ${cell.stratum} is a member of merged stratum ${merged}`);
    });
    for (const stratum of new Set(owner.values())) {
      if (!plan.cells.some((c) => c.stratum === stratum)) bad(`split ${plan.split}: merged stratum ${stratum} has no cell`);
    }
    const total = plan.cells.reduce((sum, c) => sum + c.quota, 0);
    if (total !== plan.teams) bad(`split ${plan.split}: teams ${plan.teams} is not the sum of its quotas ${total}`);
  }
  const { config, labelSet } = context;
  if (config !== undefined) {
    if (manifest.d_max !== config.d_max) bad(`d_max ${manifest.d_max} differs from the config's ${config.d_max}`);
    if (manifest.pilot_draws !== config.pilot_draws) bad(`pilot_draws ${manifest.pilot_draws} differs from the config's ${config.pilot_draws}`);
    if (manifest.generator.config_hash !== recordHash(config)) bad("config_hash is not the hash of the given config");
  }
  if (labelSet !== undefined) {
    if (manifest.generator.label_set_hash !== recordHash(labelSet)) bad("label_set_hash is not the hash of the given label set");
    const cells = labelCellCount(labelSet);
    const numbers = [
      ...manifest.splits.flatMap((p) => p.cells.map((c) => c.stratum)),
      ...(manifest.merged_cells ?? []).flatMap((g) => g.cells),
    ];
    for (const n of numbers) if (n >= cells) bad(`cell ${n} is outside the label set's ${cells} cells`);
  }
  return manifest;
}

/**
 * Parse a TeamSnapshotManifest: the schema, then everything the profile cannot
 * say: sorted and unique dataset hashes, shards in fixed order with unique
 * logical hashes, each shard's object key equal to its file hash, its cell
 * ranges tiling it, the cell totals equal to the shards' sums, and
 * `logical_snapshot_hash` recomputed.
 */
export function readTeamSnapshotManifest(value: unknown): TeamSnapshotManifest {
  const snapshot = parsed<TeamSnapshotManifest>(validators.snapshot, value, "team snapshot manifest");
  ordered(snapshot.dataset_hashes, "dataset_hashes");
  const fixed = (a: TeamSnapshotManifest["shards"][number]): string => `${SPLIT_ORDER[a.split]}:${a.logical_hash}`;
  ordered(snapshot.shards.map(fixed), "shards");
  distinct(snapshot.shards.map((s) => s.logical_hash), "shard logical hashes");
  const totals = new Map<string, number>();
  for (const shard of snapshot.shards) {
    if (shard.object_key !== `pub/v1/shards/${shard.file_sha256}`) bad(`shard ${shard.logical_hash}: object_key is not pub/v1/shards/<file_sha256>`);
    let next = 0;
    let previous = -1;
    for (const range of shard.cell_ranges) {
      if (range.stratum <= previous) bad(`shard ${shard.logical_hash}: cell ranges are repeated or out of stratum order`);
      if (range.first_row !== next) bad(`shard ${shard.logical_hash}: cell ranges do not tile the shard`);
      next += range.rows;
      previous = range.stratum;
      const key = `${shard.split}:${range.stratum}`;
      totals.set(key, (totals.get(key) ?? 0) + range.rows);
    }
    if (next !== shard.rows) bad(`shard ${shard.logical_hash}: rows ${shard.rows} is not the sum of its cell ranges ${next}`);
  }
  const keys = snapshot.cells.map((c) => `${SPLIT_ORDER[c.split]}:${String(c.stratum).padStart(6, "0")}`);
  ordered(keys, "cells");
  if (snapshot.cells.length !== totals.size) bad("cells do not list exactly the cells the shards hold");
  for (const cell of snapshot.cells) {
    if (totals.get(`${cell.split}:${cell.stratum}`) !== cell.rows) bad(`cell ${cell.split}:${cell.stratum}: rows is not the shards' sum`);
  }
  if (snapshot.logical_snapshot_hash !== logicalSnapshotHash(snapshot)) bad("logical_snapshot_hash is not recomputed from the shards, cells, datasets and manifest");
  return snapshot;
}
