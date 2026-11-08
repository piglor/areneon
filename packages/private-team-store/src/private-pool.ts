// The private team pool of the generation loop (ADR-015 erratum E1, item 3): the
// active records of ONE pinned snapshot, read from its verified local copy
// (`private-data/snapshots/<hash>/`) through the one snapshot check
// (`readPrivateSnapshot`), each team normalised through the pinned validator on
// a copy (as `datasetTeamPool` does), sorted by canonical team hash and kept with
// its hold-out flag. `withPrivatePool` joins it to the synthetic pool a run
// already has, as a second component: the bridge cannot import this package, so
// the result is an ordinary `PoolSource` for `runGenerationCli`. The staged
// directory is never a source (it has no hold-out flag). Nothing here writes a
// file; every refusal is a fixed code and at most a 1-based team position, never
// text of a lower layer, a paste or a team field.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type Engine,
  importPaste,
  type PokemonSet,
  type PoolSource,
  privateDataRoot,
  resolveFormat,
  teamHash,
  teamVocabularyHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { loadFixtureInputs } from "@areneon/team-dataset";
import { readPrivateSnapshot } from "./snapshot-reader.ts";

export const PRIVATE_POOL_REFUSALS = {
  source_unreadable: "the snapshot's local copy cannot be read",
  snapshot_unverified: "the snapshot does not verify",
  source_empty: "the snapshot has no usable active team",
  team_invalid: "a team is not legal in the format",
  share_invalid: "the share is not a whole number of permille from 1 to 1000",
  pool_has_private_component: "the synthetic pool already has a private component",
  synthetic_overlap: "a private team is also in the synthetic pool",
} as const;
export type PrivatePoolRefusalCode = keyof typeof PRIVATE_POOL_REFUSALS;

export class PrivatePoolRefusal extends Error {
  readonly code: PrivatePoolRefusalCode;
  readonly position: number | undefined;
  constructor(code: PrivatePoolRefusalCode, position?: number) {
    super(`private-pool: ${code}${position === undefined ? "" : ` ${position}`}`);
    this.name = "PrivatePoolRefusal";
    this.code = code;
    this.position = position;
  }
}

/** The active teams of one snapshot, in the order an index names them (canonical team hash), played form. */
export interface PrivatePool {
  snapshot_hash: string;
  /** Normalised by the pinned validator, the form a battle starts with. */
  teams: readonly PokemonSet[][];
  /** `teamHash` of each played form, the key `known_synthetic` would hold for a synthetic team (and never holds for these). */
  hashes: readonly string[];
  /** Each team's hold-out flag, as the snapshot's reader recomputed it. */
  holdout: readonly boolean[];
}

export interface LoadPrivatePoolInput {
  engine: Engine;
  format_id: string;
  /** The `private-data/snapshots` directory holding the verified local copies. */
  snapshotsDir: string;
  snapshot_hash: string;
  /** The run config's `generation.excluded_abilities`: a team holding one is left out, as in the synthetic pool. */
  excluded_abilities?: readonly string[];
}

/** Read and verify the pinned snapshot's local copy and build its pool. */
export async function loadPrivatePool(input: LoadPrivatePoolInput): Promise<PrivatePool> {
  const { engine, format_id, snapshotsDir, snapshot_hash } = input;
  if (!/^[0-9a-f]{64}$/.test(snapshot_hash)) throw new PrivatePoolRefusal("source_unreadable");
  let dir: string;
  let manifest: unknown;
  try {
    privateDataRoot(snapshotsDir); // the copy lies inside a private-data directory
    dir = join(snapshotsDir, snapshot_hash);
    manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
  } catch {
    throw new PrivatePoolRefusal("source_unreadable");
  }
  let active;
  try {
    const inputs = loadFixtureInputs(engine, format_id);
    const read = await readPrivateSnapshot({
      engine,
      manifest,
      // A copy in a plain Uint8Array: the decode worker takes ownership of the buffer it is given.
      shard: (index) => new Uint8Array(readFileSync(join(dir, "shards", `${index}.parquet`))),
      labelSet: inputs.labelSet,
      expectedSnapshotHash: snapshot_hash,
      expected: { manifest: resolveFormat(engine, format_id).manifest, vocabularyHash: teamVocabularyHash(inputs.vocabulary) },
    });
    active = read.records.filter((record) => record.status === "active");
  } catch {
    throw new PrivatePoolRefusal("snapshot_unverified");
  }
  const format = resolveFormat(engine, format_id);
  const excluded = new Set(input.excluded_abilities ?? []);
  const kept: { canonical: string; played: string; sets: PokemonSet[]; holdout: boolean }[] = [];
  let position = 0;
  for (const record of active) {
    position++;
    // The reader has round-tripped the paste, so it imports; validateTeam normalises the sets in place: this is the copy the battle plays.
    const sets = importPaste(engine, record.paste);
    if (!validateTeam(engine, format, sets).valid) throw new PrivatePoolRefusal("team_invalid", position);
    if (sets.some((set) => excluded.has(set.ability))) continue;
    // The record's own canonical hash (checked against its paste by the reader): the order does not depend on how the validator normalises.
    // (Active records have distinct canonical hashes: the reader's record-set check.)
    kept.push({ canonical: record.canonical_team_hash, played: teamHash(sets), sets, holdout: record.holdout });
  }
  if (kept.length === 0) throw new PrivatePoolRefusal("source_empty");
  kept.sort((a, b) => (a.canonical < b.canonical ? -1 : a.canonical > b.canonical ? 1 : 0));
  return { snapshot_hash, teams: kept.map((k) => k.sets), hashes: kept.map((k) => k.played), holdout: kept.map((k) => k.holdout) };
}

/**
 * The synthetic pool a run already has plus the private one as a second component. `share_permille` is the share of seats that
 * draw from the private component (`league-team-source/1`). `known_synthetic` is the synthetic pool's, untouched, so `stampTeam`
 * makes every private team `private`; a private team that is also in that set is refused, never stamped synthetic.
 */
export function withPrivatePool(synthetic: PoolSource, pool: PrivatePool, share_permille: number): PoolSource {
  if (!Number.isInteger(share_permille) || share_permille < 1 || share_permille > 1000) throw new PrivatePoolRefusal("share_invalid");
  return (engine, request) => {
    const base = synthetic(engine, request);
    if (base.private !== undefined) throw new PrivatePoolRefusal("pool_has_private_component");
    pool.hashes.forEach((hash, index) => {
      if (base.known_synthetic.has(hash)) throw new PrivatePoolRefusal("synthetic_overlap", index + 1);
    });
    return {
      teams: [...base.teams, ...pool.teams],
      hashes: [...base.hashes, ...pool.hashes],
      known_synthetic: base.known_synthetic,
      private: { count: pool.teams.length, share_permille, holdout: pool.holdout },
    };
  };
}
