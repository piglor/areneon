// The training pool of a run, listed for the training-data inspector (ADR-017
// section 4a). It is the pool the loop itself builds, through the loop's own
// functions (`datasetPoolDetail` for a run that names a team dataset,
// `interimPoolSource` for the interim pool), plus what a reader needs to show
// it: the readable sets, the generator labels and the dataset's own hash. It
// reads, never writes, and adds no rule: a Mega Stone is whatever the pinned
// engine's data marks as one for the format's mod.

import {
  type Engine,
  exportPaste,
  interimPoolSource,
  importPaste,
  type PokemonSet,
  type TeamRecord,
  type TrainingRunConfig,
} from "@areneon/showdown-bridge";
import { datasetPoolDetail } from "./loop-pool.ts";

export interface ListedSet {
  name: string;
  species: string;
  item: string;
  ability: string;
  nature: string;
  gender: string;
  level: number;
  moves: string[];
  evs: Record<string, number>;
  /** The engine's data says the item is a Mega Stone. */
  mega_stone: boolean;
}

export interface ListedTeam {
  /** The played-form team hash: the one `plan.json` indices, `team-set.json` and Episodes use. */
  hash: string;
  canonical_team_hash: string | null;
  labels: { label_id: string; value: string }[];
  split: string | null;
  /** The engine's canonical paste of the sets. */
  paste: string;
  sets: ListedSet[];
}

export interface PoolListing {
  schema_version: 1;
  source: "dataset" | "interim";
  dataset_hash: string | null;
  records_before: number;
  excluded_abilities: string[];
  /** In pool order: the index a plan names is the position here. */
  teams: ListedTeam[];
}

function listSets(engine: Engine, mod: string, sets: readonly PokemonSet[]): ListedSet[] {
  const dex = engine.sim.Dex.mod(mod);
  return sets.map((set) => ({
    name: set.name,
    species: set.species,
    item: set.item,
    ability: set.ability,
    nature: set.nature,
    gender: set.gender,
    level: set.level,
    moves: [...set.moves],
    evs: { ...set.evs },
    mega_stone: dex.items.get(set.item).megaStone !== undefined,
  }));
}

export function listPool(engine: Engine, config: TrainingRunConfig): PoolListing {
  const mod = engine.sim.Dex.formats.get(config.format_id).mod;
  const excluded = [...config.generation.excluded_abilities];
  if (config.team_dataset !== undefined) {
    const detail = datasetPoolDetail(engine, config);
    const teams = detail.pool.hashes.map((hash): ListedTeam => {
      const record = detail.records.get(hash) as TeamRecord; // `datasetPoolDetail` keys one record per pool team
      return {
        hash,
        canonical_team_hash: record.canonical_team_hash,
        labels: record.labels.map(({ label_id, value }) => ({ label_id, value })),
        split: record.origin.split,
        paste: record.paste,
        sets: listSets(engine, mod, importPaste(engine, record.paste)),
      };
    });
    return { schema_version: 1, source: "dataset", dataset_hash: detail.dataset_hash, records_before: detail.records_before, excluded_abilities: excluded, teams };
  }
  const pool = interimPoolSource(engine, config);
  const before = interimPoolSource(engine, { ...config, generation: { ...config.generation, excluded_abilities: [] } }).hashes.length;
  const teams = pool.hashes.map(
    (hash, index): ListedTeam => ({
      hash,
      canonical_team_hash: null,
      labels: [],
      split: null,
      paste: exportPaste(engine, pool.teams[index] as PokemonSet[]),
      sets: listSets(engine, mod, pool.teams[index] as PokemonSet[]),
    }),
  );
  return { schema_version: 1, source: "interim", dataset_hash: null, records_before: before, excluded_abilities: excluded, teams };
}
