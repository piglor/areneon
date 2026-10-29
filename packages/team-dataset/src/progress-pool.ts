// The progress evaluation's team pool (ADR-017 section 2): the VALIDATION split
// of the run's team dataset and nothing else. The directory goes through
// `readDataset` (manifest against config, label set and the engine's version
// manifest; every record round-tripped through the engine and the validator),
// its hash must be the one the run config names, and only validation records
// are taken, so a progress game never plays a team of the training pool
// (`loop-pool.ts` takes the dev and training records and refuses validation
// duplicates; the two pools are disjoint by `readDataset`'s own checks).
//
// As in the loop's pool, a team starts the battle as the engine's validator
// normalises the paste's sets, and a team with an excluded ability (the run
// config's list, a stop-gap for a reducer gap) is left out.

import {
  type Engine,
  importPaste,
  type PokemonSet,
  ProgressError,
  type ProgressPool,
  resolveFormat,
  syntheticTeamHashes,
  teamHash,
  type TrainingRunConfig,
  validateTeam,
} from "@areneon/showdown-bridge";
import { datasetHash, readDataset } from "./dataset-reader.ts";
import { loadFixtureInputs } from "./fixtures.ts";

/** The pool a run config's `team_dataset` names, validation split only, sorted by canonical hash. */
export function datasetValidationPool(engine: Engine, config: TrainingRunConfig): ProgressPool {
  const named = config.team_dataset;
  if (named === undefined) throw new ProgressError("the run config names no team dataset");
  const dataset = readDataset(named.path, loadFixtureInputs(engine, config.format_id, named.path));
  if (datasetHash(dataset) !== named.dataset_hash) throw new ProgressError("the dataset's hash is not the one the run config names");
  const format = resolveFormat(engine, config.format_id);
  const excluded = new Set(config.generation.excluded_abilities);
  const teams = new Map<string, PokemonSet[]>();
  for (const record of dataset.splits.validation ?? []) {
    const played = importPaste(engine, record.paste); // readDataset already round-tripped the paste
    validateTeam(engine, format, played); // normalises in place
    if (!played.some((set) => excluded.has(set.ability))) teams.set(record.canonical_team_hash, played);
  }
  if (teams.size === 0) throw new ProgressError("the validation split is empty after the excluded abilities");
  const hashes = [...teams.values()].map(teamHash);
  return {
    dataset_hash: named.dataset_hash,
    teams: [...teams.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([canonical_team_hash, sets]) => ({ canonical_team_hash, sets })),
    known_synthetic: syntheticTeamHashes(hashes),
  };
}

/**
 * The hashes of every dev and training team of the run's own dataset (the splits its loop draws from, whether or not a team was
 * drawn): the canonical hash of each record and the played-form `teamHash` of its sets, in one set. An evaluation dataset
 * (ADR-020 E-17.1) must hold none of them, so a larger evaluation-only dataset cannot lend the run a team it trained on.
 * The validation split is left out: it is the run's own evaluation split, not training data.
 */
export function datasetTrainingHashes(engine: Engine, config: TrainingRunConfig): Set<string> {
  const named = config.team_dataset;
  if (named === undefined) throw new ProgressError("the run config names no team dataset");
  const dataset = readDataset(named.path, loadFixtureInputs(engine, config.format_id, named.path));
  if (datasetHash(dataset) !== named.dataset_hash) throw new ProgressError("the dataset's hash is not the one the run config names");
  const format = resolveFormat(engine, config.format_id);
  const hashes = new Set<string>();
  for (const record of [...dataset.splits.dev, ...(dataset.training ?? [])]) {
    hashes.add(record.canonical_team_hash);
    const played = importPaste(engine, record.paste);
    validateTeam(engine, format, played); // normalises in place, as the loop's pool does
    hashes.add(teamHash(played));
  }
  return hashes;
}
