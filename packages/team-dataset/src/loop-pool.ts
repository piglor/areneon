// The generation loop's team pool from a generated, verified dataset (ADR-010
// section 1, ADR-013). The bridge cannot import this package, so the loop's
// script (`scripts/play-generation.ts`) hands `datasetPoolSource` to
// `runGenerationCli`.
//
// Nothing is taken on trust: the directory goes through `readDataset`, its
// manifest hash must be the one the run config names, and `trainingPool` takes
// only the dev and training records and refuses validation duplicates. The
// `known_synthetic` set is built from the positional hash of the teams in the
// pool and nothing else, so no other team can be stamped synthetic through it.
//
// The team a battle starts with is the one the engine's validator normalises
// the paste's sets to (a set that names a Mega forme starts as its base forme
// with the base ability; the runner plays that normalised copy). The pool
// holds that form, because it is the team the player's own Team Preview
// request shows, which an observation is checked against. The stamp is by the
// hash of that form, derived from a verified record by the engine, never given.

import {
  type Engine,
  GenerationError,
  type GenerationTeamPool,
  importPaste,
  type PoolSource,
  type PokemonSet,
  resolveFormat,
  syntheticTeamHashes,
  teamHash,
  type TrainingRunConfig,
  validateTeam,
} from "@areneon/showdown-bridge";
import { drawTrainingTeam, readDataset, trainingPool } from "./dataset-reader.ts";
import { loadFixtureInputs } from "./fixtures.ts";

/** The pool a run config's `team_dataset` names, sorted by team hash so an index names the same team everywhere. */
export function datasetTeamPool(engine: Engine, config: TrainingRunConfig): GenerationTeamPool {
  const named = config.team_dataset;
  if (named === undefined) throw new GenerationError("the run config names no team dataset");
  const dataset = readDataset(named.path, loadFixtureInputs(engine, config.format_id));
  if (dataset.manifest_hash !== named.manifest_hash) throw new GenerationError("the dataset's manifest hash is not the one the run config names");
  const pool = trainingPool([dataset], engine);
  const format = resolveFormat(engine, config.format_id);
  const excluded = new Set(config.generation.excluded_abilities);
  const byHash = new Map<string, PokemonSet[]>();
  for (let index = 0; index < pool.records.length; index++) {
    const record = drawTrainingTeam(pool, index);
    // readDataset already round-tripped the paste, checked both hashes and had the validator accept it.
    const played = importPaste(engine, record.paste);
    validateTeam(engine, format, played); // normalises in place
    if (!played.some((set) => excluded.has(set.ability))) byHash.set(teamHash(played), played);
  }
  const hashes = [...byHash.keys()].sort();
  if (hashes.length === 0) throw new GenerationError("the team pool is empty after the excluded abilities");
  return { teams: hashes.map((hash) => byHash.get(hash) as PokemonSet[]), hashes, known_synthetic: syntheticTeamHashes(hashes) };
}

/** The port for `runGenerationCli`: the dataset pool when the config names a dataset, else the interim pool. */
export const datasetPoolSource =
  (interim: PoolSource): PoolSource =>
  (engine, config) =>
    config.team_dataset === undefined ? interim(engine, config) : datasetTeamPool(engine, config);
