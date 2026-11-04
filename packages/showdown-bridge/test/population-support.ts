import { readFileSync } from "node:fs";
import { join } from "node:path";
import { SAMPLES_DIR } from "../src/samples.ts";
import { readTrainingRunConfig, type TrainingRunConfig } from "../src/training-run-config.ts";

export const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/valid/${name}.json`), "utf8"));

/** The tiny config as a population of `members` with the given seats and shares (no past pool unless asked). */
export function populationConfig(members: number, seats: number, shares: [number, number, number] = [500_000, 500_000, 0], games = 2): TrainingRunConfig {
  const base = sample("population") as { population: Record<string, unknown>; generation: Record<string, unknown> };
  const raw = {
    ...base,
    generation: { ...base.generation, baseline_games: [{ actor: "random-legal", policy_id: "random-legal@tiny", games }] },
    population: {
      ...base.population,
      members,
      seats_per_member: seats,
      shares: { ladder_ppm: shares[0], self_play_ppm: shares[1], past_pool_ppm: shares[2] },
      selection: { ...(base.population.selection as object), min_roots: 1 },
      explore: { bounds: { ...((base.population.explore as { bounds: object }).bounds), ladder_ppm: { min: 0, max: 1_000_000 }, self_play_ppm: { min: 0, max: 1_000_000 }, past_pool_ppm: { min: 0, max: 1_000_000 } } },
    },
  };
  return readTrainingRunConfig(raw);
}


export const populationOf = (config: TrainingRunConfig): NonNullable<TrainingRunConfig["population"]> => {
  if (config.population === undefined) throw new Error("no population");
  return config.population;
};
