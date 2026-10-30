// The TrainingRunConfig reader (ADR-010 §5): the run's seeds, generation size,
// model sizes and PPO settings are a reviewed JSON record, validated fail-closed
// and named only by the SHA-256 of its JCS bytes. The schema checks the shape;
// the three rules the contract profile cannot say are checked here, and again
// by the Python reader (`areneon_ml.learner.config`).

import { ajvFor } from "@areneon/schema/ajv";
import trainingRunConfigSchema from "@areneon/schema/contracts/v1/training-run-config.schema.json" with { type: "json" };
import type { TrainingRunConfig } from "@areneon/schema/types/v1/training-run-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { nodeSha256 } from "./player-channel.ts";

export type { TrainingRunConfig };

export class TrainingRunConfigError extends Error {
  override name = "TrainingRunConfigError";
}

const validConfig = ajvFor("contract").compile(trainingRunConfigSchema);

/**
 * Read a TrainingRunConfig: the schema, then the rules it cannot say: the
 * attention heads divide the token width, a Team Preview order does not name
 * more positions than a side holds, each scripted baseline appears once, and a warm start names different
 * checkpoints (a checkpoint holds the network of one role; the Battle role alone is allowed), a value warm-up needs a warm start, and so does a KL anchor, a shared trunk excludes a two-role warm start (a Battle-only one is allowed), and an explicit play temperature of 1,000,000 is refused (the default is omitted).
 */
export function readTrainingRunConfig(value: unknown): TrainingRunConfig {
  if (!validConfig(value)) throw new TrainingRunConfigError(`invalid TrainingRunConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as TrainingRunConfig;
  const { model, generation } = config;
  if (model.d_model % model.n_heads !== 0) {
    throw new TrainingRunConfigError(`d_model ${model.d_model} must be a multiple of n_heads ${model.n_heads}`);
  }
  if (model.encoder.max_picks > model.encoder.max_team) {
    throw new TrainingRunConfigError(`max_picks ${model.encoder.max_picks} cannot exceed max_team ${model.encoder.max_team}`);
  }
  const actors = generation.baseline_games.map((baseline) => baseline.actor);
  if (new Set(actors).size !== actors.length) throw new TrainingRunConfigError("a scripted baseline may be listed once");
  if (config.warm_start?.preview !== undefined && config.warm_start.preview === config.warm_start.battle) {
    throw new TrainingRunConfigError("warm_start names one checkpoint for both roles: a checkpoint holds one role's network");
  }
  if (config.warm_start?.preview !== undefined && config.model.options?.shared_trunk !== undefined) {
    throw new TrainingRunConfigError("model.options.shared_trunk cannot be combined with a warm_start that names the preview role: two cloned trunks conflict (use a Battle-only warm start)");
  }
  if (config.ppo.value_warmup !== undefined && config.warm_start === undefined) {
    throw new TrainingRunConfigError("ppo.value_warmup needs warm_start: only a cloned start has an untrained value head to warm up");
  }
  if (config.ppo.anchor !== undefined && config.warm_start === undefined) {
    throw new TrainingRunConfigError("ppo.anchor needs warm_start: the anchor is the warm-start pair");
  }
  if (generation.play?.temperature_ppm === 1_000_000) {
    throw new TrainingRunConfigError("generation.play.temperature_ppm 1000000 is the default: omit generation.play (one rule, one encoding)");
  }
  return config;
}

/** The hash `CheckpointRecord.run_config_hash` holds: SHA-256 of the record's JCS bytes. */
export const runConfigHash = (config: TrainingRunConfig): Promise<string> =>
  nodeSha256(new TextEncoder().encode(jcs(config as unknown as Json)));
