// The CloneRunConfig reader (ADR-019 track 1, ADR-010 erratum A2): the teacher
// TrainingSet, seeds, model sizes and cross-entropy settings of one supervised
// cloning run are a reviewed JSON record, validated fail-closed and named only
// by the SHA-256 of its JCS bytes (`CheckpointRecord.run_config_hash`). The
// schema checks the shape; the two model rules it cannot say are checked here,
// and again by the Python reader (`areneon_ml.learner.clone_config`).

import { ajvFor } from "@areneon/schema/ajv";
import cloneRunConfigSchema from "@areneon/schema/contracts/v1/clone-run-config.schema.json" with { type: "json" };
import type { CloneRunConfig } from "@areneon/schema/types/v1/clone-run-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { nodeSha256 } from "./player-channel.ts";

export type { CloneRunConfig };

export class CloneRunConfigError extends Error {
  override name = "CloneRunConfigError";
}

const validConfig = ajvFor("contract").compile(cloneRunConfigSchema);

/** Read a CloneRunConfig: the schema, then the heads divide the token width and the picks fit the team. */
export function readCloneRunConfig(value: unknown): CloneRunConfig {
  if (!validConfig(value)) throw new CloneRunConfigError(`invalid CloneRunConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as CloneRunConfig;
  const { model } = config;
  if (model.d_model % model.n_heads !== 0) {
    throw new CloneRunConfigError(`d_model ${model.d_model} must be a multiple of n_heads ${model.n_heads}`);
  }
  if (model.encoder.max_picks > model.encoder.max_team) {
    throw new CloneRunConfigError(`max_picks ${model.encoder.max_picks} cannot exceed max_team ${model.encoder.max_team}`);
  }
  return config;
}

/** The hash `CheckpointRecord.run_config_hash` holds for a cloned checkpoint: SHA-256 of the record's JCS bytes. */
export const cloneRunConfigHash = (config: CloneRunConfig): Promise<string> =>
  nodeSha256(new TextEncoder().encode(jcs(config as unknown as Json)));
