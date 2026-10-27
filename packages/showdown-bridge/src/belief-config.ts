// The belief filter's reviewed configuration (ADR-025 sections 3 and 5): reviewed data, never code.
// The schema checks the shape; the reader adds sorted and unique ids and that every identity-hiding
// ability is an ability of the team vocabulary. `beliefConfigHash` is the `belief_config_hash` of
// the belief identity.

import { ajvFor } from "@areneon/schema/ajv";
import beliefConfigSchema from "@areneon/schema/contracts/v1/belief-config.schema.json" with { type: "json" };
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

export type { BeliefConfig };

export class BeliefConfigError extends Error {
  override name = "BeliefConfigError";
}

const validConfig = ajvFor("contract").compile(beliefConfigSchema);

/** Read a config: the schema, strictly ascending ability ids, and each id in the vocabulary. */
export function readBeliefConfig(value: unknown, vocabulary: TeamVocabulary): BeliefConfig {
  if (!validConfig(value)) throw new BeliefConfigError(`invalid BeliefConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as BeliefConfig;
  const known = new Set(vocabulary.roster.flatMap((r) => r.abilities));
  config.identity_hiding_abilities.forEach((id, k) => {
    if (k > 0 && !((config.identity_hiding_abilities[k - 1] as string) < id)) throw new BeliefConfigError(`identity_hiding_abilities: ${id} is repeated or out of order`);
    if (!known.has(id)) throw new BeliefConfigError(`ability ${id} is not in the team vocabulary`);
  });
  return config;
}

/** The hash an identity names its config by: SHA-256 of the config's JCS bytes. */
export const beliefConfigHash = (config: BeliefConfig): string => sha256Hex(jcs(config as unknown as Json));
