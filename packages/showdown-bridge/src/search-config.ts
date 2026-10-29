// The decision-time search's reviewed configuration and its identity (ADR-026 section 5, slice R part 1).
//
// `SearchConfig` is reviewed data (`config/search/<algorithm>.v<N>.json`), never code; the schema checks the
// shape and `readSearchConfig` adds what the profile cannot say. `searchConfigHash` is the `search_config_hash`
// of the search identity, and `searchIdentityHash` the hash a search policy id carries as `search: <hash>`
// (`policyId` in `policy-protocol.ts`). The identity names EVERY input that changes what a search plays: the
// config, the belief identity, the mechanics and damage-aware configs, the reviewed world writers, the format
// version of each module whose behaviour the search depends on, the shared reader and the version manifest.
// Nothing here runs a search: the actor is part 2.

import { readFileSync } from "node:fs";
import { ajvFor } from "@areneon/schema/ajv";
import searchConfigSchema from "@areneon/schema/contracts/v1/search-config.schema.json" with { type: "json" };
import type { SearchConfig } from "@areneon/schema/types/v1/search-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import type { VersionManifest } from "./formats.ts";

export type { SearchConfig };

/** The label of a policy that plays with decision-time search (ADR-026 section 5). */
export const SEARCH_POLICY_LABEL = "learned pair with decision-time search (one-ply re-rank)";

/**
 * Bumped when the hypothetical-channel construction for the value-head leaf, or the non-learned HP and
 * knock-out leaf (`search-actor.ts`, `search-leaf.ts`), changes (ADR-026 section 5). The constant starts at 1 and is
 * the identity's `leaf_format`.
 */
export const LEAF_FORMAT = 1;

export class SearchConfigError extends Error {
  override name = "SearchConfigError";
}

const validConfig = ajvFor("contract").compile(searchConfigSchema);

const SEARCH_CONFIG_DIR = new URL("../../../config/search/", import.meta.url);

/**
 * Read a config: the schema, then the rules it cannot say. W_min cannot exceed N (a decision could never be
 * searched), the max-damage weight and the uniform draws must leave each uniform reply an exact share of the
 * rest (no rounding in the weights).
 */
export function readSearchConfig(value: unknown): SearchConfig {
  if (!validConfig(value)) throw new SearchConfigError(`invalid SearchConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as SearchConfig;
  if (config.min_valid_worlds > config.worlds.n) throw new SearchConfigError("min_valid_worlds exceeds the worlds per decision");
  const { max_damage_ppm, uniform_draws } = config.replies;
  if (uniform_draws === 0 && max_damage_ppm !== 1_000_000) throw new SearchConfigError("with no uniform draw the max-damage reply must carry the whole weight");
  if (uniform_draws > 0 && (1_000_000 - max_damage_ppm) % uniform_draws !== 0) throw new SearchConfigError("the uniform replies must share the rest of the weight in exact millionths");
  return config;
}

/** The hash the search identity names the config by: SHA-256 of the config's JCS bytes. */
export const searchConfigHash = (config: SearchConfig): string => sha256Hex(jcs(config as unknown as Json));

/** Read `config/search/<name>.json` (default `rerank-1.v1`) for a format. */
export function loadSearchConfig(formatId: string, name = "rerank-1.v1", dir: URL = SEARCH_CONFIG_DIR): SearchConfig {
  const config = readSearchConfig(JSON.parse(readFileSync(new URL(`${name}.json`, dir), "utf8")) as unknown);
  if (config.format_id !== formatId) throw new SearchConfigError(`the config is for ${config.format_id}, not ${formatId}`);
  return config;
}

/** Everything that changes what a search plays, besides the config's own values (which `search_config_hash` covers). */
export interface SearchIdentityInputs {
  /** ADR-025's `belief_identity_hash` of the belief in use; must equal the config's. */
  belief_identity_hash: string;
  mechanics_config_hash: string;
  /** The probe's `MECHANICS_ADAPTER_FORMAT`: a probe answer read differently is another search. */
  mechanics_adapter_format: number;
  /** The reviewed weights `damageValue` uses (ADR-018). */
  damage_aware_config_hash: string;
  damage_value_format: number;
  world_writers_hash: string;
  world_builder_format: number;
  /** The public-state world-facts module's format string (`WORLD_FACTS_FORMAT`). */
  world_facts_format: string;
  reply_model_format: number;
  leaf_format: number;
  reader_sha256: string;
  version_manifest: VersionManifest;
}

const HEX64 = /^[0-9a-f]{64}$/;
const HASH_FIELDS = ["belief_identity_hash", "mechanics_config_hash", "damage_aware_config_hash", "world_writers_hash", "reader_sha256"] as const;

/** The identity preimage: the inputs plus `search_config_hash`, checked. */
export function searchIdentity(config: SearchConfig, inputs: SearchIdentityInputs): Record<string, Json> {
  for (const field of HASH_FIELDS) if (!HEX64.test(inputs[field])) throw new SearchConfigError(`${field} is a SHA-256: 64 lowercase hex digits`);
  if (inputs.belief_identity_hash !== config.belief_identity_hash) throw new SearchConfigError("the belief identity is not the one the config names");
  for (const field of ["mechanics_adapter_format", "damage_value_format", "world_builder_format", "reply_model_format", "leaf_format"] as const) {
    if (!Number.isInteger(inputs[field]) || inputs[field] < 1) throw new SearchConfigError(`${field} is a positive integer`);
  }
  if (typeof inputs.world_facts_format !== "string" || inputs.world_facts_format === "") throw new SearchConfigError("world_facts_format is a non-empty string");
  return {
    search_config_hash: searchConfigHash(config),
    belief_identity_hash: inputs.belief_identity_hash,
    mechanics_config_hash: inputs.mechanics_config_hash,
    mechanics_adapter_format: inputs.mechanics_adapter_format,
    damage_aware_config_hash: inputs.damage_aware_config_hash,
    damage_value_format: inputs.damage_value_format,
    world_writers_hash: inputs.world_writers_hash,
    world_builder_format: inputs.world_builder_format,
    world_facts_format: inputs.world_facts_format,
    reply_model_format: inputs.reply_model_format,
    leaf_format: inputs.leaf_format,
    reader_sha256: inputs.reader_sha256,
    version_manifest: inputs.version_manifest as unknown as Json,
  };
}

/** SHA-256 of the identity's JCS bytes: the `search` member of a search policy id. */
export const searchIdentityHash = (config: SearchConfig, inputs: SearchIdentityInputs): string => sha256Hex(jcs(searchIdentity(config, inputs) as Json));
