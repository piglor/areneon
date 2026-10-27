// The damage-aware baseline's assumption record (ADR-018 sections 3 to 5): reviewed data in
// config/baselines/damage-aware.<version>.json, never code. The schema checks the shape; the reader
// adds what the profile cannot say by checking every id against the pinned engine's team vocabulary,
// and `damageAwareConfigHash` is the `baseline_config_hash` of the baseline identity. The record
// holds no game rule: it only chooses among values the engine allows.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { ajvFor } from "@areneon/schema/ajv";
import configSchema from "@areneon/schema/contracts/v1/damage-aware-config.schema.json" with { type: "json" };
import configSchemaV2 from "@areneon/schema/contracts/v1/damage-aware-config-2.schema.json" with { type: "json" };
import type { DamageAwareConfig } from "@areneon/schema/types/v1/damage-aware-config.ts";
import type { DamageAwareConfigV2 } from "@areneon/schema/types/v1/damage-aware-config-2.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { sha256Hex } from "./rules.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

export type { DamageAwareConfig, DamageAwareConfigV2 };

/** Either record: v1 (field rules) or v2 (the belief). They share the format, the rounding, the scoring and the exploration rate. */
export type AnyDamageAwareConfig = DamageAwareConfig | DamageAwareConfigV2;

export const DAMAGE_AWARE_CONFIG_DIR = fileURLToPath(new URL("../../../config/baselines", import.meta.url));

export class DamageAwareConfigError extends Error {
  override name = "DamageAwareConfigError";
}

const validConfig = ajvFor("contract").compile(configSchema);
const validConfigV2 = ajvFor("contract").compile(configSchemaV2);

/**
 * Read a config: the schema (closed objects, integers only), then every id against `vocabulary`:
 * the format, the nature, a fixed item, and a Stat Point spread that fits the format's limit.
 */
export function readDamageAwareConfig(value: unknown, vocabulary: TeamVocabulary): DamageAwareConfig {
  if (!validConfig(value)) throw new DamageAwareConfigError(`invalid DamageAwareConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as DamageAwareConfig;
  if (config.format_id !== vocabulary.format_id) throw new DamageAwareConfigError("format_id is not the vocabulary's format");
  const { item, nature, stat_points } = config.assumed_set;
  if (!vocabulary.natures.some((entry) => entry.id === nature)) throw new DamageAwareConfigError(`nature ${nature} is not in the team vocabulary`);
  if (item.kind === "fixed" && !vocabulary.items.includes(item.item)) throw new DamageAwareConfigError(`item ${item.item} is not in the team vocabulary`);
  const total = Object.values(stat_points).reduce((sum, points) => sum + points, 0);
  if (total > vocabulary.rules.ev_limit) throw new DamageAwareConfigError(`stat_points sum to ${total}, over the format's limit of ${vocabulary.rules.ev_limit}`);
  return config;
}

/** The hash an identity names its config by: SHA-256 of the config's JCS bytes (`baseline_config_hash`). */
export const damageAwareConfigHash = (config: AnyDamageAwareConfig): string => sha256Hex(jcs(config as unknown as Json));

/** Read `<dir>/damage-aware.<version>.json` against `vocabulary`. */
export function loadDamageAwareConfig(version: string, vocabulary: TeamVocabulary, dir: string = DAMAGE_AWARE_CONFIG_DIR): DamageAwareConfig {
  return readDamageAwareConfig(JSON.parse(readFileSync(join(dir, `damage-aware.${version}.json`), "utf8")) as unknown, vocabulary);
}

/**
 * Read a v2 record: the schema, then the format against the vocabulary. The prior and the belief config
 * it names by hash are loaded by the composition root and checked against these hashes by the actor.
 */
export function readDamageAwareConfigV2(value: unknown, vocabulary: TeamVocabulary): DamageAwareConfigV2 {
  if (!validConfigV2(value)) throw new DamageAwareConfigError(`invalid DamageAwareConfigV2: ${JSON.stringify(validConfigV2.errors)}`);
  const config = structuredClone(value) as unknown as DamageAwareConfigV2;
  if (config.format_id !== vocabulary.format_id) throw new DamageAwareConfigError("format_id is not the vocabulary's format");
  return config;
}

/** Read `<dir>/damage-aware.v2.json` (the belief record) against `vocabulary`. */
export function loadDamageAwareConfigV2(vocabulary: TeamVocabulary, dir: string = DAMAGE_AWARE_CONFIG_DIR): DamageAwareConfigV2 {
  return readDamageAwareConfigV2(JSON.parse(readFileSync(join(dir, "damage-aware.v2.json"), "utf8")) as unknown, vocabulary);
}
