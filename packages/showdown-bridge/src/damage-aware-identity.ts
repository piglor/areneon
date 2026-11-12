// The identity of the damage-aware baseline (ADR-018 section 5). Its policy id is `damage-aware@<commit>`
// like every baseline's, but its play also depends on the config records, the probe's reading, the
// public-state reader and the pinned engine, so every game also records a baseline identity hash.
// Two games with different hashes are different baselines and are never pooled; a change of any part
// is a new baseline, never an edit.

import { HISTORY_FACTS_FORMAT, historyFactsSha256, WORLD_FACTS_FORMAT } from "@areneon/public-state";
import { FEATURE_SET_OUTCOME, featureSetHash } from "./candidate-features.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { type AnyDamageAwareConfig, damageAwareConfigHash } from "./damage-aware-config.ts";
import type { VersionManifest } from "./formats.ts";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { OUTCOME_FORMAT } from "./mechanics-outcome.ts";
import { MECHANICS_ADAPTER_FORMAT } from "./mechanics-result.ts";
import { type OutcomeConfig, outcomeConfigHash } from "./outcome-config.ts";
import { sha256Hex } from "./rules.ts";

export const DAMAGE_AWARE_NAME = "damage-aware";

/** The version of the situational scorer of `damage-aware@v4` (`damage-aware-situational.ts`); bump it when it scores a switch or a stalling move differently for the same input. Part of the identity only for a record with `situational` weights, so v1, v2 and v3 keep their bytes. */
export const SITUATIONAL_SCORE_FORMAT = 1;

/** The version of the success-scaled scorer of `damage-aware@v5` (`damage-aware-success.ts` and the actor's scaling and tie rules); bump it when it scores or orders candidates differently for the same input. Part of the identity only for a record with `success_scaling`. */
export const SUCCESS_SCORE_FORMAT = 1;

/**
 * The parts a baseline's play depends on besides its policy id: five, and a sixth only when the config
 * record names a belief (ADR-025 section 5, ADR-018 erratum E-18.3), so v1's identity bytes are unchanged.
 */
export interface DamageAwareIdentity {
  baseline_config_hash: string;
  mechanics_config_hash: string;
  adapter_format: number;
  reader_sha256: string;
  version_manifest: VersionManifest;
  /** `belief_identity_hash` of the belief the record names; absent for v1. */
  belief_identity_hash?: string;
  /** `SITUATIONAL_SCORE_FORMAT`; present only for a record with `situational` weights (v4). */
  situational_format?: number;
  /** v4 only: its choices read the world facts (the Protect streak) and the `probe-damage/2` arrays, so their formats are part of the baseline. */
  world_facts_format?: string;
  /** v4 only: `feature_set_hash` of the `probe-damage/2` set this checkout computes for the record's own assumed-set config. */
  feature_set_hash?: string;
  /** `SUCCESS_SCORE_FORMAT`; present only for a record with `success_scaling` (v5). */
  success_format?: number;
  /** v5 only: its damage values and success chances declare the player's own channel history, read by this facts module (and the probe's `move_outcome` format). */
  history_facts_format?: string;
  outcome_format?: number;
}

export interface IdentityParts {
  config: AnyDamageAwareConfig;
  mechanics: MechanicsConfig;
  manifest: VersionManifest;
  /** SHA-256 of the public-state reader's file (`readerSha256()` of `@areneon/public-state`). */
  reader_sha256: string;
  /** Required exactly when the config names a belief (a v2 record). */
  belief_identity_hash?: string;
  /** Required exactly when the config has `success_scaling` (v5): the reviewed `probe-outcome/1` record. */
  outcome_config?: OutcomeConfig;
}

const HEX64 = /^[0-9a-f]{64}$/;

export function damageAwareIdentity(parts: IdentityParts): DamageAwareIdentity {
  const named = parts.config.schema_version === 2;
  if (named !== (parts.belief_identity_hash !== undefined)) throw new RangeError("a belief identity hash goes with a config record that names a belief, and only with it");
  if (parts.belief_identity_hash !== undefined && !HEX64.test(parts.belief_identity_hash)) throw new RangeError("belief_identity_hash is a SHA-256 in lower-case hexadecimal");
  const scaling = parts.config.schema_version === 1 && parts.config.success_scaling !== undefined;
  if (scaling !== (parts.outcome_config !== undefined)) throw new RangeError("an outcome config goes with a config record that has success scaling, and only with it");
  return {
    baseline_config_hash: damageAwareConfigHash(parts.config),
    mechanics_config_hash: mechanicsConfigHash(parts.mechanics),
    adapter_format: MECHANICS_ADAPTER_FORMAT,
    reader_sha256: parts.reader_sha256,
    version_manifest: parts.manifest,
    ...(parts.belief_identity_hash === undefined ? {} : { belief_identity_hash: parts.belief_identity_hash }),
    ...(parts.config.schema_version === 1 && parts.config.situational !== undefined && parts.outcome_config !== undefined
      ? {
          situational_format: SITUATIONAL_SCORE_FORMAT,
          success_format: SUCCESS_SCORE_FORMAT,
          history_facts_format: HISTORY_FACTS_FORMAT,
          outcome_format: OUTCOME_FORMAT,
          feature_set_hash: featureSetHash({
            feature_set: FEATURE_SET_OUTCOME,
            mechanics_config_hash: mechanicsConfigHash(parts.mechanics),
            assumed_set_config_hash: damageAwareConfigHash(parts.config),
            adapter_format: MECHANICS_ADAPTER_FORMAT,
            reader_sha256: parts.reader_sha256,
            version_manifest: parts.manifest,
            outcome_format: OUTCOME_FORMAT,
            history_facts_format: HISTORY_FACTS_FORMAT,
            history_facts_sha256: historyFactsSha256(),
            outcome_config_hash: outcomeConfigHash(parts.outcome_config),
          }),
        }
      : parts.config.schema_version === 1 && parts.config.situational !== undefined
      ? {
          situational_format: SITUATIONAL_SCORE_FORMAT,
          world_facts_format: WORLD_FACTS_FORMAT,
          feature_set_hash: featureSetHash({
            feature_set: "probe-damage/2",
            mechanics_config_hash: mechanicsConfigHash(parts.mechanics),
            assumed_set_config_hash: damageAwareConfigHash(parts.config),
            adapter_format: MECHANICS_ADAPTER_FORMAT,
            reader_sha256: parts.reader_sha256,
            version_manifest: parts.manifest,
          }),
        }
      : {}),
  };
}

/** The baseline identity hash: SHA-256 of the JCS bytes of the identity. */
export const damageAwareIdentityHash = (identity: DamageAwareIdentity): string => sha256Hex(jcs(identity as unknown as Json));

const TAG = /^[A-Za-z0-9._-]+$/;

/** `damage-aware@<commit>`; the tag is the run config's `<name>@<commit>` part. */
export function damageAwarePolicyId(commit: string): string {
  if (!TAG.test(commit)) throw new RangeError("a policy id's commit part is letters, digits, `.`, `_` or `-`");
  return `${DAMAGE_AWARE_NAME}@${commit}`;
}
