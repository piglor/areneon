// The identity of the damage-aware baseline (ADR-018 section 5). Its policy id is `damage-aware@<commit>`
// like every baseline's, but its play also depends on the config records, the probe's reading, the
// public-state reader and the pinned engine, so every game also records a baseline identity hash.
// Two games with different hashes are different baselines and are never pooled; a change of any part
// is a new baseline, never an edit.

import { type Json, jcs } from "./canonical-json.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import { damageAwareConfigHash } from "./damage-aware-config.ts";
import type { VersionManifest } from "./formats.ts";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { MECHANICS_ADAPTER_FORMAT } from "./mechanics-result.ts";
import { sha256Hex } from "./rules.ts";

export const DAMAGE_AWARE_NAME = "damage-aware";

/** The five parts a baseline's play depends on besides its policy id. */
export interface DamageAwareIdentity {
  baseline_config_hash: string;
  mechanics_config_hash: string;
  adapter_format: number;
  reader_sha256: string;
  version_manifest: VersionManifest;
}

export interface IdentityParts {
  config: DamageAwareConfig;
  mechanics: MechanicsConfig;
  manifest: VersionManifest;
  /** SHA-256 of the public-state reader's file (`readerSha256()` of `@areneon/public-state`). */
  reader_sha256: string;
}

export const damageAwareIdentity = (parts: IdentityParts): DamageAwareIdentity => ({
  baseline_config_hash: damageAwareConfigHash(parts.config),
  mechanics_config_hash: mechanicsConfigHash(parts.mechanics),
  adapter_format: MECHANICS_ADAPTER_FORMAT,
  reader_sha256: parts.reader_sha256,
  version_manifest: parts.manifest,
});

/** The baseline identity hash: SHA-256 of the JCS bytes of the identity. */
export const damageAwareIdentityHash = (identity: DamageAwareIdentity): string => sha256Hex(jcs(identity as unknown as Json));

const TAG = /^[A-Za-z0-9._-]+$/;

/** `damage-aware@<commit>`; the tag is the run config's `<name>@<commit>` part. */
export function damageAwarePolicyId(commit: string): string {
  if (!TAG.test(commit)) throw new RangeError("a policy id's commit part is letters, digits, `.`, `_` or `-`");
  return `${DAMAGE_AWARE_NAME}@${commit}`;
}
