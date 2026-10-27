// Identity and determinism (ADR-025 section 5). `BELIEF_FORMAT` names how the filter and the sampler
// compute a value; `beliefIdentityHash` binds a belief to everything it is a function of:
//
//   SHA-256 of the JCS of {prior_hash, belief_config_hash, belief_format, vocabulary_hash,
//                          reader_sha256, version_manifest}
//
// Through `prior_hash` it binds the source and the truncation limits of the usage prior, through
// `vocabulary_hash` the Item Clause and team-size values, through `reader_sha256` the shared reader's file,
// through `belief_format` this package's arithmetic. A new prior month, filter version, belief config,
// reader change or pin bump is a new identity; nothing is edited in place.
//
// `beliefDigest` and `worldsDigest` are the SHA-256 of the canonical JSON of a belief and of a list of
// sampled worlds. The belief holds big integers with a per-slot scale, so its canonical encoding writes
// each bigint as its decimal string; the golden fixture pins both digests.

import type { VersionManifest } from "@areneon/schema/types/v1/version-manifest.ts";
import { digest } from "./canonical.ts";
import type { SampledWorlds } from "./sampler.ts";
import { BELIEF_FORMAT, type OpponentBelief } from "./types.ts";

export interface BeliefIdentityParts {
  /** SHA-256 of the JCS of the usage prior record. */
  readonly prior_hash: string;
  /** SHA-256 of the JCS of the `BeliefConfig`. */
  readonly belief_config_hash: string;
  /** The team vocabulary's hash. */
  readonly vocabulary_hash: string;
  /** SHA-256 of the shared reader's file (`readerSha256()` of `@areneon/public-state`). */
  readonly reader_sha256: string;
  readonly version_manifest: VersionManifest;
}

const HEX64 = /^[0-9a-f]{64}$/;

export function beliefIdentityHash(parts: BeliefIdentityParts): string {
  for (const key of ["prior_hash", "belief_config_hash", "vocabulary_hash", "reader_sha256"] as const) {
    if (!HEX64.test(parts[key])) throw new RangeError(`${key} is a SHA-256 in lower-case hexadecimal`);
  }
  return digest({
    prior_hash: parts.prior_hash,
    belief_config_hash: parts.belief_config_hash,
    belief_format: BELIEF_FORMAT,
    vocabulary_hash: parts.vocabulary_hash,
    reader_sha256: parts.reader_sha256,
    version_manifest: parts.version_manifest,
  });
}

/** SHA-256 of the canonical JSON of a belief (bigints as decimal strings). */
export const beliefDigest = (belief: OpponentBelief): string => digest(belief);

/** SHA-256 of the canonical JSON of the sampled worlds (index, weight, redraws, world). */
export const worldsDigest = (sampled: SampledWorlds): string => digest(sampled);
