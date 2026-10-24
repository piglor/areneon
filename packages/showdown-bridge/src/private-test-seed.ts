// Seeds, pairing and the commitment of the private test set (ADR-015 erratum
// E1, item 2). Everything a private test game is built from is a keyed
// function of the per-test 256-bit salt (HMAC-SHA256 under the salt's bytes)
// and a purpose string that no other derivation uses (`league-seed/1` and
// `progress-seed/1` hash a run key without a key), so a private test shares no
// preimage with a training or a progress game, and a rerun with the same salt
// replays every game. The salt stays in the private root; the record holds
// only the commitment. Like every seed, these values are the harness's: a seed
// never reaches a policy worker or an observation.

import { createHash, createHmac } from "node:crypto";
import { type Json, jcs } from "./canonical-json.ts";

/** Names the seed derivation in every PrivateTestRecord. A new byte layout gets a new number. */
export const PRIVATE_TEST_SEED_DERIVATION = "private-test-seed/1";
/** Names the pairing rule. */
export const PRIVATE_TEST_PAIRING = "private-test-pairing/1";

const SALT = /^[0-9a-f]{64}$/;

/** Refuse anything but 256 bits in lower-case hex. The message never repeats the value. */
export function checkSalt(salt: string): void {
  if (!SALT.test(salt)) throw new RangeError("the salt must be 64 lower-case hex digits");
}

const keyed = (salt: string, value: Json): Buffer => {
  checkSalt(salt);
  return createHmac("sha256", Buffer.from(salt, "hex")).update(jcs(value)).digest();
};

/** What a private test game is built from. */
export interface PrivateTestStreams {
  /** The battle seed, in the engine's `sodium,<32 hex>` form: the digest's first 16 bytes. */
  seed: string;
  /** Seeds a scripted baseline's stream, an unsigned 32-bit integer (digest bytes 16 to 19, big endian). */
  baseline_seed: number;
}

/** HMAC-SHA256 under the salt over the JCS of `{pair, purpose: "private-test-seed/1"}`, cut into the two values above. */
export function privateTestStreams(salt: string, pair: number): PrivateTestStreams {
  if (!Number.isSafeInteger(pair) || pair < 0) throw new RangeError("pair must be an integer of at least 0");
  const digest = keyed(salt, { pair, purpose: "private-test-seed" });
  return { seed: `sodium,${digest.subarray(0, 16).toString("hex")}`, baseline_seed: digest.readUInt32BE(16) };
}

/**
 * Disjoint ordered pairs of canonical team hashes: the order is HMAC-SHA256 under the salt over each hash, ties
 * (none in practice) by the hash, and the pairs are consecutive. An odd team out is left unpaired.
 */
export function privateTestPairs(salt: string, teams: readonly string[]): [string, string][] {
  if (new Set(teams).size !== teams.length) throw new RangeError("a team may appear once in the private test set");
  const ordered = teams
    .map((team) => [keyed(salt, { canonical_team_hash: team, purpose: "private-test-pairing" }).toString("hex") + team, team] as const)
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([, team]) => team);
  const pairs: [string, string][] = [];
  for (let i = 0; i + 1 < ordered.length; i += 2) pairs.push([ordered[i] as string, ordered[i + 1] as string]);
  return pairs;
}

/** SHA-256 of the JCS of `{salt, teams: sorted canonical team hashes}`: names the set without a team in the record. */
export function privateTestCommitment(salt: string, teams: readonly string[]): string {
  checkSalt(salt);
  if (new Set(teams).size !== teams.length) throw new RangeError("a team may appear once in the private test set");
  return createHash("sha256").update(jcs({ salt, teams: [...teams].sort() } as Json)).digest("hex");
}
