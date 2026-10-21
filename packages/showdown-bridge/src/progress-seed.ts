// Seeds and pairing of the progress evaluation (ADR-017 section 2): everything a
// progress game is built from is a pure function of the run key and the game
// index, so a game replays from the run key alone and every match, mode and
// generation is played on the same plan (protocol 5.3). The purpose string
// differs from `league-seed/1`'s, so no preimage is shared with a training
// game; the command also checks every seed against the run's committed plans.
//
// Like a league seed, these values are the harness's: a seed never reaches a
// policy worker or an observation.

import { createHash } from "node:crypto";
import { type Json, jcs } from "./canonical-json.ts";

/** Names the seed derivation in every ProgressEvaluationRecord. A new byte layout gets a new number. */
export const PROGRESS_SEED_DERIVATION = "progress-seed/1";
/** Names the pairing rule. */
export const PROGRESS_PAIRING = "progress-pairing/1";

/** What a progress game is built from. */
export interface ProgressStreams {
  /** The battle seed, in the engine's `sodium,<32 hex>` form: the digest's first 16 bytes. */
  seed: string;
  /** Seeds a scripted baseline's stream, an unsigned 32-bit integer (digest bytes 16 to 19, big endian). */
  baseline_seed: number;
}

/** SHA-256 over the JCS of `{run_key, game, purpose: "progress-game"}`, cut into the two values above. */
export function progressGameStreams(run_key: string, game: number): ProgressStreams {
  if (!Number.isSafeInteger(game) || game < 0) throw new RangeError("game must be an integer of at least 0");
  const digest = createHash("sha256").update(jcs({ run_key, game, purpose: "progress-game" } as Json)).digest();
  return { seed: `sodium,${digest.subarray(0, 16).toString("hex")}`, baseline_seed: digest.readUInt32BE(16) };
}

/** Disjoint ordered pairs of items: the order is by `progress-pairing/1`, the pairs are consecutive. */
export function progressPairs(run_key: string, teams: readonly string[]): [string, string][] {
  if (new Set(teams).size !== teams.length) throw new RangeError("a team may appear once in the progress pool");
  // The sort key is the digest, then the team hash: the order is total, so it never depends on the input order.
  const keyed = teams.map((team) => [createHash("sha256").update(jcs({ run_key, canonical_team_hash: team, purpose: "progress-pairing" } as Json)).digest("hex") + team, team] as const);
  const ordered = keyed.sort(([a], [b]) => (a < b ? -1 : 1)).map(([, team]) => team);
  const pairs: [string, string][] = [];
  for (let i = 0; i + 1 < ordered.length; i += 2) pairs.push([ordered[i] as string, ordered[i + 1] as string]);
  return pairs;
}
