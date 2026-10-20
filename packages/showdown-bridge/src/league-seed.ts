// League seed derivation (ADR-009 §3, ADR-010 §4 protocol D9): every number a
// league game is built from is a pure function of the run key, the generation
// and the game index, so a game, and a generation that was killed, replay from
// the TrainingRunConfig alone. The derivation is named in the LeagueRunRecord
// (`LEAGUE_SEED_DERIVATION`); a change to any byte it hashes is a new version.
//
// These values are the harness's and the evaluation's: a seed or a stream
// label never reaches a policy worker or an observation. The worker is given
// only the sampler's draw `u`, which `sampleU` derives from a seat's label.

import { createHash } from "node:crypto";
import { type Json, jcs } from "./canonical-json.ts";

/** Names the derivation below in every LeagueRunRecord. A new byte layout gets a new number. */
export const LEAGUE_SEED_DERIVATION = "league-seed/1";

/** What a league game is built from. All of it comes from one digest. */
export interface GameStreams {
  /** The battle seed, in the engine's `sodium,<32 hex>` form: the digest's first 16 bytes. */
  seed: string;
  /** The two sides' team draws, unsigned 32-bit integers (digest bytes 16 to 23); the caller takes them modulo its pool size. */
  team_draws: { p1: number; p2: number };
  /** Seeds a scripted baseline's stream, an unsigned 32-bit integer (digest bytes 24 to 27). */
  baseline_seed: number;
}

const requireIndex = (name: string, value: number, least: number): void => {
  if (!Number.isSafeInteger(value) || value < least) throw new RangeError(`${name} must be an integer of at least ${least}`);
};

/**
 * SHA-256 over the JCS of `{run_key, generation, game, purpose: "league-game"}`,
 * cut into the three values above. `generation` counts from 1 and `game` from
 * 0 within the generation's match plan.
 */
export function leagueGameStreams(run_key: string, generation: number, game: number): GameStreams {
  requireIndex("generation", generation, 1);
  requireIndex("game", game, 0);
  const digest = createHash("sha256").update(jcs({ run_key, generation, game, purpose: "league-game" } as Json)).digest();
  return {
    seed: `sodium,${digest.subarray(0, 16).toString("hex")}`,
    team_draws: { p1: digest.readUInt32BE(16), p2: digest.readUInt32BE(20) },
    baseline_seed: digest.readUInt32BE(24),
  };
}

/** A league game's battle seed. */
export const leagueBattleSeed = (run_key: string, generation: number, game: number): string => leagueGameStreams(run_key, generation, game).seed;

/**
 * The label of one seat's policy stream (protocol D9, with the game index
 * recorded in the protocol erratum): the battle seed, the game index, the seat
 * and the seat's policy id. Two seats, two policies, or two games that share a
 * battle seed never share a stream. The sampler adds the request number.
 */
export interface PolicyStreamLabel {
  seed: string;
  game: number;
  player: "p1" | "p2";
  policy_id: string;
}

export const policyStreamLabel = (seed: string, game: number, player: "p1" | "p2", policy_id: string): PolicyStreamLabel => ({ seed, game, player, policy_id });
