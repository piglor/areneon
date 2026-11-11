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
  return derive(run_key, generation, game);
}

/**
 * The same derivation at generation 0, which no training plan uses: the anchor calibration of a population run
 * (ADR-021 section 4, slice 7) draws its pairs here, so a calibration pair never shares a seed with a training pair.
 * `pair` counts from 0 within the calibration plan.
 */
export const calibrationGameStreams = (run_key: string, pair: number): GameStreams => derive(run_key, 0, pair);

function derive(run_key: string, generation: number, game: number): GameStreams {
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

/** Names the team-source draw below: which component (the synthetic pool or the private one) a seat's team comes from. */
export const LEAGUE_TEAM_SOURCE_PURPOSE = "league-team-source/1";

/**
 * Whether one seat of one game takes its team from the private component (ADR-015 erratum E1 item 3): SHA-256 over the JCS of
 * `{run_key, generation, game, seat, purpose: "league-team-source/1"}`, its first four bytes read big-endian as n, private when
 * `n * 1000 < share_permille * 2^32` (exact in integers; 0 never, 1000 always). It is a separate digest from `league-seed/1`, so
 * the battle seed, the team draws and the baseline stream of every game are what they were without a private pool.
 */
export function privateTeamSource(run_key: string, generation: number, game: number, seat: "p1" | "p2", share_permille: number): boolean {
  requireIndex("generation", generation, 1);
  requireIndex("game", game, 0);
  if (!Number.isInteger(share_permille) || share_permille < 0 || share_permille > 1000) throw new RangeError("a share is a whole number of permille from 0 to 1000");
  const digest = createHash("sha256").update(jcs({ run_key, generation, game, seat, purpose: LEAGUE_TEAM_SOURCE_PURPOSE } as Json)).digest();
  return digest.readUInt32BE(0) * 1000 < share_permille * 2 ** 32;
}
