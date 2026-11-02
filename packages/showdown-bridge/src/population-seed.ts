// Population seed derivation (ADR-021 section 5): every draw of a population
// run that is not a battle seed or a team draw (those keep `league-seed/1`)
// is a pure function of the run key, the generation, the member, a purpose and
// an index, so a population generation, and one that was killed, replay from
// the TrainingRunConfig and the committed bytes alone. A change to any byte it
// hashes is a new version.
//
// Like `league-seed/1`, these values are the harness's: none reaches a policy
// worker or an observation.

import { createHash } from "node:crypto";
import { type Json, jcs } from "./canonical-json.ts";

/** Names the derivation below. A new byte layout gets a new number. */
export const POPULATION_SEED_DERIVATION = "population-seed/1";

/** What a draw is for. Each purpose has its own stream, so no two uses share a value. */
export const POPULATION_SEED_PURPOSES = ["member-init", "member-learner", "opponent-draw", "explore", "bootstrap"] as const;
export type PopulationSeedPurpose = (typeof POPULATION_SEED_PURPOSES)[number];

/** A learning member `m<k>` or an exploiter seat `x<k>` (ADR-021 section 2), k a decimal without leading zeros. */
const MEMBER = /^[mx](0|[1-9][0-9]{0,1})$/;

const requireIndex = (name: string, value: number, least: number): void => {
  if (!Number.isSafeInteger(value) || value < least) throw new RangeError(`${name} must be an integer of at least ${least}`);
};

/**
 * SHA-256 over the JCS of `{run_key, generation, member, purpose, index}`; the
 * first 8 bytes, big endian, as an unsigned 64-bit integer. How a use reduces
 * it (modulo a count, a share of 2^64) is that use's rule. `generation` counts
 * from 0 (the initialisation draws are generation 0's), `index` from 0 within
 * a purpose.
 */
export function populationDraw(run_key: string, generation: number, member: string, purpose: PopulationSeedPurpose, index: number): bigint {
  requireIndex("generation", generation, 0);
  requireIndex("index", index, 0);
  if (!MEMBER.test(member)) throw new RangeError("member must be m<k> or x<k>");
  if (!POPULATION_SEED_PURPOSES.includes(purpose)) throw new RangeError("purpose must be one of the population seed purposes");
  const digest = createHash("sha256").update(jcs({ run_key, generation, member, purpose, index } as Json)).digest();
  return digest.readBigUInt64BE(0);
}
