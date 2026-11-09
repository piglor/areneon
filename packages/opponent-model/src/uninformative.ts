// The uninformative belief mode (ADR-025 erratum E4, EXP-B1's same-shape control arm, #1309). The opponent sheet block and the
// `probe-damage/3` belief threats of this mode are computed from the PRIOR ALONE: the belief reads the opponent half as it stands at
// Team Preview, with no revealed move, item, ability, forme, level, gender, Mega, settled slot or flag, so every block and every
// threat list of a game is the same whatever the channel shows. The arm has the same shape as the informative one (same contract,
// same encoder columns, same k values) and none of its information about the game.
//
// The mode has its own identity: `uninformativeIdentityHash` replaces the belief identity in the feature set preimages, so the sheet
// source hash (`sheetFeatureSetHash`) and the `probe-damage/3` hash differ from the informative arm's and a checkpoint made for one
// cannot be read as the other's. The informative path is untouched (no byte or hash of it moves).

import type { PublicMon, PublicSide } from "@areneon/public-state";
import { digest } from "./canonical.ts";

/** Bumped when the blank opponent is built differently. Part of the uninformative identity. */
export const UNINFORMATIVE_FORMAT = "belief-uninformative/1";

/** The value of a factory context's `evidence` that selects this mode (absent: the informative belief). */
export const EVIDENCE_NONE = "none";

/** A roster entry as the Team Preview gives it, less the details the preview shows: the preview species and nothing else. */
function blank(mon: PublicMon): PublicMon {
  return {
    own: false,
    name: null,
    base_species: mon.base_species,
    species: null,
    temp_species: null,
    level: null,
    gender: null,
    shiny: false,
    hp_num: null,
    hp_den: null,
    status: null,
    fainted: false,
    brought: null,
    active_slot: null,
    boosts: {},
    volatiles: [],
    moves_revealed: [],
    item: null,
    item_known: false,
    item_lost: false,
    item_received: false,
    ability: null,
    ability_received: false,
    ability_suppressed: false,
    mega: false,
    mega_stone: null,
    stats: null,
    moves_own: null,
    pp: {},
    can_mega_evo: false,
  };
}

/**
 * The opponent half with nothing revealed: the preview species of each non-phantom roster entry, in roster order (a phantom entry is
 * a record of an Illusion twin, evidence by definition), no active slot, no condition. Pure; the input is not changed.
 */
export function uninformativeOpponent(opponent: PublicSide): PublicSide {
  return {
    player: opponent.player,
    username: null,
    teamsize: opponent.teamsize,
    roster: opponent.roster.filter((mon) => mon.phantom !== true).map(blank),
    slots: {},
    conditions: {},
  };
}

/** SHA-256 of the JCS of the mode's identity: the belief identity it reads the prior through, the mode and the builder format. */
export function uninformativeIdentityHash(belief_identity_hash: string): string {
  if (!/^[0-9a-f]{64}$/.test(belief_identity_hash)) throw new RangeError("belief_identity_hash is a SHA-256 in lower-case hexadecimal");
  return digest({ belief_identity_hash, evidence: EVIDENCE_NONE, uninformative_format: UNINFORMATIVE_FORMAT });
}
