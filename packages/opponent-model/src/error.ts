// The typed refusal of the belief filter (ADR-025 section 3): a consumer catches it, falls back and
// counts it by code. A refusal is never a guess: the filter either holds a sound belief or throws.

export const BELIEF_ERROR_CODES = [
  "reader_flagged", // a line the shared reader could not take (its known limit is an Illusion twin)
  "identity_hiding_species", // a previewed species that can hide its identity (the Illusion-aware belief is part 2)
  "format_mismatch", // the prior or config is for another format or engine data than the vocabulary
  "item_clause_unsupported", // a rule table value other than "absent" or 1
  "team_size", // the opponent roster is not a team of the format, or the picked size does not fit it
  "species_not_in_prior", // a previewed species the usage prior has no entry for
  "forme_not_in_prior", // a shown forme (or Mega Evolution) no forme of the species' prior entry explains
  "brought_exceeds_picked", // more settled Pokemon than the picked team size
  "evidence_has_no_support", // the evidence leaves no world with a positive weight
  "empty_distribution", // a prior list with no mass at all (every weight unmapped)
] as const;

export type BeliefErrorCode = (typeof BELIEF_ERROR_CODES)[number];

export class BeliefError extends Error {
  override name = "BeliefError";
  readonly code: BeliefErrorCode;
  /** The 0-based roster position the refusal is about, when it is about one. */
  readonly slot: number | null;
  constructor(code: BeliefErrorCode, message: string, slot: number | null = null) {
    super(message);
    this.code = code;
    this.slot = slot;
  }
}
