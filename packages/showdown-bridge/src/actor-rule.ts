import type { LegalChoiceSet } from "@areneon/schema/types/v1/legal-choice-set.ts";

/** The legal choice set has no candidate at all, so the actor rule has nothing to play. */
export class NoCandidateError extends Error {
  override name = "NoCandidateError";
}

/**
 * The shared A1/A2 actor rule (ADR-005 §1, ADR-006): play the first
 * non-pending candidate in canonical order, else the first pending one.
 * A pending candidate the engine refuses as `[Unavailable choice]` brings
 * a new request, so the rule never aborts while any candidate exists. It
 * throws `NoCandidateError` only for an empty set.
 *
 * `choices` and `pending` each keep the enumerator's canonical order, so a
 * pending candidate that comes earlier in that order is still passed over
 * for a non-pending one. Pure: it reads only the player's own legal choice
 * set, never a verdict on a pending candidate.
 */
export function actorRule(legal: Pick<LegalChoiceSet, "choices" | "pending">): string {
  const choice = legal.choices[0] ?? legal.pending[0];
  if (choice === undefined) throw new NoCandidateError("the legal choice set has no candidate");
  return choice;
}
