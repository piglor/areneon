// Speed-order evidence (ADR-025 section 7, ticket #1133; finding N9, owner answer Q6). In a turn with nothing that changes
// Speed before both actions, the order in which an opponent Pokemon and one of the player's own Pokemon used moves of the
// same priority is evidence about the opponent's Speed-relevant fields: nature, Speed Stat Points, item (Choice Scarf) and
// ability (Swift Swim, Chlorophyll). The check is the ADR-016 `speed_order` probe behind the `SpeedCheck` port that the
// bridge implements over the pinned engine (this package stays engine-free): for a candidate set the port says whether the
// probe's order, with the engine's own tie groups, can give the order that was observed.
//
// It is the same indicator as the damage evidence and shares its candidate space and reweighing (`damage.ts`): a candidate that passes,
// or that is not checked (beyond the bound, an "other spread" that names no nature or Stat Points, an unmodelled candidate,
// a probe answer `unsupported` such as a random draw), keeps its weight; a failing candidate gets weight 0; the marginals
// are those of the exact posterior over the candidate space with every unchecked candidate passing (mean field), under all of
// a slot's orders at once (`applyJoint`: a turn can show up to four, and applying them one after the other multiplies the size
// of every number the belief then reads); an event that rules out every checked candidate, alone or with the events before it,
// is dropped and counted. No Speed formula, priority rule or tie rule is written here.

import { type DamageCandidate, type DamageCounts, type DamageForme, type DamageRules, type EvidenceCheck, type EvidenceEvent, applyJoint, emptyDamageCounts } from "./damage.ts";

/** One observed order that concerns one opponent slot: the slot's Pokemon acted `first` or `second` relative to one of the player's own. */
export interface SpeedEvent extends EvidenceEvent {
  readonly role: "first" | "second";
}

/** The port the bridge implements over the `speed_order` probe: `check` as for `DamageCheck`. */
export interface SpeedCheck extends EvidenceCheck<SpeedEvent> {
  readonly events: readonly SpeedEvent[];
}

/** The reviewed values the belief reads (`BeliefConfig.speed_evidence`): the same two as the damage evidence's. */
export type SpeedRules = DamageRules;

/** What the evidence did, counted per belief (the fields of `DamageCounts`, read for speed events). */
export type SpeedCounts = DamageCounts;

export const emptySpeedCounts = emptyDamageCounts;

export type { DamageCandidate as SpeedCandidate };

/** Apply the speed events of one slot to its formes; see `applyJoint`. The input formes are not changed. */
export function applySpeed<F extends DamageForme>(formes: readonly F[], species: string, events: readonly SpeedEvent[], check: SpeedCheck, rules: SpeedRules, counts: SpeedCounts): { formes: F[]; notes: string[] } {
  return applyJoint(formes, species, events, check, rules, counts);
}
