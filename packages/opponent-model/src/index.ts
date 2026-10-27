// The opponent model (ADR-025): an engine-free belief over the opponent's hidden team, computed
// exactly in integers from one player's own view. Slice 2, parts 1 and 2: `belief`, with the exact
// mixture over disguise assignments while an identity-hiding species is unsettled. `sample`, the
// legality port and the belief identity hash are later parts.

export { belief } from "./belief.ts";
export { enumerate as enumerateAssignments, hasMatching, type Options as AssignmentOptions } from "./disguise.ts";
export { BELIEF_ERROR_CODES, BeliefError, type BeliefErrorCode } from "./error.ts";
export { EVIDENCE_FIELDS, IGNORED_FIELDS, slotEvidence, type SlotEvidence } from "./evidence.ts";
export { solveItemClause, type ClauseResult, type ClauseWeights } from "./item-clause.ts";
export { elementary, moveModel, type MoveModel } from "./moves.ts";
export { type Ratio } from "./ratio.ts";
export { topWorlds, type TopWorld } from "./top-worlds.ts";
export { BELIEF_FORMAT, type BeliefInput, type BeliefVocabulary, type Cell, type DisguiseAssignment, type DisguiseMixture, type FormeBelief, type Marginal, type OpponentBelief, type SlotBelief, type SlotMarginal } from "./types.ts";
