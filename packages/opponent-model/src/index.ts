// The opponent model (ADR-025): an engine-free belief over the opponent's hidden team, computed
// exactly in integers from one player's own view. Slice 2, part 1: `belief` for every team
// without an identity-hiding species. `sample`, the legality port and the belief identity hash
// are later parts.

export { belief } from "./belief.ts";
export { BELIEF_ERROR_CODES, BeliefError, type BeliefErrorCode } from "./error.ts";
export { EVIDENCE_FIELDS, IGNORED_FIELDS, slotEvidence, type SlotEvidence } from "./evidence.ts";
export { solveItemClause, type ClauseResult, type ClauseWeights } from "./item-clause.ts";
export { elementary, moveModel, type MoveModel } from "./moves.ts";
export { type Ratio } from "./ratio.ts";
export { topWorlds, type TopWorld } from "./top-worlds.ts";
export { BELIEF_FORMAT, type BeliefInput, type BeliefVocabulary, type Cell, type FormeBelief, type Marginal, type OpponentBelief, type SlotBelief } from "./types.ts";
