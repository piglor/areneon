// The opponent model (ADR-025): an engine-free belief over the opponent's hidden team, computed
// exactly in integers from one player's own view. Slice 2: `belief`, with the exact mixture over
// disguise assignments while an identity-hiding species is unsettled; the exact seeded sampler
// (`createSampler`: `sample`, `sampleWorlds`) behind the `SetLegality` and `SpreadSampler` ports; and the
// belief identity hash.

export { belief } from "./belief.ts";
export { enumerate as enumerateAssignments, hasMatching, type Options as AssignmentOptions } from "./disguise.ts";
export { applyDamage, applyJoint, emptyDamageCounts, type EvidenceCheck, type EvidenceEvent, type DamageCandidate, type DamageCheck, type DamageCounts, type DamageEvent, type DamageForme, type DamageRules } from "./damage.ts";
export { BitStream, INDEX_PURPOSE, STREAM_BYTES, TEAM_INDEX_PURPOSE, TEAM_PURPOSE, WORLD_PURPOSE, blocksOf, teamStream, worldStream, type BlockSource } from "./draw.ts";
export { BELIEF_ERROR_CODES, BeliefError, type BeliefErrorCode } from "./error.ts";
export { EVIDENCE_FIELDS, IGNORED_FIELDS, slotEvidence, trapEvidence, type SlotEvidence } from "./evidence.ts";
export { beliefDigest, beliefIdentityHash, worldsDigest, type BeliefIdentityParts } from "./identity.ts";
export { solveItemClause, type ClauseResult, type ClauseWeights } from "./item-clause.ts";
export { elementary, moveModel, type MoveModel } from "./moves.ts";
export { designDeviation, type FormeDeviation } from "./prior-deviation.ts";
export { STAT_IDS, createSampler, type OpponentTeam, type OpponentWorld, type SampledTeam, type SampledTeams, type SampledWorld, type SampledWorlds, type SamplerPorts, type SetLegality, type ShownIdentifiers, type Sampler, type SpreadSampler, type StatPoints, type WorldSet, type WorldSlot } from "./sampler.ts";
export { DEFAULT_ITERATIONS, SOLVER_FORMAT, SOLVER_KINDS, SolverError, solve as solvePreview, type Solution as PreviewSolution, type SolverKind } from "./preview-solver.ts";
export { type Ratio } from "./ratio.ts";
export { topWorlds, type TopWorld } from "./top-worlds.ts";
export { TrappingCheckError, trapExclusions, type TrapCandidate, type TrapExclusions, type TrapLevel, type TrappingCheck, type TrapSlot } from "./trapping.ts";
export { BELIEF_FORMAT, type BeliefInput, type BeliefVocabulary, type Cell, type DisguiseAssignment, type DisguiseMixture, type FormeBelief, type Marginal, type OpponentBelief, type SlotBelief, type SlotMarginal } from "./types.ts";
export { applySpeed, emptySpeedCounts, type SpeedCandidate, type SpeedCheck, type SpeedCounts, type SpeedEvent, type SpeedRules } from "./speed.ts";
export { OPPONENT_SHEET_FORMAT, SHEET_DEFECT_CODES, SHEET_FEATURE_SET, SHEET_K, SHEET_REFUSAL_CODES, opponentSheet, refusedSheet, roundPpm, sheetFeatureSetHash, sheetRefusalOf, type SheetK, type SheetRefusalCode } from "./sheet.ts";
