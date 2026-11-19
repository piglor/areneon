// The engine-free core of the differential harness (ADR-030 groundwork G1): the backend port, the byte comparison, the corpus
// tallies, the admission ledger (`BackendAdmission`) and its reader, the fresh-corpus rule, the shadow sampler and the report.
// The pinned engine reaches it only through a level adapter and an oracle backend that the bridge declares structurally and a
// composition root (packages/team-dataset) joins; this package imports no engine, bridge or network module.
export type { Json } from "./canonical.ts";
export { CanonicalError, jcs } from "./canonical.ts";
export { key64, sha256Jcs } from "./hash.ts";
export type { Backend, Level, Settled } from "./backend.ts";
export { compareBatch, compareItem, firstDifference, HarnessError } from "./compare.ts";
export type { BatchComparison, Difference, DifferenceReason, ItemResult } from "./compare.ts";
export { addBatch, addItem, newTally, pairKey, TallyError } from "./tally.ts";
export type { CorpusKind, CorpusSpec, CorpusTally, EntityTally, PairTally } from "./tally.ts";
export { runCorpus, RunError } from "./harness.ts";
export {
  ADMISSION_RULE,
  AdmissionError,
  admissionHash,
  admissionRouter,
  admits,
  admitsPair,
  buildLedger,
  FRESH_PURPOSE,
  freshCorpus,
  freshCorpusId,
  freshSeed,
  frequent,
  N_EXERCISED,
  N_PAIR,
  PAIR_ODDS,
  readBackendAdmission,
  withCorpus,
} from "./admission.ts";
export type { BackendAdmission, BackendIdentity, LedgerInput, Refusal } from "./admission.ts";
export { LEVEL_1_RULE, LEVEL_2_RULE, SHADOW_PURPOSE, ShadowDivergence, shadowCheck, shadowKey, shadowSample, shadowSize } from "./shadow.ts";
export type { SizeRule } from "./shadow.ts";
export { EntityError, mechanicsEntities, mechanicsExercised, mechanicsLevel } from "./mechanics-level.ts";
export type { MechanicsDescriptor, MechanicsFacts, MonFacts } from "./mechanics-level.ts";
export { differentialReport, HARNESS_FORMAT, LISTED_DIFFERENCES, renderReport } from "./report.ts";
export type { CorpusSummary, DifferentialReport, EntitySummary } from "./report.ts";
