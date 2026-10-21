export { AssignError, assignSplits, type PoolEntry } from "./assign-splits.ts";
export { checkTeamConfigs, labelProblems, TEAMS_CONFIG_DIR } from "./check-labels.ts";
export { recordHashes, recordsFile, writeDataset } from "./dataset-io.ts";
export { Binner, BinError, type Offer } from "./binner.ts";
export { type CliIo, runCli } from "./command.ts";
export { DedupeError, FileStage, KnownHashes, MemoryStage, referencePoolHash, sortedUnique } from "./dedupe.ts";
export { type Accepted, canonicalOrder, type Draw, Drawer, type Rejected } from "./draw.ts";
export { type CoverageReport, generate, type GenerateInput, GenerateError, type GenerateResult, type ReferencePool } from "./generate.ts";
export { cellCount, cellLabels, cellOf, LabelError, labelTeam, type LabelValue } from "./labeller.ts";
export { type Pilot, PilotError, readPilot, runPilot } from "./pilot.ts";
export { groupKey, manifestPlan, planQuotas, PlanError, type QuotaPlan, type Split, SPLITS, type SplitSizes, type Stratum } from "./plan.ts";
export { Proposal, ProposalError } from "./proposal.ts";
export { fillQuantile, provision, type Provision } from "./provision.ts";
export { rangeSummary } from "./range.ts";
export { RejectionLog } from "./rejection-log.ts";
export { drawSeed, engineRng, pilotSeed, type Rng, StreamError } from "./stream.ts";
export {
  COMMITTED_SPLITS,
  type CommittedSplit,
  type DatasetContext,
  DatasetReadError,
  datasetHash,
  datasetSyntheticHashes,
  drawTrainingTeam,
  HASHES_FILE,
  type HashIndex,
  hashIndex,
  MANIFEST_FILE,
  readDataset,
  splitHashes,
  type SplitHashes,
  teamsFile,
  TrainingDrawError,
  type TrainingPool,
  trainingPool,
  type VerifiedDataset,
} from "./dataset-reader.ts";
export { datasetPoolDetail, datasetPoolSource, datasetTeamPool } from "./loop-pool.ts";
export {
  committedDatasetHashes,
  differing,
  FixtureError,
  type FixtureInputs,
  type FixtureRun,
  loadFixtureInputs,
  readFixtureRun,
  recordedPilot,
  regenerate,
  writeFixtures,
} from "./fixtures.ts";
export {
  fillSealed,
  type FillArgs,
  type FillResult,
  generateSealedSet,
  type KnownTeam,
  loadSealedTeamHashes,
  NearIndex,
  readStoredSet,
  referenceTeams,
  SEALED_FILES,
  type SealedGenerateInput,
  type SealedGenerateResult,
  SealedError,
  type SealedPlan,
  sealedBudget,
  sealedPlan,
  sealedQuotas,
  SealedShortfallError,
  type SealedTeam,
  SealedTeamHashes,
  setDirOf,
  type StoredSet,
  teamDistance,
} from "./sealed.ts";
export { readRevealDir, REVEAL_FILES, retireSealedSet, revealHash, revealOf, SealedRetireError, SealedVerifyError, verifyRetiredSealed, type VerifyInput, type VerifyReport } from "./sealed-retire.ts";
export { assertPrivateStore, SealedStoreError } from "./sealed-store.ts";
export { runSealedCli, type SealedCliIo } from "./sealed-command.ts";
export * from "./shards/index.ts";
export { listPool, type ListedSet, type ListedTeam, type PoolListing } from "./pool-listing.ts";
