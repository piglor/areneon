// The private team store (ADR-015). It is never shipped: no image, Coolify
// build or other workspace package may depend on it (test/packaging.test.ts).
// Built so far: the pure import planner (import-plan.ts, staged-file.ts) and
// the private snapshot writer and reader (snapshot-writer.ts, snapshot-reader.ts).
export const PRIVATE_TEAM_STORE_PACKAGE = "@areneon/private-team-store";
export { IMPORT_REFUSALS, ImportPlanError, type ImportRefusal, type RefusalPosition } from "./import-error.ts";
export {
  assertNoActiveRecordLost,
  type ChangeSummary,
  type ImportPlan,
  type ImportPlanInput,
  planImport,
} from "./import-plan.ts";
export { MAX_TAGS, parseStagedFile, type ParsedStagedFile, type StagedFile } from "./staged-file.ts";
export { SNAPSHOT_IO_REFUSALS, SnapshotIoError, type SnapshotIoPosition, type SnapshotIoRefusal } from "./snapshot-io-error.ts";
export { type ReadSnapshotInput, readPrivateSnapshot, type SnapshotRead } from "./snapshot-reader.ts";
export { labelPlan } from "./snapshot-plan.ts";
export { type WriteSnapshotInput, type WrittenSnapshot, writePrivateSnapshot } from "./snapshot-writer.ts";
export { assertWorkDir, type WorkDirProbe } from "./work-dir.ts";
