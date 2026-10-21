// The private team store (ADR-015). It is never shipped: no image, Coolify
// build or other workspace package may depend on it (test/packaging.test.ts).
// Built so far: the pure import planner (import-plan.ts, staged-file.ts), the
// private snapshot writer and reader (snapshot-writer.ts, snapshot-reader.ts),
// the object store port with its in-memory and R2 implementations and the
// publish sequence (object-store.ts, memory-store.ts, r2-client.ts,
// r2-settings.ts, publish.ts). The import and export commands are not built.
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
export { type CurrentPointer, encodeCurrent, readCurrentPointer } from "./current-pointer.ts";
export { InMemoryObjectStore } from "./memory-store.ts";
export {
  CURRENT_KEY,
  LOCK_CHECK_PREFIX,
  MAX_OBJECT_BYTES,
  OBJECT_STORE_REFUSALS,
  type ObjectHead,
  type ObjectStore,
  ObjectStoreError,
  type ObjectStoreRefusal,
  snapshotKeys,
  SNAPSHOTS_PREFIX,
} from "./object-store.ts";
export { type FetchLike, R2ObjectStore, type R2ClientOptions } from "./r2-client.ts";
export { CREDENTIAL_REFUSALS, CredentialError, type CredentialRefusal, R2Settings } from "./r2-settings.ts";
export { type Published, type PublishInput, publishSnapshot } from "./publish.ts";
export { type ExpectedVersion, INSPECT_ONLY } from "./snapshot-reader.ts";
export { type FetchedSnapshot, type FetchSnapshotInput, fetchSnapshot, listSnapshotHashes, readCurrent } from "./snapshot-store.ts";
export { checkLocalCopyTarget, ensurePrivateDirectory, writeLocalCopy } from "./local-copy.ts";
