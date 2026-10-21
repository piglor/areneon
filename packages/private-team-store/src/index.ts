// The private team store (ADR-015). It is never shipped: no image, Coolify
// build or other workspace package may depend on it (test/packaging.test.ts).
// Built so far: the pure import planner (import-plan.ts, staged-file.ts).
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
