// Print this checkout's rules snapshot hash and the engine commit it is
// built for, without validating or loading the engine:
//   node packages/showdown-bridge/scripts/rules-hash.ts
// check-rules.ts runs this file from old commits (a bare `git archive`, no
// node_modules) so each historical snapshot is rebuilt by its own builder.
// Keep it free of non-built-in imports.
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { buildSnapshot, readRuleRecords, type RuleRecords, rulesSnapshotHash } from "../src/rules.ts";

const raw = readRuleRecords();
const records = {
  formats: raw.formats.map(({ value }) => value),
  regulations: raw.regulations.map(({ value }) => value),
} as RuleRecords;
const snapshot = buildSnapshot(records, PINNED_SHOWDOWN_COMMIT);
console.log(JSON.stringify({ hash: rulesSnapshotHash(snapshot), showdown_commit: PINNED_SHOWDOWN_COMMIT }));
