// Compile-time checks for quicktype output (tsc --noEmit).
import type { DecisionRecord, Identity, Observation } from "../dist/ts/quicktype.ts";

// quicktype merges the oneOf branches into one interface: after narrowing on
// `kind`, branch-only fields are still optional.
export const logLength = (o: Observation): number =>
  // @ts-expect-error public_log is possibly undefined even when kind === "battle"
  o.kind === "battle" ? o.public_log.length : 0;

const id: Identity = { request_seq: 1, request_hash: "0".repeat(64) };
// @ts-expect-error unknown field (excess property check)
export const extra: Identity = { ...id, omniscient_log: [] };

declare const decision: DecisionRecord;
// const false is widened to boolean: no error here.
export const autoChoose: DecisionRecord = { ...decision, server_auto_choose_detected: true };
