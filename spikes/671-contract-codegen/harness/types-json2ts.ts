// Compile-time checks for json-schema-to-typescript output (tsc --noEmit).
// Every `@ts-expect-error` must fire, or tsc fails ("unused directive").
import type {
  DecisionRecord,
  Observation,
  PreviewObservation,
  RequestIdentity,
} from "../dist/ts/json2ts.ts";

// The const-tagged oneOf is a discriminated union: narrowing on `kind` works.
export const logLength = (o: Observation): number =>
  o.kind === "battle" ? o.public_log.length : o.opponent_preview.length;

const id: RequestIdentity = { request_seq: 1, request_hash: "0".repeat(64) };
// @ts-expect-error unknown field on a closed object (excess property check)
export const extra: RequestIdentity = { ...id, omniscient_log: [] };
// @ts-expect-error optional is not nullable (exactOptionalPropertyTypes)
export const rqidNull: RequestIdentity = { ...id, rqid: null };

declare const preview: PreviewObservation;
// @ts-expect-error unknown union tag
export const badTag: Observation = { ...preview, kind: "spectator" };

declare const decision: DecisionRecord;
const { bank_seconds_after: _bank, ...noBank } = decision;
// @ts-expect-error required-nullable field missing
export const missing: DecisionRecord = noBank;
export const nullable: DecisionRecord = { ...decision, bank_seconds_after: null };
// @ts-expect-error const false
export const autoChoose: DecisionRecord = { ...decision, server_auto_choose_detected: true };
