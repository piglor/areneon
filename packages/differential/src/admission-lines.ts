// The committed form of a BackendAdmission ledger (#1457): a header line (the ledger without its entities, pairs and
// revocations), then one line per entity, per pair and per revocation, in the ledger's own order, each the JCS bytes of
// `{"<kind>": value}`. It reads back to the very ledger (so `admission_hash`, over the JCS of the whole ledger, is unchanged),
// diffs line by line, and a format bump adds no single-line blob. The Python reader (`areneon_contracts.backend_admission`)
// reads the same lines.
import { type BackendAdmission, AdmissionError, readBackendAdmission } from "./admission.ts";
import { type Json, jcs } from "./canonical.ts";

const SECTIONS = ["entity", "pair", "revoked"] as const;
const FIELD = { entity: "entities", pair: "pairs", revoked: "revoked" } as const;

/** The lines of a ledger (each ends with a newline). */
export function admissionLines(ledger: BackendAdmission): string {
  const { entities, pairs, revoked, ...header } = ledger;
  const lines = [jcs({ header } as unknown as Json), ...entities.map((v) => jcs({ entity: v } as unknown as Json)), ...pairs.map((v) => jcs({ pair: v } as unknown as Json)), ...revoked.map((v) => jcs({ revoked: v } as unknown as Json))];
  return `${lines.join("\n")}\n`;
}

/** Read the lines back: a header, then entities, pairs and revocations in that order, every line canonical; then the ledger's reader. */
export function readAdmissionLines(text: string, expected: { showdown_commit: string }): BackendAdmission {
  const fail = (message: string): never => {
    throw new AdmissionError(message);
  };
  if (!text.endsWith("\n")) fail("the admission lines end with a newline");
  const lines = text.slice(0, -1).split("\n");
  const parsed = lines.map((line) => {
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      return fail("an admission line is not JSON");
    }
    if (value === null || typeof value !== "object" || Array.isArray(value) || Object.keys(value).length !== 1) fail("an admission line holds one kind of value");
    if (jcs(value as Json) !== line) fail("an admission line is not canonical (RFC 8785)");
    return value as Record<string, unknown>;
  });
  const first = parsed[0];
  if (first === undefined || !("header" in first)) return fail("the admission lines start with the header");
  const ledger: Record<string, unknown> = { ...(first.header as object), entities: [], pairs: [], revoked: [] };
  let section = 0;
  for (const line of parsed.slice(1)) {
    const kind = Object.keys(line)[0] as string;
    const at = (SECTIONS as readonly string[]).indexOf(kind);
    if (at < 0) fail("an admission line holds one kind of value: entity, pair or revoked after the header");
    if (at < section) fail("the admission lines hold entities, then pairs, then revocations, in that order");
    section = at;
    (ledger[FIELD[kind as (typeof SECTIONS)[number]]] as unknown[]).push(line[kind]);
  }
  return readBackendAdmission(ledger, expected);
}
