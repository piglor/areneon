import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalJson, type Json } from "./request-identity.ts";

// Pure snapshot building and hashing (ADR-007). This module imports only
// Node built-ins so the historical channel check can run an old commit's
// own builder from a bare `git archive`, without that commit's node_modules.
// Schema validation lives in rules-load.ts.

/** Areneon's own rules records (ADR-007), authored in the repo. */
export const DEFAULT_RULES_DIR = fileURLToPath(new URL("../../../rules", import.meta.url));

export const RULES_SCHEMA_VERSION = 1;

export interface ForbiddenRule {
  rule: string;
  reason: string;
}

/** `rules/formats/<format_id>.json`. */
export interface FormatRecord {
  schema_version: number;
  format_id: string;
  regulation: string;
  game_type: "doubles";
  season: { start: string; end: string };
  forbidden_rules: ForbiddenRule[];
}

/** `rules/regulations/<regulation>.json`. */
export interface RegulationRecord {
  schema_version: number;
  regulation: string;
  sources: { url: string; verified_on: string }[];
}

export interface RuleRecords {
  formats: FormatRecord[];
  regulations: RegulationRecord[];
}

/** One published, content-hashed set of rules. Its five keys are the whole hash preimage. */
export interface RulesSnapshot {
  schema_version: number;
  showdown_commit: string;
  formats: Record<string, Omit<FormatRecord, "schema_version" | "format_id">>;
  regulations: Record<string, Omit<RegulationRecord, "schema_version" | "regulation">>;
  gaps: Record<string, never>;
}

/**
 * What a battle's replay identity binds: only the rules that change how the
 * format plays. Reasons, seasons, sources and gaps stay out, so editing them
 * never changes a version manifest.
 */
export interface FormatRulesRecord {
  schema_version: number;
  format_id: string;
  regulation: string;
  game_type: string;
  forbidden_rules: string[];
}

/** `rules/channels/<channel>.json`. */
export interface ChannelFile {
  channel: string;
  seq: number;
  hash: string;
}

export class RulesError extends Error {
  override name = "RulesError";
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

function assertIJson(value: Json): void {
  if (typeof value === "string") {
    if (LONE_SURROGATE.test(value)) throw new RangeError("jcs: lone surrogate in string");
  } else if (Array.isArray(value)) {
    value.forEach(assertIJson);
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      assertIJson(key);
      assertIJson(item);
    }
  }
}

/**
 * RFC 8785 (JCS) canonical form. `canonicalJson` already sorts keys by UTF-16
 * code units and prints numbers the ECMAScript way; this adds the I-JSON
 * checks it lacks. `canonicalJson` is left unchanged so `request_hash` and
 * `ruleset_hash` keep their bytes.
 */
export function jcs(value: Json): string {
  assertIJson(value);
  return canonicalJson(value);
}

export const sha256Hex = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");

function readJsonDir(dir: string): { file: string; value: unknown }[] {
  return readdirSync(dir)
    .filter((file) => file.endsWith(".json"))
    .sort()
    .map((file) => ({ file, value: JSON.parse(readFileSync(join(dir, file), "utf8")) as unknown }));
}

/** Read the raw records. Nothing is validated here; see `loadRulesSnapshot`. */
export function readRuleRecords(dir: string = DEFAULT_RULES_DIR): {
  formats: { file: string; value: unknown }[];
  regulations: { file: string; value: unknown }[];
} {
  return {
    formats: readJsonDir(join(dir, "formats")),
    regulations: readJsonDir(join(dir, "regulations")),
  };
}

/** Assemble validated records into a snapshot for `showdownCommit`. */
export function buildSnapshot(records: RuleRecords, showdownCommit: string): RulesSnapshot {
  const formats: RulesSnapshot["formats"] = {};
  for (const { schema_version: _v, format_id, ...rest } of records.formats) {
    formats[format_id] = rest;
  }
  const regulations: RulesSnapshot["regulations"] = {};
  for (const { schema_version: _v, regulation, ...rest } of records.regulations) {
    regulations[regulation] = rest;
  }
  return {
    schema_version: RULES_SCHEMA_VERSION,
    showdown_commit: showdownCommit,
    formats,
    regulations,
    gaps: {},
  };
}

/** The exact published bytes: UTF-8 of the JCS form. */
export const snapshotBytes = (snapshot: RulesSnapshot): Buffer =>
  Buffer.from(jcs(snapshot as unknown as Json), "utf8");

export const rulesSnapshotHash = (snapshot: RulesSnapshot): string =>
  sha256Hex(snapshotBytes(snapshot));

/** The record a battle on `formatId` binds into its version manifest. */
export function formatRulesRecord(snapshot: RulesSnapshot, formatId: string): FormatRulesRecord {
  const format = snapshot.formats[formatId];
  if (!format) throw new RulesError(`${formatId} is not in the rules snapshot`);
  return {
    schema_version: snapshot.schema_version,
    format_id: formatId,
    regulation: format.regulation,
    game_type: format.game_type,
    forbidden_rules: format.forbidden_rules.map(({ rule }) => rule),
  };
}

export const formatRulesHash = (record: FormatRulesRecord): string =>
  sha256Hex(Buffer.from(jcs(record as unknown as Json), "utf8"));

export function readChannel(channel: string, dir: string = DEFAULT_RULES_DIR): unknown {
  return JSON.parse(readFileSync(join(dir, "channels", `${channel}.json`), "utf8")) as unknown;
}
