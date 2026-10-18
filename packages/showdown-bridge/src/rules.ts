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

/** Where a gap's behaviour lives: engine source (relative to the submodule) or a fixture. */
export type GapEvidence =
  | { kind: "engine"; path: string; symbols: string[] }
  | { kind: "fixture"; path: string };

/** `rules/gaps/<gap_id>.json`: one row of the gap ledger. */
export interface GapRecord {
  schema_version: number;
  gap_id: string;
  area: string;
  status: "verified" | "unverified" | "unsupported";
  description: string;
  evidence: GapEvidence[];
}

export interface RuleRecords {
  formats: FormatRecord[];
  regulations: RegulationRecord[];
  gaps: GapRecord[];
}

/** One published, content-hashed set of rules. Its five keys are the whole hash preimage. */
export interface RulesSnapshot {
  schema_version: number;
  showdown_commit: string;
  formats: Record<string, Omit<FormatRecord, "schema_version" | "format_id">>;
  regulations: Record<string, Omit<RegulationRecord, "schema_version" | "regulation">>;
  gaps: Record<string, Omit<GapRecord, "schema_version" | "gap_id">>;
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
  game_type: FormatRecord["game_type"];
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

/** UTF-8 of the JCS form: the exact bytes Areneon hashes and publishes. */
export const jcsBytes = (value: object): Buffer => Buffer.from(jcs(value as Json), "utf8");

function readJson(path: string, where: string): unknown {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as unknown;
  } catch (error) {
    throw new RulesError(`${where}: not valid JSON (${(error as Error).message})`);
  }
}

/** Every entry must be a `.json` record, so nothing is silently skipped. */
function readJsonDir(dir: string, kind: string): { file: string; value: unknown }[] {
  return readdirSync(dir)
    .sort()
    .map((file) => {
      if (!file.endsWith(".json")) throw new RulesError(`${kind}/${file} is not a .json record`);
      return { file, value: readJson(join(dir, file), `${kind}/${file}`) };
    });
}

/** Read the raw records. Nothing is validated here; see `loadRulesSnapshot`. */
export function readRuleRecords(dir: string = DEFAULT_RULES_DIR): {
  formats: { file: string; value: unknown }[];
  regulations: { file: string; value: unknown }[];
  gaps: { file: string; value: unknown }[];
} {
  return {
    formats: readJsonDir(join(dir, "formats"), "formats"),
    regulations: readJsonDir(join(dir, "regulations"), "regulations"),
    gaps: readJsonDir(join(dir, "gaps"), "gaps"),
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
  const gaps: RulesSnapshot["gaps"] = {};
  for (const { schema_version: _v, gap_id, ...rest } of records.gaps) {
    gaps[gap_id] = rest;
  }
  return {
    schema_version: RULES_SCHEMA_VERSION,
    showdown_commit: showdownCommit,
    formats,
    regulations,
    gaps,
  };
}

/** The exact published bytes: UTF-8 of the JCS form. */
export const snapshotBytes = (snapshot: RulesSnapshot): Buffer => jcsBytes(snapshot);

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

export const formatRulesHash = (record: FormatRulesRecord): string => sha256Hex(jcsBytes(record));

export function readChannel(channel: string, dir: string = DEFAULT_RULES_DIR): unknown {
  return readJson(join(dir, "channels", `${channel}.json`), `channels/${channel}.json`);
}
