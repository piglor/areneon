import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Ajv2020, type ValidateFunction } from "ajv/dist/2020.js";
import { jcs, type Json } from "./canonical-json.ts";
import {
  buildSnapshot,
  type ChannelFile,
  DEFAULT_RULES_DIR,
  type FormatRecord,
  type GapRecord,
  readChannel,
  readRuleRecords,
  type RegulationRecord,
  RULES_SCHEMA_VERSION,
  RulesError,
  type RulesSnapshot,
} from "./rules.ts";

/** Every published schema version stays here, so old snapshots stay readable (ADR-007). */
export const RULES_SCHEMA_DIR = fileURLToPath(
  new URL("../../schema/rules", import.meta.url),
);

type Kind = "format" | "regulation" | "gap" | "channel" | "channel-artifact";

const ajv = new Ajv2020({ allErrors: true, strict: true });
const validators = new Map<string, ValidateFunction>();

function schemaFor(version: number, kind: Kind): object {
  try {
    return JSON.parse(readFileSync(`${RULES_SCHEMA_DIR}/v${version}/${kind}.schema.json`, "utf8")) as object;
  } catch {
    throw new RulesError(`no ${kind} schema for schema_version ${version}`);
  }
}

function validator(version: number, kind: Kind): ValidateFunction {
  const key = `v${version}/${kind}`;
  let validate = validators.get(key);
  if (!validate) {
    validate = ajv.compile(schemaFor(version, kind));
    validators.set(key, validate);
  }
  return validate;
}

function check<T>(value: unknown, version: unknown, kind: Kind, where: string): T {
  if (typeof version !== "number" || !Number.isInteger(version)) {
    throw new RulesError(`${where}: schema_version must be an integer`);
  }
  const validate = validator(version, kind);
  if (!validate(value)) {
    throw new RulesError(`${where}: ${ajv.errorsText(validate.errors, { dataVar: kind })}`);
  }
  return value as T;
}

const versionOf = (value: unknown): unknown =>
  value !== null && typeof value === "object" ? (value as { schema_version?: unknown }).schema_version : undefined;

type Entry = { file: string; value: unknown };

/**
 * Validate raw records and assemble them into the snapshot for
 * `showdownCommit`. Fails closed on any malformed, unknown-version or
 * unknown-key record (formats, regulations, gaps), a file named differently
 * from its ID, an unsorted or duplicated forbidden rule, or a format naming
 * an unknown regulation.
 */
function assembleSnapshot(
  raw: { formats: Entry[]; regulations: Entry[]; gaps: Entry[] },
  showdownCommit: string,
  where: string,
  prefix = "",
): RulesSnapshot {
  const regulations = raw.regulations.map(({ file, value }) => {
    const record = check<RegulationRecord>(value, versionOf(value), "regulation", `${prefix}regulations/${file}`);
    if (file !== `${record.regulation}.json`) {
      throw new RulesError(`${prefix}regulations/${file} must be named ${record.regulation}.json`);
    }
    return record;
  });
  const known = new Set(regulations.map((record) => record.regulation));
  const formats = raw.formats.map(({ file, value }) => {
    const record = check<FormatRecord>(value, versionOf(value), "format", `${prefix}formats/${file}`);
    if (file !== `${record.format_id}.json`) {
      throw new RulesError(`${prefix}formats/${file} must be named ${record.format_id}.json`);
    }
    if (!known.has(record.regulation)) {
      throw new RulesError(`${prefix}formats/${file}: unknown regulation ${record.regulation}`);
    }
    const rules = record.forbidden_rules.map(({ rule }) => rule);
    if (rules.some((rule, i) => i > 0 && rule <= (rules[i - 1] as string))) {
      throw new RulesError(`${prefix}formats/${file}: forbidden_rules must be sorted by rule and unique`);
    }
    return record;
  });
  if (formats.length === 0) throw new RulesError(`no format records in ${where}`);
  const gaps = raw.gaps.map(({ file, value }) => {
    const record = check<GapRecord>(value, versionOf(value), "gap", `${prefix}gaps/${file}`);
    if (file !== `${record.gap_id}.json`) {
      throw new RulesError(`${prefix}gaps/${file} must be named ${record.gap_id}.json`);
    }
    return record;
  });
  return buildSnapshot({ formats, regulations, gaps }, showdownCommit);
}

/** Read, validate and assemble the rules records into the snapshot for `showdownCommit`. */
export function loadRulesSnapshot(
  showdownCommit: string,
  dir: string = DEFAULT_RULES_DIR,
): RulesSnapshot {
  return assembleSnapshot(readRuleRecords(dir), showdownCommit, `${dir}/formats`);
}

const SNAPSHOT_KEYS = ["formats", "gaps", "regulations", "schema_version", "showdown_commit"];
const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

/**
 * Validate a parsed, published snapshot (e.g. a released artifact) against
 * the record schemas of its own `schema_version`, with every invariant
 * `loadRulesSnapshot` enforces. Each keyed entry is checked as the record it
 * was built from, so an unknown version, key or record fails closed.
 */
export function parseSnapshot(value: unknown, where: string): RulesSnapshot {
  if (!isPlainObject(value) || Object.keys(value).sort().join() !== SNAPSHOT_KEYS.join()) {
    throw new RulesError(`${where}: a snapshot has exactly the keys ${SNAPSHOT_KEYS.join(", ")}`);
  }
  const { schema_version: version, showdown_commit: commit } = value;
  if (typeof commit !== "string" || !/^[0-9a-f]{40}$/.test(commit)) {
    throw new RulesError(`${where}: showdown_commit must be a 40-hex commit`);
  }
  const entries = (part: string, id: string): Entry[] => {
    const records = value[part];
    if (!isPlainObject(records)) throw new RulesError(`${where}: ${part} must be an object`);
    return Object.keys(records)
      .sort()
      .map((key) => {
        const record = records[key];
        if (!isPlainObject(record)) throw new RulesError(`${where}: ${part}.${key} must be an object`);
        return { file: `${key}.json`, value: { ...record, schema_version: version, [id]: key } };
      });
  };
  const snapshot = assembleSnapshot(
    { formats: entries("formats", "format_id"), regulations: entries("regulations", "regulation"), gaps: entries("gaps", "gap_id") },
    commit,
    `${where} formats`,
    `${where}: `,
  );
  // Rebuilding must give back the same document, so a stray key inside a
  // record (its own ID, a schema_version) or an unbuilt version cannot pass.
  if (jcs(snapshot as unknown as Json) !== jcs(value as Json)) {
    throw new RulesError(`${where}: not the snapshot its records build`);
  }
  return snapshot;
}

/** `jcs({schema_version, channel, seq, hash, source_commit})`: one released channel step. */
export interface ChannelArtifactBody extends ChannelFile {
  schema_version: number;
  source_commit: string;
}

/** Validate a parsed channel artifact against the schema of its own `schema_version`. */
export function parseChannelArtifact(value: unknown, where: string): ChannelArtifactBody {
  return check<ChannelArtifactBody>(value, versionOf(value), "channel-artifact", where);
}

/** Validate a parsed `rules/channels/<channel>.json`, e.g. one read from another commit. */
export function parseChannel(channel: string, value: unknown, where: string): ChannelFile {
  const file = check<ChannelFile>(value, RULES_SCHEMA_VERSION, "channel", where);
  if (file.channel !== channel) throw new RulesError(`${where} names channel ${file.channel}`);
  return file;
}

/** Read and validate `rules/channels/<channel>.json`. */
export function loadChannel(channel: string, dir: string = DEFAULT_RULES_DIR): ChannelFile {
  return parseChannel(channel, readChannel(channel, dir), `channels/${channel}.json`);
}

/** The channels Areneon serves: the channel schema's closed `channel` enum. */
export function knownChannels(schema: object = schemaFor(RULES_SCHEMA_VERSION, "channel")): string[] {
  const channels = (schema as { properties?: { channel?: { enum?: unknown } } }).properties?.channel?.enum;
  if (!Array.isArray(channels) || channels.length === 0 || !channels.every((c) => typeof c === "string")) {
    throw new RulesError("the channel schema must list its channels as a non-empty string enum");
  }
  return [...(channels as string[])].sort();
}

/**
 * Read and validate every channel file. `rules/channels/` must hold exactly
 * one `<channel>.json` per known channel, so a stray or misspelt file fails
 * instead of being silently ignored.
 */
export function loadChannels(dir: string = DEFAULT_RULES_DIR): ChannelFile[] {
  const known = knownChannels();
  let files: string[];
  try {
    files = readdirSync(join(dir, "channels")).sort();
  } catch {
    throw new RulesError("rules/channels/ is missing");
  }
  for (const file of files) {
    if (!known.some((channel) => file === `${channel}.json`)) {
      throw new RulesError(`channels/${file} is not a known channel (${known.join(", ")})`);
    }
  }
  return known.map((channel) => {
    if (!files.includes(`${channel}.json`)) throw new RulesError(`channels/${channel}.json is missing`);
    return loadChannel(channel, dir);
  });
}
