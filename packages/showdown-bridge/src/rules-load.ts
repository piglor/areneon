import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Ajv2020, type ValidateFunction } from "ajv/dist/2020.js";
import {
  buildSnapshot,
  type ChannelFile,
  DEFAULT_RULES_DIR,
  type FormatRecord,
  readChannel,
  readRuleRecords,
  type RegulationRecord,
  RulesError,
  type RulesSnapshot,
} from "./rules.ts";

/** Every published schema version stays here, so old snapshots stay readable (ADR-007). */
export const RULES_SCHEMA_DIR = fileURLToPath(
  new URL("../../schema/rules", import.meta.url),
);

type Kind = "format" | "regulation" | "channel";

const ajv = new Ajv2020({ allErrors: true, strict: true });
const validators = new Map<string, ValidateFunction>();

function validator(version: number, kind: Kind): ValidateFunction {
  const key = `v${version}/${kind}`;
  let validate = validators.get(key);
  if (!validate) {
    let schema: unknown;
    try {
      schema = JSON.parse(readFileSync(`${RULES_SCHEMA_DIR}/${key}.schema.json`, "utf8"));
    } catch {
      throw new RulesError(`no ${kind} schema for schema_version ${version}`);
    }
    validate = ajv.compile(schema as object);
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

/**
 * Read, validate and assemble the rules records into the snapshot for
 * `showdownCommit`. Fails closed on any malformed, unknown-version or
 * unknown-key record, a file named differently from its ID, an unsorted
 * or duplicated forbidden rule, or a format naming an unknown regulation.
 */
export function loadRulesSnapshot(
  showdownCommit: string,
  dir: string = DEFAULT_RULES_DIR,
): RulesSnapshot {
  const raw = readRuleRecords(dir);
  const regulations = raw.regulations.map(({ file, value }) => {
    const record = check<RegulationRecord>(value, versionOf(value), "regulation", `regulations/${file}`);
    if (file !== `${record.regulation}.json`) {
      throw new RulesError(`regulations/${file} must be named ${record.regulation}.json`);
    }
    return record;
  });
  const known = new Set(regulations.map((record) => record.regulation));
  const formats = raw.formats.map(({ file, value }) => {
    const record = check<FormatRecord>(value, versionOf(value), "format", `formats/${file}`);
    if (file !== `${record.format_id}.json`) {
      throw new RulesError(`formats/${file} must be named ${record.format_id}.json`);
    }
    if (!known.has(record.regulation)) {
      throw new RulesError(`formats/${file}: unknown regulation ${record.regulation}`);
    }
    const rules = record.forbidden_rules.map(({ rule }) => rule);
    if (rules.some((rule, i) => i > 0 && rule <= (rules[i - 1] as string))) {
      throw new RulesError(`formats/${file}: forbidden_rules must be sorted by rule and unique`);
    }
    return record;
  });
  if (formats.length === 0) throw new RulesError(`no format records in ${dir}/formats`);
  return buildSnapshot({ formats, regulations }, showdownCommit);
}

/** Read and validate `rules/channels/<channel>.json`. */
export function loadChannel(channel: string, dir: string = DEFAULT_RULES_DIR): ChannelFile {
  const file = check<ChannelFile>(readChannel(channel, dir), 1, "channel", `channels/${channel}.json`);
  if (file.channel !== channel) {
    throw new RulesError(`channels/${channel}.json names channel ${file.channel}`);
  }
  return file;
}
