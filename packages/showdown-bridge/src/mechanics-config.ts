// The mechanics probe's configuration (ADR-016 section 3): reviewed data in
// config/mechanics/<format>.json, outside rules/ and outside the published
// rules snapshot. The schema checks the shape; this reader adds what the
// profile cannot say (sorted and unique lists), `checkMechanicsConfig` checks
// the file against the pinned engine and the gap ledger, and
// `mechanicsConfigHash` is the identity every MechanicsResult carries.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { ajvFor } from "@areneon/schema/ajv";
import configSchema from "@areneon/schema/contracts/v1/mechanics-config.schema.json" with { type: "json" };
import type { MechanicsConfig } from "@areneon/schema/types/v1/mechanics-config.ts";
import { type Json, jcs } from "./canonical-json.ts";
import type { Engine } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { checkScenario, ScenarioError } from "./mechanics-scenario.ts";
import { MEGA_GAP_ID } from "./mechanics-speed.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { type RulesSnapshot, sha256Hex } from "./rules.ts";

export type { MechanicsConfig };

export const MECHANICS_CONFIG_DIR = fileURLToPath(new URL("../../../config/mechanics", import.meta.url));

export class MechanicsConfigError extends Error {
  override name = "MechanicsConfigError";
}

const validConfig = ajvFor("contract").compile(configSchema);

const sortedUnique = (list: readonly string[]): boolean => list.every((entry, index) => index === 0 || entry > (list[index - 1] as string));

/** Read a config: the schema, then the sorted-and-unique rule of every list. */
export function readMechanicsConfig(value: unknown): MechanicsConfig {
  if (!validConfig(value)) throw new MechanicsConfigError(`invalid MechanicsConfig: ${JSON.stringify(validConfig.errors)}`);
  const config = structuredClone(value) as unknown as MechanicsConfig;
  const lists: [string, readonly string[]][] = [
    ["scenario.removed_rules", config.scenario.removed_rules],
    ["scenario.removed_rule_keys", config.scenario.removed_rule_keys],
    ["scope.query_kinds", config.scope.query_kinds.map((entry) => entry.kind)],
    ["ko_blockers.abilities", config.ko_blockers.abilities],
    ["ko_blockers.items", config.ko_blockers.items],
    ["ko_non_blockers.abilities", (config.ko_non_blockers?.abilities ?? []).map((entry) => entry.id)],
    ["ko_non_blockers.items", (config.ko_non_blockers?.items ?? []).map((entry) => entry.id)],
    ...config.scope.query_kinds.map((entry): [string, readonly string[]] => [`gap_ids of ${entry.kind}`, entry.gap_ids]),
  ];
  for (const [name, list] of lists) {
    if (!sortedUnique(list)) throw new MechanicsConfigError(`${name} must be sorted and unique`);
  }
  return config;
}

/** The hash a result names its config by: SHA-256 of the config's JCS bytes. */
export const mechanicsConfigHash = (config: MechanicsConfig): string => sha256Hex(jcs(config as unknown as Json));

/** Read `<dir>/<format_id>.json`. */
export function loadMechanicsConfig(formatId: string, dir: string = MECHANICS_CONFIG_DIR): MechanicsConfig {
  const config = readMechanicsConfig(JSON.parse(readFileSync(join(dir, `${formatId}.json`), "utf8")) as unknown);
  if (config.format_id !== formatId) throw new MechanicsConfigError(`${formatId}.json is the config of another format`);
  return config;
}

/**
 * Check `config` against the pinned engine and the rules snapshot: the format
 * is a supported one, every gap id is a ledger row, every blocker and reviewed
 * non-blocker is an engine item or ability, and the scenario format's rule tables differ from the
 * target's by exactly the listed keys. Returns the problems; none means ok.
 */
export function checkMechanicsConfig(engine: Engine, config: MechanicsConfig, snapshot: RulesSnapshot = loadRulesSnapshot(engine.commit)): string[] {
  const problems: string[] = [];
  try {
    resolveFormat(engine, config.format_id, snapshot);
  } catch (error) {
    return [(error as Error).message];
  }
  for (const entry of config.scope.query_kinds) {
    for (const gap of entry.gap_ids) {
      if (!(gap in snapshot.gaps)) problems.push(`${entry.kind}: gap ${gap} is not in the gap ledger`);
    }
  }
  // a speed order names the Mega row itself when a Mega Evolution is involved (the probe's constant, not the config's)
  if (config.scope.query_kinds.some((entry) => entry.kind === "speed_order") && !(MEGA_GAP_ID in snapshot.gaps)) problems.push(`speed_order: gap ${MEGA_GAP_ID} is not in the gap ledger`);
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(config.format_id).mod);
  for (const id of config.ko_blockers.items) if (!dex.items.get(id).exists) problems.push(`ko_blockers: ${id} is not an item`);
  for (const id of config.ko_blockers.abilities) if (!dex.abilities.get(id).exists) problems.push(`ko_blockers: ${id} is not an ability`);
  for (const entry of config.ko_non_blockers?.items ?? []) if (!dex.items.get(entry.id).exists) problems.push(`ko_non_blockers: ${entry.id} is not an item`);
  for (const entry of config.ko_non_blockers?.abilities ?? []) if (!dex.abilities.get(entry.id).exists) problems.push(`ko_non_blockers: ${entry.id} is not an ability`);
  try {
    checkScenario(engine, config);
  } catch (error) {
    problems.push(...(error as ScenarioError).problems);
  }
  return problems;
}
