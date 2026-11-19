// The scenario format and the scenario battle (ADR-016 section 3): the target
// format minus the non-mechanics rules the mechanics config lists. A scenario
// battle starts at once with two Pokemon a side and no Team Preview, so the
// probe can read the engine's own values from declared sets. The start-up
// check refuses to run unless the engine's rule tables differ by exactly the
// listed keys; that is what ties the probe's numbers to the target format.

import type { Engine, PokemonSet, SimBattle, SimFormat, SimPokemon, SimRuleTable } from "./engine.ts";
import type { MechanicsConfig } from "@areneon/schema/types/v1/mechanics-config.ts";

export class ScenarioError extends Error {
  override name = "ScenarioError";
  /** Every difference found, in words (never put into a result). */
  readonly problems: string[];

  constructor(problems: string[]) {
    super(`the scenario format differs from the target: ${problems.join("; ")}`);
    this.problems = problems;
  }
}

/** The scenario format id: the target's, then `!<rule>` for each removed rule. */
export const scenarioFormatId = (config: MechanicsConfig): string =>
  `${config.format_id}@@@${config.scenario.removed_rules.map((rule) => `!${rule}`).join(",")}`;

// Rule-table fields that stay equal; the team-size fields are what the removed rules set.
export const SHARED_FIELDS = ["adjustLevel", "adjustLevelDown", "evLimit", "minLevel", "maxLevel", "defaultLevel", "maxMoveCount", "minSourceGen", "maxTotalLevel"] as const;

/** A format's `battle` options as comparable text (a function by its source). */
export const battleOptions = (format: SimFormat): string =>
  JSON.stringify(Object.entries(format.battle ?? {}).sort(([a], [b]) => (a < b ? -1 : 1)).map(([key, value]) => [key, typeof value === "function" ? String(value) : value]));

function differences(target: SimFormat, scenario: SimFormat, tt: SimRuleTable, ts: SimRuleTable, removedKeys: readonly string[]): string[] {
  const problems: string[] = [];
  if (scenario.mod !== target.mod) problems.push(`mod ${scenario.mod} is not ${target.mod}`);
  if (scenario.gameType !== target.gameType) problems.push(`game type ${scenario.gameType} is not ${target.gameType}`);
  if (battleOptions(scenario) !== battleOptions(target)) problems.push("the battle options differ");
  const removed = [...tt.keys()].filter((key) => !ts.has(key)).sort();
  if (removed.join() !== [...removedKeys].sort().join()) problems.push(`the target's keys missing from the scenario are [${removed.join(", ")}], not the listed [${removedKeys.join(", ")}]`);
  const own = [...ts.keys()].filter((key) => !tt.has(key));
  if (own.length > 0) problems.push(`the scenario has keys of its own: [${own.join(", ")}]`);
  for (const [key, source] of ts) if (tt.has(key) && tt.get(key) !== source) problems.push(`rule ${key} comes from another source`);
  for (const [key, value] of ts.valueRules) if (tt.valueRules.get(key) !== value) problems.push(`value rule ${key} differs`);
  for (const key of SHARED_FIELDS) if (ts[key] !== tt[key]) problems.push(`${key} differs`);
  return problems;
}

/**
 * The start-up check. Returns the scenario format id when the engine's rule
 * tables for it equal the target's minus exactly the config's removed keys,
 * with no key of its own, the same mod, game type and battle options, and the
 * same values for every shared rule; throws a `ScenarioError` naming the
 * differences otherwise.
 */
export function checkScenario(engine: Engine, config: MechanicsConfig): string {
  const formats = engine.sim.Dex.formats;
  const id = scenarioFormatId(config);
  const target = formats.get(config.format_id);
  let problems: string[];
  try {
    const scenario = formats.get(id);
    problems = scenario.exists
      ? differences(target, scenario, formats.getRuleTable(target), formats.getRuleTable(scenario), config.scenario.removed_rule_keys)
      : ["the scenario format does not exist"];
  } catch (error) {
    // The engine itself refuses a custom rule that is unknown or not in effect.
    problems = [`the engine refuses the scenario format: ${(error as Error).message}`];
  }
  if (problems.length > 0) throw new ScenarioError(problems);
  return id;
}

/** A fixed battle seed: a scenario battle draws nothing the stats probe reads. */
export const SCENARIO_SEED = "sodium,00000000000000000000000000000000";

/**
 * Switches for the probe caches below. They never change a result (a cached value is a pure function of its key and every
 * caller builds from the kept serialisation, never from a live object), so the switch is a test seam: the caches-off path
 * is the reference the equality tests compare with.
 */
export const probeCaches = { enabled: true };

/** At most this many resolved scenario formats and kept starts are held per engine; a table past it is emptied (forgetting only costs time). */
const FORMAT_LIMIT = 16;
const START_LIMIT = 256;

type FormatGet = (name: string, isTrusted?: boolean) => SimFormat;
const formatTables = new WeakMap<Engine, Map<string, SimFormat>>();
const startTables = new WeakMap<Engine, Map<string, string>>();
/** Diagnostics, like `scenarioStartStats` of the damage probe. */
export const scenarioCacheStats = { format_resolved: 0, format_reused: 0, start_built: 0, start_reused: 0 };

/** Empty this engine's kept formats and kept starts (a test seam: the next probe is a cold one). */
export function clearProbeCaches(engine: Engine): void {
  formatTables.get(engine)?.clear();
  startTables.get(engine)?.clear();
}

/**
 * Run `fn` with the engine's trusted format lookup (`formats.get(id, true)`, which `new Battle` and `Battle.fromJSON` call) answered
 * from a per-engine table. A format with custom rules (the scenario's `@@@!rule` id) is a new object every lookup, so the engine
 * rebuilt its rule table for every battle; the kept object carries the table the engine caches on it. The patch is scoped to the call.
 */
function withKeptFormats<T>(engine: Engine, fn: () => T): T {
  if (!probeCaches.enabled) return fn();
  const formats = engine.sim.Dex.formats as unknown as { get: FormatGet };
  const original = formats.get;
  const table = formatTables.get(engine) ?? new Map<string, SimFormat>();
  formatTables.set(engine, table);
  formats.get = function (this: unknown, name: string, isTrusted?: boolean): SimFormat {
    if (isTrusted !== true) return original.call(this, name, isTrusted);
    const kept = table.get(name);
    if (kept !== undefined) {
      scenarioCacheStats.format_reused++;
      return kept;
    }
    const format = original.call(this, name, true);
    scenarioCacheStats.format_resolved++;
    if (table.size >= FORMAT_LIMIT) table.clear();
    table.set(name, format);
    return format;
  };
  try {
    return fn();
  } finally {
    formats.get = original;
  }
}

/**
 * A scenario battle of `scenarioFormatId` with the two teams as declared (no
 * validation: the caller has judged each set), started at once.
 */
export function startScenario(engine: Engine, scenarioId: string, teams: { p1: PokemonSet[]; p2: PokemonSet[] }): SimBattle {
  return withKeptFormats(engine, () => {
    const battle = new engine.sim.Battle({ formatid: scenarioId, seed: SCENARIO_SEED });
    battle.setPlayer("p1", { name: "p1", team: engine.sim.Teams.pack(teams.p1) });
    battle.setPlayer("p2", { name: "p2", team: engine.sim.Teams.pack(teams.p2) });
    return battle;
  });
}

/**
 * A battle rebuilt from a kept serialisation (the damage probe's kept start, below): the resolved format is the kept one too.
 */
export function restoreScenario(engine: Engine, json: string): SimBattle {
  return withKeptFormats(engine, () => engine.sim.Battle.fromJSON(json));
}

/**
 * The started scenario of two declared teams, before any declared state. The engine starts it once per
 * (scenario format, teams) and the serialised start is kept; every caller, the first included, gets its own battle from that
 * serialisation, so no result depends on the table's contents. With the caches off it starts the scenario itself every time.
 */
export function keptScenario(engine: Engine, scenarioId: string, teams: { p1: PokemonSet[]; p2: PokemonSet[] }): SimBattle {
  if (!probeCaches.enabled) return startScenario(engine, scenarioId, teams);
  const table = startTables.get(engine) ?? new Map<string, string>();
  startTables.set(engine, table);
  // The sets are the validated, canonical ones, which serialise the same way every time; a different order of keys would only miss.
  const key = JSON.stringify([scenarioId, teams.p1, teams.p2]);
  let json = table.get(key);
  if (json === undefined) {
    scenarioCacheStats.start_built++;
    json = JSON.stringify(startScenario(engine, scenarioId, teams).toJSON());
    if (table.size >= START_LIMIT) table.clear();
    table.set(key, json);
  } else scenarioCacheStats.start_reused++;
  return restoreScenario(engine, json);
}

/** The engine's Pokemon at `index` of a side, as the probe reads it. */
export const scenarioPokemon = (battle: SimBattle, side: number, index: number): SimPokemon =>
  (battle.sides[side]?.pokemon[index]) as SimPokemon;
