// The mechanics probe (ADR-016 section 1): Areneon computes no stat, Speed or
// damage value. It builds a scenario battle from declared sets and reads the
// pinned engine's own values. This slice answers the `stats` kind; the other
// kinds are in the contract and stay `unsupported` until the mechanics config
// lists them (slices 2 and 3). Decision order (ADR-016 figure 4): set check
// (`refused`), scope check, scenario check (`unsupported`), then the probe.

import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { checkScenario, ScenarioError, scenarioPokemon, startScenario } from "./mechanics-scenario.ts";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { declaredSets, mechanicsQueryHash, type ParsedMechanicsQuery } from "./mechanics-query.ts";
import { buildResult, type MechanicsResult, type Outcome, type ResultIdentity } from "./mechanics-result.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import { importPaste } from "./teams.ts";

export class MechanicsProbeError extends Error {
  override name = "MechanicsProbeError";
}

export interface MechanicsContext {
  engine: Engine;
  config: MechanicsConfig;
  /** The rules snapshot; read from `rules/` when absent. */
  snapshot?: RulesSnapshot;
}

export interface ProbeOutcome {
  result: MechanicsResult;
  /**
   * The engine's and the validator's own words about a refusal or an
   * unsupported scenario, for the caller's stderr. They may quote a nickname,
   * so they are never part of a result.
   */
  diagnostics: string[];
}

type Declared = { set: PokemonSet } | { problems: string[] };

/** One declared set: the engine's import round trip, exactly one Pokemon, then the target format's validator on a copy. */
function declaredSet(engine: Engine, formatId: string, paste: string): Declared {
  let team: PokemonSet[];
  try {
    team = importPaste(engine, paste);
  } catch (error) {
    return { problems: [(error as Error).message] };
  }
  if (team.length !== 1) return { problems: [`a set is one Pokémon, the paste has ${team.length}`] };
  const set = team[0] as PokemonSet;
  const problems = engine.sim.TeamValidator.get(formatId).validateSet(structuredClone(set), {}) ?? [];
  return problems.length > 0 ? { problems } : { set };
}

/** The engine's six stats of `set`, read from a scenario battle that holds it. */
function statsOf(engine: Engine, scenarioId: string, set: PokemonSet): { hp: number; atk: number; def: number; spa: number; spd: number; spe: number } {
  const battle = startScenario(engine, scenarioId, { p1: [set, set], p2: [set, set] });
  const pokemon = scenarioPokemon(battle, 0, 0);
  const { atk, def, spa, spd, spe } = pokemon.storedStats;
  return { hp: pokemon.maxhp, atk, def, spa, spd, spe };
}

/**
 * Answer one query. A query that fails the contract schema never gets here
 * (`readMechanicsQuery` throws); a format the rules snapshot does not support,
 * or a config of another format, throws too, as there is no manifest to name.
 */
export function probeMechanics(context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome {
  const { engine, config } = context;
  if (config.format_id !== query.format_id) throw new MechanicsProbeError("the mechanics config is for another format");
  const manifest = resolveFormat(engine, query.format_id, context.snapshot ?? loadRulesSnapshot(engine.commit)).manifest;
  const kind = query.query.kind;
  const identity: ResultIdentity = {
    data_class: query.data_class,
    query_hash: mechanicsQueryHash(query),
    manifest,
    mechanics_config_hash: mechanicsConfigHash(config),
    gap_ids: [...(config.scope.query_kinds.find((entry) => entry.kind === kind)?.gap_ids ?? [])],
  };
  const done = (outcome: Outcome, diagnostics: string[] = []): ProbeOutcome => ({ result: buildResult(identity, outcome), diagnostics });

  const sets: PokemonSet[] = [];
  for (const [position, paste] of declaredSets(query).entries()) {
    const declared = declaredSet(engine, query.format_id, paste);
    if ("problems" in declared) return done({ kind: "refused", code: "invalid_set", position }, declared.problems);
    sets.push(declared.set);
  }
  if (!config.scope.query_kinds.some((entry) => entry.kind === kind)) return done({ kind: "unsupported", reason: "query_kind", position: null });
  let scenarioId: string;
  try {
    scenarioId = checkScenario(engine, config);
  } catch (error) {
    return done({ kind: "unsupported", reason: "scenario_rule_table", position: null }, (error as ScenarioError).problems);
  }
  if (kind !== "stats") throw new MechanicsProbeError(`the config lists ${kind}, which this probe cannot answer yet`);
  return done({ kind: "computed", answer: { kind: "stats", stats: statsOf(engine, scenarioId, sets[0] as PokemonSet) } });
}
