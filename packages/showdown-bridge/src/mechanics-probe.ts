// The mechanics probe (ADR-016 section 1): Areneon computes no stat, Speed or
// damage value. It builds a scenario battle from declared sets and reads the
// pinned engine's own values. It answers the `stats`, `damage` and `speed_order` kinds. Decision order (ADR-016 figure 4): set check
// (`refused`), scope check, scenario check (`unsupported`), then the probe.

import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat } from "./formats.ts";
import { type DamagePath, DamageStateError, probeDamage } from "./mechanics-damage.ts";
import { MechanicsProbeError } from "./mechanics-error.ts";
import { type HistoryDeclaration, historyVariants, type ResolvedHistory } from "./mechanics-history.ts";
import { probeMoveOutcome } from "./mechanics-outcome.ts";
import { checkScenario, ScenarioError, scenarioPokemon, startScenario } from "./mechanics-scenario.ts";
import { type MechanicsConfig, mechanicsConfigHash } from "./mechanics-config.ts";
import { declaredSets, mechanicsQueryHash, type ParsedMechanicsQuery } from "./mechanics-query.ts";
import { buildResult, type MechanicsResult, type Outcome, type ResultIdentity } from "./mechanics-result.ts";
import { MEGA_GAP_ID, megaInvolved, probeSpeedOrder } from "./mechanics-speed.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import type { RulesSnapshot } from "./rules.ts";
import { importPaste } from "./teams.ts";
import { statePlan } from "./world-state.ts";
import { loadWorldWriters } from "./world-writers.ts";

export { MechanicsProbeError };

export interface MechanicsContext {
  engine: Engine;
  config: MechanicsConfig;
  /** The rules snapshot; read from `rules/` when absent. */
  snapshot?: RulesSnapshot;
  /** The `damage` implementation; both give the same bytes (the default is the one-clone path). */
  damage_path?: DamagePath;
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

/** At most this many remembered answers per table; a table past it is emptied (a probe is a pure function of its inputs, so forgetting only costs time). */
const MEMORY_LIMIT = 4096;

/**
 * What a query's identity and scenario need that depends on the engine, the config, the rules snapshot and the
 * format alone: the version manifest, the config hash and the scenario format id, resolved once instead of per query.
 * A hit requires the same engine and snapshot objects and equal config bytes; a failure (an unsupported format, a scenario that differs) is never remembered.
 */
interface Prepared {
  snapshot: RulesSnapshot;
  manifest: ResultIdentity["manifest"];
  config_hash: string;
  scenario_id: string | null;
}
const prepared = new WeakMap<Engine, Map<string, Prepared>>();

function prepare(engine: Engine, config: MechanicsConfig, formatId: string, snapshot: RulesSnapshot): Prepared {
  const key = `${formatId}\n${JSON.stringify(config)}`;
  const table = prepared.get(engine) ?? new Map<string, Prepared>();
  prepared.set(engine, table);
  const hit = table.get(key);
  if (hit !== undefined && hit.snapshot === snapshot) return hit;
  const entry: Prepared = { snapshot, manifest: resolveFormat(engine, formatId, snapshot).manifest, config_hash: mechanicsConfigHash(config), scenario_id: null };
  if (table.size >= MEMORY_LIMIT) table.clear();
  table.set(key, entry);
  return entry;
}

/** The engine condition a declared stall counter is written to: the reviewed world writers record names it (the world builder's Protect counter). */
const stallConditions = new WeakMap<Engine, Map<string, string>>();

function stallCondition(engine: Engine, formatId: string): string {
  const table = stallConditions.get(engine) ?? new Map<string, string>();
  stallConditions.set(engine, table);
  let id = table.get(formatId);
  if (id === undefined) {
    id = statePlan(loadWorldWriters(engine, formatId)).conditionOf("protect_counter");
    table.set(formatId, id);
  }
  return id;
}

/** Declared sets by (engine, format, paste): the engine's verdict on a paste is a pure function of the pinned engine, so it is asked once. */
const declared = new WeakMap<Engine, Map<string, Declared>>();

function declaredSetMemo(engine: Engine, formatId: string, paste: string): Declared {
  const table = declared.get(engine) ?? new Map<string, Declared>();
  declared.set(engine, table);
  const key = `${formatId}\n${paste}`;
  let value = table.get(key);
  if (value === undefined) {
    value = declaredSet(engine, formatId, paste);
    if (table.size >= MEMORY_LIMIT) table.clear();
    table.set(key, value);
  }
  // A copy: the scenario battle may fill defaults into a set, and no query may see another's.
  return "set" in value ? { set: structuredClone(value.set) } : value;
}

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

/** The gap ledger rows a result of this query lists: the config's for the kind, and the Mega row when a speed order or a move outcome involves a Mega Evolution. */
function gapIds(config: MechanicsConfig, query: ParsedMechanicsQuery): string[] {
  const entry = config.scope.query_kinds.find((candidate) => candidate.kind === query.query.kind);
  if (entry === undefined) return [];
  const extra = (query.query.kind === "speed_order" || query.query.kind === "move_outcome") && megaInvolved(query.query) ? [MEGA_GAP_ID] : [];
  return [...new Set([...entry.gap_ids, ...extra])].sort();
}

/**
 * Answer one query. A query that fails the contract schema never gets here
 * (`readMechanicsQuery` throws); a format the rules snapshot does not support,
 * or a config of another format, throws too, as there is no manifest to name.
 */
export function probeMechanics(context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome {
  const { engine, config } = context;
  if (config.format_id !== query.format_id) throw new MechanicsProbeError("the mechanics config is for another format");
  const entry = prepare(engine, config, query.format_id, context.snapshot ?? loadRulesSnapshot(engine.commit));
  const kind = query.query.kind;
  const identity: ResultIdentity = {
    data_class: query.data_class,
    query_hash: mechanicsQueryHash(query),
    manifest: structuredClone(entry.manifest),
    mechanics_config_hash: entry.config_hash,
    gap_ids: gapIds(config, query),
  };
  const done = (outcome: Outcome, diagnostics: string[] = []): ProbeOutcome => ({ result: buildResult(identity, outcome), diagnostics });

  const sets: PokemonSet[] = [];
  for (const [position, paste] of declaredSets(query).entries()) {
    const checked = declaredSetMemo(engine, query.format_id, paste);
    if ("problems" in checked) return done({ kind: "refused", code: "invalid_set", position }, [...checked.problems]);
    sets.push(checked.set);
  }
  if (!config.scope.query_kinds.some((entry) => entry.kind === kind)) return done({ kind: "unsupported", reason: "query_kind", position: null });
  let scenarioId: string;
  try {
    scenarioId = entry.scenario_id ?? checkScenario(engine, config);
    entry.scenario_id = scenarioId;
  } catch (error) {
    return done({ kind: "unsupported", reason: "scenario_rule_table", position: null }, (error as ScenarioError).problems);
  }
  // The query's history, one resolution per end of an unknown fact (the two ends rule): equal answers are kept, different ones are `unsupported`.
  const variants = historyVariants(query.query);
  const stall = variants.some((variant) => variant !== null) ? stallCondition(engine, query.format_id) : "";
  // A declared volatile whose start handler draws is asked at both ends of its draws (`low`, then `high`): equal answers are kept.
  let drew = false;
  const declaration = (resolved: ResolvedHistory | null): HistoryDeclaration | null => (resolved === null ? null : { resolved, stall, on_draw: () => void (drew = true) });
  const answer = (resolved: ResolvedHistory | null): Outcome => {
    const history = declaration(resolved);
    const asked = query.query;
    if (asked.kind === "damage") {
      try {
        return probeDamage(engine, scenarioId, asked, sets[0] as PokemonSet, sets[1] as PokemonSet, config.ko_blockers, context.damage_path, history);
      } catch (error) {
        if (error instanceof DamageStateError) throw new MechanicsProbeError(error.message);
        throw error;
      }
    }
    if (asked.kind === "speed_order") return probeSpeedOrder(engine, scenarioId, asked, sets, history);
    if (asked.kind === "move_outcome") return probeMoveOutcome(engine, scenarioId, asked, sets, history);
    return { kind: "computed", answer: { kind: "stats", stats: statsOf(engine, scenarioId, sets[0] as PokemonSet) } };
  };
  const first = answer(variants[0] as ResolvedHistory | null);
  const differs = (variant: ResolvedHistory | null): boolean => JSON.stringify(answer(variant)) !== JSON.stringify(first);
  for (const variant of variants.slice(1)) {
    if (differs(variant)) return done({ kind: "unsupported", reason: "history_unknown", position: null });
  }
  // The draws of a declared volatile's start (a confusion's length): the other end of every draw, for every history variant that drew.
  if (drew) {
    for (const variant of variants) if (variant !== null && differs({ ...variant, draws: "high" })) return done({ kind: "unsupported", reason: "history_unknown", position: null });
  }
  return done(first);
}
