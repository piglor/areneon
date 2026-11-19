// The composition root of the differential harness at level 1, the mechanics-query level (ADR-030 groundwork G1, #1461): it
// joins the engine-free core (`@areneon/differential`) with the bridge's oracle side (`probeOracle`, `queryFacts`), which does
// not import the core. Its queries are the ones the feature builders themselves issue: every request of each seat of a
// committed battle is built with `probe-damage/2` and `probe-outcome/1` (`buildFeaturesG`, the builders' generator form) and
// every query they yield is recorded, as `feature-bench.ts` replays real decisions. Evaluation code never imports this module.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type Backend,
  type CorpusTally,
  type DifferentialReport,
  differentialReport,
  HARNESS_FORMAT,
  type Json,
  type Level,
  mechanicsLevel,
  runCorpus,
  sha256Jcs,
} from "@areneon/differential";
import {
  buildFeaturesG,
  createFeatureMemo,
  damageAwareConfigHash,
  decodeRequest,
  driveSync,
  type Engine,
  enumerateCandidates,
  FEATURE_SET_OUTCOME,
  type FeatureConfigs,
  FeatureBuildError,
  FIDELITY_DIR,
  GOLDEN_FORMAT,
  goldenTeamVocabulary,
  loadDamageAwareConfig,
  loadMechanicsConfig,
  loadOutcomeConfig,
  loadRandomCorpus,
  loadRulesSnapshot,
  mechanicsConfigHash,
  type MechanicsResult,
  mechanicsQueryHash,
  type OracleBackend,
  outcomeConfigHash,
  type ParsedMechanicsQuery,
  type Player,
  playerChannel,
  type PokemonSet,
  probeOracle,
  queryFacts,
  RANDOM_DIR,
  type RandomBattleFile,
  type Settled,
  UNAVAILABLE_DIR,
  withMoveOutcome,
} from "@areneon/showdown-bridge";

/** The feature sets whose queries the harness compares (`/2` holds `/1`'s rows by the same code; `probe-outcome/1` adds the move-outcome queries and the history declarations). */
export const DIFFERENTIAL_FEATURE_SETS = ["probe-damage/2", FEATURE_SET_OUTCOME] as const;

/** The committed corpora the self-differential runs over. */
export const COMMITTED_CORPORA: readonly { id: string; dir: string }[] = [
  { id: "random", dir: RANDOM_DIR },
  { id: "search-fidelity", dir: FIDELITY_DIR },
  { id: "unavailable", dir: UNAVAILABLE_DIR },
];

/** Where the committed self-differential report lives, and its title. */
export const SELF_REPORT_DIR = fileURLToPath(new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/differential/", import.meta.url));
export const SELF_REPORT_TITLE = "Differential harness: the pinned probe against itself (level 1)";

const BATTLE_FILE = /^b\d+\.json$/;
const REQUEST = "|request|";

/** The configs the builders and the oracle use: one rules snapshot; the oracle answers with the outcome set's mechanics config (a superset of `/2`'s query kinds). */
export interface HarnessConfigs {
  sets: Record<(typeof DIFFERENTIAL_FEATURE_SETS)[number], FeatureConfigs>;
  /** The mechanics config every probe answer is made under. */
  oracle: FeatureConfigs["mechanics"];
  snapshot: NonNullable<FeatureConfigs["snapshot"]>;
}

export function harnessConfigs(engine: Engine, format: string = GOLDEN_FORMAT): HarnessConfigs {
  const vocabulary = goldenTeamVocabulary();
  const config = loadDamageAwareConfig("v1", vocabulary);
  const mechanics = loadMechanicsConfig(format);
  const snapshot = loadRulesSnapshot(engine.commit);
  const base = { vocabulary, config, snapshot, data_class: "synthetic" as const };
  const oracle = withMoveOutcome(mechanics);
  return {
    sets: {
      "probe-damage/2": { ...base, mechanics },
      [FEATURE_SET_OUTCOME]: { ...base, mechanics: oracle, outcome: { config: loadOutcomeConfig(format) } },
    } as HarnessConfigs["sets"],
    oracle,
    snapshot,
  };
}

/** A backend that remembers the outcome of each query (by its hash): the builders' run and the comparison share one probe call. */
export function remembering(base: OracleBackend): OracleBackend {
  const seen = new Map<string, Settled>();
  return {
    id: base.id,
    backend_format: base.backend_format,
    backend_commit: base.backend_commit,
    run(batch) {
      const missing = batch.filter((q) => !seen.has(mechanicsQueryHash(q)));
      base.run(missing).forEach((settled, i) => seen.set(mechanicsQueryHash(missing[i] as ParsedMechanicsQuery), settled));
      return batch.map((q) => seen.get(mechanicsQueryHash(q)) as Settled);
    },
  };
}

/** What collecting a corpus's queries found: the distinct queries in the order the builders first asked them, and the requests seen. */
export interface Collected {
  queries: ParsedMechanicsQuery[];
  /** Requests built (both seats, every phase but `wait`, each feature set). */
  builds: number;
  /** Builds the builder refused (`FeatureBuildError`, by code). */
  refused: Record<string, number>;
}

/**
 * Every distinct query the feature builders issue at every request of both seats of `battles`, the oracle answering each as it is
 * asked (one memo per game and set, as in play). A query's answer is a pure function of the query, so recording it is enough.
 */
export function collectQueries(engine: Engine, configs: HarnessConfigs, battles: readonly RandomBattleFile[], oracle: OracleBackend): Collected {
  const queries: ParsedMechanicsQuery[] = [];
  const known = new Set<string>();
  const refused: Record<string, number> = {};
  let builds = 0;
  const ask = (query: ParsedMechanicsQuery): MechanicsResult => {
    const hash = mechanicsQueryHash(query);
    if (!known.has(hash)) {
      known.add(hash);
      queries.push(query);
    }
    const settled = oracle.run([query])[0] as Settled;
    if (!settled.ok) throw settled.error;
    return settled.result;
  };
  for (const battle of battles) {
    for (const player of ["p1", "p2"] as Player[]) {
      const lines = battle.channels[player].lines;
      const team = battle.teams[player] as PokemonSet[];
      const memos = DIFFERENTIAL_FEATURE_SETS.map(() => createFeatureMemo());
      lines.forEach((line, i) => {
        if (!line.startsWith(REQUEST)) return;
        const decoded = decodeRequest(line.slice(REQUEST.length));
        if (decoded.phase === "wait") return;
        const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
        const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player, lines: lines.slice(0, i + 1) });
        DIFFERENTIAL_FEATURE_SETS.forEach((set, k) => {
          builds += 1;
          try {
            driveSync(buildFeaturesG(engine, configs.sets[set], channel, team, candidates, memos[k] as ReturnType<typeof createFeatureMemo>, set), ask);
          } catch (error) {
            if (!(error instanceof FeatureBuildError)) throw error;
            refused[error.code] = (refused[error.code] ?? 0) + 1;
          }
        });
      });
    }
  }
  return { queries, builds, refused };
}

/** The battles of a committed corpus, the first `limit` of them when given. */
export const corpusBattles = (dir: string, limit?: number): RandomBattleFile[] => loadRandomCorpus(dir, BATTLE_FILE).slice(0, limit);

/** SHA-256 over the names and bytes of a corpus's battle files: the hash naming a regression corpus and the report's input. */
export function corpusHash(dir: string, limit?: number): string {
  const names = readdirSync(dir)
    .filter((name) => BATTLE_FILE.test(name))
    .sort()
    .slice(0, limit);
  return sha256Jcs(names.map((name) => [name, createHash("sha256").update(readFileSync(join(dir, name))).digest("hex")]));
}

/** The inputs a self-differential report is a function of (a test recomputes them: a stale report fails). */
export function reportInputs(engine: Engine, configs: HarnessConfigs, limit?: number): Json {
  return {
    harness_format: HARNESS_FORMAT,
    showdown_commit: engine.commit,
    feature_sets: [...DIFFERENTIAL_FEATURE_SETS],
    mechanics_config_hash: mechanicsConfigHash(configs.oracle),
    damage_aware_config_hash: damageAwareConfigHash(configs.sets["probe-damage/2"].config),
    outcome_config_hash: outcomeConfigHash((configs.sets[FEATURE_SET_OUTCOME].outcome as { config: Parameters<typeof outcomeConfigHash>[0] }).config),
    corpora: Object.fromEntries(COMMITTED_CORPORA.map((c) => [c.id, corpusHash(c.dir, limit)])),
    limit: limit ?? null,
  };
}

/** The level-1 adapter, reading facts with the pinned engine. */
export const mechanicsHarnessLevel = (engine: Engine): Level<ParsedMechanicsQuery, MechanicsResult> => mechanicsLevel(queryFacts(engine));

export interface SelfDifferential {
  report: DifferentialReport;
  tallies: CorpusTally[];
  collected: Record<string, Collected>;
}

/**
 * The probe against itself (gate 1 of #1461): the oracle on `engine`, the backend the same probe on `second` (another engine
 * object, so its own per-engine probe tables), every committed corpus as a regression corpus. `limit` takes the first battles of
 * each corpus (the CI subset); the full run is the committed report.
 */
export function selfDifferential(engine: Engine, second: Engine, limit?: number, batchSize = 256): SelfDifferential {
  const configs = harnessConfigs(engine);
  const oracle = remembering(probeOracle(engine, { config: configs.oracle, snapshot: configs.snapshot }));
  const backend = probeOracle(second, { config: configs.oracle, snapshot: loadRulesSnapshot(second.commit), id: "showdown-probe-second" });
  const level = mechanicsHarnessLevel(engine);
  const tallies: CorpusTally[] = [];
  const collected: Record<string, Collected> = {};
  for (const corpus of COMMITTED_CORPORA) {
    const found = collectQueries(engine, configs, corpusBattles(corpus.dir, limit), oracle);
    collected[corpus.id] = found;
    tallies.push(runCorpus(level, oracle as Backend<ParsedMechanicsQuery, MechanicsResult>, backend as Backend<ParsedMechanicsQuery, MechanicsResult>, { id: corpus.id, kind: "regression", seed: corpusHash(corpus.dir, limit) }, found.queries, batchSize));
  }
  return { report: differentialReport(reportInputs(engine, configs, limit), tallies), tallies, collected };
}
