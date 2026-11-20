// The damage kernel through the differential harness (ADR-030 step 3, #1457): the composition root that joins the engine-free
// core (`@areneon/differential`), the pinned probe as the oracle and the kernel backend (`kernelBackend`), at level 1.
//
// The kernel answers only the queries it covers; an abstained query routes to the engine probe and is no evidence either way, so
// only COVERED queries are compared and tallied (the harness would read an abstention as a difference). Coverage (the share of
// the builders' damage queries the kernel covers) is counted beside the tally. Queries come from the feature builders themselves
// (`collectQueries`, as the harness's self-differential), the damage ones kept.
//
// - Regression corpora: the committed random, search-fidelity and unavailable corpora (debugging evidence, never admission).
// - The fresh corpus: seeded battles on the generator pool, drawn from the kernel version's fresh seed (ADR-030 rule 7), so it
//   cannot exist before the kernel's commit is frozen; battles are played until the corpus holds `MIN_FRESH_ITEMS` distinct
//   covered queries.
// Evaluation code never imports this module (a source scan in its test).
import {
  addBatch,
  type Backend,
  buildLedger,
  type BackendAdmission,
  compareBatch,
  type CorpusSpec,
  type CorpusTally,
  freshCorpusId,
  freshSeed,
  type Level,
  MIN_FRESH_ITEMS,
  newTally,
  sha256Jcs,
} from "@areneon/differential";
import {
  abstained,
  type Engine,
  type GenerationTeamPool,
  type KernelBackend,
  kernelBackend,
  type MechanicsResult,
  mechanicsQueryHash,
  type ParsedMechanicsQuery,
  type ProbeBackend,
  probeBackend,
  RANDOM_MAX_TURNS,
  type RandomBattleFile,
  type Settled,
} from "@areneon/showdown-bridge";
import { collectQueries, COMMITTED_CORPORA, corpusBattles, corpusHash, type HarnessConfigs, harnessConfigs, mechanicsHarnessLevel, remembering } from "./differential-mechanics.ts";
import { playSweepBattle } from "./reducer-sweep.ts";

/** The backend kind the ledger names. */
export const KERNEL_KIND = "damage-kernel";

/** What one corpus through the kernel gave: the tally of the covered queries and the coverage counts. */
export interface KernelCorpusRun {
  tally: CorpusTally;
  /** Distinct damage queries the builders issued, and how many the kernel covered. */
  damage_queries: number;
  covered: number;
  /** Abstentions by entity (`<kind>:<id>`). */
  abstained: Map<string, number>;
  battles: number;
}

/** The kernel and the oracle of one harness run. */
export interface KernelHarness {
  engine: Engine;
  configs: HarnessConfigs;
  kernel: KernelBackend;
  level: Level<ParsedMechanicsQuery, MechanicsResult>;
}

export function kernelHarness(engine: Engine, backendCommit: string): KernelHarness {
  const configs = harnessConfigs(engine);
  return { engine, configs, kernel: kernelBackend(engine, { config: configs.oracle, snapshot: configs.snapshot, backend_commit: backendCommit }), level: mechanicsHarnessLevel(engine) };
}

/** A run's state while queries are added: the tally, the distinct keys seen and the coverage counts. */
interface Accumulator {
  run: KernelCorpusRun;
  seen: Set<string>;
}

const accumulator = (spec: CorpusSpec): Accumulator => ({ run: { tally: newTally(spec), damage_queries: 0, covered: 0, abstained: new Map(), battles: 0 }, seen: new Set() });

/**
 * Add one battle's damage queries: each distinct query once; the kernel's abstentions counted (by entity) and left out; the
 * covered ones compared with the oracle in batches. `limit` stops at that many covered queries (the fresh corpus's size).
 */
function addBattle(h: KernelHarness, acc: Accumulator, battle: RandomBattleFile, oracle: () => ProbeBackend, limit: number, batchSize: number): void {
  const probe = oracle();
  const found = collectQueries(h.engine, h.configs, [battle], probe);
  acc.run.battles += 1;
  const covered: ParsedMechanicsQuery[] = [];
  for (const query of found.queries) {
    if (query.query.kind !== "damage") continue;
    const key = mechanicsQueryHash(query);
    if (acc.seen.has(key)) continue;
    acc.seen.add(key);
    acc.run.damage_queries += 1;
    const [settled] = h.kernel.run([query]) as [Settled];
    if (abstained(settled)) {
      const entity = (settled as { error: { entity: string } }).error.entity;
      acc.run.abstained.set(entity, (acc.run.abstained.get(entity) ?? 0) + 1);
      continue;
    }
    if (acc.run.covered + covered.length >= limit) continue;
    covered.push(query);
  }
  acc.run.covered += covered.length;
  for (let first = 0; first < covered.length; first += batchSize) {
    addBatch(acc.run.tally, compareBatch(h.level, probe as Backend<ParsedMechanicsQuery, MechanicsResult>, h.kernel as Backend<ParsedMechanicsQuery, MechanicsResult>, covered.slice(first, first + batchSize), acc.run.tally.items));
  }
}

export interface RunOptions {
  batchSize?: number;
  /** The oracle of one battle (default: the pinned probe, remembering its answers). */
  oracle?: () => ProbeBackend;
}

const defaultOracle = (h: KernelHarness) => () => remembering(probeBackend(h.engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));

/** One committed corpus as a regression corpus of the kernel (the first `limit` battles when given). */
export function kernelRegression(h: KernelHarness, corpus: { id: string; dir: string }, limit?: number, options: RunOptions = {}): KernelCorpusRun {
  const source = corpusHash(corpus.dir, limit);
  const acc = accumulator({ id: corpus.id, kind: "regression", seed: source, pool: source });
  for (const battle of corpusBattles(corpus.dir, limit)) addBattle(h, acc, battle, options.oracle ?? defaultOracle(h), Infinity, options.batchSize ?? 256);
  return acc.run;
}

/** Every committed corpus as a regression corpus. */
export const kernelRegressions = (h: KernelHarness, limit?: number, options: RunOptions = {}): KernelCorpusRun[] => COMMITTED_CORPORA.map((c) => kernelRegression(h, c, limit, options));

/** The pool a fresh corpus is drawn from, by its identity (the team hashes in pool order). */
export const poolHash = (pool: GenerationTeamPool): string => sha256Jcs({ hashes: [...pool.hashes], purpose: "damage-kernel-pool/1" });

/** The battle a fresh corpus plays at `index`: a function of the kernel version's fresh seed and the pool alone. */
export async function freshBattle(h: KernelHarness, pool: GenerationTeamPool, index: number, maxTurns: number = RANDOM_MAX_TURNS): Promise<RandomBattleFile | null> {
  const battle = await playSweepBattle(h.engine, h.configs.oracle.format_id, pool, { run_key: `fresh-${freshSeed(h.kernel)}`, mix: "random-legal", index, max_turns: maxTurns });
  if (battle.failure !== undefined) return null;
  const channel = (player: "p1" | "p2") => ({ schema_version: 1 as const, data_class: "synthetic" as const, player, lines: battle.channels[player] });
  return { schema_version: 1, data_class: "synthetic", provenance: "fresh", id: `f${index}`, seed: battle.seed, actor_seeds: { p1: 0, p2: 0 }, teams: battle.teams, winner: battle.winner, turns: battle.turns, channels: { p1: channel("p1"), p2: channel("p2") } };
}

/**
 * The fresh corpus of the kernel version: battles `0, 1, ...` of the fresh draw on `pool`, every distinct covered damage query
 * compared, until `size` (default the level's minimum, 200,000) are held. `onBattle` reports progress.
 */
export async function kernelFresh(h: KernelHarness, pool: GenerationTeamPool, size: number = MIN_FRESH_ITEMS["mechanics-query"], options: RunOptions & { onBattle?: (run: KernelCorpusRun) => void; maxBattles?: number } = {}): Promise<KernelCorpusRun> {
  const spec: CorpusSpec = { id: freshCorpusId(h.kernel), kind: "fresh", seed: freshSeed(h.kernel), pool: poolHash(pool) };
  const acc = accumulator(spec);
  for (let index = 0; acc.run.covered < size && index < (options.maxBattles ?? Infinity); index++) {
    const battle = await freshBattle(h, pool, index);
    if (battle !== null) addBattle(h, acc, battle, options.oracle ?? defaultOracle(h), size, options.batchSize ?? 256);
    options.onBattle?.(acc.run);
  }
  return acc.run;
}

/** The admission ledger of the kernel version from its corpora (one fresh corpus, any regression corpora). */
export function kernelLedger(h: KernelHarness, runs: readonly KernelCorpusRun[]): BackendAdmission {
  return buildLedger({
    level: "mechanics-query",
    backend: { kind: KERNEL_KIND, backend_format: h.kernel.backend_format, backend_commit: h.kernel.backend_commit },
    showdown_commit: h.engine.commit,
    tallies: runs.map((r) => r.tally),
  });
}
