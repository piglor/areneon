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
// - The targeted corpus (`fixtures/backends/<format>/damage-kernel-targeted.json`): every case a review, a fresh corpus or a shadow
//   check found, as a reviewed query spec with its reason; it must be fully equal or abstained before any version is admitted.
// - The interaction sweep: one query for every reachable (move with an unusual stat or event rule, or one ordinary move per type)
//   x (every ability, item and condition of the effects record that acts on that move's events), derived from the record and the
//   engine's data, never by hand; the rare pairs a random draw misses.
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
  exportPaste,
  goldenTeamVocabulary,
  kernelData,
  type KernelEffects,
  loadKernelEffects,
  readMechanicsQuery,
  type PokemonSet,
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
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { collectQueries, corpusBattles, corpusHash, type HarnessConfigs, harnessConfigs, mechanicsHarnessLevel, remembering } from "./differential-mechanics.ts";
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

const BATCH = 256;

const defaultOracle = (h: KernelHarness) => () => remembering(probeBackend(h.engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));

/** One committed corpus as a regression corpus of the kernel (the first `limit` battles when given). */
export function kernelRegression(h: KernelHarness, corpus: { id: string; dir: string }, limit?: number): KernelCorpusRun {
  const source = corpusHash(corpus.dir, limit);
  const acc = accumulator({ id: corpus.id, kind: "regression", seed: source, pool: source });
  for (const battle of corpusBattles(corpus.dir, limit)) addBattle(h, acc, battle, defaultOracle(h), Infinity, BATCH);
  return acc.run;
}

/** The pool a fresh corpus is drawn from, by its identity (the team hashes in pool order). */
export const poolHash = (pool: GenerationTeamPool): string => sha256Jcs({ hashes: [...pool.hashes], purpose: "damage-kernel-pool/1" });

/** The battle a fresh corpus plays at `index`: a function of the kernel version's fresh seed and the pool alone. */
export async function freshBattle(h: KernelHarness, pool: GenerationTeamPool, index: number, maxTurns: number = RANDOM_MAX_TURNS): Promise<RandomBattleFile> {
  const battle = await playSweepBattle(h.engine, h.configs.oracle.format_id, pool, { run_key: `fresh-${freshSeed(h.kernel)}`, mix: "random-legal", index, max_turns: maxTurns });
  // A battle that did not finish keeps no channels, so it issues no query (the draw is still the next index's).
  const channel = (player: "p1" | "p2") => ({ schema_version: 1 as const, data_class: "synthetic" as const, player, lines: battle.channels[player] });
  return { schema_version: 1, data_class: "synthetic", provenance: "fresh", id: `f${index}`, seed: battle.seed, actor_seeds: { p1: 0, p2: 0 }, teams: battle.teams, winner: battle.winner, turns: battle.turns, channels: { p1: channel("p1"), p2: channel("p2") } };
}

/**
 * The fresh corpus of the kernel version: battles `0, 1, ...` of the fresh draw on `pool`, every distinct covered damage query
 * compared, until `size` (default the level's minimum, 200,000) are held, or `maxBattles` were played.
 */
export async function kernelFresh(h: KernelHarness, pool: GenerationTeamPool, size: number = MIN_FRESH_ITEMS["mechanics-query"], maxBattles = Number.POSITIVE_INFINITY): Promise<KernelCorpusRun> {
  const spec: CorpusSpec = { id: freshCorpusId(h.kernel), kind: "fresh", seed: freshSeed(h.kernel), pool: poolHash(pool) };
  const acc = accumulator(spec);
  for (let index = 0; acc.run.covered < size && index < maxBattles; index++) addBattle(h, acc, await freshBattle(h, pool, index), defaultOracle(h), size, BATCH);
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

// ---------------------------------------------------------------- inputs, the committed ledger and the isolation scan

/** Every file whose bytes can change a kernel answer: the kernel, the probe envelope it shares, the KO counts, its data. */
export const KERNEL_INPUTS: readonly string[] = [
  "config/backends/gen9championsvgc2026regmc.damage-kernel.effects.json",
  "fixtures/showdown/gen9championsvgc2026regmc/mechanics-inventory.json",
  "packages/showdown-bridge/src/damage-kernel-backend.ts",
  "packages/showdown-bridge/src/damage-kernel-data.ts",
  "packages/showdown-bridge/src/damage-kernel-effects.ts",
  "packages/showdown-bridge/src/damage-kernel.ts",
  "packages/showdown-bridge/src/mechanics-ko.ts",
  "packages/showdown-bridge/src/mechanics-probe.ts",
];

/** The committed ledger of the kernel version (its lines form, `admissionLines`) and the hash it must have. */
export const KERNEL_LEDGER = "config/backends/gen9championsvgc2026regmc.damage-kernel.admission.jsonl";
export const KERNEL_ADMISSION_HASH = "7d49b2b64da169e89857790dc38977350a73844f9d9b1ff73066085bafbaa091";

/** The names the bridge's index exports from the kernel's modules (read from the index, so a new export is scanned too). */
export function kernelExportNames(indexSource: string): string[] {
  const names = new Set<string>();
  for (const m of indexSource.matchAll(/export\s*\{([^}]*)\}\s*from\s*"\.\/damage-kernel[a-z-]*\.ts"/g)) {
    for (const part of (m[1] as string).split(",")) {
      const name = part.replace(/^\s*type\s+/, "").trim().split(/\s+as\s+/)[0]?.trim();
      if (name) names.add(name);
    }
  }
  return [...names].sort();
}

/** Whether a source imports the kernel: one of its modules (or the composition root) by path, or one of its names from the index. */
export function importsKernel(source: string, names: readonly string[]): boolean {
  // A module specifier (static import or re-export, dynamic import, require) that names a kernel module or this composition root.
  if (/(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)["'][^"']*(damage-kernel|differential-kernel)[^"']*["']/.test(source)) return true;
  for (const m of source.matchAll(/import\s*(?:type\s*)?\{([^}]*)\}\s*from\s*["']@areneon\/showdown-bridge["']/g)) {
    const imported = (m[1] as string).split(",").map((p) => p.replace(/^\s*type\s+/, "").trim().split(/\s+as\s+/)[0]?.trim());
    if (imported.some((n) => n !== undefined && names.includes(n))) return true;
  }
  return false;
}

// ---------------------------------------------------------------- the targeted corpus and the interaction sweep

/** One Pokemon of a query spec, in engine ids. */
export interface SideSpec {
  species: string;
  ability: string;
  item?: string;
  moves: string[];
  nature?: string;
  status?: string | null;
  hp?: number | null;
  mega?: boolean;
  boosts?: Partial<Record<"atk" | "def" | "spa" | "spd" | "spe", number>>;
}
export interface QuerySpec {
  id: string;
  reason: string;
  attacker: SideSpec;
  defender: SideSpec;
  move: string;
  state: { weather?: string | null; terrain?: string | null; trick_room?: boolean; sides?: Partial<Record<"aurora_veil" | "light_screen" | "reflect" | "tailwind", boolean>>[] };
}

export const TARGETED_FILE = fileURLToPath(new URL("../../../fixtures/backends/gen9championsvgc2026regmc/damage-kernel-targeted.json", import.meta.url));

interface NameDex {
  species: { get(id: string): { name: string } };
  moves: { get(id: string): { name: string } };
  abilities: { get(id: string): { name: string } };
  items: { get(id: string): { name: string } };
}

/** The parsed query of a spec: the sets written by the engine's own export (level 50, 11 Stat Points each, Hardy unless given). */
export function specQuery(engine: Engine, spec: QuerySpec, formatId: string): ParsedMechanicsQuery {
  const dex = (engine.sim.Dex as unknown as { mod(m: string): NameDex }).mod(engine.sim.Dex.formats.get(formatId).mod);
  const paste = (s: SideSpec): string =>
    exportPaste(engine, [
      {
        name: "",
        species: dex.species.get(s.species).name,
        item: s.item === undefined ? "" : dex.items.get(s.item).name,
        ability: dex.abilities.get(s.ability).name,
        moves: s.moves.map((m) => dex.moves.get(m).name),
        nature: s.nature ?? "Hardy",
        gender: "",
        evs: { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 },
        ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
        level: 50,
      } as PokemonSet,
    ]);
  const zero = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  const combatant = (s: SideSpec) => ({ set: paste(s), boosts: { ...zero, ...s.boosts }, status: s.status ?? null, hp: s.hp ?? null, mega: s.mega ?? false });
  const side = (v: Partial<Record<string, boolean>> = {}) => ({ aurora_veil: false, light_screen: false, reflect: false, tailwind: false, ...v });
  return readMechanicsQuery({
    schema_version: 1,
    data_class: "synthetic",
    format_id: formatId,
    query: {
      kind: "damage",
      attacker: combatant(spec.attacker),
      defender: combatant(spec.defender),
      move: spec.move,
      state: { weather: spec.state.weather ?? null, terrain: spec.state.terrain ?? null, trick_room: spec.state.trick_room ?? false, sides: [side(spec.state.sides?.[0]), side(spec.state.sides?.[1])] },
    },
  });
}

/** The committed targeted specs. */
export const targetedSpecs = (): QuerySpec[] => (JSON.parse(readFileSync(TARGETED_FILE, "utf8")) as { entries: QuerySpec[] }).entries;

interface Roster {
  id: string;
  abilities: string[];
  moves: string[];
  required_item?: string;
}

/**
 * The interaction sweep: for every move whose data gives it an unusual stat or event rule (a stat override, Foul Play's target,
 * ignored defence, a set or raised crit, a spread target, priority, ignored immunity) and one ordinary damaging move per type,
 * every ability, item and condition of the effects record, on the side it acts from: a holder of the ability or item that knows
 * the move (attacker side) or a holder as the defender against a user of the move; a condition in the declared state, with a
 * defender of the condition's type for a weather's stat rule. Only combinations the roster can hold (the validator decides at the
 * probe); each spec once.
 */
export function interactionSweep(engine: Engine, formatId: string, effects: KernelEffects = loadKernelEffects(formatId, engine.commit)): QuerySpec[] {
  const data = kernelData(engine, formatId);
  const roster = (goldenTeamVocabulary().roster as Roster[]).filter((r) => r.required_item === undefined);
  const inertAbility = (r: Roster): string | undefined => r.abilities.find((a) => (data.abilityBlockers(a, false) ?? ["x"]).length === 0);
  const damaging = (id: string): boolean => {
    const m = data.move(id);
    return m !== null && m.category !== "Status" && !m.outside_scope && m.base_power > 0;
  };
  const learnable = [...new Set(roster.flatMap((r) => r.moves))].filter(damaging).sort();
  const unusual = learnable.filter((id) => {
    const m = data.move(id);
    return m !== null && (m.offensive_stat !== null || m.offensive_target || m.defensive_stat !== null || m.ignore_defensive || m.will_crit || m.crit_ratio > 1 || m.target === "allAdjacentFoes" || m.target === "allAdjacent" || m.priority > 0 || m.ignore_immunity !== null);
  });
  const perType = new Map<string, string>();
  for (const id of learnable) {
    const m = data.move(id);
    if (m !== null && m.blockers.length === 0 && !perType.has(m.type)) perType.set(m.type, id);
  }
  const moves = [...new Set([...unusual, ...perType.values()])].sort();
  const userOf = (move: string, ability?: string): SideSpec | null => {
    const r = roster.find((x) => x.moves.includes(move) && (ability === undefined ? inertAbility(x) !== undefined : x.abilities.includes(ability)));
    return r === undefined ? null : { species: r.id, ability: ability ?? (inertAbility(r) as string), moves: [move] };
  };
  const holderOf = (ability?: string, type?: string): SideSpec | null => {
    const r = roster.find((x) => (ability === undefined ? inertAbility(x) !== undefined : x.abilities.includes(ability)) && (type === undefined || (data.species(x.id)?.types ?? []).includes(type)) && x.moves.some(damaging));
    return r === undefined ? null : { species: r.id, ability: ability ?? (inertAbility(r) as string), moves: [r.moves.find(damaging) as string] };
  };
  const plainDefender = holderOf();
  const out = new Map<string, QuerySpec>();
  const add = (id: string, attacker: SideSpec | null, defender: SideSpec | null, move: string, state: QuerySpec["state"] = {}): void => {
    if (attacker === null || defender === null || out.has(id)) return;
    out.set(id, { id, reason: "interaction sweep", attacker, defender, move, state });
  };
  for (const move of moves) {
    for (const ability of Object.keys(effects.abilities)) {
      add(`a:${ability}:${move}`, userOf(move, ability), plainDefender, move);
      add(`d:${ability}:${move}`, userOf(move), holderOf(ability), move);
    }
    for (const item of Object.keys(effects.items)) {
      const user = userOf(move);
      add(`ai:${item}:${move}`, user === null ? null : { ...user, item }, plainDefender, move);
      add(`di:${item}:${move}`, userOf(move), plainDefender === null ? null : { ...plainDefender, item }, move);
    }
    for (const [id, c] of Object.entries(effects.conditions)) {
      const state: QuerySpec["state"] =
        c.kind === "weather_damage" || c.kind === "weather_stat"
          ? { weather: id }
          : c.kind === "terrain"
            ? { terrain: id }
            : id === "trickroom"
              ? { trick_room: true }
              : { sides: [{}, { [id === "lightscreen" ? "light_screen" : id === "auroraveil" ? "aurora_veil" : id]: true }] };
      add(`c:${id}:${move}`, userOf(move), c.kind === "weather_stat" ? holderOf(undefined, c.type) : plainDefender, move, state);
    }
  }
  return [...out.values()];
}

/** A corpus of query specs through the kernel: each query distinct, abstentions counted, the covered ones compared with the oracle. */
export function kernelSpecs(h: KernelHarness, id: string, specs: readonly QuerySpec[]): KernelCorpusRun {
  const source = sha256Jcs(specs as never);
  const acc = accumulator({ id, kind: "regression", seed: source, pool: source });
  const probe = defaultOracle(h)();
  const covered: ParsedMechanicsQuery[] = [];
  for (const spec of specs) {
    const query = specQuery(h.engine, spec, h.configs.oracle.format_id);
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
    covered.push(query);
  }
  acc.run.covered = covered.length;
  for (let first = 0; first < covered.length; first += BATCH) {
    addBatch(acc.run.tally, compareBatch(h.level, probe as Backend<ParsedMechanicsQuery, MechanicsResult>, h.kernel as Backend<ParsedMechanicsQuery, MechanicsResult>, covered.slice(first, first + BATCH), acc.run.tally.items));
  }
  return acc.run;
}
