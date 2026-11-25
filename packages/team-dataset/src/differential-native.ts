// The native probe through the differential harness (ADR-030 level 1, #1534): the composition root that joins the engine-free core
// (`@areneon/differential`), the pinned probe as the oracle and the native probe backend (`nativeProbeBackend`: the declared-state
// layer of the vendored Ninjask fork), at the mechanics-query level. The backend answers `damage`, `speed_order` and `stats` queries;
// what it abstains on (a history block, a move whose pre-hit steps it does not model, a state the engine refuses) routes to the pinned
// probe and is no evidence either way, so only ANSWERED queries are compared and tallied (the harness would read an abstention as a
// difference); coverage is counted beside the tally. Evaluation code never imports this module (a source scan in its test).
//
// Corpora (the same rules as the damage kernel's, `differential-kernel.ts`):
// - regression: the committed random, search-fidelity and unavailable corpora, queried as the `probe-damage/2` builder issues them;
//   the move sweep (every damaging move of the roster through eleven states) and the speed sweep (seeded random `speed_order`
//   queries over every speed-relevant ability, item, status, field and Mega state); the damage kernel's interaction sweep and its
//   targeted corpus. Debugging evidence, never admission.
// - fresh: seeded random-legal battles on the generator pool, drawn from the native backend version's fresh seed (ADR-030 rule 7).
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
  buildFeaturesG,
  createFeatureMemo,
  decodeRequest,
  driveSync,
  type Engine,
  enumerateCandidates,
  exportPaste,
  FeatureBuildError,
  type GenerationTeamPool,
  goldenTeamVocabulary,
  mechanicsQueryHash,
  type MechanicsResult,
  nativeAbstained,
  type NativeProbeBackend,
  type NativeProbeCommand,
  nativeProbeBackend,
  type ParsedMechanicsQuery,
  type PokemonSet,
  type Player,
  playerChannel,
  type ProbeBackend,
  probeBackend,
  RANDOM_MAX_TURNS,
  type RandomBattleFile,
  readMechanicsQuery,
  type Settled,
} from "@areneon/showdown-bridge";
import { type HarnessConfigs, harnessConfigs, mechanicsHarnessLevel, remembering, COMMITTED_CORPORA, corpusBattles, corpusHash } from "./differential-mechanics.ts";
import { playSweepBattle } from "./reducer-sweep.ts";

/** The backend kind the ledger names. */
export const NATIVE_KIND = "native-probe";

/** The feature set whose queries the native backend is asked: `probe-damage/2` (it holds `/1`'s by the same code). */
export const NATIVE_FEATURE_SET = "probe-damage/2" as const;

/** What one corpus through the native backend gave: the tally of the answered queries and the coverage counts. */
export interface NativeCorpusRun {
  tally: CorpusTally;
  /** Distinct queries the builders (or the sweep) issued, by kind, and how many the native backend answered (by kind). */
  queries: Record<string, number>;
  answered: Record<string, number>;
  /** Abstentions by entity (`<kind>:<id>`). */
  abstained: Map<string, number>;
  battles: number;
}

export interface NativeHarness {
  engine: Engine;
  configs: HarnessConfigs;
  native: NativeProbeBackend;
  level: Level<ParsedMechanicsQuery, MechanicsResult>;
}

export function nativeHarness(engine: Engine, backendCommit: string, command: NativeProbeCommand): NativeHarness {
  const configs = harnessConfigs(engine);
  return { engine, configs, native: nativeProbeBackend(engine, { config: configs.oracle, snapshot: configs.snapshot, command, backend_commit: backendCommit }), level: mechanicsHarnessLevel(engine) };
}

const BATCH = 256;
const bump = (table: Record<string, number>, key: string): void => void (table[key] = (table[key] ?? 0) + 1);

interface Accumulator {
  run: NativeCorpusRun;
  seen: Set<string>;
}

const accumulator = (spec: CorpusSpec): Accumulator => ({ run: { tally: newTally(spec), queries: {}, answered: {}, abstained: new Map(), battles: 0 }, seen: new Set() });

/**
 * Add queries: each distinct query once; the native backend's abstentions counted (by entity) and left out; the answered ones
 * compared with the oracle in batches.
 */
function addQueries(h: NativeHarness, acc: Accumulator, queries: readonly ParsedMechanicsQuery[], oracle: ProbeBackend): void {
  const answered: ParsedMechanicsQuery[] = [];
  for (let first = 0; first < queries.length; first += BATCH) {
    const slice: ParsedMechanicsQuery[] = [];
    for (const q of queries.slice(first, first + BATCH)) {
      const key = mechanicsQueryHash(q);
      if (acc.seen.has(key)) continue;
      acc.seen.add(key);
      bump(acc.run.queries, q.query.kind);
      slice.push(q);
    }
    h.native.run(slice).forEach((settled, i) => {
      const q = slice[i] as ParsedMechanicsQuery;
      if (nativeAbstained(settled)) {
        const entity = (settled as { error: { entity: string } }).error.entity;
        acc.run.abstained.set(entity, (acc.run.abstained.get(entity) ?? 0) + 1);
      } else {
        bump(acc.run.answered, q.query.kind);
        answered.push(q);
      }
    });
  }
  for (let first = 0; first < answered.length; first += BATCH) {
    addBatch(acc.run.tally, compareBatch(h.level, oracle as Backend<ParsedMechanicsQuery, MechanicsResult>, h.native as Backend<ParsedMechanicsQuery, MechanicsResult>, answered.slice(first, first + BATCH), acc.run.tally.items));
  }
}

/** A build the builder itself refuses (`reader_flagged`, `assumed_set`) is no query; any other error is a defect. */
export function refusedBuild(error: unknown): void {
  if (!(error instanceof FeatureBuildError)) throw error;
}

/** The queries (each once per game and seat's memo; `nativeQueries` and the accumulator keep each once) the `probe-damage/2` builder issues at every request of both seats of `battles`, the oracle answering each as it is asked (one memo per game, as in play). */
export function builderQueries(h: NativeHarness, battles: readonly RandomBattleFile[], oracle: ProbeBackend): ParsedMechanicsQuery[] {
  const queries: ParsedMechanicsQuery[] = [];
  const ask = (query: ParsedMechanicsQuery): MechanicsResult => {
    queries.push(query);
    const settled = oracle.run([query])[0] as Settled;
    if (!settled.ok) throw settled.error;
    return settled.result;
  };
  for (const battle of battles) {
    for (const player of ["p1", "p2"] as Player[]) {
      const lines = battle.channels[player].lines;
      const team = battle.teams[player] as PokemonSet[];
      const memo = createFeatureMemo();
      lines.forEach((line, i) => {
        if (!line.startsWith("|request|")) return;
        const decoded = decodeRequest(line.slice("|request|".length));
        if (decoded.phase === "wait") return;
        const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
        const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player, lines: lines.slice(0, i + 1) });
        try {
          driveSync(buildFeaturesG(h.engine, h.configs.sets[NATIVE_FEATURE_SET], channel, team, candidates, memo, NATIVE_FEATURE_SET), ask);
        } catch (error) {
          refusedBuild(error);
        }
      });
    }
  }
  return queries;
}

const defaultOracle = (h: NativeHarness) => () => remembering(probeBackend(h.engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));

/** One committed corpus as a regression corpus of the native backend (the first `limit` battles when given). */
export function nativeRegression(h: NativeHarness, corpus: { id: string; dir: string }, limit?: number): NativeCorpusRun {
  const source = corpusHash(corpus.dir, limit);
  const acc = accumulator({ id: corpus.id, kind: "regression", seed: source, pool: source });
  for (const battle of corpusBattles(corpus.dir, limit)) {
    const oracle = defaultOracle(h)();
    addQueries(h, acc, builderQueries(h, [battle], oracle), oracle);
    acc.run.battles += 1;
  }
  return acc.run;
}

/** The pool a fresh corpus is drawn from, by its identity (the team hashes in pool order). */
export const poolHash = (pool: GenerationTeamPool): string => sha256Jcs({ hashes: [...pool.hashes], purpose: "native-probe-pool/1" });

/** The battle a fresh corpus plays at `index`: a function of the backend version's fresh seed and the pool alone. */
export async function freshBattle(h: NativeHarness, pool: GenerationTeamPool, index: number, maxTurns: number = RANDOM_MAX_TURNS): Promise<RandomBattleFile> {
  const battle = await playSweepBattle(h.engine, h.configs.oracle.format_id, pool, { run_key: `fresh-${freshSeed(h.native)}`, mix: "random-legal", index, max_turns: maxTurns });
  const channel = (player: "p1" | "p2") => ({ schema_version: 1 as const, data_class: "synthetic" as const, player, lines: battle.channels[player] });
  return { schema_version: 1, data_class: "synthetic", provenance: "fresh", id: `f${index}`, seed: battle.seed, actor_seeds: { p1: 0, p2: 0 }, teams: battle.teams, winner: battle.winner, turns: battle.turns, channels: { p1: channel("p1"), p2: channel("p2") } };
}

/** A corpus of explicit queries through the backend (a sweep): each distinct, abstentions counted, the answered ones compared. */
export function nativeQueries(h: NativeHarness, id: string, queries: readonly ParsedMechanicsQuery[]): NativeCorpusRun {
  const source = sha256Jcs(queries.map((q) => mechanicsQueryHash(q)));
  const acc = accumulator({ id, kind: "regression", seed: source, pool: source });
  addQueries(h, acc, queries, defaultOracle(h)());
  return acc.run;
}

/** The admission ledger of the backend version from its corpora (one fresh corpus, any regression corpora). */
export function nativeLedger(h: NativeHarness, runs: readonly NativeCorpusRun[]): BackendAdmission {
  return buildLedger({ level: "mechanics-query", backend: { kind: NATIVE_KIND, backend_format: h.native.backend_format, backend_commit: h.native.backend_commit }, showdown_commit: h.engine.commit, tallies: runs.map((r) => r.tally) });
}

export { COMMITTED_CORPORA, MIN_FRESH_ITEMS, freshCorpusId, freshSeed };

// ---------------------------------------------------------------- the sweeps

interface Roster {
  id: string;
  abilities: string[];
  moves: string[];
  required_item?: string;
}

interface NameDex {
  species: { get(id: string): { name: string; baseSpecies: string; id: string } };
  moves: { get(id: string): { name: string; category: string; priority: number; exists: boolean } };
  abilities: { get(id: string): { name: string } };
  items: { get(id: string): { name: string; exists: boolean } };
}

const SIDE = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const ZERO = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const STAT_POINTS = { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 };

function nameDex(engine: Engine, format: string): NameDex {
  return (engine.sim.Dex as unknown as { mod(m: string): NameDex }).mod(engine.sim.Dex.formats.get(format).mod);
}

const setPaste = (engine: Engine, dex: NameDex, s: { species: string; ability: string; moves: string[] }): string =>
  exportPaste(engine, [
    {
      name: "",
      species: dex.species.get(s.species).name,
      item: "",
      ability: dex.abilities.get(s.ability).name,
      moves: s.moves.map((m) => dex.moves.get(m).name),
      nature: "Hardy",
      gender: "",
      evs: STAT_POINTS,
      ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
      level: 50,
    } as PokemonSet,
  ]);

/**
 * The move sweep: every damaging move of the roster (one or two users each) against three defenders through eleven states (plain,
 * both terrains that read the move, sun, rain, one and two screens, Aurora Veil, a sleeping user and Trick Room). The roster, the
 * moves and the defenders come from the pinned vocabulary and engine; no move is named in code but the three defenders' species,
 * which are only the targets.
 */
export function moveSweep(engine: Engine, format: string): ParsedMechanicsQuery[] {
  const dex = nameDex(engine, format);
  const roster = (goldenTeamVocabulary().roster as Roster[]).filter((r) => r.required_item === undefined);
  const moves = [...new Set(roster.flatMap((r) => r.moves))].filter((m) => dex.moves.get(m).category !== "Status").sort();
  const defenders = ["garchomp", "blissey", "gengar"].map((id) => roster.find((r) => r.id === id)).filter((r): r is Roster => r !== undefined);
  const states: { state: { weather?: string; terrain?: string; trick_room?: boolean; sides?: object[] }; status?: string }[] = [
    { state: {} },
    { state: { terrain: "psychicterrain" } },
    { state: { terrain: "grassyterrain" } },
    { state: { weather: "sunnyday" } },
    { state: { weather: "raindance" } },
    { state: { sides: [{}, { reflect: true }] } },
    { state: { sides: [{}, { reflect: true, light_screen: true }] } },
    { state: { sides: [{}, { aurora_veil: true }] } },
    { state: {}, status: "slp" },
    { state: { trick_room: true } },
  ];
  const out: ParsedMechanicsQuery[] = [];
  for (const move of moves) {
    for (const user of roster.filter((r) => r.moves.includes(move)).slice(0, 2)) {
      for (const d of defenders) {
        for (const st of states) {
          try {
            const side = (i: number) => ({ ...SIDE, ...((st.state.sides?.[i] as object | undefined) ?? {}) });
            const query = readMechanicsQuery({
              schema_version: 1,
              data_class: "synthetic",
              format_id: format,
              query: {
                kind: "damage",
                attacker: { set: setPaste(engine, dex, { species: user.id, ability: user.abilities[0] as string, moves: [move] }), boosts: ZERO, status: st.status ?? null, hp: null, mega: false },
                defender: { set: setPaste(engine, dex, { species: d.id, ability: d.abilities[0] as string, moves: [d.moves[0] as string] }), boosts: ZERO, status: null, hp: null, mega: false },
                move,
                state: { weather: st.state.weather ?? null, terrain: st.state.terrain ?? null, trick_room: st.state.trick_room ?? false, sides: [side(0), side(1)] },
              },
            });
            out.push(query);
          } catch {
            // the roster cannot hold that set
          }
        }
      }
    }
  }
  return out;
}

/**
 * The speed sweep: `n` seeded random `speed_order` queries over valid sets of the roster (the pinned validator judges each set), their
 * abilities and items (the speed-relevant ones weighted), Mega-capable species and stones, statuses, stages, HP, fields, Trick Room and
 * Tailwind, with priority moves and switches. A function of `(engine, format, n, seed)` alone.
 */
export function speedSweep(engine: Engine, format: string, n: number, seed: number): ParsedMechanicsQuery[] {
  let state = seed >>> 0;
  const rnd = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rnd() * xs.length)] as T;
  const dex = nameDex(engine, format);
  const vocab = goldenTeamVocabulary() as unknown as { roster: Roster[]; items: string[] };
  const roster = vocab.roster;
  const validator = engine.sim.TeamValidator.get(format);
  // Speed-relevant items by the engine's own data: those with a speed hook would need a list; the weighted picks are plain ids the engine checks exist.
  const itemPool = vocab.items.filter((i) => dex.items.get(i).exists);
  const pool: { paste: string; moves: string[]; stone: boolean }[] = [];
  const make = (stone: boolean): { paste: string; moves: string[]; stone: boolean } | null => {
    const r = pick(roster.filter((x) => (x.required_item !== undefined) === stone));
    const base = stone ? dex.species.get(dex.species.get(r.id).baseSpecies) : null;
    const baseRoster = base === null ? null : roster.find((x) => x.id === base.id);
    if (stone && baseRoster === undefined) return null;
    const moves = [...new Set([pick(r.moves), pick(r.moves), pick(r.moves), pick(r.moves)])];
    const item = r.required_item ?? (rnd() < 0.6 ? pick(itemPool) : "");
    const ability = baseRoster ? pick(baseRoster.abilities) : pick(r.abilities);
    const set = { name: "", species: base === null ? dex.species.get(r.id).name : base.name, item: item === "" ? "" : dex.items.get(item).name, ability: dex.abilities.get(ability).name, moves: moves.map((m) => dex.moves.get(m).name), nature: pick(["Timid", "Jolly", "Brave", "Quiet", "Hardy", "Relaxed"]), gender: "", evs: { ...STAT_POINTS, spe: Math.floor(rnd() * 32) }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, level: 50 } as PokemonSet;
    if ((validator.validateSet(structuredClone(set), {}) ?? []).length > 0) return null;
    return { paste: exportPaste(engine, [set]), moves, stone };
  };
  const setOf = (stone: boolean): { paste: string; moves: string[]; stone: boolean } => {
    const own = pool.filter((p) => p.stone === stone);
    if (own.length > 400) return pick(own);
    for (;;) {
      const made = make(stone);
      if (made !== null) {
        pool.push(made);
        return made;
      }
    }
  };
  const out: ParsedMechanicsQuery[] = [];
  while (out.length < n) {
    const sides: object[][] = [[], []];
    const moves: string[][][] = [[], []];
    const stones: boolean[][] = [[], []];
    for (let s = 0; s < 2; s++) {
      const count = rnd() < 0.7 ? 2 : 1;
      for (let i = 0; i < count; i++) {
        const stone = rnd() < 0.45 && !stones[s]?.includes(true);
        const set = setOf(stone);
        sides[s]?.push({ set: set.paste, boosts: { ...ZERO, spe: rnd() < 0.25 ? Math.floor(rnd() * 13) - 6 : 0 }, status: rnd() < 0.2 ? pick(["par", "brn", "slp", "psn", "tox", "frz"]) : null, hp: rnd() < 0.3 ? 1 + Math.floor(rnd() * 100) : null, mega: stone && rnd() < 0.5 });
        moves[s]?.push(set.moves);
        stones[s]?.push(set.stone);
      }
    }
    const actions: object[] = [];
    for (let s = 0; s < 2; s++) {
      const mine = sides[s] as { mega: boolean }[];
      for (let i = 0; i < mine.length; i++) {
        const declaredMega = (mine[i] as { mega: boolean }).mega;
        const move = rnd() < 0.15 ? null : pick(moves[s]?.[i] as string[]);
        actions.push({ side: s, position: i, move, mega: move !== null && !declaredMega && (stones[s]?.[i] as boolean) && rnd() < 0.5 });
      }
    }
    const side = () => ({ aurora_veil: rnd() < 0.05, light_screen: rnd() < 0.05, reflect: rnd() < 0.05, tailwind: rnd() < 0.2 });
    try {
      const query = readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: format, query: { kind: "speed_order", sides, actions, state: { weather: rnd() < 0.4 ? pick(["sunnyday", "raindance", "sandstorm", "snowscape"]) : null, terrain: rnd() < 0.3 ? pick(["grassyterrain", "psychicterrain", "electricterrain", "mistyterrain"]) : null, trick_room: rnd() < 0.2, sides: [side(), side()] } } });
      out.push(query);
    } catch {
      // a combination the contract refuses
    }
  }
  return out;
}
