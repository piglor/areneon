// The evaluation-only prediction-quality harness of the opponent model (ADR-025 section 13, slice 3 part 3).
//
// It compares the belief with the opponent's REAL sets over recorded games whose two teams are known, at every
// request of both seats: per field (item, ability, move) the top-1 accuracy and log-loss of the belief, of the
// prior alone (the belief at the seat's Team Preview request, which holds no evidence) and of `damage-aware@v1`'s
// fixed rule (a point prediction: no log-loss); a reliability table (calibration) of the belief; and the joint
// consistency of sampled worlds (share whose item AND moves agree with the real set, against the product of the
// single-field rates). The real team is read here as a test-side oracle, as `CloneJudge` verdicts are read only by
// evaluation code: this module is imported by its script and tests and by NO actor, bridge or opponent-model module
// (a source scan holds that), and nothing it returns holds a set, a name or a hash of a team: counts and integer ppm.
//
// Conventions (descriptive, never deciding):
// - A field is judged for the opponent Pokemon the real player brought; an Evolved Mega's ability is not judged.
// - Top-1 is the expected accuracy under a uniform random tie-break among the options tied for the highest
//   probability (integer ppm, floor of 10^6 / tied options): a flat belief scores 1 / options.
// - Log-loss is -ln(p) of the true value in micro-nats, rounded; a true value with probability zero is counted in
//   `zero_mass` and left out of the mean (the soundness test says it must not happen).
// - Moves: only the true moves not yet revealed at the request are judged, on the option list without the revealed
//   ones. The "top-1" of a slot asks whether the most likely not-yet-revealed move is one of the real moves; the
//   log-loss is of the move's inclusion probability. v1 predicts no unrevealed move, so it has no move row.
// - The joint rate counts (world, brought slot) pairs where the world also brings the slot; "moves agree" means
//   the world's four moves are exactly the real four.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type BeliefVocabulary, type OpponentBelief, type Ratio, type SlotMarginal, BeliefError, belief } from "@areneon/opponent-model";
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import {
  type AssumedSetContext,
  type BeliefWorldsPort,
  type Engine,
  type PokemonSet,
  type TeamVocabulary,
  buildAssumedSet,
  loadDamageAwareConfig,
  loadDamageAwareConfigV2,
  loadRulesSnapshot,
  readBeliefConfig,
  readUsagePrior,
  resolveFormat,
  teamVocabularyHash,
} from "@areneon/showdown-bridge";
import { readerSha256 } from "../../public-state/src/index.ts";
import { beliefWorldsFiles, createBeliefWorlds } from "./damage-aware-worlds.ts";
import { verifiedDatasetHash } from "./dataset-hash.ts";
import { DATASETS_DIR } from "./fixtures.ts";
import { datasetTeamPool } from "./loop-pool.ts";
import { playSweepBattle } from "./reducer-sweep.ts";
import { type PublicFlag, type PublicMon, type PublicState, interpretChannel, toId } from "../../public-state/src/index.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";

export type Seat = "p1" | "p2";
export const FIELDS = ["item", "ability", "move"] as const;
export type Field = (typeof FIELDS)[number];
export const METHODS = ["belief", "prior", "v1"] as const;
export type Method = (typeof METHODS)[number];
export type Phase = "preview" | "battle";
export const CALIBRATION_BINS = 10;

/** A recorded game with both teams known: the oracle's input. */
export interface OracleGame {
  readonly id: string;
  readonly teams: Readonly<Record<Seat, readonly PokemonSet[]>>;
  readonly channels: Readonly<Record<Seat, readonly string[]>>;
}

export interface QualityContext {
  readonly engine: Engine;
  readonly prior: UsagePrior;
  readonly vocabulary: BeliefVocabulary;
  readonly config: BeliefConfig;
  /** v1's fixed rule (`buildAssumedSet`): the engine, the team vocabulary and the v1 record. */
  readonly v1: AssumedSetContext;
  /** The v2 port the baseline samples worlds from (the same prior and belief config). */
  readonly port: BeliefWorldsPort;
  /** Worlds drawn per request. */
  readonly worlds: number;
}

/** One option of a distribution: the value and its probability. */
export interface Option {
  readonly id: string | null;
  readonly p: Ratio;
}

/** The context of the committed v2 record: its prior and belief config (read the way the baseline reads them), v1's rule and the v2 port. */
export function qualityContext(engine: Engine, vocabulary: TeamVocabulary): QualityContext {
  const configV2 = loadDamageAwareConfigV2(vocabulary);
  const files = beliefWorldsFiles(configV2.format_id);
  const json = (name: string): unknown => JSON.parse(readFileSync(join(files.model_dir, name), "utf8")) as unknown;
  const manifest = resolveFormat(engine, configV2.format_id, loadRulesSnapshot(engine.commit)).manifest;
  return {
    engine,
    prior: readUsagePrior(json("generator-matched.uniform-v1.json"), vocabulary),
    vocabulary: { ...vocabulary, vocabulary_hash: teamVocabularyHash(vocabulary) },
    config: readBeliefConfig(json("belief.v1.json"), vocabulary),
    v1: { engine, vocabulary, config: loadDamageAwareConfig("v1", vocabulary) },
    port: createBeliefWorlds({ engine, vocabulary, config: configV2, reader_sha256: readerSha256(), manifest }, files),
    worlds: configV2.assumed_set.worlds,
  };
}

/** The run key of the declared seeded set: the games are a function of it, the committed dev pool and the index. */
export const ORACLE_RUN_KEY = "prediction-quality-1";

/**
 * Seeded A1 games, random-legal against random-legal, on teams of the committed dev split (`fixtures-v1`), both teams
 * recorded: game `index` is a function of the run key, the index and the pool alone. A game that did not finish throws.
 */
export async function oracleGames(engine: Engine, formatId: string, start: number, count: number, maxTurns = 40): Promise<OracleGame[]> {
  const dir = join(DATASETS_DIR(formatId), "fixtures-v1");
  const pool = datasetTeamPool(engine, { format_id: formatId, team_dataset: { path: dir, dataset_hash: verifiedDatasetHash(engine, dir, formatId) }, generation: { excluded_abilities: [] } });
  const out: OracleGame[] = [];
  for (let index = start; index < start + count; index++) {
    const battle = await playSweepBattle(engine, formatId, pool, { run_key: ORACLE_RUN_KEY, mix: "random-legal", index, max_turns: maxTurns });
    if (battle.failure !== undefined) throw new Error(`oracle game ${index} did not finish: ${battle.failure.kind}`);
    out.push({ id: `${ORACLE_RUN_KEY}/${index}`, teams: battle.teams, channels: battle.channels });
  }
  return out;
}

// ---- the primitives (hand-checked in the tests) ----

/** The probability as a number in [0, 1], to 12 digits, from the exact ratio. */
export const probability = (p: Ratio): number => Math.max(Number((p.num * 1_000_000_000_000n) / p.den), 1) / 1e12;

export interface Judged {
  /** The expected top-1 hit in ppm: 10^6 / k when the true value is among the k options tied for the highest probability, else 0. */
  readonly hit_ppm: number;
  /** -ln(p) in micro-nats, or null when the true value has no mass. */
  readonly loss_micro: number | null;
}

const atLeast = (a: Ratio, b: Ratio): boolean => a.num * b.den >= b.num * a.den;

/** Judge a distribution against the true value. */
export function judge(options: readonly Option[], truth: string | null): Judged {
  const mine = options.find((o) => o.id === truth);
  const top = options.filter((o) => options.every((q) => atLeast(o.p, q.p)));
  const hit = mine !== undefined && mine.p.num > 0n && top.some((o) => o.id === truth);
  return {
    hit_ppm: hit ? Math.floor(1_000_000 / top.length) : 0,
    loss_micro: mine === undefined || mine.p.num === 0n ? null : Math.round(-Math.log(probability(mine.p)) * 1_000_000) + 0,
  };
}

export interface Bin {
  n: number;
  /** Sum of the options' probabilities in ppm (floored per option). */
  p_ppm: number;
  /** How many of the options were the true value. */
  true: number;
}

export const emptyBins = (): Bin[] => Array.from({ length: CALIBRATION_BINS }, () => ({ n: 0, p_ppm: 0, true: 0 }));

/** The reliability table: every option with a probability above zero goes in the bin of its probability (bin 9 holds 0.9 to 1). */
export function calibrate(bins: Bin[], options: readonly Option[], isTrue: (id: string | null) => boolean): void {
  for (const o of options) {
    if (o.p.num === 0n) continue;
    const at = Math.min(CALIBRATION_BINS - 1, Number((o.p.num * BigInt(CALIBRATION_BINS)) / o.p.den));
    const bin = bins[at] as Bin;
    bin.n += 1;
    bin.p_ppm += Number((o.p.num * 1_000_000n) / o.p.den);
    if (isTrue(o.id)) bin.true += 1;
  }
}

// ---- the accumulator ----

interface Cell {
  n: number;
  hit_ppm: number;
  loss_n: number;
  loss_micro: number;
  zero_mass: number;
}
const emptyCell = (): Cell => ({ n: 0, hit_ppm: 0, loss_n: 0, loss_micro: 0, zero_mass: 0 });

export interface Accumulator {
  games: number;
  requests: number;
  /** Requests the belief answered, by phase. */
  scored: Record<Phase, number>;
  /** Requests with no belief (a refusal), by code; and requests with no one-to-one truth (an Illusion twin in the roster). */
  refusals: Record<string, number>;
  no_truth: number;
  /** v1 could not build a set for a Pokemon: counted, left out of the v1 rows. */
  v1_failures: number;
  cells: Map<string, Cell>;
  bins: Record<"belief" | "prior", Record<Field, Bin[]>>;
  joint: { pairs: number; item: number; moves: number; both: number; overlap_ppm: number; worlds: number; redraws: number; sample_refusals: Record<string, number> };
}

export function accumulator(): Accumulator {
  const bins = () => ({ item: emptyBins(), ability: emptyBins(), move: emptyBins() });
  return {
    games: 0,
    requests: 0,
    scored: { preview: 0, battle: 0 },
    refusals: {},
    no_truth: 0,
    v1_failures: 0,
    cells: new Map(),
    bins: { belief: bins(), prior: bins() },
    joint: { pairs: 0, item: 0, moves: 0, both: 0, overlap_ppm: 0, worlds: 0, redraws: 0, sample_refusals: {} },
  };
}

const bump = (counts: Record<string, number>, key: string): void => {
  counts[key] = (counts[key] ?? 0) + 1;
};

function cellOf(acc: Accumulator, phase: Phase, field: Field, method: Method): Cell {
  const key = `${phase}/${field}/${method}`;
  let cell = acc.cells.get(key);
  if (cell === undefined) acc.cells.set(key, (cell = emptyCell()));
  return cell;
}

function add(cell: Cell, judged: Judged): void {
  cell.n += 1;
  cell.hit_ppm += judged.hit_ppm;
  if (judged.loss_micro === null) cell.zero_mass += 1;
  else {
    cell.loss_n += 1;
    cell.loss_micro += judged.loss_micro;
  }
}

// ---- the oracle: truth of a request ----

interface Truth {
  readonly team: readonly PokemonSet[];
  readonly brought: ReadonlySet<number>;
}

/** The set's species id and its base species id, from the pinned engine's own data (no naming rule is written here). */
let dexOf: ((name: string) => { name: string; baseSpecies: string }) | undefined;
const formeIds = (set: PokemonSet): string[] => {
  const species = (dexOf as NonNullable<typeof dexOf>)(set.species);
  return [toId(species.name), toId(species.baseSpecies)];
};

const idOf = (value: string): string | null => (value === "" ? null : toId(value));

/** The preview position of each roster entry in the real team, by species (a team holds a species once); null when it does not match one to one. */
function positions(roster: readonly { base_species: string }[], team: readonly PokemonSet[]): number[] | null {
  const at = roster.map((m) => team.findIndex((set) => formeIds(set).includes(m.base_species)));
  return at.some((i) => i < 0) || new Set(at).size !== at.length ? null : at;
}

interface Point {
  readonly seat: Seat;
  readonly request: number;
  readonly state: PublicState;
  readonly flags: readonly PublicFlag[];
  readonly truth: Truth | null;
}

function pointsOf(game: OracleGame): Point[] {
  const out: Point[] = [];
  for (const seat of ["p1", "p2"] as const) {
    const other: Seat = seat === "p1" ? "p2" : "p1";
    const run = interpretChannel(seat, game.channels[seat] as string[]);
    const theirs = interpretChannel(other, game.channels[other] as string[]);
    const final = (theirs.frames[theirs.frames.length - 1] as (typeof theirs.frames)[number]).state.own.roster;
    const team = game.teams[other];
    const brought = new Set<number>();
    for (const m of final) {
      if (m.brought !== true) continue;
      const own = toId(m.name ?? "");
      const i = team.findIndex((set) => toId(set.name === "" ? set.species : set.name) === own || formeIds(set).includes(m.base_species));
      if (i >= 0) brought.add(i);
    }
    let request = 0;
    for (const frame of run.frames) {
      if (frame.kind !== "request") continue;
      const roster = frame.state.opponent.roster;
      const at = roster.some((m) => m.phantom === true) ? null : positions(roster, team);
      out.push({
        seat,
        request,
        state: frame.state,
        flags: run.flags.filter((f) => f.index <= frame.index),
        truth: at === null ? null : { team: at.map((i) => team[i] as PokemonSet), brought: new Set(at.flatMap((i, p) => (brought.has(i) ? [p] : []))) },
      });
      request += 1;
    }
  }
  return out;
}

const streamOf = (game: string, seat: Seat, request: number): Uint8Array => createHash("sha256").update(JSON.stringify({ game, seat, request, purpose: "prediction-quality" })).digest();

const optionsOf = <T extends string | null>(list: readonly { id: T; p: Ratio }[]): Option[] => list.map((m) => ({ id: m.id, p: m.p }));
const point = (id: string | null): Option[] => [{ id, p: { num: 1n, den: 1n } }];

/** The fields of one brought slot against its distributions. Exported for the hand-checked tests. */
export function scoreSlot(
  acc: Accumulator,
  phase: Phase,
  slots: { belief: SlotMarginal; prior: SlotMarginal; v1: { item: string | null; ability: string } | null },
  real: { item: string | null; ability: string; moves: readonly string[] },
  mon: Pick<PublicMon, "mega" | "moves_revealed">,
): void {
  const methods: [Method, SlotMarginal][] = [["belief", slots.belief], ["prior", slots.prior]];
  for (const [method, slot] of methods) {
    const items = optionsOf(slot.items);
    add(cellOf(acc, phase, "item", method), judge(items, real.item));
    calibrate(acc.bins[method as "belief" | "prior"].item, items, (id) => id === real.item);
    if (!mon.mega) {
      const abilities = optionsOf(slot.abilities);
      add(cellOf(acc, phase, "ability", method), judge(abilities, real.ability));
      calibrate(acc.bins[method as "belief" | "prior"].ability, abilities, (id) => id === real.ability);
    }
    const unrevealed = optionsOf(slot.moves).filter((o) => !mon.moves_revealed.includes(o.id as string));
    const hidden = real.moves.filter((m) => !mon.moves_revealed.includes(m));
    if (unrevealed.length > 0 && hidden.length > 0) {
      const top = unrevealed.filter((o) => unrevealed.every((q) => atLeast(o.p, q.p)));
      const hit = top.filter((o) => hidden.includes(o.id as string)).length;
      const cell = cellOf(acc, phase, "move", method);
      cell.n += 1;
      cell.hit_ppm += Math.floor((hit * 1_000_000) / top.length);
      for (const m of hidden) {
        const mine = unrevealed.find((o) => o.id === m);
        if (mine === undefined || mine.p.num === 0n) cell.zero_mass += 1;
        else {
          cell.loss_n += 1;
          cell.loss_micro += Math.round(-Math.log(probability(mine.p)) * 1_000_000);
        }
      }
      calibrate(acc.bins[method as "belief" | "prior"].move, unrevealed, (id) => hidden.includes(id as string));
    }
  }
  if (slots.v1 !== null) {
    add(cellOf(acc, phase, "item", "v1"), judge(point(slots.v1.item), real.item));
    if (!mon.mega) add(cellOf(acc, phase, "ability", "v1"), judge(point(slots.v1.ability), real.ability));
  }
}

/** Score every request of both seats of one game. The real team is read here and nowhere else. */
export function scoreGame(ctx: QualityContext, game: OracleGame, acc: Accumulator): void {
  acc.games += 1;
  const dex = ctx.engine.sim.Dex.mod(ctx.engine.sim.Dex.formats.get(ctx.v1.config.format_id).mod);
  dexOf = (name) => dex.species.get(name);
  const priorAlone = new Map<Seat, OpponentBelief>();
  for (const p of pointsOf(game)) {
    acc.requests += 1;
    const phase: Phase = p.request === 0 ? "preview" : "battle";
    if (p.truth === null) {
      acc.no_truth += 1;
      continue;
    }
    let b: OpponentBelief;
    try {
      b = belief({ prior: ctx.prior, vocabulary: ctx.vocabulary, config: ctx.config, opponent: p.state.opponent, flags: p.flags });
    } catch (e) {
      if (!(e instanceof BeliefError)) throw e;
      bump(acc.refusals, e.code);
      continue;
    }
    if (p.request === 0) priorAlone.set(p.seat, b);
    // flags only accumulate, so a seat whose Team Preview was refused is refused at every later request: the prior alone exists here
    const base = priorAlone.get(p.seat) as OpponentBelief;
    acc.scored[phase] += 1;
    const roster = p.state.opponent.roster;
    roster.forEach((mon, i) => {
      if (!p.truth?.brought.has(i)) return;
      const set = p.truth.team[i] as PokemonSet;
      let v1: { item: string | null; ability: string } | null = null;
      try {
        const assumed = buildAssumedSet(ctx.v1, mon).set;
        v1 = { item: idOf(assumed.item), ability: toId(assumed.ability) };
      } catch {
        acc.v1_failures += 1;
      }
      scoreSlot(acc, phase, { belief: b.slots[i] as SlotMarginal, prior: base.slots[i] as SlotMarginal, v1 }, { item: idOf(set.item), ability: toId(set.ability), moves: set.moves.map(toId) }, mon);
    });
    // sampled worlds: the joint consistency of item and moves
    try {
      const sampled = ctx.port.sample({ opponent: p.state.opponent, flags: p.flags, stream: streamOf(game.id, p.seat, p.request), n: ctx.worlds });
      acc.joint.worlds += sampled.worlds.length;
      acc.joint.redraws += sampled.redraws;
      for (const { world } of sampled.worlds) {
        for (const slot of world.slots) {
          if (!slot.brought || slot.set === null || !p.truth.brought.has(slot.index)) continue;
          const real = p.truth.team[slot.index] as PokemonSet;
          const realMoves = real.moves.map(toId).sort();
          const itemOk = slot.set.item === idOf(real.item);
          const movesOk = slot.set.moves.length === realMoves.length && [...slot.set.moves].sort().every((m, k) => m === realMoves[k]);
          acc.joint.pairs += 1;
          if (itemOk) acc.joint.item += 1;
          if (movesOk) acc.joint.moves += 1;
          if (itemOk && movesOk) acc.joint.both += 1;
          acc.joint.overlap_ppm += Math.floor((slot.set.moves.filter((m) => realMoves.includes(m)).length * 1_000_000) / realMoves.length);
        }
      }
    } catch (e) {
      const code = (e as { code?: unknown }).code;
      if (!(e instanceof Error) || e.name !== "BeliefError" || typeof code !== "string") throw e;
      bump(acc.joint.sample_refusals, code);
    }
  }
}

// ---- the report: counts and integer ppm only ----

const share = (num: number, den: number): number | null => (den === 0 ? null : Math.floor((num * 1_000_000) / den));

export interface Row {
  readonly phase: Phase;
  readonly field: Field;
  readonly method: Method;
  readonly n: number;
  readonly top1_ppm: number;
  /** Mean -ln(p) of the true value in micro-nats; null for v1 (a point prediction) or when nothing had mass. */
  readonly log_loss_micro: number | null;
  /** True values the distribution gave no mass (null for v1: a point prediction has no mass to give). */
  readonly zero_mass: number | null;
}

export interface Report {
  readonly games: number;
  readonly requests: number;
  readonly scored: Record<Phase, number>;
  readonly refusals: Record<string, number>;
  readonly no_truth: number;
  readonly v1_failures: number;
  readonly rows: readonly Row[];
  readonly calibration: Record<"belief" | "prior", Record<Field, readonly Bin[]>>;
  readonly joint: {
    readonly worlds: number;
    readonly pairs: number;
    readonly item_ppm: number | null;
    readonly moves_ppm: number | null;
    readonly product_ppm: number | null;
    readonly joint_ppm: number | null;
    readonly move_overlap_ppm: number | null;
    readonly redraws: number;
    readonly redraw_ppm: number | null;
    readonly sample_refusals: Record<string, number>;
  };
}

export function report(acc: Accumulator): Report {
  const rows: Row[] = [];
  for (const phase of ["preview", "battle"] as const) {
    for (const field of FIELDS) {
      for (const method of METHODS) {
        const c = acc.cells.get(`${phase}/${field}/${method}`);
        if (c === undefined) continue;
        rows.push({ phase, field, method, n: c.n, top1_ppm: Math.floor(c.hit_ppm / c.n), log_loss_micro: method === "v1" || c.loss_n === 0 ? null : Math.round(c.loss_micro / c.loss_n), zero_mass: method === "v1" ? null : c.zero_mass });
      }
    }
  }
  const j = acc.joint;
  const item = share(j.item, j.pairs);
  const moves = share(j.moves, j.pairs);
  return {
    games: acc.games,
    requests: acc.requests,
    scored: acc.scored,
    refusals: acc.refusals,
    no_truth: acc.no_truth,
    v1_failures: acc.v1_failures,
    rows,
    calibration: acc.bins,
    joint: {
      worlds: j.worlds,
      pairs: j.pairs,
      item_ppm: item,
      moves_ppm: moves,
      product_ppm: item === null || moves === null ? null : Math.floor((item * moves) / 1_000_000),
      joint_ppm: share(j.both, j.pairs),
      move_overlap_ppm: j.pairs === 0 ? null : Math.floor(j.overlap_ppm / j.pairs),
      redraws: j.redraws,
      redraw_ppm: share(j.redraws, j.worlds),
      sample_refusals: j.sample_refusals,
    },
  };
}
