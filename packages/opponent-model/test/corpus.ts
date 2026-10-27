// The committed corpora as belief inputs, with the opponent's real team kept apart for the test oracle
// (ADR-025 section 8). A `Point` is one request of one seat's own channel: the shared reader's state there
// and the flags of the whole channel, which is all `belief` is ever given. The `Truth` beside it is read
// by the soundness test and the rank statistic only; no function here passes it to the belief.

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { type PublicFlag, type PublicState, interpretChannel, toId } from "@areneon/public-state";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { type BeliefInput, belief } from "../src/index.ts";
import type { DisguiseAssignment, OpponentBelief } from "../src/index.ts";
import { DIR, megasOf, realVocabulary, uniformPrior } from "./real.ts";
import { config } from "./support.ts";

export type Seat = "p1" | "p2";
export interface Set6 {
  readonly name: string;
  readonly species: string;
  readonly item: string;
  readonly ability: string;
  readonly moves: readonly string[];
  readonly nature: string;
  readonly gender: string;
  readonly evs: Readonly<Record<string, number>>;
  readonly level: number;
}
export interface Game {
  readonly id: string;
  readonly teams: Readonly<Record<Seat, readonly Set6[]>>;
  readonly channels: Readonly<Record<Seat, readonly string[]>>;
}

interface BattleFile {
  readonly teams: Record<Seat, Set6[]>;
  readonly channels: Record<Seat, { lines: string[] }>;
}

const read = <T>(...path: string[]): T => JSON.parse(readFileSync(join(DIR, ...path), "utf8")) as T;
const fromBattle = (id: string, file: BattleFile): Game => ({ id, teams: file.teams, channels: { p1: file.channels.p1.lines, p2: file.channels.p2.lines } });

/** The golden battle, the random-play corpus, `unavailable/`, the Illusion battle and the other committed single battles. */
export function games(options: { random?: number } = {}): Game[] {
  const out: Game[] = [];
  const team = read<Set6[]>("synthetic-team.json");
  out.push({ id: "golden", teams: { p1: team, p2: team }, channels: { p1: read<{ channel: { lines: string[] } }>("channels", "p1.json").channel.lines, p2: read<{ channel: { lines: string[] } }>("channels", "p2.json").channel.lines } });
  const numbered = (dir: string, limit = Infinity): string[] => readdirSync(join(DIR, dir)).filter((f) => /^b\d+\.json$/.test(f)).sort().slice(0, limit);
  for (const f of numbered("random", options.random)) out.push(fromBattle(`random/${f.slice(0, -5)}`, read<BattleFile>("random", f)));
  for (const f of numbered("unavailable")) out.push(fromBattle(`unavailable/${f.slice(0, -5)}`, read<BattleFile>("unavailable", f)));
  for (const dir of ["illusion", "fling", "baton-pass", "baton-pass-volatiles", "transform", "revival"]) {
    if (existsSync(join(DIR, dir, "battle.json"))) out.push(fromBattle(dir, read<BattleFile>(dir, "battle.json")));
  }
  return out;
}

/** What the oracle knows: the opponent's real sets in preview order and the positions it really brought. */
export interface Truth {
  readonly team: readonly Set6[];
  readonly brought: ReadonlySet<number>;
}

export interface Point {
  readonly game: string;
  readonly seat: Seat;
  /** 0-based ordinal among the seat's requests. */
  readonly request: number;
  readonly state: PublicState;
  readonly flags: readonly PublicFlag[];
  /** Null while the reader holds an Illusion twin (a phantom entry): the belief refuses there and the roster has no one-to-one truth. */
  readonly truth: Truth | null;
}

/** The preview position of each real set, matched by species id (a team holds a species once). */
function positions(roster: readonly { base_species: string }[], team: readonly Set6[]): number[] {
  const at = roster.map((m) => team.findIndex((set) => toId(set.species) === m.base_species || toId(set.species.split("-")[0] ?? "") === m.base_species));
  if (at.some((i) => i < 0) || new Set(at).size !== at.length) throw new Error(`the preview roster does not match the real team by species: ${roster.map((m) => m.base_species).join(",")} / ${team.map((s) => s.species).join(",")}`);
  return at;
}

/** Every request of both seats of a game, the reader's state at it and the oracle's truth. */
export function points(game: Game): Point[] {
  const out: Point[] = [];
  for (const seat of ["p1", "p2"] as const) {
    const other: Seat = seat === "p1" ? "p2" : "p1";
    const run = interpretChannel(seat, game.channels[seat] as string[]);
    // The real brought set: the opponent's own channel says which of its Pokemon it brought (its own requests list them).
    const theirs = interpretChannel(other, game.channels[other] as string[]);
    const final = theirs.frames[theirs.frames.length - 1]!.state.own.roster;
    const team = game.teams[other];
    const brought = new Set<number>();
    for (const m of final) {
      if (m.brought !== true) continue;
      // by nickname (the own roster keeps the preview name), else by species, a Mega forme starting with its species id
      const own = toId(m.name ?? "");
      const i = team.findIndex((set) => toId(set.name === "" ? set.species : set.name) === own || m.base_species.startsWith(toId(set.species.split("-")[0] ?? "")));
      if (i < 0) throw new Error(`a brought Pokemon is not in the real team: ${m.base_species} / ${team.map((s) => s.species).join(",")}`);
      brought.add(i);
    }
    let request = 0;
    for (const frame of run.frames) {
      if (frame.kind !== "request") continue;
      const roster = frame.state.opponent.roster;
      if (roster.some((m) => m.phantom === true)) {
        out.push({ game: game.id, seat, request, state: frame.state, flags: run.flags, truth: null });
        request += 1;
        continue;
      }
      // The preview roster is the real team in order; a state before the previews has no roster to judge.
      const at = positions(roster, team);
      const ordered = at.map((i) => team[i] as Set6);
      out.push({ game: game.id, seat, request, state: frame.state, flags: run.flags, truth: { team: ordered, brought: new Set(at.flatMap((i, p) => (brought.has(i) ? [p] : []))) } });
      request += 1;
    }
  }
  return out;
}

export const hiding = { ...config, identity_hiding_abilities: realVocabulary.roster.some((r) => r.abilities.includes("illusion")) ? ["illusion"] : [] };

/** The belief of a point under `prior`; the real team is not an argument. */
export function beliefAt(p: Point, prior: UsagePrior, change: Partial<BeliefInput> = {}): OpponentBelief {
  return belief({ prior, vocabulary: realVocabulary, config: hiding, opponent: p.state.opponent, flags: p.flags, ...change });
}

export const idsOf = (p: Point): string[] => p.state.opponent.roster.map((m) => m.base_species);

type Forme = UsagePrior["species"][number]["formes"][number];

/**
 * A prior that is not uniform and not derived from any team: per species the base forme weighs 3 and each Mega forme 1;
 * an ability list names the first listed ability (3 of 4); an item list names the first three items of the vocabulary
 * (3, 2, 2) and "no item" (2) with a residual of 1 over the rest; one named spread; moves name the first six of the pool.
 * Every list keeps a residual, so every legal set keeps a weight above zero.
 */
export function skewedPrior(ids: readonly string[]): UsagePrior {
  const natures = realVocabulary.natures;
  const items = realVocabulary.items;
  const species = [...ids].sort().map((id) => {
    const entries = [...realVocabulary.roster.filter((r) => r.id === id), ...megasOf(id)];
    const formes = entries.map((r): Forme => {
      const pool = r.moves.slice(0, 6);
      return {
        id: r.id,
        weight: r.id === id ? 3 : 1,
        abilities: { denominator: 4, entries: [{ id: r.abilities[0] as string, weight: 3 }], residual: 1, unmapped: 0 },
        items: { denominator: 10, entries: [{ id: items[0] as string, weight: 3 }, { id: items[1] as string, weight: 2 }, { id: items[2] as string, weight: 2 }], no_item: 2, residual: 1, unmapped: 0 },
        spreads: { denominator: 10, entries: [{ nature: natures[0]!.id, stat_points: { hp: 2, atk: 32, def: 0, spa: 0, spd: 0, spe: 0 }, weight: 6 }], residual: 4, unmapped: 0 },
        moves: { denominator: 10, slots: 4, entries: pool.map((m) => ({ id: m, inclusion: 6 })), empty_slots: 0, residual: 4, unmapped: 0 },
      };
    });
    return { id, denominator: formes.reduce((s, f) => s + f.weight, 0), formes: formes as [Forme, ...Forme[]] };
  });
  return { ...uniformPrior(ids), species: species as UsagePrior["species"] };
}

// ---- the oracle ----

const idOf = (value: string): string | null => (value === "" ? null : toId(value));
const roster = new Map(realVocabulary.roster.map((r) => [r.id, r]));

/**
 * The formes a real set may hold before any Evolution: the Mega forme for a stone holder, else the species. A species whose
 * Mega carries a gender letter (meowstic: meowsticmmega and meowsticfmega, one stone) has two, and the set's gender is not
 * kept by the roster ids, so both count.
 */
function trueFormes(set: Set6): string[] {
  const stone = idOf(set.item);
  const species = toId(set.species);
  if (stone === null) return [species];
  const found = megasOf(species).filter((r) => r.required_item === stone);
  const lettered = found.length > 0 ? [] : realVocabulary.roster.filter((r) => r.required_item === stone && r.id.startsWith(species) && r.id.includes("mega"));
  const all = [...found, ...lettered].map((r) => r.id);
  return all.length > 0 ? all : [species];
}
const trueForme = (set: Set6): string => trueFormes(set)[0] as string;

export interface FieldVerdict {
  readonly forme: boolean;
  readonly ability: boolean | null;
  readonly item: boolean;
  readonly moves: boolean;
}

/** Whether `assignment` gives the real set of slot `i` a probability above zero, field by field (the ability of an evolved Mega is not judged: null). */
export function admits(assignment: DisguiseAssignment, i: number, set: Set6, evolved: boolean): FieldVerdict {
  const s = assignment.slots[i] as DisguiseAssignment["slots"][number];
  const item = idOf(set.item);
  return {
    forme: s.formes.some((f) => trueFormes(set).includes(f.id)),
    ability: evolved ? null : s.abilities.some((m) => m.id === toId(set.ability)),
    item: s.items.some((m) => m.id === item),
    moves: set.moves.every((mv) => s.moves.some((m) => m.id === toId(mv))),
  };
}

/** The assignments under which the whole real team and the real brought set are possible. */
export function sound(b: OpponentBelief, p: Point & { truth: Truth }): DisguiseAssignment[] {
  return b.assignments.filter((a) =>
    p.truth.team.every((set, i) => {
      const v = admits(a, i, set, p.state.opponent.roster[i]!.mega);
      return v.forme && v.ability !== false && v.item && v.moves;
    }) && a.required.every((i) => p.truth.brought.has(i)),
  );
}

// ---- the rank statistic (N6) ----

export const FIELDS = ["forme", "ability", "item", "move"] as const;
export type RankField = (typeof FIELDS)[number];
export interface Rank {
  readonly field: RankField;
  /** The number of options at least as likely as the true one, itself included: ties count against the belief, so a flat belief ranks last. */
  readonly rank: number;
  /** How many options have a probability above zero. */
  readonly options: number;
}

const rankIn = <T>(list: readonly { id: T; p: { num: bigint; den: bigint } }[], id: T): { rank: number; options: number } | null => {
  const hit = list.find((m) => m.id === id);
  if (hit === undefined || hit.p.num === 0n) return null;
  // Compare a/b >= c/d by cross-multiplication (exact).
  const atLeast = list.filter((m) => m.p.num * hit.p.den >= hit.p.num * m.p.den).length;
  return { rank: atLeast, options: list.filter((m) => m.p.num > 0n).length };
};

/** The true set's rank in each field of the (marginal) belief, for the slots it brought. */
export function ranks(b: OpponentBelief, p: Point & { truth: Truth }): Rank[] {
  const out: Rank[] = [];
  b.slots.forEach((s, i) => {
    const set = p.truth.team[i] as Set6;
    const evolved = p.state.opponent.roster[i]!.mega;
    const add = (field: RankField, r: { rank: number; options: number } | null) => {
      if (r !== null) out.push({ field, ...r });
    };
    add("forme", rankIn(s.formes, trueForme(set)));
    if (!evolved) add("ability", rankIn(s.abilities, toId(set.ability)));
    add("item", rankIn(s.items, idOf(set.item)));
    for (const mv of set.moves) add("move", rankIn(s.moves, toId(mv)));
  });
  return out;
}

export interface RankRow {
  readonly phase: "preview" | "battle";
  readonly field: RankField;
  readonly n: number;
  /** Share of true values that are the most likely option, in ppm (integer arithmetic). */
  readonly top1_ppm: number;
  readonly top3_ppm: number;
  readonly median_rank: number;
  /** Mean of rank / options, in ppm: 500000 is what a blind guess gives. */
  readonly mean_rel_ppm: number;
}

/** The uniform-prior rank table (N6): per phase and field, how the true value ranks among the belief's options. */
export function rankTable(all: readonly { phase: "preview" | "battle"; rank: Rank }[]): RankRow[] {
  const rows: RankRow[] = [];
  for (const phase of ["preview", "battle"] as const) {
    for (const field of FIELDS) {
      const list = all.filter((r) => r.phase === phase && r.rank.field === field).map((r) => r.rank);
      if (list.length === 0) continue;
      const sorted = list.map((r) => r.rank).sort((a, b) => a - b);
      const share = (hits: number) => Math.floor((hits * 1_000_000) / list.length);
      rows.push({
        phase,
        field,
        n: list.length,
        top1_ppm: share(list.filter((r) => r.rank === 1).length),
        top3_ppm: share(list.filter((r) => r.rank <= 3).length),
        median_rank: sorted[Math.floor(sorted.length / 2)] as number,
        mean_rel_ppm: Math.floor(list.reduce((t, r) => t + Math.floor((r.rank * 1_000_000) / r.options), 0) / list.length),
      });
    }
  }
  return rows;
}

export const ROSTER = roster;
