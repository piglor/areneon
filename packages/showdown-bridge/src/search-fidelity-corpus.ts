// The targeted fidelity corpus (ADR-026 Evidence 1, finding N-A, slice T): seeded
// A1 battles built to reach the rare states a search world must reproduce (Fake
// Out, Protect chains, Encore and Disable, Last Respects, Counter and Mirror
// Coat, weather, screen, Trick Room and Tailwind expiry, Toxic, Intimidate).
// Teams come from synthetic datasets and are filtered to hold the moves or
// abilities a target names, in the manner of the reducer sweep's `--only-move`.
// The targets are a reviewed input file (`targets.json`), not constants here.
// CI replays each committed battle from its seed, teams and actor seeds; the
// search below is run by `scripts/capture-search-fidelity-corpus.ts` only.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "./golden.ts";
import { pendingFirstActor, RANDOM_MAX_TURNS, type RandomBattleFile } from "./random-corpus.ts";
import { randomLegalActor } from "./random-legal.ts";
import { type Actor, type BattleRecord, type Player, runBattle } from "./runner.ts";

export const FIDELITY_DIR = join(GOLDEN_DIR, "search-fidelity");
export const FIDELITY_TARGETS = "targets.json";
export const FIDELITY_SUMMARY = "summary.json";

export const FIDELITY_PROVENANCE =
  "synthetic: seeded A1 battles on synthetic generator teams filtered to hold the targeted moves or abilities (capture-search-fidelity-corpus.ts)";

/** The two actors of the corpus (src/random-corpus.ts, src/random-legal.ts). */
export const ACTOR_KINDS = ["random-legal", "pending-first"] as const;
export type ActorKind = (typeof ACTOR_KINDS)[number];

const ACTOR_FACTORIES: Record<ActorKind, (seed: number) => Actor> = {
  "random-legal": randomLegalActor,
  "pending-first": pendingFirstActor,
};

export interface FidelityTarget {
  id: string;
  description: string;
  /** A team is drawn for the target when one of its sets holds one of these moves or abilities. */
  moves: string[];
  abilities: string[];
  /** Held items that also draw a team for the target (a start-trigger item such as White Herb); absent means none. */
  items?: string[];
  /** Regular expressions over the public channel lines; the first group, when `consecutive`, names the actor. */
  occurs: string[];
  /** A turn counts only if the same actor matched in the turn before as well. */
  consecutive?: boolean;
  /** Occurrence turns the search keeps battles for (an occurrence floor, not the slice-F yield below). */
  min_turns: number;
  /** The team opposite the target's team must hold one of these too (a state that needs both sides, such as an item an opposing ability triggers). */
  against?: { moves: string[]; abilities: string[]; items: string[] };
  /** Battles phase 1 plays for this target instead of the file's `attempts_per_target` (a rare state needs more). */
  attempts?: number;
  /**
   * The slice-F sense (ADR-026 Evidence 1): the item of the fidelity report this target exists to feed and the number of
   * EQUAL exercising turns it needs in F1 and in F2; `fidelity.test.ts` holds the committed report to it.
   */
  report?: { item: string; min_equal: number };
}

export interface UnreachableTarget {
  id: string;
  description: string;
  moves: string[];
  abilities: string[];
  reason: string;
}

export interface FidelityTargets {
  schema_version: 1;
  data_class: "synthetic";
  provenance: string;
  attempts_per_target: number;
  max_battles: number;
  targets: FidelityTarget[];
  unreachable: UnreachableTarget[];
}

export class FidelityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FidelityError";
  }
}

const idOf = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, "");

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const strings = (value: unknown, what: string): string[] => {
  if (!Array.isArray(value) || value.some((x) => typeof x !== "string" || x === "")) {
    throw new FidelityError(`${what} is not a list of non-empty strings`);
  }
  return value as string[];
};

const wholeAtLeast1 = (value: unknown, what: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new FidelityError(`${what} is not an integer of at least 1`);
  }
  return value;
};

/** Reads the reviewed target list; every shape it does not know is refused. */
export function readFidelityTargets(value: unknown): FidelityTargets {
  if (!isObject(value) || value.schema_version !== 1 || value.data_class !== "synthetic") {
    throw new FidelityError("targets: not a synthetic version 1 record");
  }
  if (typeof value.provenance !== "string" || !Array.isArray(value.targets) || !Array.isArray(value.unreachable)) {
    throw new FidelityError("targets: provenance, targets or unreachable is missing");
  }
  const ids = new Set<string>();
  const claim = (id: unknown): string => {
    if (typeof id !== "string" || !/^[a-z][a-z0-9-]*$/.test(id) || ids.has(id)) {
      throw new FidelityError("targets: an id is not a unique lower-case token");
    }
    ids.add(id);
    return id;
  };
  const targets = value.targets.map((raw): FidelityTarget => {
    if (!isObject(raw)) throw new FidelityError("targets: a target is not an object");
    const id = claim(raw.id);
    const moves = strings(raw.moves, `${id}.moves`);
    const abilities = strings(raw.abilities, `${id}.abilities`);
    const items = raw.items === undefined ? undefined : strings(raw.items, `${id}.items`);
    if (moves.length + abilities.length + (items?.length ?? 0) === 0) throw new FidelityError(`${id}: no move or ability to filter teams by`);
    const occurs = strings(raw.occurs, `${id}.occurs`);
    if (occurs.length === 0) throw new FidelityError(`${id}: no occurrence pattern`);
    for (const pattern of occurs) new RegExp(pattern); // a bad pattern throws a SyntaxError here
    if (raw.consecutive !== undefined && typeof raw.consecutive !== "boolean") {
      throw new FidelityError(`${id}.consecutive is not a boolean`);
    }
    if (typeof raw.description !== "string") throw new FidelityError(`${id}.description is not a string`);
    let against: FidelityTarget["against"];
    if (raw.against !== undefined) {
      const a = raw.against;
      if (!isObject(a)) throw new FidelityError(`${id}.against is malformed`);
      against = {
        moves: strings(a.moves, `${id}.against.moves`),
        abilities: strings(a.abilities, `${id}.against.abilities`),
        items: strings(a.items, `${id}.against.items`),
      };
      if (against.moves.length + against.abilities.length + against.items.length === 0) throw new FidelityError(`${id}.against names nothing`);
    }
    let report: FidelityTarget["report"];
    if (raw.report !== undefined) {
      const r = raw.report;
      if (!isObject(r) || typeof r.item !== "string" || r.item === "") throw new FidelityError(`${id}.report is malformed`);
      report = { item: r.item, min_equal: wholeAtLeast1(r.min_equal, `${id}.report.min_equal`) };
    }
    return {
      id,
      description: raw.description,
      moves,
      abilities,
      ...(items === undefined ? {} : { items }),
      occurs,
      ...(raw.consecutive === true ? { consecutive: true } : {}),
      min_turns: wholeAtLeast1(raw.min_turns, `${id}.min_turns`),
      ...(against === undefined ? {} : { against }),
      ...(raw.attempts === undefined ? {} : { attempts: wholeAtLeast1(raw.attempts, `${id}.attempts`) }),
      ...(report === undefined ? {} : { report }),
    };
  });
  const unreachable = value.unreachable.map((raw): UnreachableTarget => {
    if (!isObject(raw) || typeof raw.description !== "string" || typeof raw.reason !== "string") {
      throw new FidelityError("targets: an unreachable entry is malformed");
    }
    return {
      id: claim(raw.id),
      description: raw.description,
      moves: strings(raw.moves, "unreachable.moves"),
      abilities: strings(raw.abilities, "unreachable.abilities"),
      reason: raw.reason,
    };
  });
  return {
    schema_version: 1,
    data_class: "synthetic",
    provenance: value.provenance,
    attempts_per_target: wholeAtLeast1(value.attempts_per_target, "attempts_per_target"),
    max_battles: wholeAtLeast1(value.max_battles, "max_battles"),
    targets,
    unreachable,
  };
}

export function loadFidelityTargets(dir: string = FIDELITY_DIR): FidelityTargets {
  return readFidelityTargets(JSON.parse(readFileSync(join(dir, FIDELITY_TARGETS), "utf8")));
}

/** A target that the roster says no legal team can reach must be unreachable in fact, so a pin bump cannot hide a change. */
export function checkUnreachable(
  unreachable: readonly UnreachableTarget[],
  roster: readonly { id: string; abilities: readonly string[]; moves: readonly string[] }[],
): void {
  for (const entry of unreachable) {
    const wanted = new Set([...entry.moves, ...entry.abilities].map(idOf));
    const holder = roster.find((r) => [...r.abilities, ...r.moves].some((x) => wanted.has(x)));
    if (holder !== undefined) {
      throw new FidelityError(`${entry.id} is marked unreachable but ${holder.id} of the roster can hold it: make it a target`);
    }
  }
}

/** Whether a team can be drawn for a target: one of its sets holds a named move or ability. */
export function holdsTarget(team: readonly PokemonSet[], target: Pick<FidelityTarget, "moves" | "abilities" | "items">): boolean {
  const moves = new Set(target.moves.map(idOf));
  const abilities = new Set(target.abilities.map(idOf));
  const items = new Set((target.items ?? []).map(idOf));
  return team.some((set) => abilities.has(idOf(set.ability)) || items.has(idOf(set.item)) || set.moves.some((m) => moves.has(idOf(m))));
}

/** The turns of a public channel in which a target occurs, ascending. Lines before `|turn|1` are turn 0. */
export function occurrenceTurns(lines: readonly string[], target: FidelityTarget): number[] {
  const patterns = target.occurs.map((p) => new RegExp(p));
  const byTurn = new Map<number, Set<string>>();
  let turn = 0;
  for (const line of lines) {
    if (line.startsWith("|turn|")) {
      turn = Number(line.slice("|turn|".length));
      continue;
    }
    for (const pattern of patterns) {
      const match = pattern.exec(line);
      if (match === null) continue;
      const actor = target.consecutive === true ? (match[1] ?? "") : "";
      byTurn.set(turn, (byTurn.get(turn) ?? new Set()).add(actor));
    }
  }
  const turns = [...byTurn.keys()].sort((a, b) => a - b);
  if (target.consecutive !== true) return turns;
  return turns.filter((t) => [...(byTurn.get(t) ?? [])].some((actor) => byTurn.get(t - 1)?.has(actor) === true));
}

/** What a battle needs to be replayed: the random corpus's spec plus the seats' actors and where the teams came from. */
export interface FidelitySpec {
  id: string;
  seed: string;
  actor_seeds: Record<Player, number>;
  teams: Record<Player, PokemonSet[]>;
  actors: Record<Player, ActorKind>;
  origin: Record<Player, TeamOrigin>;
}

export interface TeamOrigin {
  dataset_id: string;
  canonical_team_hash: string;
}

export interface FidelityBattleFile extends Omit<RandomBattleFile, "teams" | "id" | "seed" | "actor_seeds"> {
  id: string;
  seed: string;
  actor_seeds: Record<Player, number>;
  teams: Record<Player, PokemonSet[]>;
  actors: Record<Player, ActorKind>;
  origin: Record<Player, TeamOrigin>;
}

export function playFidelityBattle(engine: Engine, spec: FidelitySpec): Promise<BattleRecord> {
  return runBattle({
    engine,
    formatId: GOLDEN_FORMAT,
    seed: spec.seed,
    teams: spec.teams,
    actors: {
      p1: ACTOR_FACTORIES[spec.actors.p1](spec.actor_seeds.p1),
      p2: ACTOR_FACTORIES[spec.actors.p2](spec.actor_seeds.p2),
    },
    maxTurns: RANDOM_MAX_TURNS,
  });
}

/** The committed file of one played battle. Its teams are synthetic by construction. */
export function fidelityBattleFile(spec: FidelitySpec, record: BattleRecord): FidelityBattleFile {
  const channel = (player: Player) => ({
    schema_version: 1 as const,
    data_class: "synthetic" as const,
    player,
    lines: record.channels[player],
  });
  return {
    schema_version: 1,
    data_class: "synthetic",
    provenance: FIDELITY_PROVENANCE,
    ...spec,
    winner: record.winner,
    turns: record.turns,
    channels: { p1: channel("p1"), p2: channel("p2") },
  };
}

/** One team of a pool a battle can be drawn from. */
export interface PoolTeam {
  team: PokemonSet[];
  origin: TeamOrigin;
}

export interface FidelityCoverageItem {
  id: string;
  min_turns: number;
  /** Turns in which the target occurs, over the whole corpus. */
  turns: number;
  /** Battles in which it occurs at least once. */
  battles: number;
  reached: boolean;
  /** Per battle id: the turns (sorted) of that battle in which it occurs. */
  by_battle: Record<string, number[]>;
}

export interface FidelitySummary {
  schema_version: 1;
  data_class: "synthetic";
  /** Battles in the corpus and the turns they play in all (the denominator slice F counts against). */
  battles: number;
  battle_turns: number;
  items: FidelityCoverageItem[];
  unreachable: { id: string; reason: string }[];
}

/** The coverage of a corpus: a pure function of the committed public channels and the target list. */
export function fidelitySummary(battles: readonly FidelityBattleFile[], targets: FidelityTargets): FidelitySummary {
  const items = targets.targets.map((target): FidelityCoverageItem => {
    const by_battle: Record<string, number[]> = {};
    let turns = 0;
    for (const battle of battles) {
      const hit = occurrenceTurns(battle.channels.p1.lines, target);
      if (hit.length > 0) by_battle[battle.id] = hit;
      turns += hit.length;
    }
    return {
      id: target.id,
      min_turns: target.min_turns,
      turns,
      battles: Object.keys(by_battle).length,
      reached: turns >= target.min_turns,
      by_battle,
    };
  });
  return {
    schema_version: 1,
    data_class: "synthetic",
    battles: battles.length,
    battle_turns: battles.reduce((sum, b) => sum + b.turns, 0),
    items,
    unreachable: targets.unreachable.map(({ id, reason }) => ({ id, reason })),
  };
}

const ACTOR_MIXES: Record<Player, ActorKind>[] = [
  { p1: "random-legal", p2: "random-legal" },
  { p1: "pending-first", p2: "random-legal" },
  { p1: "random-legal", p2: "pending-first" },
  { p1: "pending-first", p2: "pending-first" },
];

/** Mulberry-free deterministic spread: attempt `j` of a pool of `n` teams, a stride coprime-ish to typical sizes. */
const pick = (j: number, salt: number, n: number): number => ((Math.imul(j + 1, 2654435761) >>> 0) + salt * 40503) % n;

/**
 * Searches for the corpus in two phases. (1) For each target, plays
 * `attempts_per_target` seeded battles (a team holding the target on a seat that
 * alternates, any team opposite, the four actor mixes in turn) and pools those in
 * which the target occurs; a battle that fails to enumerate (Revival Blessing) is
 * skipped. (2) Greedy over the whole pool: while some target is short of its
 * `min_turns`, keep the battle with the highest gain per turn played, the gain
 * being the turns it adds to every short target (capped at the shortfall), the
 * earlier attempt winning a tie; stop when no battle adds anything. At most
 * `max_battles` are kept. A target left short is logged, never lowered. Everything
 * is a function of the pool, the targets and the options, so a re-run gives the
 * same corpus.
 */
export async function captureFidelityCorpus(
  engine: Engine,
  pool: readonly PoolTeam[],
  targets: FidelityTargets,
  options: { seed_base: number; log?: (line: string) => void; existing?: readonly FidelityBattleFile[] } = { seed_base: 0x6000 },
): Promise<FidelityBattleFile[]> {
  const candidates: { file: FidelityBattleFile; turns: Map<string, number> }[] = [];
  const existing = options.existing ?? [];
  const have = new Map(targets.targets.map((t) => [t.id, 0]));
  const kept: FidelityBattleFile[] = [...existing];
  for (const file of existing) {
    for (const t of targets.targets) have.set(t.id, (have.get(t.id) ?? 0) + occurrenceTurns(file.channels.p1.lines, t).length);
  }
  let attempt = 0;
  for (const target of targets.targets) {
    if (existing.length > 0 && (have.get(target.id) ?? 0) >= target.min_turns) continue;
    const holders = pool.filter((p) => holdsTarget(p.team, target));
    if (holders.length === 0) {
      options.log?.(`${target.id}: no team in the pool holds it`);
      continue;
    }
    const opposite = target.against === undefined ? pool : pool.filter((p) => holdsTarget(p.team, target.against as NonNullable<FidelityTarget["against"]>));
    if (opposite.length === 0) {
      options.log?.(`${target.id}: no team in the pool holds what the opposite team needs`);
      continue;
    }
    for (let k = 0; k < (target.attempts ?? targets.attempts_per_target); k++, attempt++) {
      const seat: Player = attempt % 2 === 0 ? "p1" : "p2";
      const other: Player = seat === "p1" ? "p2" : "p1";
      const focus = holders[pick(attempt, 1, holders.length)] as PoolTeam;
      const rest = opposite[pick(attempt, 2, opposite.length)] as PoolTeam;
      const spec: FidelitySpec = {
        id: "",
        seed: `sodium,${(options.seed_base + attempt).toString(16).padStart(32, "0")}`,
        actor_seeds: { p1: (2 * attempt + 1) * 2654435761 >>> 0, p2: (2 * attempt + 2) * 2654435761 >>> 0 },
        teams: { [seat]: focus.team, [other]: rest.team } as Record<Player, PokemonSet[]>,
        actors: ACTOR_MIXES[attempt % ACTOR_MIXES.length] as Record<Player, ActorKind>,
        origin: { [seat]: focus.origin, [other]: rest.origin } as Record<Player, TeamOrigin>,
      };
      let file: FidelityBattleFile;
      try {
        file = fidelityBattleFile(spec, await playFidelityBattle(engine, spec));
      } catch (error) {
        options.log?.(`${target.id}: skipped attempt ${attempt}: ${(error as Error).message}`);
        continue;
      }
      const turns = new Map(targets.targets.map((t) => [t.id, occurrenceTurns(file.channels.p1.lines, t).length]));
      if ((turns.get(target.id) ?? 0) > 0) candidates.push({ file, turns });
    }
  }
  while (kept.length < targets.max_battles) {
    const deficit = (t: FidelityTarget): number => Math.max(0, t.min_turns - (have.get(t.id) ?? 0));
    const gainOf = (c: { turns: Map<string, number> }): number =>
      targets.targets.reduce((sum, t) => sum + Math.min(deficit(t), c.turns.get(t.id) ?? 0), 0);
    let best = -1;
    let bestScore = 0;
    candidates.forEach((c, i) => {
      const score = gainOf(c) / c.file.turns;
      if (score > bestScore) {
        best = i;
        bestScore = score;
      }
    });
    if (best < 0) break;
    const [chosen] = candidates.splice(best, 1);
    const { file, turns } = chosen as (typeof candidates)[number];
    file.id = `b${String(kept.length).padStart(2, "0")}`;
    kept.push(file);
    for (const [id, n] of turns) have.set(id, (have.get(id) ?? 0) + n);
    options.log?.(`kept ${file.id} (seed ${file.seed}, ${file.turns} turns played)`);
  }
  for (const t of targets.targets) {
    if ((have.get(t.id) ?? 0) < t.min_turns) options.log?.(`${t.id}: ${have.get(t.id) ?? 0} of ${t.min_turns} turns reached`);
  }
  return kept;
}
