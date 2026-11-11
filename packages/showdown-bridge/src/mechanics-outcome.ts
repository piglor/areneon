// The `move_outcome` probe (ADR-022 erratum E-22.5.3, slice H2): whether the asked move works, and with what chance.
// Areneon decides nothing about a move: the scenario battle holds the declared state and history (`mechanics-history.ts`),
// the branch's other actions are queued as the engine queues them, and the engine's own move step (`useMoveInner`)
// runs for the asked action with a random source that answers every `randomChance` with success and records each
// `(n, d)` (`mechanics-random.ts`). The move step stops at its hit step:
// - the precondition chance is the product of the draws made before the hit step (the stall draw of Protect), 1 when there is none;
// - the hit chance is the accuracy draw of the asked target (1 when the engine draws none); a move of a side or the field has no accuracy step and hits;
// - a move that never reaches its hit step (a failed `onTry`, no action left after Protect, a target the engine removed) fails.
// No move, ability, item or condition id is named here: whether a move reads `ateBerry` or `activeMoveActions` is
// the engine's business. Any draw that is not a `randomChance` makes the result `unsupported` (`random_effect`).
//
// The chance draws answer `true`. That is "success" for the draws the probe reads (the stall draw, accuracy); a draw that
// means a failure when true (full paralysis, confusion) belongs to the turn-start events, which `useMoveInner` does not run.

import type { Engine, PokemonSet } from "./engine.ts";
import { draw, firstTarget, need, refusingSource, RandomDraw } from "./mechanics-guard.ts";
import { declareHistory, type HBattle, type HistoryDeclaration } from "./mechanics-history.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import { type Draw, recordingRandom } from "./mechanics-random.ts";
import type { Outcome } from "./mechanics-result.ts";
import { startScenario } from "./mechanics-scenario.ts";
import { declare, evolve, type Mon, type ProbeBattle, type Queued, resolveActions, restoreStagesAndItems, retireCopies, teams } from "./mechanics-speed.ts";

/** Bumped when the probe computes an outcome differently for the same query (named by the `probe-outcome/1` feature set's identity). */
export const OUTCOME_FORMAT = 1;

type MoveOutcomeQuery = Extract<MechanicsQuery["query"], { kind: "move_outcome" }>;

const MILLION = 1_000_000;

interface OutcomeMon extends Mon {
  activeMoveActions: number;
}
interface OutcomeBattle extends ProbeBattle {
  actions: ProbeBattle["actions"] & {
    useMoveInner(move: string, pokemon: OutcomeMon, options: { target: OutcomeMon }): unknown;
    hitStepAccuracy(targets: OutcomeMon[], pokemon: OutcomeMon, move: object): boolean[];
    moveHit(...args: unknown[]): unknown;
  };
  queue: ProbeBattle["queue"] & { list: unknown[] };
}

/** What the engine's move step had drawn when it reached its hit step. */
export interface Reached {
  /** Every draw before the hit step. */
  pre: Draw[];
  /** Per target, the accuracy draws and whether the target was hit; null for a move of a side or the field (no accuracy step). */
  targets: Map<unknown, { draws: Draw[]; hit: boolean }> | null;
}

/** Thrown from the replaced hit step to end the move step where the probe has read what it needs. */
export class HitStepReached extends Error {
  readonly reached: Reached;
  constructor(reached: Reached) {
    super("the move step reached its hit step");
    this.reached = reached;
  }
}

/** The chance of a list of draws, in parts per million, rounded down: the exact product of the engine's own `randomChance` pairs (the caller has refused any other draw). */
export function chancePpm(draws: readonly Draw[]): number {
  let numerator = 1n;
  let denominator = 1n;
  for (const entry of draws) {
    if (entry.kind !== "randomChance") continue;
    numerator *= BigInt(Math.max(0, Math.min(entry.numerator, entry.denominator)));
    denominator *= BigInt(entry.denominator);
  }
  return Number((numerator * BigInt(MILLION)) / denominator);
}

/** The hit chance of the asked target's accuracy window: 0 for a target the accuracy step never met or refused, the product of its draws otherwise (1 million with none). */
export const hitChance = (window: { draws: Draw[]; hit: boolean } | undefined): number => (window === undefined || !window.hit ? 0 : chancePpm(window.draws));

/** The state the engine's move step had reached, or the error when it ended some other way. */
export function reachedOf(error: unknown): Reached {
  if (error instanceof HitStepReached) return error.reached;
  throw error;
}

const fails = (): Outcome => ({ kind: "computed", answer: { kind: "move_outcome", precondition: "fails", precondition_ppm: 0, hit_ppm: 0 } });

/** The engine's `BeforeMove` event of the asked action, as the probe runs it before the move step (see `attempt`). */
interface EventBattle {
  runEvent(name: string, target: unknown, source: unknown, effect: unknown): unknown;
  setActiveMove(move: object, pokemon: object, target: object): void;
  dex: { getActiveMove(id: string): object };
}

/** One `randomChance(n, d)` the before-move event drew, and the answer this path gave it (true is the chance's "yes"). */
export interface Chance {
  numerator: number;
  denominator: number;
  answer: boolean;
}

/** What the move step reads after the event passed: the draws before its hit step and the asked target's accuracy chance, or that it never reaches the hit step. */
export type Tail = { fails: true } | { fails: false; pre: ChanceDraw[]; hit: number };
type ChanceDraw = Extract<Draw, { kind: "randomChance" }>;

export type Leaf = { kind: "leaf"; chances: Chance[]; pass: boolean; tail: Tail | null } | { kind: "unsupported"; outcome: Outcome };

/** The most before-move outcomes (paths of draws) the probe explores; more is `random_effect`. */
const MAX_LEAVES = 16;

/**
 * Answer a `move_outcome` query on fresh scenario battles. `sets` are the query's declared sets in the contract's
 * position order, each already judged by the target format's validator. `history` is the resolved history of the query (one end of
 * every unknown fact), declared through the engine's own fields. Returns `computed`, or `unsupported` (`random_effect`) when the engine
 * draws anything but a chance, or when the asked action ties in the engine's order with another (the engine would shuffle them);
 * throws a `MechanicsProbeError` for a declared state or action the engine refuses.
 *
 * The engine's own `BeforeMove` event of the asked action runs first, as `runMove` runs it: whatever it blocks (a status's
 * full-paralysis or self-hit draw, a volatile that forbids the move) makes the move not work. Its `randomChance` draws are explored
 * as the engine's code path gives them: the event is run once for every combination of answers (a `true` is the chance's "yes"), the
 * probability of a path is the product of its draws, and the move's success chance is that of the paths on which the event passes,
 * times the chances the move step draws after it. No status, volatile or move is named here. A path of the event that draws
 * anything but a chance and passes, or more than `MAX_LEAVES` paths, is `random_effect`.
 */
export function probeMoveOutcome(engine: Engine, scenarioId: string, query: MoveOutcomeQuery, sets: PokemonSet[], history: HistoryDeclaration | null = null, maxLeaves: number = MAX_LEAVES): Outcome {
  const leaves: Leaf[] = [];
  const todo: boolean[][] = [[]];
  for (let script = todo.pop(); script !== undefined; script = todo.pop()) {
    if (leaves.length >= maxLeaves) return { kind: "unsupported", reason: "random_effect", position: query.asked };
    const attempted = attempt(engine, scenarioId, query, sets, history, script);
    if (attempted.kind === "unsupported") return attempted.outcome;
    leaves.push(attempted);
    // every draw the path answered by default is a branch: the other answer is another path
    for (let at = script.length; at < attempted.chances.length; at += 1) todo.push([...attempted.chances.slice(0, at).map((chance) => chance.answer), !(attempted.chances[at] as Chance).answer]);
  }
  return combine(leaves, query.asked);
}

/** The answer of the explored paths: the chance the event passes (exact), times the chances the move step drew before its hit step, and the accuracy chance. */
export function combine(leaves: readonly Leaf[], asked: number): Outcome {
  // a path with probability 0 (a draw answered against a chance of 0 or 1) is no path: it takes no part in the sum or in the equal-tails check
  const chanceOf = (leaf: Leaf & { kind: "leaf" }): { n: bigint; d: bigint } => {
    let n = 1n;
    let d = 1n;
    for (const chance of leaf.chances) {
      const yes = BigInt(Math.max(0, Math.min(chance.numerator, chance.denominator)));
      n *= chance.answer ? yes : BigInt(chance.denominator) - yes;
      d *= BigInt(chance.denominator);
    }
    return { n, d };
  };
  const passing = (leaves.filter((leaf) => leaf.kind === "leaf" && leaf.pass) as (Leaf & { kind: "leaf" })[]).filter((leaf) => chanceOf(leaf).n > 0n);
  if (passing.length === 0) return fails();
  const tail = passing[0]?.tail as Tail;
  if (passing.some((leaf) => JSON.stringify(leaf.tail) !== JSON.stringify(tail))) return { kind: "unsupported", reason: "random_effect", position: asked };
  if (tail.fails) return fails();
  let numerator = 0n;
  let denominator = 1n;
  for (const leaf of passing) {
    const { n, d } = chanceOf(leaf);
    numerator = numerator * d + n * denominator;
    denominator *= d;
  }
  for (const entry of tail.pre) {
    numerator *= BigInt(Math.max(0, Math.min(entry.numerator, entry.denominator)));
    denominator *= BigInt(entry.denominator);
  }
  // A chance below one part per million stays a chance of 1 ppm: 0 would read as a failed precondition, which only a move that never reaches its hit step is.
  const ppm = numerator === denominator ? MILLION : Math.max(1, Number((numerator * BigInt(MILLION)) / denominator));
  return { kind: "computed", answer: { kind: "move_outcome", precondition: ppm === MILLION ? "passes" : "chance", precondition_ppm: ppm, hit_ppm: tail.hit } };
}

/**
 * One path: a fresh scenario battle, the declared state and history, the branch queued, the asked action's `BeforeMove` event with
 * `script` answering its chances in order (a missing answer is `true`), then, if it passed, the engine's move step to its hit step.
 */
function attempt(engine: Engine, scenarioId: string, query: MoveOutcomeQuery, sets: PokemonSet[], history: HistoryDeclaration | null, script: boolean[]): Leaf {
  const battle = startScenario(engine, scenarioId, teams(query, sets)) as unknown as OutcomeBattle;
  retireCopies(battle, query);
  const refusing = refusingSource(battle.prng);
  battle.prng = refusing;
  // A spread move queued with a target location is still resolved by the engine's `getRandomTarget`, which draws: answered with the first candidate (as in the speed probe), whatever source is installed.
  const resolveTarget = battle.getRandomTarget.bind(battle);
  battle.getRandomTarget = (...args) => {
    const current = battle.prng;
    battle.prng = firstTarget;
    try {
      return resolveTarget(...args);
    } finally {
      battle.prng = current;
    }
  };
  let at = 0;
  const setAt = (place: number): void => {
    at = place;
  };
  const recorder = recordingRandom({ roll: 0, chance: true });
  try {
    declare(battle, query, setAt);
    if (history !== null) {
      declareHistory(battle as unknown as HBattle, history.resolved, { stall: history.stall, at: setAt, on_draw: history.on_draw });
      restoreStagesAndItems(battle, query);
    }
    const resolved = resolveActions(battle, query, setAt);
    evolve(battle, resolved, setAt);
    for (const item of resolved) {
      at = item.index;
      battle.getActionSpeed(item.own);
    }
    const asked = resolved[query.asked];
    need(asked !== undefined && asked.own.choice === "move", "the asked action is not a move");
    const target = battle.sides[query.target.side]?.pokemon[query.target.position] as OutcomeMon | undefined;
    need(target !== undefined && target.hp > 0, "the asked target is not a Pokemon in play");
    // The engine's own order decides what stays queued after the asked action (willAct, willMove read it): the actions that act later. A tie would be shuffled by the engine.
    const queued: Queued[] = [];
    for (const item of [...resolved].sort((a, b) => battle.comparePriority(a.own, b.own))) {
      if (item === asked) continue;
      const order = battle.comparePriority(item.own, (asked as { own: Queued }).own);
      if (order === 0) return { kind: "unsupported", outcome: { kind: "unsupported", reason: "random_effect", position: item.index } };
      if (order > 0) queued.push(item.own);
    }
    battle.queue.list.push(...queued);
    const user = (asked as { pokemon: Mon }).pokemon as OutcomeMon;
    const moveId = (asked as { own: Queued & { moveid?: string } }).own.moveid as string;

    // The before-move event, as `runMove` runs it: whatever blocks the move here makes it not work. Its chances are answered by the script.
    const chances: Chance[] = [];
    let other = false;
    battle.prng = {
      randomChance: (numerator: number, denominator: number): boolean => {
        const answer = script[chances.length] ?? true;
        chances.push({ numerator, denominator, answer });
        return answer;
      },
      random: (): number => {
        other = true; // a draw that is no chance (a self-hit's damage roll): harmless on a path that is blocked, `random_effect` on one that passes
        return 0;
      },
      sample: draw,
      shuffle: draw,
      getSeed: draw,
      setSeed: draw,
      clone: draw,
    };
    const events = battle as unknown as EventBattle;
    const move = events.dex.getActiveMove(moveId);
    // `runMove` counts the action first (Fake Out and First Impression read it, and so do before-move handlers); the probe counts it here.
    user.activeMoveActions += 1;
    // `OverrideAction` (an Encore-forced move; the Champions mod removes it from Encore, so no condition of this pin has one): the move used would be
    // another, which the probe does not follow, so the answer is `unsupported`.
    if (moveId !== "struggle") {
      const changed = events.runEvent("OverrideAction", user, target, move);
      if (changed && changed !== true) return { kind: "unsupported", outcome: { kind: "unsupported", reason: "query_kind", position: query.asked } };
    }
    events.setActiveMove(move, user, target as OutcomeMon);
    let pass = Boolean(events.runEvent("BeforeMove", user, target, move));
    // then, as `runMove` does: the move's own `beforeMoveCallback` (it blocks when it returns true). The PP check of `runMove` (no PP left blocks the move)
    // is not run: the scenario's PP is full and PP is never declared (an opponent's is unknown, an own move with no PP is not offered), so it cannot block here.
    const callback = (move as { beforeMoveCallback?: (this: unknown, ...args: unknown[]) => unknown }).beforeMoveCallback;
    if (pass && callback !== undefined && callback.call(battle, user, target, move)) pass = false;
    if (!pass) return { kind: "leaf", chances, pass: false, tail: null };
    if (other) return { kind: "unsupported", outcome: { kind: "unsupported", reason: "random_effect", position: query.asked } };

    battle.prng = recorder;
    const accuracy = battle.actions.hitStepAccuracy;
    battle.actions.hitStepAccuracy = function (this: unknown, targets, pokemon, moveObject) {
      const pre = [...recorder.draws];
      const perTarget = new Map<unknown, { draws: Draw[]; hit: boolean }>();
      // The engine's accuracy step is per target; running it one target at a time gives each target its own draws.
      for (const each of targets) {
        const start = recorder.draws.length;
        const results = accuracy.call(this, [each], pokemon, moveObject);
        perTarget.set(each, { draws: recorder.draws.slice(start), hit: Boolean(results[0]) });
      }
      throw new HitStepReached({ pre, targets: perTarget });
    };
    battle.actions.moveHit = () => {
      throw new HitStepReached({ pre: [...recorder.draws], targets: null });
    };
    let reached: Reached | null = null;
    try {
      battle.actions.useMoveInner(moveId, user, { target: target as OutcomeMon });
    } catch (error) {
      reached = reachedOf(error);
    }
    if (recorder.draws.some((entry) => entry.kind !== "randomChance")) return { kind: "unsupported", outcome: { kind: "unsupported", reason: "random_effect", position: query.asked } };
    if (reached === null) return { kind: "leaf", chances, pass: true, tail: { fails: true } };
    // The engine aims a move of its own side at the user whatever target the query names.
    const hit = reached.targets === null ? MILLION : hitChance(reached.targets.get(target) ?? reached.targets.get(user));
    return { kind: "leaf", chances, pass: true, tail: { fails: false, pre: reached.pre.filter((entry): entry is ChanceDraw => entry.kind === "randomChance"), hit } };
  } catch (error) {
    if (error instanceof RandomDraw) return { kind: "unsupported", outcome: { kind: "unsupported", reason: "random_effect", position: at } };
    throw error;
  }
}

const ignore = (): void => undefined;

/**
 * The scenario battle with the declared state and history and the branch queued, for a test that compares the declared fields with a
 * live battle's. Nothing is played.
 */
export function declaredOutcomeScenario(engine: Engine, scenarioId: string, query: Pick<MoveOutcomeQuery, "sides" | "actions" | "state">, sets: PokemonSet[], history: HistoryDeclaration | null): HBattle {
  const battle = startScenario(engine, scenarioId, teams(query, sets)) as unknown as OutcomeBattle;
  retireCopies(battle, query);
  battle.prng = refusingSource(battle.prng);
  declare(battle, query, ignore);
  if (history !== null) {
    declareHistory(battle as unknown as HBattle, history.resolved, { stall: history.stall, at: ignore, on_draw: history.on_draw });
    restoreStagesAndItems(battle, query);
  }
  return battle as unknown as HBattle;
}
