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
import { firstTarget, need, refusingSource, RandomDraw } from "./mechanics-guard.ts";
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
interface Reached {
  /** Every draw before the hit step. */
  pre: Draw[];
  /** Per target, the accuracy draws and whether the target was hit; null for a move of a side or the field (no accuracy step). */
  targets: Map<unknown, { draws: Draw[]; hit: boolean }> | null;
}

/** Thrown from the replaced hit step to end the move step where the probe has read what it needs. */
class HitStepReached extends Error {
  readonly reached: Reached;
  constructor(reached: Reached) {
    super("the move step reached its hit step");
    this.reached = reached;
  }
}

/** The chance of a list of `randomChance` draws, in parts per million, rounded down: the exact product of the engine's own pairs. */
function chancePpm(draws: readonly Draw[]): number {
  let numerator = 1n;
  let denominator = 1n;
  for (const entry of draws) {
    if (entry.kind !== "randomChance" || entry.denominator <= 0) throw new Error("not a chance draw");
    numerator *= BigInt(Math.max(0, Math.min(entry.numerator, entry.denominator)));
    denominator *= BigInt(entry.denominator);
  }
  return Number((numerator * BigInt(MILLION)) / denominator);
}

const fails = (): Outcome => ({ kind: "computed", answer: { kind: "move_outcome", precondition: "fails", precondition_ppm: 0, hit_ppm: 0 } });

/**
 * Answer a `move_outcome` query on a fresh scenario battle. `sets` are the query's declared sets in the contract's
 * position order, each already judged by the target format's validator. `history` is the resolved history of the query (one end of
 * every unknown fact), declared through the engine's own fields. Returns `computed`, or `unsupported` (`random_effect`) when the engine
 * draws anything but a chance, or when the asked action ties in the engine's order with another (the engine would shuffle them);
 * throws a `MechanicsProbeError` for a declared state or action the engine refuses.
 */
export function probeMoveOutcome(engine: Engine, scenarioId: string, query: MoveOutcomeQuery, sets: PokemonSet[], history: HistoryDeclaration | null = null): Outcome {
  const battle = startScenario(engine, scenarioId, teams(query, sets)) as unknown as OutcomeBattle;
  retireCopies(battle, query);
  const refusing = refusingSource(battle.prng);
  battle.prng = refusing;
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
  const recorder = recordingRandom({ roll: 0, chance: true });
  try {
    declare(battle, query, (position) => (at = position));
    if (history !== null) {
      declareHistory(battle as unknown as HBattle, history.resolved, { stall: history.stall, at: (position) => (at = position) });
      restoreStagesAndItems(battle, query);
    }
    const resolved = resolveActions(battle, query, (index) => (at = index));
    evolve(battle, resolved, (index) => (at = index));
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
    for (const item of resolved) {
      if (item === asked) continue;
      const order = battle.comparePriority(item.own, (asked as { own: Queued }).own);
      if (order === 0) return { kind: "unsupported", reason: "random_effect", position: item.index };
      if (order > 0) queued.push(item.own);
    }
    queued.sort((a, b) => battle.comparePriority(a, b));
    battle.queue.list.push(...queued);

    battle.prng = recorder;
    const accuracy = battle.actions.hitStepAccuracy;
    battle.actions.hitStepAccuracy = function (this: unknown, targets, pokemon, move) {
      const pre = [...recorder.draws];
      const perTarget = new Map<unknown, { draws: Draw[]; hit: boolean }>();
      // The engine's accuracy step is per target; running it one target at a time gives each target its own draws.
      for (const each of targets) {
        const start = recorder.draws.length;
        const results = accuracy.call(this, [each], pokemon, move);
        perTarget.set(each, { draws: recorder.draws.slice(start), hit: Boolean(results[0]) });
      }
      throw new HitStepReached({ pre, targets: perTarget });
    };
    battle.actions.moveHit = () => {
      throw new HitStepReached({ pre: [...recorder.draws], targets: null });
    };
    const user = (asked as { pokemon: Mon }).pokemon as OutcomeMon;
    // `runMove` counts the action before the move step starts; the probe enters at the move step, so it counts it here.
    user.activeMoveActions += 1;
    let reached: Reached | null = null;
    try {
      battle.actions.useMoveInner((asked as { own: Queued & { moveid?: string } }).own.moveid as string, user, { target: target as OutcomeMon });
    } catch (error) {
      if (!(error instanceof HitStepReached)) {
        // A member of the engine's random source the recorder does not answer (`sample`, `shuffle`) hands the engine a number where it wants an object.
        if (recorder.draws.some((entry) => entry.kind === "other")) return { kind: "unsupported", reason: "random_effect", position: query.asked };
        throw error;
      }
      reached = error.reached;
    }
    if (recorder.draws.some((entry) => entry.kind !== "randomChance")) return { kind: "unsupported", reason: "random_effect", position: query.asked };
    if (reached === null) return fails();
    let hitPpm = MILLION;
    if (reached.targets !== null) {
      // The engine aims a move of its own side at the user whatever target the query names.
      const own = reached.targets.get(target) ?? reached.targets.get(user);
      if (own === undefined) return fails();
      hitPpm = own.hit ? chancePpm(own.draws) : 0;
    }
    const preconditionPpm = chancePpm(reached.pre);
    const precondition = preconditionPpm === 0 ? "fails" : preconditionPpm === MILLION ? "passes" : "chance";
    return { kind: "computed", answer: { kind: "move_outcome", precondition, precondition_ppm: preconditionPpm, hit_ppm: hitPpm } };
  } catch (error) {
    if (error instanceof RandomDraw) return { kind: "unsupported", reason: "random_effect", position: at };
    throw error;
  }
}

/**
 * The scenario battle with the declared state and history and the branch queued, for a test that compares the declared fields with a
 * live battle's. Nothing is played.
 */
export function declaredOutcomeScenario(engine: Engine, scenarioId: string, query: Pick<MoveOutcomeQuery, "sides" | "actions" | "state">, sets: PokemonSet[], history: HistoryDeclaration | null): HBattle {
  const battle = startScenario(engine, scenarioId, teams(query, sets)) as unknown as OutcomeBattle;
  retireCopies(battle, query);
  battle.prng = refusingSource(battle.prng);
  declare(battle, query, () => undefined);
  if (history !== null) {
    declareHistory(battle as unknown as HBattle, history.resolved, { stall: history.stall, at: () => undefined });
    restoreStagesAndItems(battle, query);
  }
  return battle as unknown as HBattle;
}
