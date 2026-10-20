import { NoCandidateError } from "./actor-rule.ts";
import { EnumeratorDefectError } from "./choice-judge.ts";
import {
  type Actor,
  BattleAbortedError,
  type BattleOptions,
  type BattleRecord,
  type ForfeitCause,
  ForfeitSignal,
  InvalidTeamError,
  type Player,
  runBattle,
} from "./runner.ts";

/**
 * Policy forfeits and harness incidents around `runBattle` (evaluation
 * protocol D2, O12; ADR-010 §2). A policy failure ends the game as a loss
 * for that side and is recorded with its cause; a harness failure is an
 * incident, which fails the run but is returned as a row rather than thrown,
 * so one bad game does not stop the others. Nothing is retried mid-game.
 */

/** Per-decision compute limits in milliseconds (`latency_budget`, a run parameter). */
export interface LatencyBudget {
  preview_ms: number;
  battle_ms: number;
}

/** A policy worker died. `cause` says whether it ran out of memory. */
export class PolicyWorkerError extends Error {
  override name = "PolicyWorkerError";
  readonly worker_cause: "crash" | "out_of_memory";
  constructor(worker_cause: "crash" | "out_of_memory", detail: string) {
    super(detail);
    this.worker_cause = worker_cause;
  }
}

/**
 * The worker met a candidate shape its parser does not know. This is a
 * tooling gap (ADR-010 §2), a harness incident and never a policy forfeit.
 */
export class UnsupportedCandidateError extends Error {
  override name = "UnsupportedCandidateError";
}

/** A scripted baseline failed (protocol D2: a harness incident, not a forfeit). */
export class BaselineFailureError extends Error {
  override name = "BaselineFailureError";
}

export interface GuardOptions {
  /**
   * `policy`: the candidate or a learned opponent, whose failures are
   * forfeits. `scripted`: a baseline, whose failure is a harness incident.
   */
  role: "policy" | "scripted";
  latency_budget: LatencyBudget;
}

// Errors that name a harness fault, whoever the actor is.
const isHarnessFault = (error: unknown): boolean =>
  error instanceof UnsupportedCandidateError ||
  error instanceof NoCandidateError ||
  error instanceof EnumeratorDefectError ||
  error instanceof BattleAbortedError ||
  error instanceof ForfeitSignal;

const describe = (error: unknown): string => (error instanceof Error ? error.message : String(error));

function workerCause(error: unknown): ForfeitCause {
  if (error instanceof PolicyWorkerError) return error.worker_cause;
  const code = (error as { code?: unknown } | null)?.code;
  return code === "ERR_WORKER_OUT_OF_MEMORY" ? "out_of_memory" : "crash";
}

/**
 * Wrap an actor. A policy that throws, answers after its budget, or answers
 * outside the request's legal choice set forfeits (`ForfeitSignal`). Harness
 * faults, and every failure of a `scripted` actor, pass through unchanged for
 * `playGame` to classify as incidents. The actor still sees only the request.
 */
export function guardActor(actor: Actor, options: GuardOptions): Actor {
  const { role, latency_budget } = options;
  for (const ms of [latency_budget.preview_ms, latency_budget.battle_ms]) {
    if (!Number.isFinite(ms) || ms <= 0) throw new RangeError("latency budget must be a positive number of ms");
  }
  return async (request) => {
    const budget = request.phase === "preview" ? latency_budget.preview_ms : latency_budget.battle_ms;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const overrun = new Promise<"overrun">((resolve) => {
      timer = setTimeout(() => resolve("overrun"), budget);
    });
    let reply: unknown;
    try {
      const answer = await Promise.race([Promise.resolve().then(() => actor(request)), overrun]);
      if (answer === "overrun") {
        if (role === "scripted") throw new BaselineFailureError(`scripted actor exceeded ${budget} ms`);
        throw new ForfeitSignal("latency_overrun", `no reply within ${budget} ms`);
      }
      reply = answer;
    } catch (error) {
      if (error instanceof BaselineFailureError || isHarnessFault(error)) throw error;
      if (role === "scripted") throw new BaselineFailureError(describe(error));
      throw new ForfeitSignal(workerCause(error), describe(error));
    } finally {
      clearTimeout(timer);
    }
    const legal = request.legal;
    if (typeof reply !== "string" || legal === undefined || ![...legal.choices, ...legal.pending].includes(reply)) {
      const detail = `reply is not in the legal choice set: ${JSON.stringify(reply)}`;
      if (role === "scripted") throw new BaselineFailureError(detail);
      throw new ForfeitSignal("invalid_reply", detail);
    }
    return reply;
  };
}

export type IncidentClass =
  | "unsupported_candidate"
  | "enumerator_defect"
  | "invalid_team"
  | "no_candidate"
  | "baseline_failure"
  | "engine_error"
  | "unclassified";

/** One game's outcome, the per-game result row's outcome fields. */
export type GameOutcome =
  | { kind: "played"; winner: Player | null; turns: number }
  | {
      kind: "forfeit";
      /** The seat whose policy failed; it loses. */
      loser: Player;
      winner: Player;
      turns: number;
      cause: ForfeitCause;
      request_seq: number;
      detail: string;
    }
  | { kind: "incident"; incident: IncidentClass; message: string };

export interface PlayedGame {
  outcome: GameOutcome;
  /** Absent for an incident: the run fails and its games are not used. */
  record: BattleRecord | null;
}

function incidentClass(error: unknown): IncidentClass {
  if (error instanceof UnsupportedCandidateError) return "unsupported_candidate";
  if (error instanceof EnumeratorDefectError) return "enumerator_defect";
  if (error instanceof InvalidTeamError) return "invalid_team";
  if (error instanceof NoCandidateError) return "no_candidate";
  if (error instanceof BaselineFailureError) return "baseline_failure";
  if (error instanceof BattleAbortedError) return "engine_error";
  return "unclassified";
}

/**
 * Run one game through `runBattle` and return its outcome row instead of
 * throwing. The caller guards its actors with `guardActor`. A forfeit is a
 * scored loss; an incident fails the run (protocol D2).
 */
export async function playGame(options: BattleOptions): Promise<PlayedGame> {
  let record: BattleRecord;
  try {
    record = await runBattle(options);
  } catch (error) {
    return { outcome: { kind: "incident", incident: incidentClass(error), message: describe(error) }, record: null };
  }
  const { forfeit } = record;
  if (forfeit === undefined) {
    return { outcome: { kind: "played", winner: record.winner, turns: record.turns }, record };
  }
  return {
    outcome: {
      kind: "forfeit",
      loser: forfeit.player,
      winner: forfeit.player === "p1" ? "p2" : "p1",
      turns: record.turns,
      cause: forfeit.cause,
      request_seq: forfeit.request_seq,
      detail: forfeit.detail,
    },
    record,
  };
}
