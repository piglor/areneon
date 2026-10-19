import type { Budget } from "./deadline.ts";

/**
 * The deadline watchdog and the server auto-choose detector (ADR-005 §3), for
 * one battle room. Pure logic over request identities (`rqid`) and the local
 * clock: it sends nothing and decides nothing. When a deadline runs out it
 * calls `onFire` once, and the caller sends the fallback (the actor rule's
 * choice, through the same candidate/`rqid` path as any other choice).
 *
 * Two deadlines:
 *
 * - `overrun`: the policy had until `received_at + policy_budget_ms`. That is
 *   `submit_within_ms - fallback_reserve_ms` for a budget with room for the
 *   reserve, so the fallback, sent at once, still reaches the server before
 *   `submit_within`. A budget of 0 (skip the policy) fires at the receipt.
 * - `snapshot_missing`: the request waits for the `|inactive|Time left`
 *   snapshot its budget comes from, and the server does not always send one.
 *   `RoomBattleTimer.nextRequest` (server/room-battle.ts:311-322) returns
 *   before the snapshot when the player has no bank left, or when fewer than
 *   two players have time left; then no turn timer runs for this request
 *   either (the tick is armed after the snapshot, :341), so the server will
 *   not time the bot out and nothing arrives to decide on. The policy: wait
 *   `snapshot_wait_ms` from the receipt (the snapshot is written right behind
 *   the request, so only the local round trip is at stake), then decide with
 *   no timer state and a zero budget, exactly as a re-sent request does. A
 *   snapshot that arrives later is ignored: the request is answered.
 *
 * Guarantees, whatever the caller does:
 *
 * - one request is armed at a time: arming or seeing a newer request drops
 *   the older (stale) deadline, and `cancel` drops it too (request closed,
 *   connection lost);
 * - a request fires at most once, and never after `settle` (it was answered
 *   some other way): `arm*` and `inherit` refuse a request that already fired
 *   or settled.
 *
 * The auto-choose detector. With `timeoutAutoChoose` (the `VGC Timer`), a
 * turn timeout with bank left makes the server write `>pN default` to the
 * sim (`checkTimeout`, room-battle.ts:452-454) and add NOTHING to the room,
 * so no line names it. What the player's own stream does show is the effect:
 * the sim moves on and sends a newer `|request|` (a new `rqid`: every
 * sideupdate request increments it, :804-806) while the bot's previous
 * request was never answered. A request the bot answered is never
 * reported, even if the engine refused the answer: a refused pending
 * candidate is followed by an `update` request (a new `rqid`) that the
 * server did not choose. The detector uses only the
 * bot's own requests and sends; the opponent's state is never read. A
 * disconnection is not an auto-choose: the server auto-chooses only for a
 * player it knows is active (`knownActive`, :452), and a re-sent request
 * (same `rqid`) is not a new request.
 */

export type WatchdogReason = "snapshot_missing" | "overrun";

/** How long a request waits for its timer snapshot before it is decided without one. */
export const DEFAULT_SNAPSHOT_WAIT_MS = 1_000;

export interface WatchdogFire {
  rqid: number;
  reason: WatchdogReason;
  /** The local time the deadline fell due, ms. */
  fire_at: number;
}

export interface WatchdogOptions {
  /** Local monotonic time in ms; the clock `received_at` is on. */
  clock: () => number;
  /** A deadline ran out for the armed request. Called at most once per `rqid`. */
  onFire: (fire: WatchdogFire) => void;
}

export type RequestPhase = "preview" | "move" | "forceSwitch" | "wait";

/** A request the bot never answered, though the server has moved on to a newer one. */
export interface AutoChooseDetection {
  rqid: number;
  request_seq: number;
  phase: Exclude<RequestPhase, "wait">;
}

interface Deadline {
  rqid: number;
  reason: WatchdogReason;
  fire_at: number;
}

interface Armed extends Deadline {
  timer: ReturnType<typeof setTimeout>;
}

interface OpenRequest extends AutoChooseDetection {
  answered: boolean;
}

export class DeadlineWatchdog {
  readonly #clock: () => number;
  readonly #onFire: (fire: WatchdogFire) => void;
  /** Requests that fired or were settled: never armed again. */
  readonly #done = new Set<number>();
  #armed: Armed | undefined;
  /** The latest deadline, kept after it fires or settles: an `update` follow-up inherits it. */
  #latest: Deadline | undefined;
  #open: OpenRequest | undefined;

  constructor(options: WatchdogOptions) {
    this.#clock = options.clock;
    this.#onFire = options.onFire;
  }

  /** The request `rqid` is armed and has not fired. */
  get armed(): number | undefined {
    return this.#armed?.rqid;
  }

  /**
   * A request waits for its timer snapshot: fire `waitMs` after `received_at`
   * if none came. False if the request already fired or settled.
   */
  armSnapshotWait(rqid: number, received_at: number, waitMs: number): boolean {
    assertMs("snapshot wait", waitMs);
    return this.#arm({ rqid, reason: "snapshot_missing", fire_at: received_at + waitMs });
  }

  /**
   * The policy has until `received_at + budget.policy_budget_ms`; then the
   * fallback goes out, still `fallback_reserve_ms` before `submit_within`.
   * False if the request already fired or settled.
   */
  armOverrun(rqid: number, received_at: number, budget: Budget): boolean {
    return this.#arm({ rqid, reason: "overrun", fire_at: received_at + budget.policy_budget_ms });
  }

  /**
   * An `update` request, or any request that continues the previous one,
   * keeps the original deadline (ADR-005 §3: it is inherited, never
   * restarted): the older request's deadline moves to `rqid` as it stood,
   * and one already past fires at once. False if there is none to inherit,
   * or `rqid` already fired or settled.
   */
  inherit(rqid: number): boolean {
    return this.#latest === undefined ? false : this.#arm({ ...this.#latest, rqid });
  }

  /** The request was answered another way: its deadline is dropped and cannot fire. */
  settle(rqid: number): void {
    this.#done.add(rqid);
    if (this.#armed?.rqid === rqid) this.cancel();
  }

  /** Drops the armed deadline without settling its request (closed, superseded, disconnected). */
  cancel(): void {
    if (this.#armed !== undefined) clearTimeout(this.#armed.timer);
    this.#armed = undefined;
  }

  /** Ends the battle's watch. */
  dispose(): void {
    this.cancel();
  }

  /**
   * A new request (not a re-send) arrived. Drops the stale deadline and
   * returns the detection if the previous request was left unanswered, which
   * is how the server's auto-choose shows. A `wait` request asks for nothing.
   */
  onRequest(rqid: number, request_seq: number, phase: RequestPhase): AutoChooseDetection | null {
    this.cancel();
    const previous = this.#open;
    this.#open = phase === "wait" ? undefined : { rqid, request_seq, phase, answered: false };
    if (previous === undefined || previous.answered) return null;
    return { rqid: previous.rqid, request_seq: previous.request_seq, phase: previous.phase };
  }

  /** The bot sent a choice for `rqid`. */
  onAnswered(rqid: number): void {
    if (this.#open?.rqid === rqid) this.#open.answered = true;
  }

  #arm(deadline: Deadline): boolean {
    if (this.#done.has(deadline.rqid)) return false;
    this.cancel();
    this.#latest = deadline;
    const delay = Math.max(0, deadline.fire_at - this.#clock());
    const timer = setTimeout(() => this.#fire(deadline.rqid), delay);
    this.#armed = { ...deadline, timer };
    return true;
  }

  #fire(rqid: number): void {
    const armed = this.#armed as Armed;
    this.#armed = undefined;
    this.#done.add(rqid);
    this.#onFire({ rqid, reason: armed.reason, fire_at: armed.fire_at });
  }
}

function assertMs(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a whole number of ms >= 0, got ${value}`);
  }
}
