import type { Budget } from "./deadline.ts";
import type { RoomEvent } from "./protocol.ts";

/**
 * Why the bot forfeited instead of playing on with Closed Team Sheets
 * (ADR-005 §2):
 *
 * - `not_confirmed`: its own rejection line did not arrive within the
 *   preview `policy_budget`;
 * - `refused`: the server answered the rejection with an `|error|`, so the
 *   line can no longer arrive;
 * - `showteam`: a `|showteam|` line arrived, so a team sheet is open. The
 *   record keeps only its side (`|showteam|p2`), never the packed team.
 */
export type OtsFailure = "not_confirmed" | "refused" | "showteam";

/**
 * The decline failed and the bot forfeited the match (ADR-005 §2). It is
 * thrown by the call that saw the failure, after the `/forfeit` was sent,
 * and again by any later `submitPreview`; the session records `reason` and
 * `detail` and ends the match.
 */
export class OtsDeclineError extends Error {
  override name = "OtsDeclineError";
  readonly reason: OtsFailure;
  readonly detail: string;
  /** `false` if the transport refused the `/forfeit` line too (`cause` holds why). */
  readonly forfeitSent: boolean;
  constructor(reason: OtsFailure, detail: string, forfeitSent: boolean, options?: ErrorOptions) {
    super(`open team sheets not declined (${reason}): ${detail}`, options);
    this.reason = reason;
    this.detail = detail;
    this.forfeitSent = forfeitSent;
  }
}

/** Where the decline stands. */
export type OtsStatus = "idle" | "awaiting" | "confirmed" | "forfeited";

export interface OtsDeclineOptions {
  /** The battle room the bot plays in. */
  room: string;
  /**
   * The bot's own bare name, `user.name`, which the confirmation line names:
   * `userName` of the `|updateuser|` identity, without its group symbol or
   * `@!` status suffix (server/users.ts:563-567, :975).
   */
  botName: string;
  /** Sends one protocol line, e.g. `Transport.send`. */
  send: (line: string) => void;
}

/**
 * The confirmed Open Team Sheets decline of ADR-005 §2, for one battle room.
 * Pure logic over the protocol reader's events: it never connects and keeps
 * no clock; the caller passes the local receipt time.
 *
 * - On its (first) Team Preview request the bot sends
 *   `/rejectopenteamsheets`, and never `/acceptopenteamsheets`.
 * - The preview choice goes out only through `submitPreview`, which holds it
 *   until the bot's own `<name> rejected open team sheets.` line arrives.
 *   The server adds that line to the battle room only after it recorded the
 *   rejection (server/chat-commands/core.ts:980-1002, `rejectopenteamsheets`:
 *   `this.add(this.TL\`${user.name} rejected open team sheets.\`)`); in Team
 *   Preview, with the bot an undecided player in an OTS format, it always
 *   succeeds. The `|uhtmlchange|otsrequest|` line is no confirmation: the
 *   server sends it on an accept too (:971), and the sim on battle start
 *   (data/rulesets.ts:1996-2000).
 * - It forfeits, and throws `OtsDeclineError`, when that line has not arrived
 *   within the preview `policy_budget_ms` (`tick`), when the server answers
 *   the rejection with an `|error|` (core.ts:984-997 throw `Chat.ErrorMessage`,
 *   which the server sends to the bot as `|error|`), or whenever a
 *   `|showteam|` line arrives: only the sim writes it, and only once every
 *   player accepted (sim/battle.ts:3220, `showOpenTeamSheets`, run on
 *   `>show-openteamsheets`, core.ts:973-975). A human's `!showteam` is HTML,
 *   never a `|showteam|` line.
 *
 * The session must keep one `OtsDecline` per battle room across reconnects
 * (#757 wires it). A fresh one after a reconnect during Team Preview always
 * forfeits: it ignores the replayed confirmation line while idle, and its
 * second rejection gets an `|error|` (core.ts:984-997).
 *
 * The constructor throws a `TypeError` on a `botName` that is still an
 * identity: one with a leading space or a trailing `@!`.
 */
export class OtsDecline {
  readonly #options: OtsDeclineOptions;
  readonly #confirmation: string;
  #status: OtsStatus = "idle";
  #deadline = 0;
  #held: string | undefined;
  #failure: OtsDeclineError | undefined;

  constructor(options: OtsDeclineOptions) {
    if (options.botName.startsWith(" ") || options.botName.endsWith("@!")) {
      throw new TypeError(`botName must be the bare user name, not an identity: "${options.botName}"`);
    }
    this.#options = options;
    this.#confirmation = `${options.botName} rejected open team sheets.`;
  }

  get status(): OtsStatus {
    return this.#status;
  }

  /**
   * Reads one event of the bot's connection; events of other rooms are
   * ignored. `now` is the local receipt time in ms, on the clock `tick`
   * uses. `previewBudget` is the Team Preview request's budget
   * (`previewBudget(...)` of deadline.ts, from the timer state at receipt):
   * the rejection line must arrive within its `policy_budget_ms`. It is
   * required with the first preview request, and ignored otherwise.
   * Throws `OtsDeclineError` after forfeiting.
   */
  onEvent(event: RoomEvent, now: number, previewBudget?: Budget): void {
    if (event.room !== this.#options.room || this.#status === "forfeited") return;
    if (event.kind === "line" && event.line.startsWith("|showteam|")) {
      this.#forfeit("showteam", event.line.split("|", 3).join("|"));
    }
    if (event.kind === "request" && event.request.phase === "preview" && this.#status === "idle") {
      if (previewBudget === undefined) throw new TypeError("the Team Preview request needs its preview budget");
      this.#status = "awaiting";
      this.#deadline = now + previewBudget.policy_budget_ms;
      this.#options.send(`${this.#options.room}|/rejectopenteamsheets`);
      return;
    }
    if (this.#status !== "awaiting") return;
    this.tick(now);
    if (event.kind === "error") this.#forfeit("refused", event.error);
    if (event.kind === "line" && event.line === this.#confirmation) {
      this.#status = "confirmed";
      const held = this.#held;
      this.#held = undefined;
      if (held !== undefined) this.#options.send(held);
    }
  }

  /**
   * Sends the bot's preview choice (a whole protocol line, e.g.
   * `room|/choose team 1234|rqid`) once the decline is confirmed, at once if
   * it already is. After a forfeit it throws that forfeit's `OtsDeclineError`
   * again, and before the preview request it throws an `Error`: the choice
   * never goes out unconfirmed.
   */
  submitPreview(line: string): void {
    if (this.#failure !== undefined) throw this.#failure;
    if (this.#status === "idle") throw new Error("no Team Preview request yet: nothing to choose");
    if (this.#status === "confirmed") this.#options.send(line);
    else this.#held = line;
  }

  /**
   * Checks the deadline at local time `now`; `onEvent` checks it too, so a
   * line that arrives late confirms nothing. Forfeits and throws
   * `OtsDeclineError` once the preview `policy_budget` has run out without
   * the confirmation line.
   */
  tick(now: number): void {
    if (this.#status === "awaiting" && now >= this.#deadline) {
      this.#forfeit("not_confirmed", `no "${this.#confirmation}" line within the preview policy budget`);
    }
  }

  #forfeit(reason: OtsFailure, detail: string): never {
    this.#status = "forfeited";
    this.#held = undefined;
    try {
      this.#options.send(`${this.#options.room}|/forfeit`);
      this.#failure = new OtsDeclineError(reason, detail, true);
    } catch (cause) {
      this.#failure = new OtsDeclineError(reason, detail, false, { cause });
    }
    throw this.#failure;
  }
}
