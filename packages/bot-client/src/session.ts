import {
  actorRule,
  type Engine,
  legalChoiceSet,
  type LegalChoiceSet,
  type PlayerChannel,
  type PokemonSet,
  resolveFormat,
  type RulesSnapshot,
  validateTeam,
} from "@areneon/showdown-bridge";
import { A2Channel } from "./a2-channel.ts";
import { type ChallengeRefusal, decideChallenge, timerOn, toId, userName } from "./challenge.ts";
import {
  A2_BUDGET_CONFIG,
  battleBudget,
  type Budget,
  type BudgetConfig,
  estimateRemainingDecisions,
  previewBudget,
  type TimerSnapshot,
} from "./deadline.ts";
import { assertLocal } from "./local-url.ts";
import { OtsDecline, OtsDeclineError, type OtsFailure, type OtsStatus } from "./ots.ts";
import { GLOBAL_ROOM, ProtocolReader, type RoomEvent, type SequencedRequest } from "./protocol.ts";
import { classifyRefusal, type CurrentRequest, type Refusal, type RefusalMode, type Submission } from "./refusal.ts";
import type { Transport } from "./transport.ts";
import {
  type AutoChooseDetection,
  DEFAULT_SNAPSHOT_WAIT_MS,
  DeadlineWatchdog,
  type WatchdogFire,
  type WatchdogReason,
} from "./watchdog.ts";
import { connectWebSocket } from "./websocket-transport.ts";

/** A session that cannot go on: it forfeits its open battle and stops. */
export class SessionError extends Error {
  override name = "SessionError";
}

export interface A2BotOptions {
  engine: Engine;
  /** The rules snapshot for the pin, loaded once (`loadRulesSnapshot`). */
  snapshot: RulesSnapshot;
  /** The local server's WebSocket URL. Refused unless `assertLocal` passes. */
  url: string;
  /** The bot's bare user name (ADR-005 §4: it carries the session's random suffix). */
  name: string;
  /** The human named at launch: the only challenger accepted (ADR-005 §4). */
  challenger: string;
  /** The format the team is for. The team must pass the pinned validator for it. */
  format: string;
  team: PokemonSet[];
  /** `private` for a real team, `synthetic` for a fixture (ADR-008 §5). */
  data_class: PlayerChannel["data_class"];
  /** `live_match` only against a human (ADR-005 §1). Default `harness`. */
  mode?: RefusalMode;
  /** Stop once this many battles have ended. Default: never. */
  battles?: number;
  /** Stops the session; an open battle is left to the server. */
  signal?: AbortSignal;
  /**
   * Bot-vs-bot harness only: once logged in, challenge this user to `format`.
   * A sparring bot never challenges; the human does.
   */
  challenge?: string;
  /** Opens a transport to `url`. Default `connectWebSocket`. */
  connect?: (url: string) => Promise<Transport>;
  /** Local monotonic time in ms. Default `performance.now`. */
  clock?: () => number;
  /** Default `A2_BUDGET_CONFIG`. */
  budget?: BudgetConfig;
  /**
   * How long a request waits for the server's timer snapshot before it is
   * answered without one (the server skips it when the bank is 0 or only one
   * player has time left; see `watchdog.ts`). Default 1 s.
   */
  snapshotWaitMs?: number;
  /**
   * After a close the session reconnects at once, then every `delayMs`, up
   * to `attempts` times in a row without a login in between. Default 5 x 200 ms.
   */
  reconnect?: { attempts: number; delayMs: number };
  /**
   * After the bot accepts a challenge, how long it waits for the battle to
   * start before it takes the accept for failed and may accept again.
   * Default 10 s.
   */
  acceptTimeoutMs?: number;
  onLog?: (entry: SessionLog) => void;
  /** Called once per battle, when it ends. */
  onBattle?: (record: A2BattleRecord) => void;
}

/** One answered request. Only the bot's own view: its request, its candidates, its choice. */
export interface DecisionRecord {
  request_seq: number;
  /** The count of `|request|` lines in this player's channel at the decision: the observation's numbering. */
  channel_request_seq: number;
  rqid: number;
  request_hash: string;
  phase: "preview" | "move" | "forceSwitch";
  /** The candidates the choice came from (`judge: null`: the engine judges at submission). */
  legal: LegalChoiceSet;
  choice: string;
  /**
   * The server's timer snapshot for this request; `null` when none came
   * (a request re-sent on a rejoin, or the timer off), and the budget is 0.
   */
  timer: TimerSnapshot | null;
  budget: Budget;
  /** The update after an `[Unavailable choice]` keeps the original request's receipt and budget (ADR-005 §3). */
  inherited: boolean;
  /** Set when the watchdog, not a timer snapshot, made this decision: the fallback went out unbudgeted. */
  watchdog?: WatchdogReason;
  remaining_decisions_estimate?: number;
  /** Local ms from the (original) request's receipt to the send. */
  elapsed_ms: number;
}

export type BattleResult = "win" | "loss" | "tie" | "unfinished";

/** One battle as the bot saw it: its own channel and decisions, never the omniscient view. */
export interface A2BattleRecord {
  room: string;
  player: PlayerChannel["player"] | null;
  result: BattleResult;
  winner: string | null;
  ots: OtsStatus;
  ots_failure: { reason: OtsFailure; detail: string } | null;
  /** A live human match on the local server (ADR-005 §1) in which the engine refused a non-pending candidate: excluded from evaluation. */
  defect: boolean;
  /** Requests the server answered for the bot (`server_auto_choose_detected`): it must never fire (ADR-005 §3). */
  server_auto_chooses: AutoChooseDetection[];
  decisions: DecisionRecord[];
  /** This player's channel, one per connection that joined the room (each starts at its `|init|`). */
  channels: PlayerChannel[];
}

export interface A2SessionResult {
  battles: A2BattleRecord[];
}

export type SessionLog =
  | { event: "connected"; attempt: number }
  | { event: "disconnected"; code: number }
  | { event: "logged_in"; name: string }
  | { event: "challenge_accepted"; from: string; format: string }
  /** An accepted challenge whose battle did not start: the bot may accept again. */
  | { event: "accept_abandoned"; reason: "timeout" | "disconnected" }
  | {
      event: "challenge_rejected";
      from: string;
      format: string;
      reason: ChallengeRefusal | "busy" | "stopping";
      detail: string;
    }
  | { event: "battle_started"; room: string }
  | { event: "decision"; room: string; decision: DecisionRecord }
  /** A rejoin re-sent a request the server already holds a choice for: not chosen again. */
  | { event: "sentchoice"; room: string; request_seq: number; choice: string }
  /** A rejoin re-sent a request the bot had answered, without `|sentchoice|`: the choice is sent again. */
  | { event: "resent_choice"; room: string; request_seq: number; choice: string }
  | { event: "refusal"; room: string; refusal: Refusal }
  /**
   * `line` is the raw text of a line the channel did not keep. Operator-only
   * data: it can hold anything the server sent, so a CLI log shows its kind
   * only (`|kind`), and nothing but this event carries it.
   */
  | { event: "dropped"; room: string; line: string }
  | { event: "ots_failure"; room: string; reason: OtsFailure; detail: string }
  /** The watchdog sent the fallback: the snapshot never came, or the policy overran. */
  | { event: "watchdog"; room: string; rqid: number; request_seq: number; reason: WatchdogReason }
  /** The server chose for the bot: a newer request arrived while this one was unanswered. */
  | { event: "server_auto_choose_detected"; room: string; detection: AutoChooseDetection }
  | { event: "battle_ended"; room: string; result: BattleResult; winner: string | null }
  /** A `|popup|` or a global `|error|`: server text for the operator. */
  | { event: "server_message"; line: string };

/** `|inactive|` with the player's own timer state, sent after each new request (server/room-battle.ts `nextRequest`). */
const TIME_LEFT = /^\|inactive\|Time left: (\d+) sec this turn \| (\d+) sec total(?: \| (\d+) sec grace)?$/;
const TIMER_ON = "|inactive|Battle timer is ON";
const TIMER_OFF = "|inactiveoff|";
const BATTLE_INIT = "|init|battle";
/** No timer state: skip the policy and send at once. */
const ZERO_BUDGET: Budget = Object.freeze({ submit_within_ms: 0, policy_budget_ms: 0 });

/** A request waiting for its timer snapshot. */
interface Pending {
  request: SequencedRequest & { rqid: number };
  legal: LegalChoiceSet;
  channel_request_seq: number;
  received_at: number;
}

interface Deadline {
  timer: TimerSnapshot | null;
  budget: Budget;
  inherited: boolean;
  watchdog?: WatchdogReason;
  remaining_decisions_estimate?: number;
}

/** One battle room, across connections. */
interface Battle {
  room: string;
  ots: OtsDecline;
  otsTimer: ReturnType<typeof setTimeout> | undefined;
  otsFailure: A2BattleRecord["ots_failure"];
  player: PlayerChannel["player"] | null;
  /** The open request, numbered as the current connection numbers it. */
  current: (CurrentRequest & { phase: SequencedRequest["phase"] }) | null;
  pending: Pending | undefined;
  /** The deadline of `pending` and the server auto-choose detector. */
  watchdog: DeadlineWatchdog;
  submission: (Submission & { rqid: number }) | undefined;
  /** Every request seen, by rqid: its hash. */
  seen: Map<number, string>;
  /** The choice the bot made for each answered request, by rqid. */
  answered: Map<number, string>;
  /** The last decision's receipt and deadline, inherited by an update request. */
  last: { received_at: number; deadline: Deadline } | undefined;
  /** Live human match: choices for the current request the engine refused as defects. */
  refused: string[];
  /** The next request is the server's re-send on a rejoin. */
  rejoin: boolean;
  /**
   * Whether the battle timer is on, as last seen on any connection. A rejoin
   * starts from it, and the room log replayed on the rejoin then corrects it.
   */
  timer: boolean;
  result: BattleResult;
  winner: string | null;
  defect: boolean;
  autoChooses: AutoChooseDetection[];
  decisions: DecisionRecord[];
  channels: PlayerChannel[];
}

/** One battle room on one connection. */
interface RoomOnConnection {
  /** Frames since `|init|`, kept until the first request names the player's seat. */
  frames: string[];
  channel: A2Channel | undefined;
  /** How many of the channel's dropped lines are logged already: each build repeats them. */
  logged: number;
}

interface Connection {
  transport: Transport;
  reader: ProtocolReader;
  rooms: Map<string, RoomOnConnection>;
  loggedIn: boolean;
}

/**
 * Runs the A2 bot on the local server (ADR-005 §1-§4) until `battles` have
 * ended or `signal` aborts.
 *
 * - It connects only through `assertLocal`, and logs in as `name` with
 *   `/trn <name>,0,`: the launcher starts the server with `--no-security`,
 *   which makes it take an unregistered name without a token
 *   (server/config-loader.ts `noguestsecurity`, server/users.ts
 *   `validateToken`). It then sets its team with `/utm`.
 * - It accepts only challenges `decideChallenge` accepts, one battle at a time.
 * - In a battle it sends `/timer on` at `|init|battle`, declines open team
 *   sheets through one `OtsDecline` per room that lives across reconnects,
 *   and sends its preview choice only through `submitPreview`.
 * - It answers every non-`wait` request with the actor rule over the
 *   request's candidates (`legalChoiceSet`, `judge: null`): the engine judges
 *   at submission, and refusals are classified by `classifyRefusal`.
 * - Each new request's budget comes from the server's timer snapshot that
 *   follows it (`previewBudget` / `battleBudget`); the update request after
 *   an `[Unavailable choice]` inherits the original's. A request with no
 *   snapshot to come gets a zero budget, and one whose snapshot does not
 *   come within `snapshotWaitMs` is answered by the watchdog (`watchdog.ts`),
 *   which also detects the server choosing for the bot.
 * - On a close it reconnects and logs in again; the server rejoins the bot
 *   to its battle and re-sends its request, with `|sentchoice|` if it holds
 *   a choice (ADR-005 §1). Each connection's channel starts at its `|init|`.
 * - An `|error|` no rule classifies, an enumerator defect in a harness run,
 *   or a request the pinned decoder refuses ends the session: it forfeits the
 *   open battle and rejects.
 *
 * It records only the bot's own player channel and decisions.
 */
export async function runA2Bot(options: A2BotOptions): Promise<A2SessionResult> {
  assertLocal(options.url);
  const battles = options.battles ?? Infinity;
  if (battles !== Infinity && !(Number.isSafeInteger(battles) && battles >= 1)) {
    throw new RangeError(`battles must be a positive integer, not ${battles}`);
  }
  const format = resolveFormat(options.engine, options.format, options.snapshot);
  const verdict = validateTeam(options.engine, format, options.team);
  if (!verdict.valid) throw new SessionError(`the team is not valid for ${options.format}: ${verdict.problems.join("; ")}`);
  return new Session(options, battles).run();
}

class Session {
  readonly #options: A2BotOptions;
  readonly #battlesWanted: number;
  readonly #clock: () => number;
  readonly #connect: (url: string) => Promise<Transport>;
  readonly #reconnect: { attempts: number; delayMs: number };
  readonly #packedTeam: string;
  readonly #battles = new Map<string, Battle>();
  #connection: Connection | undefined;
  #failures = 0;
  #retryTimer: ReturnType<typeof setTimeout> | undefined;
  #challenged = false;
  #accepting = false;
  #acceptTimer: ReturnType<typeof setTimeout> | undefined;
  #onAbort: (() => void) | undefined;
  #ended = 0;
  #stopping = false;
  #settled = false;
  readonly #done = Promise.withResolvers<A2SessionResult>();

  constructor(options: A2BotOptions, battles: number) {
    this.#options = options;
    this.#battlesWanted = battles;
    this.#clock = options.clock ?? (() => performance.now());
    this.#connect = options.connect ?? ((url) => connectWebSocket(url));
    this.#reconnect = options.reconnect ?? { attempts: 5, delayMs: 200 };
    this.#packedTeam = options.engine.sim.Teams.pack(options.team);
  }

  run(): Promise<A2SessionResult> {
    const done = this.#done.promise;
    const { signal } = this.#options;
    if (signal?.aborted === true) {
      this.#stop();
      return done;
    }
    this.#onAbort = () => this.#stop();
    signal?.addEventListener("abort", this.#onAbort, { once: true });
    void this.#open(1);
    return done;
  }

  #log(entry: SessionLog): void {
    this.#options.onLog?.(entry);
  }

  async #open(attempt: number): Promise<void> {
    let transport: Transport;
    try {
      transport = await this.#connect(this.#options.url);
    } catch (error) {
      this.#retry(error);
      return;
    }
    if (this.#stopping) {
      transport.close();
      return;
    }
    const connection: Connection = { transport, reader: new ProtocolReader(), rooms: new Map(), loggedIn: false };
    this.#connection = connection;
    this.#guard(() => this.#log({ event: "connected", attempt }));
    transport.onClose((code) => this.#guard(() => this.#closed(connection, code)));
    transport.onFrame((frame) => this.#guard(() => this.#frame(connection, frame)));
  }

  /** A failed connect or a close before the next login: try again, or give up. */
  #retry(cause: unknown): void {
    this.#failures += 1;
    if (this.#stopping) return this.#finish();
    if (this.#failures > this.#reconnect.attempts) {
      return this.#fail(new SessionError(`no connection after ${this.#reconnect.attempts} attempts`, { cause }));
    }
    const attempt = this.#failures + 1;
    const delay = this.#failures === 1 ? 0 : this.#reconnect.delayMs;
    this.#retryTimer = setTimeout(() => void this.#open(attempt), delay);
  }

  #guard(fn: () => void): void {
    try {
      fn();
    } catch (error) {
      this.#fail(error);
    }
  }

  #send(line: string): void {
    const connection = this.#connection;
    if (connection === undefined) throw new SessionError(`not connected: cannot send ${line}`);
    connection.transport.send(line);
  }

  #closed(connection: Connection, code: number): void {
    this.#connection = undefined;
    // An accept that was waiting for its battle on this connection can no longer be told.
    this.#acceptEnded("disconnected");
    for (const [room, state] of connection.rooms) {
      const battle = this.#battles.get(room);
      if (battle !== undefined && battle.result === "unfinished") {
        // The rejoin re-sends the request, and a held one is answered then.
        battle.watchdog.cancel();
        this.#keepChannel(battle, state);
      }
    }
    if (this.#stopping) return this.#finish();
    this.#log({ event: "disconnected", code });
    this.#retry(new SessionError(`connection closed with code ${code}`));
  }

  #stop(): void {
    if (this.#stopping) return;
    this.#stopping = true;
    clearTimeout(this.#retryTimer);
    if (this.#connection === undefined) return this.#finish();
    this.#connection.transport.close();
  }

  #fail(error: unknown): void {
    this.#stopping = true;
    for (const battle of this.#battles.values()) {
      if (battle.result !== "unfinished") continue;
      try {
        this.#send(`${battle.room}|/forfeit`);
      } catch {
        // Not connected: the server's timer ends the battle.
      }
    }
    this.#settle(() => this.#done.reject(error));
    this.#connection?.transport.close();
  }

  #finish(): void {
    this.#settle(() => this.#done.resolve({ battles: [...this.#battles.values()].map((b) => this.#record(b)) }));
  }

  #settle(fn: () => void): void {
    if (this.#settled) return;
    this.#settled = true;
    clearTimeout(this.#retryTimer);
    clearTimeout(this.#acceptTimer);
    if (this.#onAbort !== undefined) this.#options.signal?.removeEventListener("abort", this.#onAbort);
    for (const battle of this.#battles.values()) {
      clearTimeout(battle.otsTimer);
      battle.watchdog.dispose();
    }
    fn();
  }

  #record(battle: Battle): A2BattleRecord {
    return {
      room: battle.room,
      player: battle.player,
      result: battle.result,
      winner: battle.winner,
      ots: battle.ots.status,
      ots_failure: battle.otsFailure,
      defect: battle.defect,
      server_auto_chooses: battle.autoChooses,
      decisions: battle.decisions,
      channels: battle.channels,
    };
  }

  #frame(connection: Connection, frame: string): void {
    const now = this.#clock();
    const events = connection.reader.read(frame);
    const room = events[0]?.room;
    if (room === undefined) return;
    if (room === GLOBAL_ROOM) {
      for (const event of events) this.#global(connection, event);
      return;
    }
    if (events.some((event) => event.kind === "line" && event.line === BATTLE_INIT)) this.#joined(connection, room);
    const battle = this.#battles.get(room);
    const state = connection.rooms.get(room);
    if (battle === undefined || state === undefined || battle.result !== "unfinished") return;
    this.#receive(battle, state, frame, events);
    events.forEach((event, i) => this.#battleEvent(battle, state, event, events.slice(i + 1), now));
  }

  #global(connection: Connection, event: RoomEvent): void {
    if (event.kind === "error") return this.#log({ event: "server_message", line: `|error|${event.error}` });
    if (event.kind !== "line") return;
    const { line } = event;
    const { name } = this.#options;
    if (line.startsWith("|challstr|")) return this.#send(`|/trn ${name},0,`);
    if (line.startsWith("|nametaken|")) {
      // Held by a connection the server has not dropped yet: close and try again.
      this.#log({ event: "server_message", line });
      return connection.transport.close();
    }
    if (line.startsWith("|popup|")) return this.#log({ event: "server_message", line });
    if (line.startsWith("|updateuser|")) return this.#updateUser(connection, line);
    this.#challenge(event);
  }

  #updateUser(connection: Connection, line: string): void {
    const [identity = "", named] = line.slice("|updateuser|".length).split("|");
    if (connection.loggedIn || named !== "1" || toId(userName(identity)) !== toId(this.#options.name)) return;
    connection.loggedIn = true;
    this.#failures = 0;
    this.#send(`|/utm ${this.#packedTeam}`);
    this.#log({ event: "logged_in", name: userName(identity) });
    const { challenge, format } = this.#options;
    if (challenge !== undefined && !this.#challenged) {
      this.#challenged = true;
      this.#send(`|/challenge ${challenge}, ${format}`);
    }
  }

  #challenge(event: RoomEvent): void {
    const { engine, snapshot, name, challenger } = this.#options;
    const decision = decideChallenge(event, { engine, snapshot, botName: name, challenger });
    if (decision === null) return;
    const { from, format } = decision.challenge;
    if (decision.kind === "reject") {
      this.#send(decision.command);
      return this.#log({ event: "challenge_rejected", from, format, reason: decision.reason, detail: decision.detail });
    }
    const open = [...this.#battles.values()].filter((b) => b.result === "unfinished").length;
    // One accept at a time: the battle it makes shows only at `|init|battle`.
    const busy = open > 0 || this.#accepting || this.#ended >= this.#battlesWanted;
    if (this.#stopping || busy) {
      this.#send(`|/reject ${toId(from)}`);
      const [reason, detail] = this.#stopping
        ? (["stopping", "the bot is stopping"] as const)
        : (["busy", "the bot plays one battle at a time"] as const);
      return this.#log({ event: "challenge_rejected", from, format, reason, detail });
    }
    this.#send(decision.command);
    // A battle that starts after the timeout is still counted by `open`, so at
    // worst one more concurrent accept is possible, never a third.
    this.#accepting = true;
    this.#acceptTimer = setTimeout(
      () => this.#guard(() => this.#acceptEnded("timeout")),
      this.#options.acceptTimeoutMs ?? 10_000,
    );
    this.#log({ event: "challenge_accepted", from, format });
  }

  /** The accepted battle started, or will not: the next challenge may be accepted. */
  #acceptEnded(reason?: "timeout" | "disconnected"): void {
    clearTimeout(this.#acceptTimer);
    const was = this.#accepting;
    this.#accepting = false;
    if (was && reason !== undefined) this.#log({ event: "accept_abandoned", reason });
  }

  /** `|init|battle`: a new channel for the room on this connection; a new battle, or a rejoin. */
  #joined(connection: Connection, room: string): void {
    connection.rooms.set(room, { frames: [], channel: undefined, logged: 0 });
    this.#acceptEnded();
    const known = this.#battles.get(room);
    if (known !== undefined) {
      known.rejoin = true;
      return;
    }
    const battle: Battle = {
      room,
      ots: new OtsDecline({ room, botName: this.#options.name, send: (line) => this.#send(line) }),
      otsTimer: undefined,
      otsFailure: null,
      player: null,
      current: null,
      pending: undefined,
      watchdog: new DeadlineWatchdog({
        clock: this.#clock,
        onFire: (fire) => this.#guard(() => this.#watchdogFired(battle, fire)),
      }),
      submission: undefined,
      seen: new Map(),
      answered: new Map(),
      last: undefined,
      refused: [],
      rejoin: false,
      timer: false,
      result: "unfinished",
      winner: null,
      defect: false,
      autoChooses: [],
      decisions: [],
      channels: [],
    };
    this.#battles.set(room, battle);
    this.#log({ event: "battle_started", room });
  }

  /** Feeds the frame to the room's A2Channel, made once the first request names the bot's seat. */
  #receive(battle: Battle, state: RoomOnConnection, frame: string, events: RoomEvent[]): void {
    if (state.channel !== undefined) {
      state.channel.receive(frame);
      return;
    }
    state.frames.push(frame);
    const request = events.find((event) => event.kind === "request");
    if (request === undefined) return;
    const player = request.request.player as PlayerChannel["player"];
    battle.player = player;
    state.channel = new A2Channel({ room: battle.room, player, data_class: this.#options.data_class });
    for (const kept of state.frames) state.channel.receive(kept);
    state.frames = [];
  }

  /** This player's channel as built now; logs each dropped line once per channel. */
  #build(battle: Battle, channel: A2Channel, state: RoomOnConnection): PlayerChannel {
    const built = channel.build();
    for (const line of built.dropped.slice(state.logged)) this.#log({ event: "dropped", room: battle.room, line });
    state.logged = built.dropped.length;
    return built.channel;
  }

  #keepChannel(battle: Battle, state: RoomOnConnection): void {
    if (state.channel !== undefined) battle.channels.push(this.#build(battle, state.channel, state));
    state.channel = undefined;
  }

  #battleEvent(battle: Battle, state: RoomOnConnection, event: RoomEvent, rest: RoomEvent[], now: number): void {
    if (battle.result !== "unfinished") return;
    // The Team Preview request reaches the decline with its budget, once the snapshot is in.
    const heldForBudget = event.kind === "request" && event.request.phase === "preview" && battle.ots.status === "idle";
    if (!heldForBudget) {
      const awaiting = battle.ots.status === "awaiting";
      this.#ots(battle, () => battle.ots.onEvent(event, now));
      // While the rejection is unconfirmed no choice is out: the error answers the rejection.
      if (event.kind === "error" && awaiting) return;
    }
    switch (event.kind) {
      case "line":
        return this.#line(battle, state, event);
      case "request":
        return this.#request(battle, state, event.request, event.resend, rest, now);
      case "request_closed":
        battle.current = null;
        battle.pending = undefined;
        battle.watchdog.cancel();
        return;
      case "error":
        return this.#refusal(battle, event.error);
      case "sentchoice":
        // Read with the re-sent request it follows.
        return;
    }
  }

  #line(battle: Battle, state: RoomOnConnection, event: RoomEvent & { kind: "line" }): void {
    const { line } = event;
    const timer = timerOn(event);
    if (timer !== null) this.#send(timer);
    if (line.startsWith(TIMER_ON)) battle.timer = true;
    if (line.startsWith(TIMER_OFF)) battle.timer = false;
    const time = TIME_LEFT.exec(line);
    if (time !== null && battle.pending !== undefined) {
      const snapshot: TimerSnapshot = {
        timer_source: "server",
        turn_seconds_left: Number(time[1]),
        bank_seconds_left: Number(time[2]),
        grace_seconds_left: Number(time[3] ?? 0),
      };
      const pending = battle.pending;
      battle.pending = undefined;
      this.#decide(battle, pending, this.#deadline(battle, state, pending, snapshot));
    }
    if (line.startsWith("|win|") || line === "|tie") this.#end(battle, state, line);
  }

  /** The budget of a new request from the server's snapshot (ADR-005 §3). */
  #deadline(battle: Battle, state: RoomOnConnection, pending: Pending, timer: TimerSnapshot): Deadline {
    const config = this.#options.budget ?? A2_BUDGET_CONFIG;
    if (pending.request.phase === "preview") return { timer, budget: previewBudget(timer, config), inherited: false };
    const remaining = estimateRemainingDecisions(this.#counts(battle, state, pending.request), config);
    return {
      timer,
      budget: battleBudget(timer, remaining, config),
      inherited: false,
      remaining_decisions_estimate: remaining,
    };
  }

  /** `R̂`'s counts, from the bot's own request and the public `|teamsize|` and `|faint|` lines. */
  #counts(battle: Battle, state: RoomOnConnection, request: SequencedRequest) {
    const side = request.request.side as { pokemon: { condition: string; active?: boolean }[] };
    const alive = side.pokemon.filter((mon) => !mon.condition.endsWith(" fnt"));
    const channel = this.#build(battle, state.channel as A2Channel, state);
    const foe = request.player === "p1" ? "p2" : "p1";
    const teamsize = channel.lines.find((line) => line.startsWith(`|teamsize|${foe}|`));
    const faints = channel.lines.filter((line) => line.startsWith(`|faint|${foe}`)).length;
    // Without a usable `|teamsize|` the foe is taken to have brought as many as the bot did.
    const told = Number.parseInt(teamsize?.split("|")[3] ?? "", 10);
    const size = Number.isFinite(told) ? told : side.pokemon.length;
    return {
      own_alive: alive.length,
      own_bench: alive.filter((mon) => mon.active !== true).length,
      opp_alive: Math.max(0, size - faints),
    };
  }

  #request(
    battle: Battle,
    state: RoomOnConnection,
    request: SequencedRequest,
    resend: boolean,
    rest: RoomEvent[],
    now: number,
  ): void {
    const { rqid } = request;
    if (rqid === undefined) throw new SessionError(`${battle.room}: a request without an rqid is not from the server`);
    const sequenced = request as Pending["request"];
    const rejoin = battle.rejoin;
    battle.rejoin = false;
    const seen = battle.seen.get(rqid) === request.request_hash;
    battle.seen.set(rqid, request.request_hash);
    const channel = this.#build(battle, state.channel as A2Channel, state);
    const channel_request_seq = channel.lines.filter((line) => line.startsWith("|request|")).length;
    const legal =
      request.phase === "wait"
        ? null
        : legalChoiceSet({ request, showdown_commit: this.#options.engine.commit, judge: null });
    battle.current = { request_seq: request.request_seq, rqid, legal, phase: request.phase };
    const resent = resend || (rejoin && seen);
    if (!resent) this.#autoChoose(battle, request);
    if (resent) return this.#resent(battle, sequenced, legal, rest, channel_request_seq);
    battle.refused = [];
    battle.pending = undefined;
    if (legal === null) return;
    const pending: Pending = { request: sequenced, legal, channel_request_seq, received_at: now };
    if (request.request.update === true) {
      const last = battle.last ?? { received_at: now, deadline: { timer: null, budget: ZERO_BUDGET, inherited: false } };
      const inherited = { ...pending, received_at: last.received_at };
      return this.#decide(battle, inherited, { ...last.deadline, inherited: true });
    }
    if (battle.timer && !rejoin) {
      battle.pending = pending;
      battle.watchdog.armSnapshotWait(rqid, now, this.#options.snapshotWaitMs ?? DEFAULT_SNAPSHOT_WAIT_MS);
      return;
    }
    this.#decide(battle, pending, { timer: null, budget: ZERO_BUDGET, inherited: false });
  }

  /** A request the bot has seen before, re-sent on a rejoin (ADR-005 §1). */
  #resent(
    battle: Battle,
    request: Pending["request"],
    legal: LegalChoiceSet | null,
    rest: RoomEvent[],
    channel_request_seq: number,
  ): void {
    const { room } = battle;
    const { rqid, request_seq } = request;
    // The connection numbers its requests afresh: the open submission keeps up.
    if (battle.submission?.rqid === rqid) battle.submission.request_seq = request_seq;
    // The server sends the re-sent request and its `|sentchoice|` together (room-battle.ts `onConnect`).
    const held = rest.find((event) => event.kind === "sentchoice");
    if (held !== undefined) return this.#log({ event: "sentchoice", room, request_seq, choice: held.choice });
    const answered = battle.answered.get(rqid);
    if (answered !== undefined) {
      this.#submit(battle, request, answered);
      return this.#log({ event: "resent_choice", room, request_seq, choice: answered });
    }
    const pending = battle.pending;
    if (pending?.request.rqid === rqid) {
      // Its snapshot went to the old connection: decide now, from the original receipt.
      battle.pending = undefined;
      const refreshed = { ...pending, request, legal: legal as LegalChoiceSet, channel_request_seq };
      this.#decide(battle, refreshed, { timer: null, budget: ZERO_BUDGET, inherited: false });
    }
  }

  /** A newer request while the previous one is unanswered: the server chose for the bot (ADR-005 §3). */
  #autoChoose(battle: Battle, request: SequencedRequest): void {
    const detection = battle.watchdog.onRequest(request.rqid as number, request.request_seq, request.phase);
    if (detection === null) return;
    battle.autoChooses.push(detection);
    this.#log({ event: "server_auto_choose_detected", room: battle.room, detection });
  }

  /**
   * A deadline ran out for the request held for its snapshot: the fallback goes out at once.
   * The zero budget and null timer are right for `snapshot_missing` only, the one deadline
   * the session arms. When a policy arms `overrun`, its real snapshot and budget must be
   * passed here instead (the pending request will have to carry its `Deadline`).
   */
  #watchdogFired(battle: Battle, fire: WatchdogFire): void {
    const pending = battle.pending as Pending;
    battle.pending = undefined;
    this.#log({
      event: "watchdog",
      room: battle.room,
      rqid: fire.rqid,
      request_seq: pending.request.request_seq,
      reason: fire.reason,
    });
    this.#decide(battle, pending, { timer: null, budget: ZERO_BUDGET, inherited: false, watchdog: fire.reason });
  }

  #decide(battle: Battle, pending: Pending, deadline: Deadline): void {
    const { request, legal, channel_request_seq, received_at } = pending;
    battle.watchdog.settle(request.rqid);
    const choice = actorRule(legal);
    if (request.phase === "preview" && battle.ots.status === "idle") {
      const event: RoomEvent = { kind: "request", room: battle.room, request, resend: false };
      this.#ots(battle, () => battle.ots.onEvent(event, received_at, deadline.budget));
      this.#watchOts(battle, received_at + deadline.budget.policy_budget_ms);
    }
    const decision: DecisionRecord = {
      request_seq: request.request_seq,
      channel_request_seq,
      rqid: request.rqid,
      request_hash: request.request_hash,
      phase: request.phase as DecisionRecord["phase"],
      legal,
      choice,
      timer: deadline.timer,
      budget: deadline.budget,
      inherited: deadline.inherited,
      ...(deadline.watchdog === undefined ? {} : { watchdog: deadline.watchdog }),
      ...(deadline.remaining_decisions_estimate === undefined
        ? {}
        : { remaining_decisions_estimate: deadline.remaining_decisions_estimate }),
      elapsed_ms: this.#clock() - received_at,
    };
    battle.answered.set(request.rqid, choice);
    battle.watchdog.onAnswered(request.rqid);
    battle.last = { received_at, deadline };
    battle.decisions.push(decision);
    this.#submit(battle, request, choice);
    this.#log({ event: "decision", room: battle.room, decision });
  }

  #submit(battle: Battle, request: Pending["request"], choice: string): void {
    battle.submission = { request_seq: request.request_seq, choice, rqid: request.rqid };
    this.#choose(battle, request.phase, choice, request.rqid);
  }

  #choose(battle: Battle, phase: SequencedRequest["phase"], choice: string, rqid: number): void {
    const line = `${battle.room}|/choose ${choice}|${rqid}`;
    if (phase === "preview") this.#ots(battle, () => battle.ots.submitPreview(line));
    else this.#send(line);
  }

  /** The decline's own deadline: no frame may come before it runs out. */
  #watchOts(battle: Battle, deadline: number): void {
    const delay = Math.max(0, deadline - this.#clock()) + 1;
    battle.otsTimer = setTimeout(() => this.#guard(() => this.#ots(battle, () => battle.ots.tick(this.#clock()))), delay);
  }

  /** Runs a decline step; a failed decline has forfeited, and the battle plays out to its end. */
  #ots(battle: Battle, step: () => void): void {
    try {
      step();
    } catch (error) {
      if (!(error instanceof OtsDeclineError) || !error.forfeitSent) throw error;
      if (battle.otsFailure !== null) return;
      battle.otsFailure = { reason: error.reason, detail: error.detail };
      this.#log({ event: "ots_failure", room: battle.room, reason: error.reason, detail: error.detail });
    }
  }

  #refusal(battle: Battle, error: string): void {
    const submission = battle.submission;
    if (submission === undefined) throw new SessionError(`${battle.room}: |error| with no choice sent: ${error}`);
    const mode = this.#options.mode ?? "harness";
    const current = battle.current;
    const refusal = classifyRefusal({ error, submission, current, mode, refused: battle.refused });
    this.#log({ event: "refusal", room: battle.room, refusal });
    if (refusal.kind !== "defect") return;
    battle.defect = true;
    battle.refused.push(submission.choice);
    const next = refusal.next;
    if (next === undefined) return;
    // A defect is only ever classified against an open request.
    const open = current as NonNullable<Battle["current"]>;
    const rqid = open.rqid as number;
    battle.submission = { request_seq: open.request_seq, choice: next.choice, rqid };
    this.#choose(battle, open.phase, next.choice, rqid);
  }

  #end(battle: Battle, state: RoomOnConnection, line: string): void {
    battle.winner = line === "|tie" ? null : line.slice("|win|".length);
    battle.result =
      battle.winner === null ? "tie" : toId(battle.winner) === toId(this.#options.name) ? "win" : "loss";
    clearTimeout(battle.otsTimer);
    battle.watchdog.dispose();
    this.#keepChannel(battle, state);
    this.#send(`${battle.room}|/leave`);
    this.#log({ event: "battle_ended", room: battle.room, result: battle.result, winner: battle.winner });
    this.#options.onBattle?.(this.#record(battle));
    this.#ended += 1;
    if (this.#ended >= this.#battlesWanted) this.#stop();
  }
}
