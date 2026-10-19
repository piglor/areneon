import { type PlayerChannel, playerChannel, stripServerLines } from "@areneon/showdown-bridge";
import { splitFrame } from "./protocol.ts";

/** A battle room stream that cannot become a PlayerChannel. */
export class A2ChannelError extends Error {
  override name = "A2ChannelError";
}

export interface A2ChannelOptions {
  /** The battle room id, as in its frames' `>ROOMID` header. */
  room: string;
  /** This player's seat in the room. */
  player: PlayerChannel["player"];
  /** `private` for a match against a human (ADR-008 §5). */
  data_class: PlayerChannel["data_class"];
}

/** A built channel and the lines of unknown kind the filter dropped from it. */
export interface A2ChannelBuild {
  channel: PlayerChannel;
  /**
   * Every line of the current channel so far whose kind neither the sim nor
   * the server list knows, for the caller to log. Built from the start each
   * time, so it repeats the lines an earlier build reported.
   */
  dropped: string[];
  /**
   * How many re-sent copies of the current request the current channel left
   * out (see `A2Channel`). A count for the caller to log; it restarts at 0
   * with every `|init|`.
   */
  resent: number;
}

const INIT = "|init|";
const BATTLE_INIT = "|init|battle";
const NOINIT = "|noinit|";
const REQUEST = "|request|";

/** One channel's state: it changes only when a whole frame is accepted. */
interface ChannelState {
  /** The raw lines since the last `|init|`. */
  history: string[];
  /** The rqid and the body without it of the last request kept, if it had an rqid. */
  last: { rqid: number; body: string } | undefined;
  /** The re-sent requests left out since the last `|init|`. */
  resent: number;
}

/** A request line's rqid and its body without the rqid, or `undefined` if it has no rqid. */
function requestIdentity(line: string): ChannelState["last"] {
  let request: unknown;
  try {
    request = JSON.parse(line.slice(REQUEST.length));
  } catch {
    return undefined; // Kept as it is: `playerChannel` refuses it on build.
  }
  if (typeof request !== "object" || request === null || !("rqid" in request)) return undefined;
  const { rqid, ...rest } = request as { rqid: unknown };
  return typeof rqid === "number" ? { rqid, body: JSON.stringify(rest) } : undefined;
}

/**
 * The A2 PlayerChannel adapter (ADR-005 §5, ADR-008 §5): one battle room's
 * stream, as this player's connection receives it, made into this player's
 * PlayerChannel.
 *
 * It takes the same frames the session loop gives `ProtocolReader.read`,
 * keeps the lines of its own room (the `>ROOMID` header routes them; frames
 * of other rooms and of the global room are ignored) and, on each `build`,
 * filters the whole history with showdown-bridge's `stripServerLines`. That
 * filter decides some lines by their place in the stream, so it always runs
 * from the channel's first line, never on a suffix. Chat, timer, join and
 * other server lines never reach the channel; `|request|` and
 * `|sentchoice|` handling (request identity, re-sends) stays with the
 * protocol reader. The result is branded by `playerChannel`, which refuses
 * an opponent's SECRET line, a `|split|` marker or another player's request.
 *
 * A channel starts at the room's `|init|battle` (ADR-004's channel rule).
 * The server sends `|init|` plus the room log whenever a connection joins
 * the room: at the start, and again on a rejoin or on a new connection. The
 * re-sent log holds no earlier `|request|` or `|error|` lines, because the
 * server sends those to the player alone and never adds them to the log
 * (server/room-battle.ts `receive`, `sideupdate`). So a stream that went
 * on across the second `|init|` would not be this player's channel: every
 * `|init|` starts a new one and the history before it is discarded, and the
 * request count of an observation built from it restarts at 1. Across
 * channels a request is matched by its `request_hash` and `rqid` (ADR-004,
 * ADR-008 §4). Use one adapter per room and connection.
 *
 * Within one channel the server can also send the current request again
 * without a new `|init|`: on a login, `onUpdateConnection` runs the game's
 * `onConnect` a second time (server/users.ts, server/room-game.ts), and
 * `onConnect` re-sends the player's stored request, with its `|sentchoice|`
 * if any (server/room-battle.ts `onConnect`). The server gives every new
 * request the next battle-wide `rqid` (`sideupdate`) and re-sends the stored
 * JSON as it is. So a `|request|` whose `rqid` and body without `rqid` both
 * equal the last kept request's is a re-send: it is left out of the history
 * and counted in `resent`, and the channel's request count matches the
 * protocol reader's `request_seq`, which a re-send keeps. A request without
 * an `rqid` is never a re-send. After an `|init|` there is no last request,
 * so the re-sent one is kept as the new channel's first.
 *
 * A frame is taken all or nothing: it is checked as a whole against a copy
 * of the channel's state, and a frame that throws leaves the state as it
 * was, even after an `|init|battle` earlier in that frame.
 */
export class A2Channel {
  readonly #options: A2ChannelOptions;
  /** The current channel, or `undefined` before the first `|init|`. */
  #state: ChannelState | undefined;

  constructor(options: A2ChannelOptions) {
    this.#options = { ...options };
  }

  /**
   * Records `frame` if it belongs to this room and returns whether it did.
   * Throws `A2ChannelError` for a `|noinit|` (the server refused the join or
   * renamed the room: this room id sends no more of the channel), a line of
   * this room before its first `|init|`, or a room that is not a battle. A
   * frame that throws changes nothing.
   */
  receive(frame: string): boolean {
    const { room, lines } = splitFrame(frame);
    if (room !== this.#options.room) return false;
    let state = this.#state && { ...this.#state, history: [...this.#state.history] };
    for (const line of lines) {
      if (line.startsWith(NOINIT)) throw new A2ChannelError(`${room}: room refused: ${line}`);
      if (line.startsWith(INIT)) {
        if (line !== BATTLE_INIT) throw new A2ChannelError(`${room} is not a battle room: ${line}`);
        state = { history: [], last: undefined, resent: 0 };
      }
      if (state === undefined) throw new A2ChannelError(`${room} sent a line before |init|: ${line}`);
      if (line.startsWith(REQUEST)) {
        const identity = requestIdentity(line);
        if (identity !== undefined && identity.rqid === state.last?.rqid && identity.body === state.last.body) {
          state.resent += 1;
          continue;
        }
        if (identity !== undefined) state.last = identity;
      }
      state.history.push(line);
    }
    this.#state = state;
    return true;
  }

  /**
   * This player's channel as received so far. Throws `A2ChannelError` before
   * the room's `|init|`, and `ObservationError` (from `playerChannel`) if
   * the filtered stream is not this player's own.
   */
  build(): A2ChannelBuild {
    if (this.#state === undefined) throw new A2ChannelError(`${this.#options.room} has not sent |init|`);
    const { lines, dropped } = stripServerLines(this.#state.history);
    const { player, data_class } = this.#options;
    return { channel: playerChannel({ schema_version: 1, data_class, player, lines }), dropped, resent: this.#state.resent };
  }
}
