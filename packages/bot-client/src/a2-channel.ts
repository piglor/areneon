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
}

const INIT = "|init|";
const BATTLE_INIT = "|init|battle";

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
 */
export class A2Channel {
  readonly #options: A2ChannelOptions;
  /** The raw lines since the last `|init|`, or `undefined` before the first. */
  #history: string[] | undefined;

  constructor(options: A2ChannelOptions) {
    this.#options = { ...options };
  }

  /**
   * Records `frame` if it belongs to this room and returns whether it did.
   * Throws `A2ChannelError` for a line of this room before its first
   * `|init|`, or a room that is not a battle.
   */
  receive(frame: string): boolean {
    const { room, lines } = splitFrame(frame);
    if (room !== this.#options.room) return false;
    for (const line of lines) {
      if (line.startsWith(INIT)) {
        if (line !== BATTLE_INIT) throw new A2ChannelError(`${room} is not a battle room: ${line}`);
        this.#history = [];
      }
      if (this.#history === undefined) throw new A2ChannelError(`${room} sent a line before |init|: ${line}`);
      this.#history.push(line);
    }
    return true;
  }

  /**
   * This player's channel as received so far. Throws `A2ChannelError` before
   * the room's `|init|`, and `ObservationError` (from `playerChannel`) if
   * the filtered stream is not this player's own.
   */
  build(): A2ChannelBuild {
    if (this.#history === undefined) throw new A2ChannelError(`${this.#options.room} has not sent |init|`);
    const { lines, dropped } = stripServerLines(this.#history);
    const { player, data_class } = this.#options;
    return { channel: playerChannel({ schema_version: 1, data_class, player, lines }), dropped };
  }
}
