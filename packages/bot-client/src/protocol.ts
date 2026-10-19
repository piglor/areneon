import { type DecodedRequest, decodeRequest } from "@areneon/showdown-bridge";

/**
 * The room of a frame without a `>ROOMID` header: PROTOCOL.md lets the
 * server omit the header for the lobby or the global room.
 */
export const GLOBAL_ROOM = "";

/** A decoded request with Areneon's per-side counter (ADR-005 §1, ADR-008 §4). */
export interface SequencedRequest extends DecodedRequest {
  /**
   * 1-based count of the requests this player has received in this room
   * since the room's last `|init|battle`, `wait` included.
   */
  request_seq: number;
}

/** One event from a server line, tagged with the room it arrived in. */
export type RoomEvent =
  | {
      kind: "request";
      room: string;
      request: SequencedRequest;
      /**
       * The server re-sent the current request (same `rqid` and same
       * `request_hash`) on the same connection: it keeps its `request_seq`.
       * After an `|init|battle` the re-sent request is a new channel's first.
       */
      resend: boolean;
    }
  /** `|request|null`: the open request is withdrawn; it is not a request. */
  | { kind: "request_closed"; room: string }
  /** The choice the server holds for the re-sent request. */
  | { kind: "sentchoice"; room: string; choice: string }
  /** An `|error|` line: classify it with `classifyRefusal` against the submission. */
  | { kind: "error"; room: string; error: string }
  /** Any other line, unparsed. */
  | { kind: "line"; room: string; line: string };

/** The reader's view of one room's request identity. */
export interface RoomRequestState {
  /** The last request received, or `undefined` before the first. */
  readonly last: SequencedRequest | undefined;
  /** `false` after `|request|null` until the next request. */
  readonly open: boolean;
}

/** A frame split into its room and its non-empty lines (PROTOCOL.md). */
export interface Frame {
  room: string;
  lines: string[];
}

/** PROTOCOL.md: `>ROOMID` on the first line, then messages; empty lines are ignored. */
export function splitFrame(frame: string): Frame {
  const lines = frame.split("\n");
  const room = lines[0]?.startsWith(">") === true ? (lines.shift() as string).slice(1) : GLOBAL_ROOM;
  return { room, lines: lines.filter((line) => line !== "") };
}

const prefixed = (line: string, type: string): string | undefined =>
  line.startsWith(`|${type}|`) ? line.slice(type.length + 2) : undefined;

/**
 * Reads server frames into room events and keeps each room's request
 * identity (ADR-005 §1). It is pure: it never sends and never connects.
 *
 * - Every `|request|` counts, `wait` included, and goes through the
 *   pinned bridge's `decodeRequest`, which validates it before hashing.
 * - `|request|null` (sent when a player leaves, server/room-battle.ts:1021)
 *   is not a request: it does not count and never reaches `decodeRequest`.
 *   The server never sends an empty `|request|`, so one fails to decode.
 * - A request whose `rqid` and `request_hash` both equal the previous
 *   request's is a re-send and keeps its `request_seq`. A request without
 *   an `rqid` is never a re-send: only the server assigns one.
 * - `|init|battle` starts a new channel for its room (ADR-004 r4.2, ADR-008
 *   §4): the server sends it on every join, then replays the room log, which
 *   holds no `|request|` (those go to the player only, server/room-battle.ts
 *   `receive`). So the room's request state is reset there: `request_seq`
 *   restarts at 1, the room has no open request, and re-send detection
 *   forgets the last identity. The request the server re-sends after a
 *   rejoin (`onConnect`, with `|sentchoice|` if a choice is held) is then
 *   request 1 of the new channel, as `buildObservation` numbers it; across
 *   channels a request is identified by `request_hash` + `rqid`.
 *
 * The counter is per room: the bot is one player in each battle room. A
 * frame is read all or nothing: its requests are decoded before any room
 * state changes, so a request that fails to decode leaves the state as it was.
 */
export class ProtocolReader {
  readonly #rooms = new Map<string, { last: SequencedRequest | undefined; open: boolean }>();

  /** A copy of the request identity of `room`, as read so far. */
  state(room: string): RoomRequestState {
    const state = this.#rooms.get(room);
    return state === undefined ? { last: undefined, open: false } : { ...state };
  }

  /**
   * Reads one frame. Throws `RequestDecodeError` (or a JSON `SyntaxError`)
   * for a request the pinned engine would not send, before changing any state.
   */
  read(frame: string): RoomEvent[] {
    const { room, lines } = splitFrame(frame);
    const decoded = lines.map((line) => {
      const body = prefixed(line, "request");
      return body === undefined || body === "null" ? undefined : decodeRequest(body);
    });
    return lines.map((line, i) => this.#event(room, line, decoded[i]));
  }

  #event(room: string, line: string, decoded: DecodedRequest | undefined): RoomEvent {
    if (line === "|init|battle") this.#rooms.delete(room);
    if (line.startsWith("|request|")) return this.#request(room, decoded);
    const choice = prefixed(line, "sentchoice");
    if (choice !== undefined) return { kind: "sentchoice", room, choice };
    const error = prefixed(line, "error");
    if (error !== undefined) return { kind: "error", room, error };
    return { kind: "line", room, line };
  }

  /** `decoded` is `undefined` for `|request|null`. */
  #request(room: string, decoded: DecodedRequest | undefined): RoomEvent {
    const state = this.#rooms.get(room) ?? { last: undefined, open: false };
    this.#rooms.set(room, state);
    if (decoded === undefined) {
      state.open = false;
      return { kind: "request_closed", room };
    }
    const last = state.last;
    const resend =
      last !== undefined &&
      decoded.rqid !== undefined &&
      decoded.rqid === last.rqid &&
      decoded.request_hash === last.request_hash;
    const request_seq = (last?.request_seq ?? 0) + (resend ? 0 : 1);
    state.last = { ...decoded, request_seq };
    state.open = true;
    return { kind: "request", room, request: state.last, resend };
  }
}
