import { type DecodedRequest, decodeRequest } from "@areneon/showdown-bridge";

/**
 * The room of a frame without a `>ROOMID` header: PROTOCOL.md lets the
 * server omit the header for the lobby or the global room.
 */
export const GLOBAL_ROOM = "";

/** A decoded request with Areneon's per-side counter (ADR-005 §1, ADR-008 §4). */
export interface SequencedRequest extends DecodedRequest {
  /** 1-based count of the requests this player has received in this room, `wait` included. */
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
       * `request_hash`, e.g. on rejoin): it keeps its `request_seq`.
       */
      resend: boolean;
    }
  /** `|request|null` or an empty `|request|`: the open request is withdrawn, not a request. */
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
 * - `|request|null` (sent when a player leaves) and an empty `|request|` are
 *   not requests: they do not count and never reach `decodeRequest`.
 * - A request whose `rqid` and `request_hash` both equal the previous
 *   request's is a re-send and keeps its `request_seq`. A request without
 *   an `rqid` is never a re-send: only the server assigns one.
 *
 * The counter is per room: the bot is one player in each battle room.
 */
export class ProtocolReader {
  readonly #rooms = new Map<string, { last: SequencedRequest | undefined; open: boolean }>();

  /** The request identity of `room`, as read so far. */
  state(room: string): RoomRequestState {
    return this.#rooms.get(room) ?? { last: undefined, open: false };
  }

  /** Reads one frame. Throws `RequestDecodeError` for a request the pinned engine would not send. */
  read(frame: string): RoomEvent[] {
    const { room, lines } = splitFrame(frame);
    return lines.map((line) => this.#event(room, line));
  }

  #event(room: string, line: string): RoomEvent {
    const body = prefixed(line, "request");
    if (body !== undefined) return this.#request(room, body);
    const choice = prefixed(line, "sentchoice");
    if (choice !== undefined) return { kind: "sentchoice", room, choice };
    const error = prefixed(line, "error");
    if (error !== undefined) return { kind: "error", room, error };
    return { kind: "line", room, line };
  }

  #request(room: string, body: string): RoomEvent {
    const state = this.#rooms.get(room) ?? { last: undefined, open: false };
    this.#rooms.set(room, state);
    if (body === "" || body === "null") {
      state.open = false;
      return { kind: "request_closed", room };
    }
    const decoded = decodeRequest(body);
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
