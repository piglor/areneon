import { decodeRequest, playerChannel, RequestDecodeError, stripServerLines } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { GLOBAL_ROOM, ProtocolReader, type RoomEvent, splitFrame } from "../src/index.ts";
import { battle, frameIndex, sentAfter, serverFrame } from "./frames.ts";

const ROOM = battle.room;
const requests = (events: RoomEvent[]) =>
  events.flatMap((e) => (e.kind === "request" ? [e] : []));
/** Reads `frames` into one reader and returns every event. */
const readAll = (reader: ProtocolReader, frames: readonly string[]): RoomEvent[] =>
  frames.flatMap((frame) => reader.read(frame));

describe("splitFrame (PROTOCOL.md framing)", () => {
  it("takes the room from the >ROOMID header and drops empty lines", () => {
    expect(splitFrame(`>${ROOM}\n|turn|1\n\n|upkeep\n`)).toEqual({ room: ROOM, lines: ["|turn|1", "|upkeep"] });
  });

  it("puts a frame without a header in the global room", () => {
    expect(splitFrame(serverFrame("global_updateuser")).room).toBe(GLOBAL_ROOM);
    expect(splitFrame("")).toEqual({ room: GLOBAL_ROOM, lines: [] });
  });
});

describe("ProtocolReader on the recorded battle", () => {
  it("counts every |request|, wait included, from 1 with no gap", () => {
    const events = requests(readAll(new ProtocolReader(), battle.frames));
    const bodies = battle.frames.flatMap((f) => f.split("\n").filter((l) => l.startsWith("|request|")));
    expect(events.map((e) => e.request.request_seq)).toEqual(bodies.map((_, i) => i + 1));
    expect(events.some((e) => e.request.phase === "wait")).toBe(true);
    expect(events.every((e) => !e.resend && e.room === ROOM)).toBe(true);
  });

  it("decodes each request through the pinned bridge: rqid kept, never hashed", () => {
    const events = requests(readAll(new ProtocolReader(), battle.frames));
    const bodies = battle.frames.flatMap((f) => f.split("\n").filter((l) => l.startsWith("|request|")));
    events.forEach((e, i) => {
      const body = (bodies[i] as string).slice("|request|".length);
      expect(e.request).toEqual({ ...decodeRequest(body), request_seq: i + 1 });
      expect(typeof e.request.rqid).toBe("number");
      const { rqid, ...withoutRqid } = JSON.parse(body) as { rqid: number };
      expect(e.request.request_hash).toBe(decodeRequest(JSON.stringify(withoutRqid)).request_hash);
    });
  });

  it("gives the updated request after [Unavailable choice] a new request_seq", () => {
    const reader = new ProtocolReader();
    const unavailable = frameIndex("|error|[Unavailable choice]");
    const before = requests(readAll(reader, battle.frames.slice(0, unavailable))).at(-1);
    const [error] = reader.read(battle.frames[unavailable] as string);
    expect(error).toMatchObject({ kind: "error", room: ROOM });
    expect(error?.kind === "error" && error.error).toMatch(/^\[Unavailable choice\] Can't switch/);
    const [after] = requests(reader.read(battle.frames[unavailable + 1] as string));
    expect(after?.request.request.update).toBe(true);
    expect(after?.resend).toBe(false);
    expect(after?.request.request_seq).toBe((before?.request.request_seq ?? 0) + 1);
  });

  it("reports other lines and errors unparsed", () => {
    const events = new ProtocolReader().read(battle.frames[frameIndex("|error|[Invalid choice]")] as string);
    expect(events).toEqual([
      { kind: "error", room: ROOM, error: "[Invalid choice] Can't move: Your Pikachu doesn't have a move 9" },
    ]);
    const turn = new ProtocolReader().read(battle.frames[frameIndex("|turn|")] as string);
    expect(turn).toContainEqual({ kind: "line", room: ROOM, line: "|turn|1" });
  });
});

describe("request identity across a rejoin", () => {
  // The re-send repeats the first move request p1 answered.
  const answered = frameIndex('|request|{"active"');

  it("does not count |request|null, nor decode it", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    expect(reader.read(serverFrame("request_null"))).toEqual([{ kind: "request_closed", room: ROOM }]);
    expect(reader.state(ROOM).open).toBe(false);
    // The body would fail decodeRequest.
    expect(() => decodeRequest("null")).toThrow();
    expect(reader.state(ROOM).last?.request_seq).toBe(2);
  });

  it("fails closed on an empty |request|, which the server never sends", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    expect(() => reader.read(`>${ROOM}\n|request|`)).toThrow(SyntaxError);
    expect(reader.state(ROOM)).toMatchObject({ open: true, last: { request_seq: 2 } });
  });

  it("keeps the request_seq of a re-sent request (same rqid and request_hash) and reports |sentchoice|", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    reader.read(serverFrame("request_null"));
    const events = reader.read(serverFrame("resend_with_sentchoice"));
    expect(events[0]).toMatchObject({ kind: "request", resend: true, request: { request_seq: 2, rqid: 3 } });
    expect(events[1]).toEqual({ kind: "sentchoice", room: ROOM, choice: sentAfter(answered) });
    expect(reader.state(ROOM).open).toBe(true);
    // A second re-send (no choice held yet) is still the same request.
    expect(requests(reader.read(serverFrame("resend_without_sentchoice")))[0]?.resend).toBe(true);
    // The battle goes on: the next request is the next one.
    const next = requests(readAll(reader, battle.frames.slice(answered + 1)))[0];
    expect(next?.request.request_seq).toBe(3);
  });

  it("counts a request with the same hash but another rqid, or no rqid, as new", () => {
    const resend = serverFrame("resend_without_sentchoice");
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    const newRqid = requests(reader.read(resend.replace('"rqid":3}', '"rqid":4}')))[0];
    expect(newRqid).toMatchObject({ resend: false, request: { request_seq: 3, rqid: 4 } });
    const sim = new ProtocolReader();
    const noRqid = resend.replace(',"rqid":3}', "}");
    sim.read(noRqid);
    expect(requests(sim.read(noRqid))[0]).toMatchObject({ resend: false, request: { request_seq: 2 } });
  });

  it("refuses a request the pinned engine would not send, without counting it", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    const odd = serverFrame("resend_without_sentchoice").replace('"rqid":3}', '"rqid":3,"extra":1}');
    expect(() => reader.read(odd)).toThrow(RequestDecodeError);
    expect(reader.state(ROOM).last?.request_seq).toBe(2);
  });

  it("reads a frame all or nothing: a bad request leaves no partial state change", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    reader.read(serverFrame("request_null"));
    const before = reader.state(ROOM);
    const good = serverFrame("resend_without_sentchoice");
    const bad = good.replace('"rqid":3}', '"rqid":3,"extra":1}').replace(`>${ROOM}\n`, "");
    // A request that would count, then one that fails to decode.
    const next = (battle.frames[answered + 2] as string).split("\n")[1] as string;
    expect(() => reader.read(`>${ROOM}\n${next}\n${bad}`)).toThrow(
      RequestDecodeError,
    );
    expect(reader.state(ROOM)).toEqual(before);
    expect(before.open).toBe(false);
  });

  it("hands out a copy of a room's state", () => {
    const reader = new ProtocolReader();
    readAll(reader, battle.frames.slice(0, answered + 1));
    const copy = reader.state(ROOM) as { open: boolean };
    copy.open = false;
    expect(reader.state(ROOM).open).toBe(true);
  });
});

describe("request identity across a new connection (|init|battle)", () => {
  const answered = frameIndex('|request|{"active"');
  const before = battle.frames.slice(0, answered + 1);
  /**
   * The join frame the pinned server sends a rejoining player
   * (server/rooms.ts `onConnect`: `|init|battle`, `|title|`, then the room
   * log). The room log holds no `|request|` or `|error|`: those go to the
   * player only (server/room-battle.ts `receive`).
   */
  const init = [
    `>${ROOM}`,
    "|init|battle",
    "|title|p1 vs. p2",
    ...before.flatMap((frame) =>
      splitFrame(frame).lines.filter((line) => !line.startsWith("|request|") && !line.startsWith("|error|")),
    ),
  ].join("\n");
  /**
   * The request_seq `buildObservation` gives the last request of the
   * channel made from `frames`, the new connection's frames: the count of
   * `|request|` lines in it, up to and including the last.
   */
  const channelSeq = (frames: readonly string[]): number => {
    const { lines } = stripServerLines(frames.flatMap((frame) => splitFrame(frame).lines));
    const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines });
    const at = channel.lines.findLastIndex((line) => line.startsWith("|request|"));
    return channel.lines.slice(0, at + 1).filter((line) => line.startsWith("|request|")).length;
  };

  it.each(["resend_with_sentchoice", "resend_without_sentchoice"])(
    "restarts request_seq at 1 for the request re-sent after a rejoin (%s)",
    (name) => {
      const reader = new ProtocolReader();
      readAll(reader, before);
      expect(reader.state(ROOM)).toMatchObject({ open: true, last: { request_seq: 2, rqid: 3 } });
      expect(reader.read(init)).toContainEqual({ kind: "line", room: ROOM, line: "|init|battle" });
      expect(reader.state(ROOM)).toEqual({ last: undefined, open: false });
      const events = reader.read(serverFrame(name));
      // Same rqid and request_hash as before the rejoin, but a new channel.
      expect(events[0]).toMatchObject({ kind: "request", resend: false, request: { request_seq: 1, rqid: 3 } });
      const seq = events[0]?.kind === "request" ? events[0].request.request_seq : undefined;
      expect(seq).toBe(channelSeq([init, serverFrame(name)]));
      if (name === "resend_with_sentchoice") {
        expect(events[1]).toEqual({ kind: "sentchoice", room: ROOM, choice: sentAfter(answered) });
      } else {
        expect(events).toHaveLength(1);
      }
      expect(reader.state(ROOM).open).toBe(true);
    },
  );

  it("keeps re-send detection within the new connection", () => {
    const reader = new ProtocolReader();
    readAll(reader, before);
    reader.read(serverFrame("request_null"));
    reader.read(init);
    const after = [serverFrame("resend_with_sentchoice"), serverFrame("resend_without_sentchoice")];
    const [first, again] = requests(readAll(reader, after));
    expect(first).toMatchObject({ resend: false, request: { request_seq: 1 } });
    expect(again).toMatchObject({ resend: true, request: { request_seq: 1 } });
  });

  it("counts on from the re-sent request as the new channel does", () => {
    const reader = new ProtocolReader();
    readAll(reader, before);
    reader.read(init);
    const resent = serverFrame("resend_with_sentchoice");
    reader.read(resent);
    // The battle goes on: the next request is the new channel's second.
    const rest = battle.frames.slice(answered + 1);
    const upTo = rest.slice(0, rest.findIndex((frame) => frame.includes("\n|request|")) + 1);
    const [next] = requests(readAll(reader, upTo));
    expect(next).toMatchObject({ resend: false, request: { request_seq: 2 } });
    expect(next?.request.request_seq).toBe(channelSeq([init, resent, ...upTo]));
  });

  it("resets only the room of the |init|battle", () => {
    const other = "battle-gen9championsvgc2026regmc-2";
    const reader = new ProtocolReader();
    readAll(reader, before);
    readAll(reader, before.map((frame) => frame.replace(`>${ROOM}\n`, `>${other}\n`)));
    reader.read(init);
    expect(reader.state(ROOM)).toEqual({ last: undefined, open: false });
    expect(reader.state(other)).toMatchObject({ open: true, last: { request_seq: 2 } });
    // A chat room's |init| is not a battle's.
    reader.read(`>${other}\n|init|chat`);
    expect(reader.state(other).last?.request_seq).toBe(2);
  });

  it("resets nothing when the frame holding |init|battle fails to decode", () => {
    const reader = new ProtocolReader();
    readAll(reader, before);
    const stateBefore = reader.state(ROOM);
    const bad = splitFrame(serverFrame("resend_without_sentchoice")).lines[0]?.replace(
      '"rqid":3}',
      '"rqid":3,"extra":1}',
    );
    expect(() => reader.read(`>${ROOM}\n|init|battle\n${bad}`)).toThrow(RequestDecodeError);
    expect(reader.state(ROOM)).toEqual(stateBefore);
  });
});

describe("room routing", () => {
  it("tags every event with its room and keeps a request_seq per room", () => {
    const other = "battle-gen9championsvgc2026regmc-2";
    const preview = battle.frames[frameIndex("|request|")] as string;
    const reader = new ProtocolReader();
    const events = readAll(reader, [
      serverFrame("global_updateuser"),
      preview,
      serverFrame("other_room"),
      preview.replace(`>${ROOM}\n`, `>${other}\n`),
      preview.replace(`>${ROOM}\n`, `>${other}\n`).replace('"rqid":1}', '"rqid":2}'),
    ]);
    expect(events.map((e) => [e.kind, e.room])).toEqual([
      ["line", GLOBAL_ROOM],
      ["request", ROOM],
      ["line", "lobby"],
      ["request", other],
      ["request", other],
    ]);
    expect(events[0]).toEqual({ kind: "line", room: GLOBAL_ROOM, line: serverFrame("global_updateuser") });
    expect(reader.state(ROOM).last?.request_seq).toBe(1);
    expect(reader.state(other).last?.request_seq).toBe(2);
    expect(reader.state("lobby")).toEqual({ last: undefined, open: false });
  });
});
