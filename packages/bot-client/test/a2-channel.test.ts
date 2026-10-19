import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ObservationError, playerChannel, stripServerLines } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { A2Channel, A2ChannelError, ProtocolReader } from "../src/index.ts";
import { battle } from "./frames.ts";

type Player = "p1" | "p2";

const ROOM = battle.room;
const GOLDEN = (player: Player): string[] =>
  (
    JSON.parse(
      readFileSync(
        fileURLToPath(
          new URL(`../../../fixtures/showdown/gen9championsvgc2026regmc/channels/${player}.json`, import.meta.url),
        ),
        "utf8",
      ),
    ) as { channel: { lines: string[] } }
  ).channel.lines;

const frame = (lines: readonly string[], room = ROOM): string => `>${room}\n${lines.join("\n")}`;

/**
 * The frame the server sends when p1's connection joins the room: `|init|battle`,
 * the title and the room log so far (server/rooms.ts `GameRoom.onConnect`).
 * `Rooms.createBattle` builds the RoomBattle and joins the players one by one,
 * before the sim's first update (`|gametype|` on) reaches the room. Its
 * players are connected, so `RoomBattle.onJoin` adds no `|player|` line, and
 * a battle room logs no `|t:|` (`noLogTimes`). So p1's log holds its own
 * join; p2's follows in the next frame.
 */
const INIT = frame(["|init|battle", "|title|p1 vs. p2", "|j|☆p1"]);
const P2_JOIN = frame(["|j|☆p2"]);

/** `RoomBattle.checkPrivacySettings` logs this before the players join when one asks for an invite-only battle. */
const INVITE_ONLY =
  "|raw|<div class=\"broadcast-red\"><strong>This battle is invite-only!</strong><br />Users must be invited with <code>/invite</code> (or be staff) to join</div>";

const isRequest = (line: string): boolean => line.startsWith("|request|");

/** A request line as the server sends it: the battle-wide `rqid` appended (server/room-battle.ts `receive`). */
const withRqid = (line: string, rqid: number): string =>
  `|request|${JSON.stringify({ ...(JSON.parse(line.slice("|request|".length)) as object), rqid })}`;

/** A request line without its `rqid`, which ADR-008 §4 leaves out of every comparison. */
const withoutRqid = (line: string): string => {
  if (!isRequest(line)) return line;
  const { rqid, ...rest } = JSON.parse(line.slice("|request|".length)) as { rqid?: number };
  return `|request|${JSON.stringify(rest)}`;
};

/**
 * A golden A1 channel as the server would send it to that player: the init
 * frame, the sim's lines in update frames with timer, wall-clock and chat
 * lines among them, and each request in a frame of its own with an `rqid`.
 */
function serverFramesOf(lines: readonly string[]): string[] {
  const frames = [INIT, P2_JOIN];
  let update: string[] = [];
  let rqid = 0;
  const flush = (): void => {
    if (update.length > 0) frames.push(frame(update));
    update = [];
  };
  for (const line of lines) {
    if (isRequest(line)) {
      flush();
      rqid += 2;
      frames.push(frame([withRqid(line, rqid)]));
      continue;
    }
    update.push(line);
    if (line.startsWith("|turn|")) {
      update.push("|inactive|Time left: 140 sec this turn | 120 sec total", "|t:|1760000001", "|c|☆p2|gl hf");
    }
  }
  flush();
  return frames;
}

const channelFrom = (frames: readonly string[], player: Player = "p1"): A2Channel => {
  const channel = new A2Channel({ room: ROOM, player, data_class: "synthetic" });
  for (const f of frames) channel.receive(f);
  return channel;
};

describe("A2Channel", () => {
  it.each(["p1", "p2"] as const)("returns the golden A1 %s channel from its server frames", (player) => {
    const golden = GOLDEN(player);
    const frames = serverFramesOf(golden);
    // Frames of other rooms and of the global room are not this channel's.
    frames.splice(2, 0, frame(["|c|Bob|hi"], "lobby"), '|updateuser| Areneon-bot-0000|0|1|{"blockChallenges":false}');
    const { channel, dropped } = channelFrom(frames, player).build();
    expect(channel).toMatchObject({ schema_version: 1, data_class: "synthetic", player });
    expect(channel.lines.some((line) => isRequest(line) && line.includes('"rqid":'))).toBe(true);
    expect(channel.lines.map(withoutRqid)).toEqual(golden);
    expect(dropped).toEqual([]);
  });

  it("keeps the server's |raw| and |player| lines around the sim's own out of the channel", () => {
    const golden = GOLDEN("p1");
    const start = golden.indexOf("|start");
    const frames = serverFramesOf(golden);
    // An invite-only battle logs its notice before the players join; p2's
    // connection leaves and comes back during Team Preview, after the sim
    // named p2 (server/room-battle.ts `onLeave`, `onJoin`).
    frames[0] = frame(["|init|battle", "|title|p1 vs. p2", INVITE_ONLY, "|j|☆p1"]);
    const preview = frames.findIndex((f) => f.includes("|teampreview|"));
    frames.splice(preview + 1, 0, frame(["|l|☆p2", "|player|p2|"]), frame(["|j|☆p2", "|player|p2|p2|2|"]));
    const raw = frames.flatMap((f) => f.split("\n"));
    expect(raw.filter((line) => line.startsWith("|player|"))).toHaveLength(4);
    expect(raw.indexOf("|player|p2|")).toBeLessThan(raw.indexOf("|start"));
    const { channel, dropped } = channelFrom(frames).build();
    expect(channel.lines.map(withoutRqid)).toEqual(golden);
    expect(channel.lines.filter((line) => line.startsWith("|player|"))).toEqual(
      golden.slice(0, start).filter((line) => line.startsWith("|player|")),
    );
    expect(dropped).toEqual([]);
  });

  it("routes by the >ROOMID header", () => {
    const channel = new A2Channel({ room: ROOM, player: "p1", data_class: "synthetic" });
    expect(channel.receive(INIT)).toBe(true);
    expect(channel.receive(frame(["|j|Bob"], "lobby"))).toBe(false);
    expect(channel.receive("|challstr|4|abc")).toBe(false);
    expect(channel.build().channel.lines).toEqual([]);
  });

  it("keeps the chat line shaped like the opponent's exact HP out of the channel", () => {
    const golden = GOLDEN("p1");
    const end = golden.findIndex(isRequest);
    const chat = "|c|Bob|p2a: lol|3/4";
    const frames = [INIT, frame([...golden.slice(0, end), chat]), frame([golden[end] as string])];
    // Negative control: unfiltered, the chat line reads as p2's SECRET HP.
    const raw = frames.flatMap((f) => f.split("\n").slice(1));
    expect(raw).toContain(chat);
    expect(() => playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: raw })).toThrow(
      /only the opponent may see/,
    );
    expect(channelFrom(frames).build().channel.lines).toEqual(golden.slice(0, end + 1));
  });

  it("holds the sim's Open Team Sheets prompt and clear once when the bot declines", () => {
    const golden = GOLDEN("p1");
    const prompt = golden.findIndex((line) => line.startsWith("|uhtml|otsrequest|"));
    const start = golden.indexOf("|start");
    expect(prompt).toBeGreaterThan(-1);
    expect(start).toBeGreaterThan(prompt);
    // The server clears the prompt for the declining player and says so
    // before |start; the sim clears it again at |start.
    const decline = frame(["|uhtmlchange|otsrequest|", "p1 rejected open team sheets."]);
    const frames = [
      INIT,
      frame(golden.slice(0, prompt + 1)),
      decline,
      ...serverFramesOf(golden.slice(prompt + 1)).slice(2),
    ];
    const raw = frames.flatMap((f) => f.split("\n"));
    expect(raw.filter((line) => line === "|uhtmlchange|otsrequest|")).toHaveLength(2);
    const { channel, dropped } = channelFrom(frames).build();
    expect(channel.lines.map(withoutRqid)).toEqual(golden);
    expect(dropped).toEqual([]);
  });

  it("surfaces lines of an unknown kind in dropped, on every build", () => {
    const golden = GOLDEN("p1");
    const end = golden.findIndex(isRequest);
    const unknown = "|modal|Something new";
    const channel = channelFrom([INIT, frame([...golden.slice(0, end), unknown, golden[end] as string])]);
    expect(channel.build()).toMatchObject({ channel: { lines: golden.slice(0, end + 1) }, dropped: [unknown] });
    channel.receive(frame(["|-message|Another new line kind follows", "|newkind|x"]));
    expect(channel.build().dropped).toEqual([unknown, "|newkind|x"]);
  });

  it("filters the whole history on each build, so a line's place in the stream still counts", () => {
    const golden = GOLDEN("p1");
    const start = golden.indexOf("|start");
    const channel = channelFrom([INIT, frame(golden.slice(0, start + 1))]);
    expect(channel.build().channel.lines).toEqual(golden.slice(0, start + 1));
    // A rejoin's |player| line after |start is the server's (stripServerLines).
    channel.receive(frame(["|player|p2|p2|2|", golden[start + 1] as string]));
    expect(channel.build().channel.lines).toEqual(golden.slice(0, start + 2));
  });

  it("starts a new channel at a re-sent |init|: the replayed room log holds no earlier request", () => {
    const golden = GOLDEN("p1");
    const requests = golden.flatMap((line, i) => (isRequest(line) ? [i] : []));
    const resent = requests[3] as number;
    const channel = channelFrom(serverFramesOf(golden.slice(0, resent)));
    expect(channel.build().channel.lines.filter(isRequest)).toHaveLength(3);
    // The server sends |init|, the room log (no |request| or |error|: those
    // went to the player alone) and then the current request with the
    // choice it holds (server/rooms.ts `onJoin`, server/room-battle.ts `onConnect`).
    const log = golden.slice(0, resent).filter((line) => !isRequest(line) && !line.startsWith("|error|"));
    const request = withRqid(golden[resent] as string, 7);
    channel.receive(frame(["|init|battle", "|title|p1 vs. p2", "|j|☆p1", "|j|☆p2"]));
    channel.receive(frame(log));
    channel.receive(frame([request, "|sentchoice|move 1 1, move 1 2"]));
    const { channel: rejoined } = channel.build();
    expect(rejoined.lines).toEqual([...log, request]);
    expect(rejoined.lines.filter(isRequest)).toEqual([request]);
  });

  it("builds a PlayerChannel from the recorded server frames of a battle", () => {
    const { channel, dropped } = channelFrom([INIT, P2_JOIN, ...battle.frames]).build();
    const lines = battle.frames.flatMap((f) => f.split("\n").slice(1));
    expect(channel.lines).toEqual(stripServerLines(lines).lines);
    expect(channel.lines.some((line) => line.startsWith("|error|[Unavailable choice]"))).toBe(true);
    expect(dropped).toEqual([]);
  });

  it("refuses a stream that is not this player's own", () => {
    expect(() => channelFrom(serverFramesOf(GOLDEN("p1")), "p2").build()).toThrow(ObservationError);
  });

  it("refuses a room before its |init|, and a room that is not a battle", () => {
    const channel = new A2Channel({ room: ROOM, player: "p1", data_class: "private" });
    expect(() => channel.build()).toThrow(A2ChannelError);
    expect(() => channel.receive(frame(["|turn|1"]))).toThrow(/before \|init\|/);
    expect(() => channel.receive(frame(["|init|chat"]))).toThrow(/not a battle room/);
    channel.receive(INIT);
    expect(channel.build().channel).toMatchObject({ data_class: "private", lines: [] });
  });

  it("refuses a |noinit| for its room: the server refused the join", () => {
    const channel = new A2Channel({ room: ROOM, player: "p1", data_class: "synthetic" });
    // server/users.ts `tryJoinRoom`: the room is gone or closed to this user.
    const refused = `|noinit|nonexistent|The room "${ROOM}" does not exist.`;
    expect(() => channel.receive(frame([refused]))).toThrow(new A2ChannelError(`${ROOM}: room refused: ${refused}`));
    expect(() => channel.receive(frame(["|noinit|joinfailed|You are banned from the room."]))).toThrow(
      /room refused: \|noinit\|joinfailed\|/,
    );
    // A rename (server/rooms.ts `rename`) moves the room away from this id.
    channel.receive(INIT);
    expect(() => channel.receive(frame([`|noinit|rename|${ROOM}-x|p1 vs. p2`]))).toThrow(
      /room refused: \|noinit\|rename\|/,
    );
    expect(channel.build().channel.lines).toEqual([]);
  });

  describe("takes a frame all or nothing", () => {
    const golden = GOLDEN("p1");
    const end = golden.findIndex(isRequest);
    const request = withRqid(golden[end] as string, 2);
    const before = (): A2Channel => channelFrom([INIT, P2_JOIN, frame([...golden.slice(0, end), request])]);

    it.each([
      ["a chat room's |init|", ["|turn|9", "|init|chat"], /not a battle room/],
      ["a refused room", ["|turn|9", "|noinit|joinfailed|x"], /room refused/],
      ["a chat room's |init| after an |init|battle", ["|init|battle", "|title|t", "|init|chat"], /not a battle room/],
      ["a refused room after an |init|battle", ["|init|battle", "|noinit|joinfailed|x"], /room refused/],
    ])("leaves the channel unchanged after %s", (_, lines, error) => {
      const channel = before();
      const built = channel.build();
      expect(built.channel.lines.filter(isRequest)).toEqual([request]);
      expect(() => channel.receive(frame(lines))).toThrow(error);
      expect(channel.build()).toEqual(built);
      // The last kept request is unchanged too: its re-send is still left out.
      channel.receive(frame([request]));
      expect(channel.build()).toEqual({ ...built, resent: 1 });
    });

    it("stays without a channel after a throwing frame before the first |init|", () => {
      const channel = new A2Channel({ room: ROOM, player: "p1", data_class: "synthetic" });
      expect(() => channel.receive(frame(["|init|battle", "|title|t", "|init|chat"]))).toThrow(/not a battle room/);
      expect(() => channel.build()).toThrow(/has not sent \|init\|/);
    });
  });

  describe("leaves a re-sent request out of the channel", () => {
    const golden = GOLDEN("p1");
    const requests = golden.flatMap((line, i) => (isRequest(line) ? [i] : []));
    const at = requests[2] as number;
    /** The server frames up to and including the third request, whose rqid is 6. */
    const upTo = serverFramesOf(golden.slice(0, at + 1));
    const third = withRqid(golden[at] as string, 6);

    it("keeps the channel's request count equal to the protocol reader's request_seq", () => {
      // A login runs onConnect again: the stored request and its choice (server/room-battle.ts `onConnect`).
      const frames = [...upTo, frame([third, "|sentchoice|move 1 1, move 1 2"])];
      const reader = new ProtocolReader();
      const seqs = frames.flatMap((f) =>
        reader.read(f).flatMap((event) => (event.kind === "request" ? [[event.request.request_seq, event.resend]] : [])),
      );
      expect(seqs.slice(-2)).toEqual([
        [3, false],
        [3, true],
      ]);
      const { channel, resent } = channelFrom(frames).build();
      expect(resent).toBe(1);
      expect(channel.lines).toEqual(channelFrom(upTo).build().channel.lines);
      expect(channel.lines.filter(isRequest)).toHaveLength(3);
      expect(channel.lines.at(-1)).toBe(third);
    });

    it("keeps a new request: a new rqid, or no rqid, is never a re-send", () => {
      const next = withRqid(golden[at] as string, 7);
      const bare = golden[at] as string;
      const { channel, resent } = channelFrom([...upTo, frame([next]), frame([bare]), frame([bare])]).build();
      expect(resent).toBe(0);
      expect(channel.lines.slice(-3)).toEqual([next, bare, bare]);
    });

    it("keeps a request with the same rqid but another body", () => {
      const other = withRqid(golden[requests[1] as number] as string, 6);
      const { channel, resent } = channelFrom([...upTo, frame([other])]).build();
      expect(resent).toBe(0);
      expect(channel.lines.at(-1)).toBe(other);
    });

    it("keeps the re-sent request after |init|battle, as the new channel's first", () => {
      const channel = channelFrom([...upTo, frame([third])]);
      expect(channel.build().resent).toBe(1);
      channel.receive(frame(["|init|battle", "|title|p1 vs. p2", "|j|☆p1", "|j|☆p2", third]));
      expect(channel.build()).toMatchObject({ channel: { lines: [third] }, resent: 0 });
    });

    it("keeps |request|null and a request that is not JSON, for playerChannel to judge", () => {
      // |request|null withdraws the request (server/room-battle.ts `onLeave`); the re-send after it is still one.
      const withdrawn = channelFrom([...upTo, frame(["|request|null", third])]);
      expect(withdrawn.build().resent).toBe(1);
      for (const bad of ['|request|{"rqid":"6"}', "|request|{"]) {
        const channel = channelFrom([...upTo, frame([bad])]);
        // Kept, so the bridge refuses the stream rather than this adapter hiding the line.
        expect(() => channel.build()).toThrow(/request|JSON/);
      }
    });
  });
});
