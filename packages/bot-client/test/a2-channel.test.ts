import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ObservationError, playerChannel, stripServerLines } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { A2Channel, A2ChannelError } from "../src/index.ts";
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

/** The frame the server sends when a connection joins the room (server/rooms.ts `onJoin`). */
const INIT = frame(["|init|battle", "|title|p1 vs. p2", "|j|☆p1", "|j|☆p2", "|t:|1760000000"]);

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
  const frames = [INIT];
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
      ...serverFramesOf(golden.slice(prompt + 1)).slice(1),
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
    channel.receive(INIT);
    channel.receive(frame(log));
    channel.receive(frame([request, "|sentchoice|move 1 1, move 1 2"]));
    const { channel: rejoined } = channel.build();
    expect(rejoined.lines).toEqual([...log, request]);
    expect(rejoined.lines.filter(isRequest)).toEqual([request]);
  });

  it("builds a PlayerChannel from the recorded server frames of a battle", () => {
    const { channel, dropped } = channelFrom([INIT, ...battle.frames]).build();
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
});
