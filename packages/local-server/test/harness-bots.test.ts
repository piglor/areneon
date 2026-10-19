import type { A2BotOptions, A2SessionResult } from "@areneon/bot-client";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { beforeEach, describe, expect, it, vi } from "vitest";

// playBotVsBot's refusals, with runA2Bot replaced: no server runs here.
const run = vi.hoisted(() => vi.fn<(options: A2BotOptions) => Promise<A2SessionResult>>());
vi.mock("@areneon/bot-client", async (original) => ({ ...(await original<typeof import("@areneon/bot-client")>()), runA2Bot: run }));

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { HarnessError, playBotVsBot, recordHarnessBattle } from "../src/harness.ts";
import { FIXTURE_FORMAT } from "../src/recordings.ts";
import { COMMITTED } from "./harness-fixtures.ts";
import { tempDir } from "./helpers.ts";

const engine = loadEngine();
const options = {
  engine,
  snapshot: loadRulesSnapshot(engine.commit),
  url: "ws://127.0.0.1:1/showdown/websocket",
  format: "f",
  names: { sparring: "s", challenger: "c" },
  teams: { sparring: [], challenger: [] },
};
const battle = (room: string) => ({ battles: [{ room }] }) as unknown as A2SessionResult;

beforeEach(() => {
  run.mockReset();
});

describe("playBotVsBot", () => {
  it("starts the challenger only after the sparring bot logged in, and returns both records", async () => {
    const order: string[] = [];
    run.mockImplementation(async (o) => {
      if (o.challenge === undefined) {
        order.push("sparring");
        o.onLog?.({ event: "logged_in", name: "s" });
        o.onLog?.({ event: "connected", attempt: 1 });
        await new Promise((done) => setTimeout(done, 20));
      } else {
        order.push(`challenger->${o.challenge}`);
      }
      return battle("battle-f-1");
    });
    const pair = await playBotVsBot(options);
    expect(order).toEqual(["sparring", "challenger->s"]);
    expect(pair.connections).toEqual({ sparring: [], challenger: [] });
    expect(pair.sparring.room).toBe(pair.challenger.room);
  });

  it("taps each bot's own connections at the transport, one list of frames per connection", async () => {
    const urls: string[] = [];
    const frameListeners: ((frame: string) => void)[] = [];
    const connect = (url: string) => {
      urls.push(url);
      return Promise.resolve({
        send: () => {},
        onFrame: (cb: (frame: string) => void) => void frameListeners.push(cb),
        onClose: () => {},
        close: () => {},
      });
    };
    run.mockImplementation(async (o) => {
      const mine = o.challenge === undefined ? "sparring" : "challenger";
      o.onLog?.({ event: "logged_in", name: "s" });
      // Two connections of the same bot (a reconnect), and one of the other.
      for (const frame of [`>r\n|${mine}|1`, `|${mine}|2`]) {
        const transport = await (o.connect as NonNullable<typeof o.connect>)(o.url);
        transport.onFrame(() => {});
        frameListeners.at(-1)?.(frame);
      }
      return battle("battle-f-1");
    });
    const pair = await playBotVsBot({ ...options, connect });
    expect(pair.connections).toEqual({
      sparring: [[">r\n|sparring|1"], ["|sparring|2"]],
      challenger: [[">r\n|challenger|1"], ["|challenger|2"]],
    });
    expect(urls).toHaveLength(4);
    expect(new Set(urls)).toEqual(new Set([options.url]));
  });

  it("fails if the sparring bot ends before it logs in", async () => {
    run.mockResolvedValue(battle("battle-f-1"));
    await expect(playBotVsBot(options)).rejects.toThrow(new HarnessError("the sparring bot ended before login"));
  });

  it("fails if the bots did not play one battle", async () => {
    run.mockImplementation((o) => {
      if (o.challenge === undefined) {
        o.onLog?.({ event: "logged_in", name: "s" });
        return Promise.resolve(battle("battle-f-1"));
      }
      return Promise.resolve(battle("battle-f-2"));
    });
    await expect(playBotVsBot(options)).rejects.toThrow("did not play the same battle");
    run.mockImplementation((o) => {
      o.onLog?.({ event: "logged_in", name: "s" });
      return Promise.resolve({ battles: [] });
    });
    await expect(playBotVsBot(options)).rejects.toThrow("did not play the same battle");
  });
});

describe("playBotVsBot with a drop plan", () => {
  const fakeTransport = () => {
    const frameListeners: ((frame: string) => void)[] = [];
    let closed = 0;
    const transport = {
      send: () => {},
      onFrame: (cb: (frame: string) => void) => void frameListeners.push(cb),
      onClose: () => {},
      close: () => void (closed += 1),
    };
    return { transport, push: (frame: string) => frameListeners.forEach((cb) => cb(frame)), closed: () => closed };
  };

  it("cuts the planned bot's first connection at the transport, so its frames stop, and keeps its next connection whole", async () => {
    const opened: ReturnType<typeof fakeTransport>[] = [];
    const connect = () => {
      const one = fakeTransport();
      opened.push(one);
      return Promise.resolve(one.transport);
    };
    run.mockImplementation(async (o) => {
      o.onLog?.({ event: "logged_in", name: "s" });
      const mine = o.challenge === undefined ? "sparring" : "challenger";
      // The first connection receives two requests (the second is the cut), then a frame that is lost;
      // the second connection (a rejoin) receives its log and the request again.
      // `connect` runs at once inside `o.connect`, so the transport it made is the one at `before`.
      const open = async (frames: string[]) => {
        const before = opened.length;
        const transport = o.connect as NonNullable<typeof o.connect>;
        const made = transport(o.url);
        const one = opened[before] as ReturnType<typeof fakeTransport>;
        (await made).onFrame(() => {});
        for (const frame of frames) one.push(frame);
      };
      await open([`>r\n|${mine}|1\n|request|{}`, `>r\n|request|{}`, ">r\n|lost|"]);
      await open([`>r\n|init|battle\n|request|{}`]);
      return battle("battle-f-1");
    });
    const pair = await playBotVsBot({ ...options, connect, drop: { who: "sparring", request: 2, when: "before_choice" } });
    expect(pair.connections.sparring).toEqual([[">r\n|sparring|1\n|request|{}", ">r\n|request|{}"], [">r\n|init|battle\n|request|{}"]]);
    // The other bot is not cut: all its frames are kept.
    expect(pair.connections.challenger).toEqual([
      [">r\n|challenger|1\n|request|{}", ">r\n|request|{}", ">r\n|lost|"],
      [">r\n|init|battle\n|request|{}"],
    ]);
    // Only the sparring bot's first connection was closed by the recorder.
    expect(opened.map((one) => one.closed()).sort()).toEqual([0, 0, 0, 1]);
  });
});

describe("recordHarnessBattle's check of the planned reconnect", () => {
  const withDrop = COMMITTED.find((c) => c.recording.drop?.when === "after_choice") as (typeof COMMITTED)[0];
  const plain = COMMITTED[0] as (typeof COMMITTED)[0];

  /** The bots are given the frames a real server sent them, connection by connection; the server's input log is a file. */
  function record(from: (typeof COMMITTED)[0], drop: Parameters<typeof recordHarnessBattle>[0]["drop"]) {
    const { fixture, recording } = from;
    const seatOf = (role: "sparring" | "challenger") => (fixture.players.p1.name === recording.names[role] ? "p1" : "p2");
    let next: string[] = [];
    const connect = () => {
      const frames = next;
      return Promise.resolve({
        send: () => {},
        onFrame: (cb: (frame: string) => void) => void frames.forEach(cb),
        onClose: () => {},
        close: () => {},
      });
    };
    run.mockImplementation(async (o) => {
      const role = o.challenge === undefined ? "sparring" : "challenger";
      if (role === "sparring") o.onLog?.({ event: "logged_in", name: o.name });
      const seat = seatOf(role);
      for (const frames of fixture.frames[seat]) {
        next = frames;
        (await (o.connect as NonNullable<typeof o.connect>)(o.url)).onFrame(() => {});
      }
      return { battles: [{ room: fixture.room, player: seat, channels: fixture.channels[seat], result: "win" }] } as unknown as A2SessionResult;
    });
    const dir = tempDir("bots-log");
    const day = join(dir, "logs", "2026-10", FIXTURE_FORMAT, "2026-10-19");
    mkdirSync(day, { recursive: true });
    const winner = fixture.winner === null ? "" : fixture.players[fixture.winner].name;
    writeFileSync(
      join(day, `${fixture.room.replace(/^battle-/, "")}.log.json`),
      JSON.stringify({ inputLog: fixture.input_log.lines, winner, turns: fixture.turns }),
    );
    return recordHarnessBattle({
      engine,
      snapshot: options.snapshot,
      server: { session: { port: 1 } } as never,
      dir,
      format: FIXTURE_FORMAT,
      names: recording.names,
      teams: { sparring: fixture.teams[seatOf("sparring")], challenger: fixture.teams[seatOf("challenger")] },
      teamLabels: recording.teams,
      provenance: fixture.provenance,
      ...(drop === undefined ? {} : { drop }),
      connect,
    });
  }
  // A request number the battle never reaches: the cut is planned, but the replayed frames are not cut.
  const never = (from: (typeof COMMITTED)[0], when: "before_choice" | "after_choice") => ({
    who: (from.recording.drop?.who ?? "sparring") as "sparring" | "challenger",
    request: 99,
    when,
  });

  it("builds a fixture that verifies, with no plan", async () => {
    const built = await record(plain, undefined);
    expect(built.frames).toEqual(plain.fixture.frames);
    expect(built.channels).toEqual(plain.fixture.channels);
  });

  it("accepts a recording whose re-sent request and |sentchoice| are the planned ones", async () => {
    const built = await record(withDrop, never(withDrop, "after_choice"));
    expect(built.frames).toEqual(withDrop.fixture.frames);
  });

  it("refuses one whose |sentchoice| is not the planned one, and one with no re-sent request at all", async () => {
    await expect(record(withDrop, never(withDrop, "before_choice"))).rejects.toThrow(/reconnect is not the planned one.*record again/);
    await expect(record(plain, never(withDrop, "after_choice"))).rejects.toThrow(/reconnect is not the planned one \(undefined\)/);
  });
});
