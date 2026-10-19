import type { A2BotOptions, A2SessionResult } from "@areneon/bot-client";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { beforeEach, describe, expect, it, vi } from "vitest";

// playBotVsBot's refusals, with runA2Bot replaced: no server runs here.
const run = vi.hoisted(() => vi.fn<(options: A2BotOptions) => Promise<A2SessionResult>>());
vi.mock("@areneon/bot-client", async (original) => ({ ...(await original<typeof import("@areneon/bot-client")>()), runA2Bot: run }));

import { HarnessError, playBotVsBot } from "../src/harness.ts";

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
