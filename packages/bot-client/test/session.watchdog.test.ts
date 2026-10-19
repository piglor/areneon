import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type A2BotOptions,
  type A2SessionResult,
  type DecisionRecord,
  type MemoryTransport,
  memoryTransport,
  runA2Bot,
  type SessionLog,
} from "../src/index.ts";
import { battle, serverFrame } from "./frames.ts";
import { fixtureTeam, FORMAT } from "./local-battle.ts";

// The deadline watchdog and the auto-choose detector (ADR-005 §3) in the
// session loop, on the in-memory transport and a fake clock: `now` is the
// session's clock, and vitest's fake timers run the watchdog's timers.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const ROOM = battle.room;
const BOT = "Areneon-bot-0000";
const URL_ = "ws://127.0.0.1:8000/showdown/websocket";

const at = (i: number): string => battle.frames[i] as string;
const room = (...lines: string[]): string => `>${ROOM}\n${lines.join("\n")}`;
const INIT = room(
  "|init|battle",
  "|title|p1 vs. p2",
  "|inactive|Battle timer is ON: inactive players will automatically lose when time's up.",
  `|j| ${BOT}`,
);
const time = (turn: number, bank: number, grace = 0): string =>
  room(`|inactive|Time left: ${turn} sec this turn | ${bank} sec total${grace > 0 ? ` | ${grace} sec grace` : ""}`);
const CONFIRM = room(`${BOT} rejected open team sheets.`);

let now = 0;
afterEach(() => {
  now = 0;
  vi.useRealTimers();
});

interface Run {
  result: Promise<A2SessionResult>;
  transports: MemoryTransport[];
  logs: SessionLog[];
  t: () => MemoryTransport;
  events: (event: SessionLog["event"]) => SessionLog[];
  decisions: () => DecisionRecord[];
}

async function start(options: Partial<A2BotOptions> = {}): Promise<Run> {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const transports: MemoryTransport[] = [];
  const logs: SessionLog[] = [];
  const result = runA2Bot({
    engine,
    snapshot,
    url: URL_,
    name: BOT,
    challenger: "Sparring Human",
    format: FORMAT,
    team: fixtureTeam("legal-synthetic.txt"),
    data_class: "synthetic",
    battles: 1,
    clock: () => now,
    reconnect: { attempts: 2, delayMs: 5 },
    connect: async () => {
      const transport = memoryTransport();
      transports.push(transport);
      return transport;
    },
    onLog: (entry) => logs.push(entry),
    ...options,
  });
  result.catch(() => {});
  await vi.waitFor(() => expect(transports).toHaveLength(1));
  return {
    result,
    transports,
    logs,
    t: () => transports.at(-1) as MemoryTransport,
    events: (event) => logs.filter((entry) => entry.event === event),
    decisions: () => logs.flatMap((entry) => (entry.event === "decision" ? [entry.decision] : [])),
  };
}

/** Logged in, preview answered with its snapshot: the next request is a move request. */
function inBattle(t: MemoryTransport): void {
  t.push("|challstr|4|0123abcd");
  t.push(`|updateuser| ${BOT}|1|1|{"blockChallenges":false}`);
  t.push(INIT);
  t.push(at(0));
  t.push(at(1));
  t.push(time(90, 420, 90));
  t.push(CONFIRM);
  t.push(at(2));
}

const chooses = (t: MemoryTransport, rqid: number): string[] =>
  t.sent.filter((line) => line.startsWith(`${ROOM}|/choose `) && line.endsWith(`|${rqid}`));

describe("snapshot never comes (server/room-battle.ts:311-322)", () => {
  it("answers the held request with the fallback after the wait, with no timer state", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    now = 10_000;
    t.push(at(3));
    vi.advanceTimersByTime(999);
    now = 10_999;
    expect(chooses(t, 3)).toEqual([]);
    vi.advanceTimersByTime(1);
    now = 11_000;
    const decision = run.decisions()[1] as DecisionRecord;
    expect(decision).toMatchObject({
      rqid: 3,
      phase: "move",
      timer: null,
      budget: { submit_within_ms: 0, policy_budget_ms: 0 },
      inherited: false,
      watchdog: "snapshot_missing",
    });
    // The fallback is the actor rule's choice through the candidate and rqid path.
    expect(decision.choice).toBe(decision.legal.choices[0]);
    expect(chooses(t, 3)).toEqual([`${ROOM}|/choose ${decision.choice}|3`]);
    expect(run.events("watchdog")).toEqual([
      { event: "watchdog", room: ROOM, rqid: 3, request_seq: decision.request_seq, reason: "snapshot_missing" },
    ]);
  });

  it("never answers twice for one rqid: a late snapshot and later time change nothing", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    vi.advanceTimersByTime(1_000);
    t.push(time(55, 400));
    vi.advanceTimersByTime(60_000);
    expect(run.decisions()).toHaveLength(2);
    expect(chooses(t, 3)).toHaveLength(1);
  });

  it("does nothing when the snapshot arrives in time", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    vi.advanceTimersByTime(999);
    t.push(time(55, 400));
    vi.advanceTimersByTime(60_000);
    expect(run.decisions()[1]).toMatchObject({ timer: { turn_seconds_left: 55 } });
    expect(run.decisions()[1]).not.toHaveProperty("watchdog");
    expect(run.events("watchdog")).toEqual([]);
    expect(chooses(t, 3)).toHaveLength(1);
  });

  it("takes the wait from the options", async () => {
    const run = await start({ snapshotWaitMs: 250 });
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    vi.advanceTimersByTime(249);
    expect(run.decisions()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(run.decisions()).toHaveLength(2);
  });

  it("never answers a stale request: a newer one drops the older deadline", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    vi.advanceTimersByTime(500);
    t.push(at(4));
    t.push(at(5));
    vi.advanceTimersByTime(999);
    expect(chooses(t, 3)).toEqual([]);
    expect(chooses(t, 5)).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(chooses(t, 3)).toEqual([]);
    expect(chooses(t, 5)).toHaveLength(1);
    expect(run.events("watchdog")).toMatchObject([{ rqid: 5 }]);
  });

  it("drops the deadline when the request is withdrawn", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    t.push(serverFrame("request_null"));
    vi.advanceTimersByTime(60_000);
    expect(run.decisions()).toHaveLength(1);
    expect(chooses(t, 3)).toEqual([]);
  });

  it("drops the deadline on a disconnect: the re-sent request is answered once, on the rejoin", async () => {
    const run = await start();
    inBattle(run.t());
    run.t().push(at(3));
    run.t().end(1006);
    vi.advanceTimersByTime(60_000);
    await vi.advanceTimersByTimeAsync(10);
    expect(run.transports).toHaveLength(2);
    expect(run.decisions()).toHaveLength(1);
    const t = run.t();
    t.push("|challstr|4|0123abcd");
    t.push(`|updateuser| ${BOT}|1|1|{"blockChallenges":false}`);
    t.push(INIT);
    t.push(at(0));
    t.push(at(2));
    t.push(at(3));
    expect(run.decisions()[1]).toMatchObject({ rqid: 3, timer: null });
    expect(run.decisions()[1]).not.toHaveProperty("watchdog");
    vi.advanceTimersByTime(60_000);
    expect(chooses(t, 3)).toHaveLength(1);
    expect(run.events("watchdog")).toEqual([]);
  });

  it("stops the deadline with the battle", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    t.push(room(`|win|${BOT}`));
    vi.advanceTimersByTime(60_000);
    await run.result;
    expect(vi.getTimerCount()).toBe(0);
  });

  it("stops the deadline when the session fails", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    t.push(room("|error|not a choice refusal"));
    await expect(run.result).rejects.toThrow();
    vi.advanceTimersByTime(60_000);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("gives the update after an [Unavailable choice] the original receipt and the fallback's budget", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    now = 1_000;
    const request = JSON.parse(at(5).split("\n")[1]?.slice("|request|".length) ?? "") as Record<string, unknown>;
    (request.active as Record<string, unknown>[])[1] = { ...(request.active as object[])[1], maybeDisabled: true };
    t.push(room(`|request|${JSON.stringify(request)}`));
    now = 2_000;
    vi.advanceTimersByTime(1_000);
    const fallback = run.decisions()[1] as DecisionRecord;
    expect(fallback).toMatchObject({ watchdog: "snapshot_missing", elapsed_ms: 1_000 });
    now = 2_500;
    t.push(room(`|error|[Unavailable choice] Can't switch: The active Pokémon is trapped`));
    t.push(at(7));
    const update = run.decisions()[2] as DecisionRecord;
    expect(update).toMatchObject({ rqid: 7, inherited: true, elapsed_ms: 1_500 });
    expect(update.budget).toEqual(fallback.budget);
    // It carries the fallback's deadline, so the record still says why it was unbudgeted.
    expect(update.watchdog).toBe("snapshot_missing");
    expect(run.events("watchdog")).toHaveLength(1);
    vi.advanceTimersByTime(60_000);
    expect(chooses(t, 7)).toHaveLength(1);
  });
});

describe("server auto-choose detector (ADR-005 §3)", () => {
  it("records server_auto_choose_detected when a newer request arrives while the bot's is unanswered", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    // The server timed the bot out and chose `default`: it adds no line, the turn just moves on.
    t.push(at(4));
    t.push(at(5));
    const detection = { rqid: 3, request_seq: (run.decisions()[0] as DecisionRecord).request_seq + 1, phase: "move" };
    expect(run.events("server_auto_choose_detected")).toEqual([
      { event: "server_auto_choose_detected", room: ROOM, detection },
    ]);
    t.push(room(`|win|${BOT}`));
    const { battles } = await run.result;
    expect(battles[0]?.server_auto_chooses).toEqual([detection]);
  });

  it("records nothing for answered requests, a wait request or a re-sent one", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    t.push(time(55, 400));
    t.push(at(4));
    t.push(at(5));
    t.push(time(55, 400));
    t.push(at(18));
    t.push(at(21));
    t.push(time(55, 400));
    // A re-send of the open request is the same request.
    t.push(at(21));
    t.push(room(`|win|${BOT}`));
    const { battles } = await run.result;
    expect(run.events("server_auto_choose_detected")).toEqual([]);
    expect(battles[0]?.server_auto_chooses).toEqual([]);
  });

  it("does not take the update after a refused pending candidate for an auto-choose", async () => {
    const run = await start({ mode: "live_match" });
    const t = run.t();
    inBattle(t);
    t.push(at(3));
    t.push(time(55, 400));
    t.push(room("|error|[Unavailable choice] Can't move: something hidden"));
    t.push(at(7));
    expect(run.events("server_auto_choose_detected")).toEqual([]);
  });

  it("detects an unanswered forceSwitch the same way", async () => {
    const run = await start();
    const t = run.t();
    inBattle(t);
    t.push(at(10));
    t.push(at(12));
    expect(run.events("server_auto_choose_detected")).toMatchObject([{ detection: { rqid: 8, phase: "forceSwitch" } }]);
  });
});
