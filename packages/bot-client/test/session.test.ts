import { continueChannel, EnumeratorDefectError, loadEngine, loadRulesSnapshot, UnsupportedFormatError } from "@areneon/showdown-bridge";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type A2BattleRecord,
  type A2BotOptions,
  type A2SessionResult,
  type DecisionRecord,
  LocalUrlError,
  type MemoryTransport,
  memoryTransport,
  RefusalError,
  runA2Bot,
  SessionError,
  type SessionLog,
} from "../src/index.ts";
import { battle, serverFrame } from "./frames.ts";
import { fixtureTeam, FORMAT, freePort, TEAMS_DIR } from "./local-battle.ts";
import { importPaste } from "@areneon/showdown-bridge";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The session loop against a scripted server on the in-memory transport.
// The battle frames are the recorded engine battle (fixtures/bot-client/frames,
// p1's own stream, with the server's rqids); the frames around them (join,
// timer, Open Team Sheets, login, challenges) are written from the pinned
// server source as cited in the other tests of this package. The scripted
// server does not play: it pushes the next frame, whatever the bot chose.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const own = fixtureTeam("legal-synthetic.txt");
const ROOM = battle.room;
const BOT = "Areneon-bot-0000";
const HUMAN = "Sparring Human";
const URL_ = "ws://127.0.0.1:8000/showdown/websocket";

const at = (i: number): string => battle.frames[i] as string;
const room = (...lines: string[]): string => `>${ROOM}\n${lines.join("\n")}`;
/** The request JSON of battle frame `i`, edited. */
const edited = (i: number, edit: (request: Record<string, unknown>) => void): string => {
  const request = JSON.parse(at(i).split("\n")[1]?.slice("|request|".length) ?? "") as Record<string, unknown>;
  edit(request);
  return room(`|request|${JSON.stringify(request)}`);
};
const INIT = room(
  "|init|battle",
  "|title|p1 vs. p2",
  "|inactive|Battle timer is ON: inactive players will automatically lose when time's up.",
  `|j| ${BOT}`,
);
const time = (turn: number, bank: number, grace = 0): string =>
  room(`|inactive|Time left: ${turn} sec this turn | ${bank} sec total${grace > 0 ? ` | ${grace} sec grace` : ""}`);
const CONFIRM = room(`${BOT} rejected open team sheets.`);
const challengeFrom = (from: string, format = FORMAT): string => `|pm| ${from}| ${BOT}|/challenge ${format}|${format}|||`;

let now = 0;
afterEach(() => {
  now = 0;
  vi.useRealTimers();
});

interface Run {
  result: Promise<A2SessionResult>;
  transports: MemoryTransport[];
  logs: SessionLog[];
  /** The current transport. */
  t: () => MemoryTransport;
  events: (event: SessionLog["event"]) => SessionLog[];
  decisions: () => DecisionRecord[];
}

async function start(options: Partial<A2BotOptions> = {}): Promise<Run> {
  const transports: MemoryTransport[] = [];
  const logs: SessionLog[] = [];
  const result = runA2Bot({
    engine,
    snapshot,
    url: URL_,
    name: BOT,
    challenger: HUMAN,
    format: FORMAT,
    team: own,
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
  const run: Run = {
    result,
    transports,
    logs,
    t: () => transports.at(-1) as MemoryTransport,
    events: (event) => logs.filter((entry) => entry.event === event),
    decisions: () =>
      logs.flatMap((entry) => (entry.event === "decision" ? [entry.decision] : [])),
  };
  if (options.signal?.aborted !== true && options.connect === undefined) await vi.waitFor(() => expect(transports).toHaveLength(1));
  return run;
}

function login(t: MemoryTransport): void {
  t.push("|challstr|4|0123abcd");
  t.push(`|updateuser| ${BOT}|1|1|{"blockChallenges":false}`);
}

/** Join, preview request and its snapshot, and the confirmed decline: the preview choice goes out. */
function preview(t: MemoryTransport): void {
  t.push(INIT);
  t.push(at(0));
  t.push(at(1));
  t.push(time(90, 420, 90));
  t.push(CONFIRM);
}

const chooseLines = (t: MemoryTransport): string[] => t.sent.filter((line) => line.startsWith(`${ROOM}|/choose `));

describe("runA2Bot: start", () => {
  it("refuses a non-local URL before connecting", async () => {
    const connect = vi.fn();
    await expect(runA2Bot({ ...base(), url: "ws://example.com:8000/", connect })).rejects.toThrow(LocalUrlError);
    expect(connect).not.toHaveBeenCalled();
  });

  it("refuses a battle count that is not a positive integer, an unknown format and an invalid team", async () => {
    await expect(runA2Bot({ ...base(), battles: 0 })).rejects.toThrow(RangeError);
    await expect(runA2Bot({ ...base(), battles: 1.5 })).rejects.toThrow(RangeError);
    await expect(runA2Bot({ ...base(), format: "gen9ou" })).rejects.toThrow(UnsupportedFormatError);
    const five = importPaste(engine, readFileSync(join(TEAMS_DIR, "illegal-five-pokemon.txt"), "utf8"));
    await expect(runA2Bot({ ...base(), team: five })).rejects.toThrow(SessionError);
  });

  it("returns at once when already aborted", async () => {
    const run = await start({ signal: AbortSignal.abort() });
    await expect(run.result).resolves.toEqual({ battles: [] });
    expect(run.transports).toEqual([]);
  });

  it("logs in with /trn on the challenge string, then sets its team", async () => {
    const run = await start();
    const t = run.t();
    t.push("|challstr|4|0123abcd");
    expect(t.sent).toEqual([`|/trn ${BOT},0,`]);
    t.push(`|updateuser| Guest 1|0|1|{}`);
    t.push(`|updateuser| Someone Else|1|1|{}`);
    expect(t.sent).toHaveLength(1);
    t.push(`|updateuser|‽${BOT}|1|1|{}`);
    t.push(`|updateuser|‽${BOT}|1|1|{}`);
    expect(t.sent).toEqual([`|/trn ${BOT},0,`, `|/utm ${engine.sim.Teams.pack(own)}`]);
    expect(run.events("logged_in")).toEqual([{ event: "logged_in", name: BOT }]);
    expect(run.events("connected")).toEqual([{ event: "connected", attempt: 1 }]);
  });

  it("challenges once in a bot-vs-bot harness", async () => {
    const run = await start({ challenge: "Other Bot" });
    login(run.t());
    expect(run.t().sent.at(-1)).toBe(`|/challenge Other Bot, ${FORMAT}`);
  });

  it("logs popups and global errors for the operator, and ignores other rooms", async () => {
    const run = await start();
    const t = run.t();
    t.push("|popup||html|Your IP (127.0.0.1) is currently locked");
    t.push("|error|/challenge - nobody is online");
    t.push(serverFrame("other_room"));
    t.push("");
    t.push(room("|j| someone"));
    t.push(`|request|null`);
    expect(run.events("server_message")).toEqual([
      { event: "server_message", line: "|popup||html|Your IP (127.0.0.1) is currently locked" },
      { event: "server_message", line: "|error|/challenge - nobody is online" },
    ]);
    expect(t.sent).toEqual([]);
  });
});

function base(): A2BotOptions {
  return {
    engine,
    snapshot,
    url: URL_,
    name: BOT,
    challenger: HUMAN,
    format: FORMAT,
    team: own,
    connect: async () => memoryTransport(),
  };
}

describe("runA2Bot: challenges (ADR-005 §2, §4)", () => {
  it("accepts the launch challenger's exact format and refuses anything else", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(challengeFrom("Stranger"));
    t.push(challengeFrom(HUMAN, `${FORMAT}@@@Best Of = 3`));
    t.push(challengeFrom(HUMAN));
    expect(t.sent.slice(2)).toEqual(["|/reject stranger", "|/reject sparringhuman", "|/accept sparringhuman"]);
    expect(run.events("challenge_rejected").map((e) => "reason" in e && e.reason)).toEqual(["challenger", "format"]);
    expect(run.events("challenge_accepted")).toEqual([
      { event: "challenge_accepted", from: HUMAN, format: FORMAT },
    ]);
  });

  it("plays one battle at a time, and none after its last", async () => {
    const run = await start({ battles: 2 });
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(challengeFrom(HUMAN));
    expect(t.sent.at(-1)).toBe("|/reject sparringhuman");
    expect(run.events("challenge_rejected").at(-1)).toMatchObject({ reason: "busy" });
    t.push(room("|tie"));
    t.push(challengeFrom(HUMAN));
    expect(t.sent.at(-1)).toBe("|/accept sparringhuman");
  });

  it("accepts one challenge until its battle starts, and takes an accept that never starts as failed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start({ battles: 3, acceptTimeoutMs: 1_000 });
    const t = run.t();
    login(t);
    t.push(challengeFrom(HUMAN));
    t.push(challengeFrom(HUMAN));
    expect(t.sent.slice(2)).toEqual(["|/accept sparringhuman", "|/reject sparringhuman"]);
    expect(run.events("challenge_rejected").at(-1)).toMatchObject({ reason: "busy" });
    vi.advanceTimersByTime(1_001);
    expect(run.events("accept_abandoned")).toEqual([{ event: "accept_abandoned", reason: "timeout" }]);
    t.push(challengeFrom(HUMAN));
    expect(t.sent.at(-1)).toBe("|/accept sparringhuman");
    // The battle starting ends the wait: no abandoned accept, and the next challenge is busy for the open battle.
    t.push(INIT);
    vi.advanceTimersByTime(5_000);
    expect(run.events("accept_abandoned")).toHaveLength(1);
    t.push(challengeFrom(HUMAN));
    expect(t.sent.at(-1)).toBe("|/reject sparringhuman");
  });

  it("forgets an accept when the connection closes before the battle starts", async () => {
    const run = await start({ battles: 3 });
    login(run.t());
    run.t().push(challengeFrom(HUMAN));
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    expect(run.events("accept_abandoned")).toEqual([{ event: "accept_abandoned", reason: "disconnected" }]);
    login(run.t());
    run.t().push(challengeFrom(HUMAN));
    expect(run.t().sent.at(-1)).toBe("|/accept sparringhuman");
  });

  it("rejects a valid challenge that arrives after the stop began", async () => {
    const controller = new AbortController();
    const inner = memoryTransport();
    // A server that has not closed the socket yet: frames still come after the stop.
    const slow = Object.create(inner, { close: { value: vi.fn() } }) as MemoryTransport;
    const run = await start({ signal: controller.signal, connect: async () => slow });
    await vi.waitFor(() => expect(slow.onFrame).toBeDefined());
    login(inner);
    inner.push(INIT);
    controller.abort();
    inner.push(challengeFrom(HUMAN));
    expect(inner.sent.at(-1)).toBe("|/reject sparringhuman");
    expect(run.events("challenge_rejected")).toEqual([
      expect.objectContaining({ reason: "stopping", from: HUMAN }),
    ]);
    expect(run.events("challenge_accepted")).toEqual([]);
    // The battle ending while the stop is under way stops nothing twice.
    inner.push(room("|tie"));
    inner.end(1000);
    await expect(run.result).resolves.toMatchObject({ battles: [{ result: "tie" }] });
  });

  it("refuses a challenge once its battles are played", async () => {
    const run = await start({ battles: 1, signal: new AbortController().signal });
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(room("|tie"));
    await run.result;
    expect(t.closed).toBe(true);
  });
});

describe("runA2Bot: a battle", () => {
  it("declines open team sheets, then sends the preview choice with its budget from the snapshot", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    expect(t.sent.at(-1)).toBe(`${ROOM}|/timer on`);
    t.push(at(0));
    t.push(at(1));
    // The budget needs the snapshot the server sends right after the request.
    expect(t.sent.at(-1)).toBe(`${ROOM}|/timer on`);
    now = 100;
    t.push(time(90, 420, 90));
    expect(t.sent.at(-1)).toBe(`${ROOM}|/rejectopenteamsheets`);
    now = 150;
    t.push(CONFIRM);
    const [decision] = run.decisions() as [DecisionRecord];
    expect(t.sent.at(-1)).toBe(`${ROOM}|/choose ${decision.choice}|1`);
    expect(decision).toMatchObject({
      request_seq: 1,
      channel_request_seq: 1,
      rqid: 1,
      phase: "preview",
      timer: { timer_source: "server", turn_seconds_left: 90, bank_seconds_left: 420, grace_seconds_left: 90 },
      budget: { submit_within_ms: 60_000, policy_budget_ms: 59_000 },
      inherited: false,
      // Received at 0, decided at the snapshot; the choice then waits for the confirmation.
      elapsed_ms: 100,
    });
    expect(decision.legal.choices).toContain(decision.choice);
  });

  it("answers a move request after its snapshot, with R̂ from its own and the public counts", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    t.push(at(3));
    t.push(time(55, 400));
    const decision = run.decisions()[1] as DecisionRecord;
    expect(decision).toMatchObject({ request_seq: 2, channel_request_seq: 2, rqid: 3, phase: "move", inherited: false });
    expect(decision.remaining_decisions_estimate).toBe(14); // ⌈1.5 × (4 + 4)⌉ + 2 bench
    expect(decision.budget).toEqual({ submit_within_ms: 20_000, policy_budget_ms: 19_000 });
    expect(chooseLines(t).at(-1)).toBe(`${ROOM}|/choose ${decision.legal.choices[0]}|3`);
    // A snapshot with no request waiting changes nothing.
    t.push(time(55, 400));
    expect(run.decisions()).toHaveLength(2);
  });

  it("sends nothing for a wait request", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    const sent = t.sent.length;
    t.push(at(18));
    expect(t.sent).toHaveLength(sent);
  });

  it("answers a request at once with a zero budget when the timer is off", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(room("|inactiveoff|Battle timer is now OFF."));
    t.push(at(0));
    t.push(at(1));
    expect(run.decisions()[0]).toMatchObject({ timer: null, budget: { submit_within_ms: 0, policy_budget_ms: 0 } });
  });

  it("plays a pending candidate when all are pending, and answers the update after [Unavailable choice]", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    // Slot 2 may be trapped and may have its moves disabled: every candidate is pending.
    now = 1_000;
    t.push(
      edited(5, (r) => {
        (r.active as Record<string, unknown>[])[1] = { ...(r.active as object[])[1], maybeDisabled: true };
      }),
    );
    now = 1_200;
    t.push(time(55, 400));
    const pending = run.decisions()[1] as DecisionRecord;
    expect(pending.legal.choices).toEqual([]);
    expect(pending.choice).toBe(pending.legal.pending[0]);
    now = 1_500;
    t.push(room(`|error|[Unavailable choice] Can't switch: The active Pokémon is trapped`));
    expect(run.events("refusal")).toEqual([
      { event: "refusal", room: ROOM, refusal: { kind: "unavailable", choice: pending.choice } },
    ]);
    // The updated request is a new request, and inherits the receipt and budget.
    t.push(at(7));
    const update = run.decisions()[2] as DecisionRecord;
    expect(update).toMatchObject({ request_seq: 3, channel_request_seq: 3, rqid: 7, inherited: true, elapsed_ms: 500 });
    expect(update.budget).toEqual(pending.budget);
    expect(chooseLines(t).at(-1)).toBe(`${ROOM}|/choose ${update.choice}|7`);
  });

  it("gives an update request with no earlier decision a zero budget", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(2));
    t.push(at(7));
    expect(run.decisions()[0]).toMatchObject({ inherited: true, timer: null, budget: { policy_budget_ms: 0 } });
  });

  it("drops a stale choice and plays on", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    preview(t);
    t.push(serverFrame("nothing_to_choose"));
    expect(run.events("refusal")).toEqual([
      {
        event: "refusal",
        room: ROOM,
        refusal: { kind: "stale_choice", choice: run.decisions()[0]?.choice, reason: "refused_as_stale" },
      },
    ]);
    t.push(serverFrame("request_null"));
    t.push(serverFrame("too_late"));
    expect(run.events("refusal").at(-1)).toMatchObject({ refusal: { reason: "no_request" } });
    t.push(room("|win|p2"));
    await expect(run.result).resolves.toMatchObject({ battles: [{ result: "loss", winner: "p2" }] });
  });

  it("fails a harness run on an enumerator defect, forfeiting the battle", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    t.push(at(3));
    t.push(time(55, 400));
    t.push(room("|error|[Invalid choice] Can't move: Your Pikachu doesn't have a move 9"));
    await expect(run.result).rejects.toThrow(EnumeratorDefectError);
    expect(t.sent.at(-1)).toBe(`${ROOM}|/forfeit`);
    expect(t.closed).toBe(true);
  });

  it("in a live match sends the next candidate after a defect and marks the battle", async () => {
    const run = await start({ mode: "live_match" });
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    t.push(at(3));
    t.push(time(55, 400));
    const legal = (run.decisions()[1] as DecisionRecord).legal;
    t.push(room("|error|[Invalid choice] Can't move: Your Pikachu doesn't have a move 9"));
    expect(chooseLines(t).at(-1)).toBe(`${ROOM}|/choose ${legal.choices[1]}|3`);
    t.push(room("|error|[Invalid choice] Can't move: again"));
    expect(chooseLines(t).at(-1)).toBe(`${ROOM}|/choose ${legal.choices[2]}|3`);
    t.push(room(`|win|${BOT}`));
    const { battles } = await run.result;
    expect(battles[0]).toMatchObject({ result: "win", defect: true });
  });

  it("sends no next candidate after a defect revealed as [Unavailable choice]", async () => {
    const run = await start({ mode: "live_match" });
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    t.push(at(3));
    t.push(time(55, 400));
    const sent = t.sent.length;
    t.push(room("|error|[Unavailable choice] Can't move: something hidden"));
    expect(t.sent).toHaveLength(sent);
    expect(run.events("refusal").at(-1)).toMatchObject({ refusal: { kind: "defect" } });
  });

  it("fails on an |error| it cannot classify, or with no choice sent", async () => {
    const first = await start();
    login(first.t());
    preview(first.t());
    first.t().push(room("|error|/timer - something else"));
    await expect(first.result).rejects.toThrow(RefusalError);

    const second = await start();
    login(second.t());
    second.t().push(INIT);
    second.t().push(room("|error|[Invalid choice] There's nothing to choose"));
    await expect(second.result).rejects.toThrow(/with no choice sent/);
  });

  it("fails on a request the pinned decoder refuses, or without an rqid", async () => {
    const first = await start();
    login(first.t());
    first.t().push(INIT);
    first.t().push(room('|request|{"wait":true}'));
    await expect(first.result).rejects.toThrow();

    const second = await start();
    login(second.t());
    second.t().push(INIT);
    second.t().push(at(0));
    second.t().push(edited(1, (r) => delete r.rqid));
    await expect(second.result).rejects.toThrow(/without an rqid/);
  });

  it("records each dropped line once, though every build repeats it", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(room("|unknownkind|x"));
    t.push(at(0));
    t.push(at(1));
    t.push(time(90, 420, 90));
    t.push(CONFIRM);
    t.push(at(2));
    t.push(at(3));
    t.push(time(55, 400));
    expect(run.events("dropped")).toEqual([{ event: "dropped", room: ROOM, line: "|unknownkind|x" }]);
  });

  it("reads the foe's |teamsize| and |faint| lines, and falls back to the bot's own team size without a usable one", async () => {
    const estimate = async (edit: (frame: string) => string): Promise<number | undefined> => {
      const run = await start();
      login(run.t());
      run.t().push(INIT);
      run.t().push(at(0));
      run.t().push(at(1));
      run.t().push(time(90, 420, 90));
      run.t().push(CONFIRM);
      run.t().push(edit(at(2)));
      run.t().push(at(3));
      run.t().push(time(55, 400));
      return run.decisions()[1]?.remaining_decisions_estimate;
    };
    // ⌈1.5 × (own alive 4 + foe alive)⌉ + 2 bench
    expect(await estimate((f) => f)).toBe(14);
    expect(await estimate((f) => f.replace("|teamsize|p2|4", "|teamsize|p2|3"))).toBe(13);
    expect(await estimate((f) => f.replace("|teamsize|p2|4\n", ""))).toBe(14);
    expect(await estimate((f) => f.replace("|teamsize|p2|4", "|teamsize|p2|many"))).toBe(14);
    expect(await estimate((f) => f.replace("|teamsize|p2|4", "|teamsize|p2|"))).toBe(14);
    // One foe fainted before the request.
    expect(await estimate((f) => `${f}\n|faint|p2a: Gengar`)).toBe(13); // foe 4 - 1 = 3
  });

  it("ends on |win| or |tie|: leaves the room, records only its own channel and ignores the rest", async () => {
    const records: unknown[] = [];
    const run = await start({ onBattle: (record) => records.push(record) });
    const t = run.t();
    login(t);
    preview(t);
    t.push(at(2));
    t.push(room("|tie", "|raw|after the end"));
    expect(t.sent.at(-1)).toBe(`${ROOM}|/leave`);
    const { battles } = await run.result;
    expect(records).toEqual(battles);
    const [record] = battles;
    expect(record).toMatchObject({ result: "tie", winner: null, player: "p1", ots: "confirmed", ots_failure: null });
    expect(record?.channels).toHaveLength(1);
    expect(record?.channels[0]?.player).toBe("p1");
    expect(record?.channels[0]?.lines).toContain("|tie");
    expect(record?.channels[0]?.lines.some((line) => line.startsWith("|inactive|"))).toBe(false);
    expect(t.closed).toBe(true);
  });

  it("stops on abort and reports an open battle as unfinished", async () => {
    const controller = new AbortController();
    const run = await start({ signal: controller.signal });
    const t = run.t();
    login(t);
    preview(t);
    controller.abort();
    const { battles } = await run.result;
    expect(battles[0]).toMatchObject({ result: "unfinished" });
    expect(battles[0]?.channels).toHaveLength(1);
    expect(t.closed).toBe(true);
  });
});

describe("runA2Bot: after a battle", () => {
  it("does not forfeit an ended battle when the session fails, and a stop after the stop is a no-op", async () => {
    const controller = new AbortController();
    const run = await start({ battles: 2, signal: controller.signal });
    const t = run.t();
    login(t);
    preview(t);
    t.push(room("|win|p2"));
    t.push(room('|request|{"wait":true}'));
    await expect(run.result).rejects.toThrow();
    expect(t.sent.filter((line) => line.endsWith("/forfeit"))).toEqual([]);
    controller.abort();
  });

  it("stops once, though the signal aborts after its last battle", async () => {
    const controller = new AbortController();
    const run = await start({ signal: controller.signal });
    login(run.t());
    run.t().push(INIT);
    run.t().push(room("|tie"));
    controller.abort();
    await expect(run.result).resolves.toMatchObject({ battles: [{ result: "tie" }] });
  });
});

describe("runA2Bot: Open Team Sheets (ADR-005 §2)", () => {
  it("fails closed when the decline's deadline runs out while it is disconnected", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start({ reconnect: { attempts: 2, delayMs: 120_000 } });
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(1));
    t.push(time(90, 420, 90));
    t.end(1006);
    now = 59_000;
    vi.advanceTimersByTime(59_001);
    await expect(run.result).rejects.toMatchObject({ name: "OtsDeclineError", forfeitSent: false });
  });

  it("records one failure when |showteam| comes before the preview choice", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(1));
    t.push(room("|showteam|p2|Gengar||focussash|||||||50|"));
    t.push(time(90, 420, 90));
    expect(run.events("ots_failure")).toEqual([expect.objectContaining({ reason: "showteam" })]);
    expect(chooseLines(t)).toEqual([]);
    expect(t.sent.some((line) => line.endsWith("/rejectopenteamsheets"))).toBe(false);
  });

  it("forfeits when the rejection is not confirmed within the preview policy budget", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(1));
    t.push(time(90, 420, 90));
    expect(chooseLines(t)).toEqual([]);
    now = 59_000;
    vi.advanceTimersByTime(59_001);
    expect(t.sent.at(-1)).toBe(`${ROOM}|/forfeit`);
    expect(run.events("ots_failure")).toEqual([
      expect.objectContaining({ event: "ots_failure", reason: "not_confirmed" }),
    ]);
    // A late confirmation releases nothing.
    t.push(CONFIRM);
    expect(chooseLines(t)).toEqual([]);
    t.push(room("|win|p2"));
    const { battles } = await run.result;
    expect(battles[0]).toMatchObject({ result: "loss", ots: "forfeited", ots_failure: { reason: "not_confirmed" } });
  });

  it("forfeits on |showteam| and on a refused rejection, recording the first failure", async () => {
    const run = await start();
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(1));
    t.push(time(90, 420, 90));
    t.push(room("|error|You can't reject open team sheets now"));
    t.push(room("|showteam|p2|Gengar||focussash|||||||50|"));
    expect(t.sent.filter((line) => line === `${ROOM}|/forfeit`)).toHaveLength(1);
    expect(run.events("ots_failure")).toEqual([expect.objectContaining({ reason: "refused" })]);
  });
});

describe("runA2Bot: reconnects (ADR-005 §1, §3)", () => {
  /** A battle whose move request rqid 3 the bot has answered on its first connection. */
  async function answeredThenDropped(): Promise<{ run: Run; choice: string }> {
    const run = await start();
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    run.t().push(at(3));
    run.t().push(time(55, 400));
    const choice = (run.decisions()[1] as DecisionRecord).choice;
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    // The server rejoins the bot: |init| and the room log, then the stored request.
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(2));
    return { run, choice };
  }

  it("does not choose again for a re-sent request the server holds a choice for", async () => {
    const { run, choice } = await answeredThenDropped();
    expect(run.events("disconnected")).toEqual([{ event: "disconnected", code: 1006 }]);
    expect(run.events("connected")).toHaveLength(2);
    run.t().push(`${at(3)}\n|sentchoice|${choice}`);
    expect(chooseLines(run.t())).toEqual([]);
    expect(run.events("sentchoice")).toEqual([{ event: "sentchoice", room: ROOM, request_seq: 1, choice }]);
    // One OtsDecline per room: the replayed log confirms nothing and nothing is rejected again.
    expect(run.t().sent.filter((line) => line.endsWith("/rejectopenteamsheets"))).toEqual([]);
    // A stale refusal is still classified against the renumbered submission.
    run.t().push(serverFrame("too_late"));
    expect(run.events("refusal").at(-1)).toMatchObject({ refusal: { reason: "refused_as_stale" } });
    run.t().push(room("|tie"));
    const { battles } = await run.result;
    expect(battles[0]?.channels).toHaveLength(2);
    expect(battles[0]?.ots).toBe("confirmed");
  });

  it("sends its choice again when the re-sent request comes without |sentchoice|", async () => {
    const { run, choice } = await answeredThenDropped();
    run.t().push(at(3));
    expect(chooseLines(run.t())).toEqual([`${ROOM}|/choose ${choice}|3`]);
    expect(run.events("resent_choice")).toEqual([{ event: "resent_choice", room: ROOM, request_seq: 1, choice }]);
  });

  it("decides at once, from the original receipt, a request whose snapshot went to the old connection", async () => {
    const run = await start();
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    now = 2_000;
    run.t().push(at(3));
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(2));
    now = 2_400;
    run.t().push(at(3));
    expect(run.decisions()[1]).toMatchObject({ rqid: 3, request_seq: 1, timer: null, elapsed_ms: 400, rejoin_without_snapshot: true });
    expect(run.events("incident")).toEqual([{ event: "incident", room: ROOM, kind: "rejoin_without_timer_snapshot", request_seq: 1 }]);
    expect(chooseLines(run.t())).toHaveLength(1);
  });

  it("decides at once a request it first sees on a rejoin, and ignores a re-sent wait", async () => {
    const { run } = await answeredThenDropped();
    run.t().push(at(10));
    expect(run.decisions()[2]).toMatchObject({ rqid: 8, phase: "forceSwitch", timer: null, request_seq: 1, rejoin_without_snapshot: true });
    run.t().push(at(18));
    run.t().push(at(18));
    expect(chooseLines(run.t())).toHaveLength(1);
  });

  it("starts a rejoin from the timer state it knew, and lets the replayed log correct it", async () => {
    const run = await start();
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    // The rejoin's log carries no timer line: the timer is still taken as on.
    run.t().push(room("|init|battle", "|title|p1 vs. p2"));
    run.t().push(at(0));
    run.t().push(at(2));
    run.t().push(at(3));
    expect(run.decisions()).toHaveLength(2);
    run.t().push(at(4));
    run.t().push(at(5));
    // A new request waits for its snapshot: no choice yet.
    expect(run.decisions()).toHaveLength(2);
    run.t().push(time(50, 400));
    expect(run.decisions()[2]).toMatchObject({ timer: { turn_seconds_left: 50 } });
    // The server says it is off now: the next request is answered at once.
    run.t().push(room("|inactiveoff|Battle timer is now OFF."));
    run.t().push(at(9));
    run.t().push(at(10));
    expect(run.decisions()[3]).toMatchObject({ timer: null });
  });

  it("fails the session when the log callback throws", async () => {
    const run = await start({
      onLog: (entry) => {
        if (entry.event === "connected") throw new Error("log sink closed");
      },
    });
    await expect(run.result).rejects.toThrow("log sink closed");
  });

  it("removes its abort listener when the session ends", async () => {
    const controller = new AbortController();
    const remove = vi.spyOn(controller.signal, "removeEventListener");
    const run = await start({ signal: controller.signal });
    login(run.t());
    run.t().push(INIT);
    run.t().push(room("|tie"));
    await run.result;
    expect(remove).toHaveBeenCalledWith("abort", expect.any(Function));
  });

  it("retries a held name and gives up after its attempts", async () => {
    const run = await start();
    run.t().push("|nametaken|Areneon-bot-0000|Someone is already using the name");
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    run.t().push("|nametaken|Areneon-bot-0000|Someone is already using the name");
    await vi.waitFor(() => expect(run.transports).toHaveLength(3));
    run.t().end(1006);
    await expect(run.result).rejects.toThrow(/no connection after 2 attempts/);
  });

  it("gives up when it cannot connect", async () => {
    const connect = vi.fn(() => Promise.reject(new Error("refused")));
    await expect(runA2Bot({ ...base(), connect, reconnect: { attempts: 1, delayMs: 1 } })).rejects.toThrow(
      SessionError,
    );
    expect(connect).toHaveBeenCalledTimes(2);
  });

  it("closes a connection that opens after the stop", async () => {
    const controller = new AbortController();
    const transport = memoryTransport();
    let open: (t: MemoryTransport) => void = () => {};
    const connect = vi.fn(() => new Promise<MemoryTransport>((done) => (open = done)));
    const run = await start({ signal: controller.signal, connect });
    await vi.waitFor(() => expect(connect).toHaveBeenCalledTimes(1));
    controller.abort();
    await expect(run.result).resolves.toEqual({ battles: [] });
    open(transport);
    await vi.waitFor(() => expect(transport.closed).toBe(true));
  });

  it("gives up on a connection that fails after the stop", async () => {
    const controller = new AbortController();
    let fail: (error: Error) => void = () => {};
    const connect = vi.fn(() => new Promise<MemoryTransport>((_done, reject) => (fail = reject)));
    const run = await start({ signal: controller.signal, connect });
    await vi.waitFor(() => expect(connect).toHaveBeenCalledTimes(1));
    controller.abort();
    fail(new Error("refused"));
    await expect(run.result).resolves.toEqual({ battles: [] });
    expect(connect).toHaveBeenCalledTimes(1);
  });

  it("connects over WebSocket by default, and retries 5 times", async () => {
    const port = await freePort();
    const options = { ...base(), url: `ws://127.0.0.1:${port}/showdown/websocket` };
    delete (options as Partial<A2BotOptions>).connect;
    await expect(runA2Bot(options)).rejects.toThrow(/no connection after 5 attempts/);
  }, 20_000);

  it("takes a fatal error while a reconnect is pending as the end: no socket opens afterwards", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start({ reconnect: { attempts: 2, delayMs: 120_000 } });
    const t = run.t();
    login(t);
    t.push(INIT);
    t.push(at(0));
    t.push(at(1));
    t.push(time(90, 420, 90));
    t.end(1006);
    // The first retry is at once; the second would wait the 120 s.
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    run.t().end(1006);
    expect(run.transports).toHaveLength(2);
    now = 59_000;
    vi.advanceTimersByTime(59_001);
    await expect(run.result).rejects.toMatchObject({ name: "OtsDeclineError" });
    vi.advanceTimersByTime(300_000);
    expect(run.transports).toHaveLength(2);
  });

  it("stops while it waits to reconnect", async () => {
    const controller = new AbortController();
    const transports: MemoryTransport[] = [];
    const connect = vi.fn(async () => {
      if (transports.length > 0) throw new Error("down");
      const transport = memoryTransport();
      transports.push(transport);
      return transport;
    });
    const run = await start({ signal: controller.signal, reconnect: { attempts: 3, delayMs: 60_000 }, connect });
    await vi.waitFor(() => expect(transports).toHaveLength(1));
    transports[0]?.end(1006);
    await vi.waitFor(() => expect(connect).toHaveBeenCalledTimes(2));
    controller.abort();
    await expect(run.result).resolves.toEqual({ battles: [] });
    expect(connect).toHaveBeenCalledTimes(2);
  });
});

describe("runA2Bot: a rejoin of a room whose battle already ended", () => {
  /** The room log as the server replays it on a join: the battle frames without the bot's own requests and errors (the server sends those to it alone). */
  const replay = battle.frames.filter((frame) => !frame.includes("|request|") && !frame.includes("|error|"));
  const END = replay.length - 1;

  /** The bot has answered request 3 on its first connection when the connection drops; it logs in again. */
  async function dropped(): Promise<{ run: Run; records: A2BattleRecord[] }> {
    const records: A2BattleRecord[] = [];
    const run = await start({ onBattle: (record) => records.push(record) });
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    run.t().push(at(3));
    run.t().push(time(55, 400));
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    return { run, records };
  }

  it("joins its unfinished room on login, records the end from the replay and leaves", async () => {
    const { run, records } = await dropped();
    // The server does not rejoin a user to a battle that ended while it was away: only a /join tells.
    expect(run.t().sent.at(-1)).toBe(`|/join ${ROOM}`);
    run.t().push(INIT);
    for (const frame of replay.slice(0, END)) run.t().push(frame);
    expect(run.events("battle_ended")).toEqual([]);
    run.t().push(replay[END] as string);
    expect(run.t().sent.filter((line) => line === `${ROOM}|/leave`)).toHaveLength(1);
    const { battles } = await run.result;
    expect(records).toEqual(battles);
    const [record] = battles as [A2BattleRecord];
    expect(record).toMatchObject({ result: "loss", winner: "p1", player: "p1", fatal: null });
    expect(record.channels).toHaveLength(2);
    expect(record.channels[0]?.lines.some((line) => line.startsWith("|win|"))).toBe(false);
    expect(record.channels[1]?.lines).toContain("|win|p1");
    // The two channels join (fail closed otherwise) into the one the Episode stores.
    expect(record.channels.reduce(continueChannel).lines).toContain("|win|p1");
    expect(record.decisions.map((d) => d.rqid)).toEqual([1, 3]);
    expect(run.events("battle_ended")).toEqual([{ event: "battle_ended", room: ROOM, result: "loss", winner: "p1" }]);
  });

  it("records a tie from the replay", async () => {
    const { run } = await dropped();
    run.t().push(INIT);
    for (const frame of replay.slice(0, END)) run.t().push(frame);
    run.t().push(room("|tie"));
    const { battles } = await run.result;
    expect(battles[0]).toMatchObject({ result: "tie", winner: null });
  });

  it("records the result of a battle that ended before the bot's first request, with no channel to keep", async () => {
    const run = await start();
    login(run.t());
    run.t().push(INIT);
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    expect(run.t().sent.at(-1)).toBe(`|/join ${ROOM}`);
    run.t().push(INIT);
    run.t().push(room("|win|p2"));
    const { battles } = await run.result;
    expect(battles[0]).toMatchObject({ result: "loss", player: null, channels: [] });
  });

  it("fails closed on a replay that does not continue the earlier channel", async () => {
    const { run, records } = await dropped();
    run.t().push(INIT);
    // The log of the earlier connection is missing from this replay.
    run.t().push(room("|win|p1"));
    await expect(run.result).rejects.toThrow();
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ result: "unfinished", fatal: { channel_complete: false } });
  });

  it("fails closed when the server refuses the join of an unfinished battle", async () => {
    const { run, records } = await dropped();
    run.t().push(room("|noinit|nonexistent|The room \"battle\" does not exist."));
    await expect(run.result).rejects.toThrow(SessionError);
    expect(records[0]).toMatchObject({ result: "unfinished", fatal: { error: "SessionError" } });
  });
});

describe("runA2Bot: negative controls for a rejoin without a timer snapshot and for an ended room", () => {
  const NO_TIMER_INIT = room("|init|battle", "|title|p1 vs. p2", `|j| ${BOT}`);

  it("marks no decision and logs no incident when the timer is off, on a rejoin or not", async () => {
    const run = await start();
    login(run.t());
    run.t().push(NO_TIMER_INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    run.t().push(CONFIRM);
    run.t().push(at(2));
    // A first decision with no snapshot, the timer off.
    expect(run.decisions()[0]).toMatchObject({ timer: null });
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    run.t().push(NO_TIMER_INIT);
    run.t().push(at(0));
    run.t().push(at(2));
    // A request first seen on the rejoin, the timer off: no snapshot would have come anyway.
    run.t().push(at(3));
    expect(run.decisions()).toHaveLength(2);
    expect(run.decisions()[1]).toMatchObject({ rqid: 3, timer: null });
    for (const decision of run.decisions()) expect(decision.rejoin_without_snapshot).toBeUndefined();
    expect(run.events("incident")).toEqual([]);
  });

  it("ignores a |noinit| for a battle that has ended, and sends no /join for it after a reconnect", async () => {
    const run = await start({ battles: 2 });
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    run.t().push(room("|tie"));
    expect(run.events("battle_ended")).toHaveLength(1);
    run.t().push(room("|noinit|nonexistent|The room does not exist."));
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    expect(run.t().sent.filter((line) => line.startsWith("|/join"))).toEqual([]);
    run.t().push(room("|noinit|nonexistent|The room does not exist."));
    const settled = await Promise.race([run.result.then(() => "resolved"), new Promise((resolve) => setTimeout(() => resolve("pending"), 50))]);
    expect(settled).toBe("pending");
    run.t().end(1000);
  });
});

describe("runA2Bot: reconnect during Team Preview (ADR-005 §1, §2)", () => {
  const rejects = (run: Run): number =>
    run.transports.flatMap((t) => t.sent).filter((line) => line.endsWith("/rejectopenteamsheets")).length;
  const forfeits = (run: Run): number =>
    run.transports.flatMap((t) => t.sent).filter((line) => line.endsWith("/forfeit")).length;

  /** The server drops the bot; it logs in again and is rejoined with the room log up to the preview request. */
  async function rejoin(run: Run): Promise<void> {
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
  }

  it("keeps the one rejection and the choice when the confirmation is replayed on the rejoin", async () => {
    const run = await start();
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    run.t().push(time(90, 420, 90));
    expect(rejects(run)).toBe(1);
    await rejoin(run);
    // The rejection reached the server: its confirmation is in the replayed log.
    run.t().push(CONFIRM);
    run.t().push(at(1));
    expect(rejects(run)).toBe(1);
    expect(forfeits(run)).toBe(0);
    expect(run.events("ots_failure")).toEqual([]);
    // The held choice goes out on the confirmation, and again with the re-sent request
    // (the server answers the second as stale; refusal.ts classifies it).
    expect(chooseLines(run.t())).toHaveLength(2);
    run.t().push(room("|tie"));
    await expect(run.result).resolves.toMatchObject({ battles: [{ ots: "confirmed", ots_failure: null }] });
  });

  it("sends nothing again when the server holds the preview choice", async () => {
    const run = await start();
    login(run.t());
    preview(run.t());
    const choice = (run.decisions()[0] as DecisionRecord).choice;
    await rejoin(run);
    run.t().push(CONFIRM);
    run.t().push(`${at(1)}\n|sentchoice|${choice}`);
    expect(rejects(run)).toBe(1);
    expect(forfeits(run)).toBe(0);
    expect(chooseLines(run.t())).toEqual([]);
    expect(run.events("sentchoice")).toHaveLength(1);
  });

  /** The connection drops after the preview request, before its timer snapshot: the rejoin re-sends the request. */
  async function droppedBeforeSnapshot(run: Run): Promise<void> {
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    expect(rejects(run)).toBe(0);
    // The snapshot went with the old connection, and the request's receipt is long past.
    now = 30_000;
    await rejoin(run);
    run.t().push(at(1));
  }

  it("declines on the rejoin with a real budget when the drop came before the preview snapshot", async () => {
    // #807: the re-sent request has no snapshot, so its decision budget is 0, but the decline
    // gets `declineWaitMs` from the rejoin instead of a deadline that passed before it was sent.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start();
    await droppedBeforeSnapshot(run);
    expect(rejects(run)).toBe(1);
    // Fail closed: the choice is held until the bot's own confirmation line.
    expect(chooseLines(run.t())).toEqual([]);
    now = 34_999;
    run.t().push(CONFIRM);
    expect(forfeits(run)).toBe(0);
    expect(run.events("ots_failure")).toEqual([]);
    const decision = run.decisions()[0] as DecisionRecord;
    expect(decision).toMatchObject({ timer: null, budget: { submit_within_ms: 0, policy_budget_ms: 0 }, elapsed_ms: 30_000 });
    expect(chooseLines(run.t())).toEqual([`${ROOM}|/choose ${decision.choice}|1`]);
    // The battle goes on.
    run.t().push(at(2));
    run.t().push(room("|tie"));
    await expect(run.result).resolves.toMatchObject({ battles: [{ ots: "confirmed", ots_failure: null }] });
  });

  it("pins the default 5 s window from the rejoin: confirmed just inside, forfeited at the boundary", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const inside = await start();
    await droppedBeforeSnapshot(inside);
    now = 34_999;
    vi.advanceTimersByTime(4_999);
    inside.t().push(CONFIRM);
    expect(forfeits(inside)).toBe(0);
    expect(chooseLines(inside.t())).toHaveLength(1);

    const boundary = await start();
    await droppedBeforeSnapshot(boundary);
    now = 35_000;
    boundary.t().push(CONFIRM);
    expect(forfeits(boundary)).toBe(1);
    expect(chooseLines(boundary.t())).toEqual([]);
  });

  it("forfeits by its own timer when no frame comes at all within the default window", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start();
    await droppedBeforeSnapshot(run);
    now = 34_999;
    vi.advanceTimersByTime(5_000);
    expect(forfeits(run)).toBe(0);
    now = 35_000;
    vi.advanceTimersByTime(1);
    expect(forfeits(run)).toBe(1);
  });

  it("still forfeits when the rejoin decline is never confirmed within its window", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start({ declineWaitMs: 2_000 });
    await droppedBeforeSnapshot(run);
    now = 31_999;
    vi.advanceTimersByTime(1_999);
    expect(forfeits(run)).toBe(0);
    now = 32_001;
    vi.advanceTimersByTime(2);
    expect(forfeits(run)).toBe(1);
    expect(run.events("ots_failure")).toEqual([expect.objectContaining({ reason: "not_confirmed" })]);
    // A late confirmation releases nothing.
    run.t().push(CONFIRM);
    expect(chooseLines(run.t())).toEqual([]);
    run.t().push(room("|win|p2"));
    await expect(run.result).resolves.toMatchObject({ battles: [{ ots: "forfeited", ots_failure: { reason: "not_confirmed" } }] });
  });

  it("forfeits a rejoin decline whose confirmation comes after the window", async () => {
    const run = await start({ declineWaitMs: 2_000 });
    await droppedBeforeSnapshot(run);
    now = 32_000;
    run.t().push(CONFIRM);
    expect(forfeits(run)).toBe(1);
    expect(run.events("ots_failure")).toEqual([expect.objectContaining({ reason: "not_confirmed" })]);
    expect(chooseLines(run.t())).toEqual([]);
  });

  it("gives the decline the same window when the watchdog answers a preview whose snapshot never came", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start();
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    now = 1_000;
    vi.advanceTimersByTime(1_000);
    expect(run.events("watchdog")).toHaveLength(1);
    expect(rejects(run)).toBe(1);
    now = 5_999;
    run.t().push(CONFIRM);
    expect(forfeits(run)).toBe(0);
    expect(chooseLines(run.t())).toHaveLength(1);
    expect(run.decisions()[0]).toMatchObject({ watchdog: "snapshot_missing", timer: null });
  });

  it("forfeits a watchdog-answered preview whose confirmation comes at the end of the window", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const run = await start();
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    now = 1_000;
    vi.advanceTimersByTime(1_000);
    now = 6_000;
    run.t().push(CONFIRM);
    expect(forfeits(run)).toBe(1);
    expect(chooseLines(run.t())).toEqual([]);
  });

  it("refuses a decline window that is not a positive whole number of ms", async () => {
    await expect(runA2Bot({ ...base(), declineWaitMs: 0 })).rejects.toThrow(RangeError);
    await expect(runA2Bot({ ...base(), declineWaitMs: 1.5 })).rejects.toThrow(RangeError);
  });
});

describe("runA2Bot: a fatal error keeps the records of its battles (#807)", () => {
  it("hands each unfinished battle to onBattle, with its channel, before it rejects", async () => {
    const records: A2BattleRecord[] = [];
    const run = await start({ onBattle: (record) => records.push(record) });
    login(run.t());
    preview(run.t());
    run.t().push(at(2));
    run.t().push(room("|error|/timer - something else"));
    await expect(run.result).rejects.toThrow(RefusalError);
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      room: ROOM,
      player: "p1",
      result: "unfinished",
      ots: "confirmed",
      fatal: { error: "RefusalError", channel_complete: true },
    });
    expect(records[0]?.decisions).toHaveLength(1);
    expect(records[0]?.channels).toHaveLength(1);
    expect(records[0]?.channels[0]?.lines.some((line) => line.startsWith("|request|"))).toBe(true);
  });

  it("keeps a forfeited decline's record, and reports a channel it could not build", async () => {
    const records: A2BattleRecord[] = [];
    const run = await start({ onBattle: (record) => records.push(record) });
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    run.t().push(time(90, 420, 90));
    run.t().push(room("|showteam|p2|Gengar||focussash|||||||50|"));
    // A request for the other seat: the channel refuses to be built from it, which is the fatal error.
    run.t().push(
      edited(3, (request) => {
        (request.side as { id: string }).id = "p2";
      }),
    );
    await expect(run.result).rejects.toThrow();
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ ots: "forfeited", ots_failure: { reason: "showteam" } });
    expect(records[0]?.fatal).toMatchObject({ channel_complete: false });
  });

  it("stores a battle that has no request yet as an empty record, and not the battles that ended", async () => {
    const records: A2BattleRecord[] = [];
    const run = await start({
      battles: 2,
      reconnect: { attempts: 1, delayMs: 0 },
      onBattle: (record) => records.push(record),
    });
    login(run.t());
    preview(run.t());
    run.t().push(room("|tie"));
    expect(records).toHaveLength(1);
    // A second battle joins, then the server goes away for good.
    const second = ">battle-gen9championsvgc2026regmc-2";
    run.t().push(`${second}\n|init|battle\n|title|p1 vs. p2`);
    run.t().end(1006);
    await vi.waitFor(() => expect(run.transports).toHaveLength(2));
    run.t().end(1006);
    await expect(run.result).rejects.toThrow(SessionError);
    expect(records).toHaveLength(2);
    expect(records[1]).toMatchObject({
      room: "battle-gen9championsvgc2026regmc-2",
      player: null,
      channels: [],
      decisions: [],
      fatal: { error: "SessionError", channel_complete: true },
    });
  });

  it("names a fatal value that is not an Error as unknown", async () => {
    const records: A2BattleRecord[] = [];
    const run = await start({
      onBattle: (record) => records.push(record),
      onLog: (entry) => {
        if (entry.event === "decision") throw "not an error";
      },
    });
    login(run.t());
    run.t().push(INIT);
    run.t().push(at(0));
    run.t().push(at(1));
    run.t().push(time(90, 420, 90));
    await expect(run.result).rejects.toBe("not an error");
    expect(records[0]?.fatal).toEqual({ error: "unknown", channel_complete: true });
  });

  it("still hands over a battle whose end failed half-way (the room could not be left)", async () => {
    const records: A2BattleRecord[] = [];
    const transports: MemoryTransport[] = [];
    const run = await start({
      onBattle: (record) => records.push(record),
      connect: async () => {
        const transport = memoryTransport();
        const send = transport.send.bind(transport);
        transport.send = (line: string) => {
          if (line.endsWith("/leave")) throw new Error("socket gone");
          send(line);
        };
        transports.push(transport);
        return transport;
      },
    });
    await vi.waitFor(() => expect(transports).toHaveLength(1));
    const t = transports[0] as MemoryTransport;
    login(t);
    preview(t);
    t.push(room("|win|p2"));
    await expect(run.result).rejects.toThrow("socket gone");
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ result: "unfinished", fatal: { error: "Error" } });
    expect(records[0]?.channels).toHaveLength(1);
  });

  it("rejects with the fatal error even when onBattle throws", async () => {
    const run = await start({
      onBattle: () => {
        throw new Error("sink");
      },
    });
    login(run.t());
    preview(run.t());
    run.t().push(room("|error|/timer - something else"));
    await expect(run.result).rejects.toThrow(RefusalError);
  });
});
