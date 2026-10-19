import { EnumeratorDefectError, loadEngine, loadRulesSnapshot, UnsupportedFormatError } from "@areneon/showdown-bridge";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
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
    data_class: "synthetic",
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
    expect(run.decisions()[1]).toMatchObject({ rqid: 3, request_seq: 1, timer: null, elapsed_ms: 400 });
    expect(chooseLines(run.t())).toHaveLength(1);
  });

  it("decides at once a request it first sees on a rejoin, and ignores a re-sent wait", async () => {
    const { run } = await answeredThenDropped();
    run.t().push(at(10));
    expect(run.decisions()[2]).toMatchObject({ rqid: 8, phase: "forceSwitch", timer: null, request_seq: 1 });
    run.t().push(at(18));
    run.t().push(at(18));
    expect(chooseLines(run.t())).toHaveLength(1);
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
