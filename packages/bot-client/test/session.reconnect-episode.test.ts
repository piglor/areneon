import {
  type Actor,
  actorRule,
  continueChannel,
  type Episode,
  episodeFromRecord,
  type LegalChoiceSet,
  loadEngine,
  loadRulesSnapshot,
  observe,
  type Observation,
  playerChannel,
  type PlayerChannel,
  readEpisode,
  resolveFormat,
  runBattle,
  stampTeam,
  type StampedTeam,
  teamHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { type HarnessFixture, type Seat } from "@areneon/local-server/harness";
import { describe, expect, it, vi } from "vitest";
import {
  a2Episode,
  A2EpisodeError,
  type A2BattleRecord,
  type A2SessionResult,
  memoryTransport,
  type MemoryTransport,
  runA2Bot,
  type SessionLog,
  timerRulesOf,
} from "../src/index.ts";
import { comparable, readFixture, SEATS, withoutRqid } from "./a1-a2-equality.ts";

// ADR-008 §3-5, ADR-005 gate 3: the real session path through a reconnect. The
// scripted server pushes the frames a recorded bot received (the committed
// bot-vs-bot fixtures, each with a real reconnect), connection by connection,
// to the session, which must observe, decide and write an Episode that
// `readEpisode` accepts and that equals the A1 Episode of the same battle.
// The choices the session makes are not scripted: the server frames are.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const FORMAT_DIR = "gen9championsvgc2026regmc";
const WITH_SENTCHOICE = `${FORMAT_DIR}/bot-vs-bot-3.json`;
const WITHOUT_SNAPSHOT = `${FORMAT_DIR}/bot-vs-bot-4.json`;

const seatOf = (fixture: HarnessFixture): Seat => SEATS.find((seat) => fixture.frames[seat].length > 1) as Seat;
const rivalOf = (seat: Seat): Seat => (seat === "p1" ? "p2" : "p1");

interface Played {
  fixture: HarnessFixture;
  seat: Seat;
  /** The session's result, or the error it ended with. */
  outcome: { result: A2SessionResult } | { error: unknown };
  records: A2BattleRecord[];
  logs: SessionLog[];
  transports: MemoryTransport[];
}

/**
 * Plays `connections` (frame lists) to a session for `seat` of `fixture`, one
 * connection after the other: a connection's frames, then the server closes it
 * and the bot logs in again on the next. Stops pushing once the session has
 * ended or failed.
 */
async function play(fixture: HarnessFixture, connections: readonly (readonly string[])[], seat = seatOf(fixture)): Promise<Played> {
  const transports: MemoryTransport[] = [];
  const logs: SessionLog[] = [];
  const records: A2BattleRecord[] = [];
  const run = runA2Bot({
    engine,
    snapshot,
    url: "ws://127.0.0.1:8000/showdown/websocket",
    name: fixture.players[seat].name,
    challenger: fixture.players[rivalOf(seat)].name,
    format: fixture.format_id,
    team: structuredClone(fixture.teams[seat]),
    known_synthetic: new Set(SEATS.map((who) => teamHash(fixture.teams[who]))),
    opponent_team: structuredClone(fixture.teams[rivalOf(seat)]),
    battles: 1,
    clock: () => 0,
    reconnect: { attempts: 2, delayMs: 5 },
    connect: async () => {
      const transport = memoryTransport();
      transports.push(transport);
      return transport;
    },
    onLog: (entry) => logs.push(entry),
    onBattle: (record) => records.push(record),
  });
  const settled = run.then(
    (result) => ({ result }),
    (error: unknown) => ({ error }),
  );
  for (const [index, frames] of connections.entries()) {
    await vi.waitFor(() => expect(transports).toHaveLength(index + 1));
    const t = transports[index] as MemoryTransport;
    t.push("|challstr|4|0123abcd");
    t.push(`|updateuser| ${fixture.players[seat].name}|1|1|{"blockChallenges":false}`);
    for (const frame of frames) {
      if (t.closed) break;
      t.push(frame);
    }
    if (index < connections.length - 1 && !t.closed) t.end(1006);
  }
  return { fixture, seat, outcome: await settled, records, logs, transports };
}

/**
 * The A1 run of the fixture's battle: its own server seed, names and teams, both
 * seats played by the shared actor rule (as the recorded bots were), not by a
 * replay of what the server logged.
 */
const actor: Actor = (request) => actorRule(request.legal as LegalChoiceSet);

async function a1Episode(fixture: HarnessFixture, seat: Seat, synthetic: Set<string>): Promise<Episode> {
  // The sets as the engine hands them to the player (the validator's defaults filled in): what an Episode stores.
  const resolved = resolveFormat(engine, fixture.format_id, snapshot);
  const filled = structuredClone(fixture.served_teams);
  for (const who of SEATS) validateTeam(engine, resolved, filled[who]);
  const known = new Set([...synthetic, ...SEATS.map((who) => teamHash(filled[who]))]);
  const record = await runBattle({
    engine,
    snapshot,
    formatId: fixture.format_id,
    seed: fixture.seed,
    teams: filled,
    players: {
      p1: { name: fixture.players.p1.name, avatar: fixture.players.p1.avatar },
      p2: { name: fixture.players.p2.name, avatar: fixture.players.p2.avatar },
    },
    actors: { p1: actor, p2: actor },
  });
  return episodeFromRecord(record, seat, {
    team: stampTeam(filled[seat] as StampedTeam["team"], known),
    opponent_team: stampTeam(filled[rivalOf(seat)] as StampedTeam["team"], known),
    known_synthetic: known,
    behaviour_policy_id: "actor-rule@a2",
  });
}

/** The observation of each decision, built from the Episode's own channel cut at the decision's request. */
async function observations(episode: Episode): Promise<Observation[]> {
  return await Promise.all(
    episode.decisions.map((decision) => {
      let seen = 0;
      const end = episode.channel.findIndex((line) => line.startsWith("|request|") && ++seen === decision.identity.request_seq);
      expect(end).toBeGreaterThanOrEqual(0);
      const channel = playerChannel({ schema_version: 1, data_class: episode.data_class, player: episode.player, lines: episode.channel.slice(0, end + 1) });
      return observe({ channel, team: episode.team, manifest: episode.provenance.manifest });
    }),
  );
}

const knownOf = (fixture: HarnessFixture): Set<string> =>
  new Set(SEATS.flatMap((seat) => [teamHash(fixture.teams[seat]), teamHash(fixture.served_teams[seat])]));

const optionsFor = (fixture: HarnessFixture, seat: Seat) => {
  const known = knownOf(fixture);
  return {
    team: stampTeam(fixture.teams[seat] as StampedTeam["team"], known),
    opponent_team: stampTeam(fixture.served_teams[rivalOf(seat)] as StampedTeam["team"], known),
    known_synthetic: known,
    manifest: fixture.manifest,
    timer_rules: timerRulesOf(engine, fixture.format_id, 5),
    behaviour_policy_id: "actor-rule@a2",
    collection: { kind: "league" as const, run_id: "reconnect-test", seed: null, opponent_policy_id: "actor-rule@a2" },
  };
};

const finished = (played: Played): A2BattleRecord => {
  if (!("result" in played.outcome)) throw played.outcome.error;
  return played.outcome.result.battles[0] as A2BattleRecord;
};

/** The channels of the record's connections joined, the way an Episode stores them. */
const joined = (record: A2BattleRecord): PlayerChannel => record.channels.reduce(continueChannel);

/** What A1 and the session agree on, from an A1 Episode and the session's record. */
function expectRecordEqualsA1(record: A2BattleRecord, a1: Episode): void {
  const channel = joined(record);
  expect(channel.lines.map(withoutRqid)).toEqual(a1.channel.map(withoutRqid));
  expect(record.decisions).toHaveLength(a1.decisions.length);
  record.decisions.forEach((decision, i) => {
    const expected = a1.decisions[i] as Episode["decisions"][number];
    expect(decision.channel_request_seq).toBe(expected.identity.request_seq);
    expect(decision.request_hash).toBe(expected.identity.request_hash);
    expect(decision.choice).toBe(expected.choice);
  });
}

describe.each([WITH_SENTCHOICE, WITHOUT_SNAPSHOT])("the session through a recorded reconnect: %s", (path) => {
  const fixture = readFixture(path);
  const seat = seatOf(fixture);
  const frames = fixture.frames[seat];

  it("observes and decides on the joined channel: the record equals A1's decisions on its one channel", async () => {
    const played = await play(fixture, frames);
    const record = finished(played);
    expect(played.transports).toHaveLength(2);
    expect(played.logs.filter((entry) => entry.event === "logged_in")).toHaveLength(2);
    expect(record.channels).toHaveLength(2);
    expect(record.fatal).toBeNull();
    expect(record.result).not.toBe("unfinished");
    // The rejoin's own channel restarts; the decisions are numbered on the joined one, strictly rising.
    const numbers = record.decisions.map((d) => d.channel_request_seq);
    expect(numbers).toEqual([...numbers].sort((a, b) => a - b));
    expect(new Set(numbers).size).toBe(numbers.length);
    expectRecordEqualsA1(record, await a1Episode(fixture, seat, knownOf(fixture)));
  }, 60_000);
});

describe("the session's Episode after a reconnect", () => {
  const fixture = readFixture(WITH_SENTCHOICE);
  const seat = seatOf(fixture);

  it("is accepted by readEpisode and equals the A1 Episode: channel, decisions, observations, result", async () => {
    const record = finished(await play(fixture, fixture.frames[seat]));
    const options = optionsFor(fixture, seat);
    const a2 = await a2Episode(record, options);
    const a1 = await a1Episode(fixture, seat, knownOf(fixture));

    expect(a2.provenance.runtime).toBe("A2");
    expect(a2.player).toBe(a1.player);
    expect(a2.channel.map(withoutRqid)).toEqual(a1.channel.map(withoutRqid));
    expect(a2.result).toEqual(a1.result);
    expect(a2.team.team.map((set) => [set.species, set.item, set.ability, set.moves])).toEqual(
      a1.team.team.map((set) => [set.species, set.item, set.ability, set.moves]),
    );
    expect(a2.decisions).toHaveLength(a1.decisions.length);
    a2.decisions.forEach((decision, i) => {
      const expected = a1.decisions[i] as Episode["decisions"][number];
      expect(decision.identity.request_seq).toBe(expected.identity.request_seq);
      expect(decision.identity.request_hash).toBe(expected.identity.request_hash);
      expect(decision.choice).toBe(expected.choice);
      // The A2 evidence A1 has no timer for.
      expect(decision.deadline?.server_auto_choose_detected).toBe(false);
    });
    const [mine, theirs] = [await observations(a2), await observations(a1)];
    expect(mine.map(comparable)).toEqual(theirs.map(comparable));
    expect(mine.length).toBeGreaterThan(0);

    // A stored copy reads back as it was written, with the proof of the played team.
    const stored = JSON.parse(JSON.stringify(a2));
    const known = new Set([...options.known_synthetic, teamHash(a2.team.team)]);
    expect(await readEpisode(stored, { known_synthetic: known })).toEqual(a2);
  }, 60_000);

  it("folds several reconnects: a second rejoin that replays the same room log", async () => {
    const [first, second] = fixture.frames[seat] as [string[], string[]];
    // The server replays the log and re-sends the stored request (and its choice) to each joining connection.
    const cut = 2;
    const played = await play(fixture, [first, second.slice(0, cut), second]);
    const record = finished(played);
    expect(record.channels).toHaveLength(3);
    const a2 = await a2Episode(record, optionsFor(fixture, seat));
    const a1 = await a1Episode(fixture, seat, knownOf(fixture));
    expect(a2.channel.map(withoutRqid)).toEqual(a1.channel.map(withoutRqid));
    expect(a2.decisions.map((d) => [d.identity.request_seq, d.identity.request_hash, d.choice])).toEqual(
      a1.decisions.map((d) => [d.identity.request_seq, d.identity.request_hash, d.choice]),
    );
    expect((await observations(a2)).map(comparable)).toEqual((await observations(a1)).map(comparable));
  }, 60_000);

  it("does not change what a run without a reconnect writes: one connection, one channel", async () => {
    const plain = readFixture(`${FORMAT_DIR}/bot-vs-bot-1.json`);
    const who = SEATS[0] as Seat;
    const record = finished(await play(plain, plain.frames[who], who));
    expect(record.channels).toHaveLength(1);
    const a2 = await a2Episode(record, optionsFor(plain, who));
    expect(a2.channel).toEqual((record.channels[0] as PlayerChannel).lines);
  }, 60_000);
});

describe("a reconnect whose re-sent request lost its timer snapshot", () => {
  const fixture = readFixture(WITHOUT_SNAPSHOT);
  const seat = seatOf(fixture);

  it("plays on, keeps the record and an incident, and writes no Episode: that decision has no deadline record to carry (ADR-008 E1b)", async () => {
    const played = await play(fixture, fixture.frames[seat]);
    const record = finished(played);
    const snapshotless = record.decisions.filter((d) => d.timer === null);
    expect(snapshotless).toHaveLength(1);
    expect(snapshotless[0]?.rejoin_without_snapshot).toBe(true);
    expect(record.decisions.filter((d) => d.rejoin_without_snapshot === true)).toHaveLength(1);
    expect(played.logs.filter((entry) => entry.event === "incident")).toEqual([
      { event: "incident", room: record.room, kind: "rejoin_without_timer_snapshot", request_seq: snapshotless[0]?.request_seq },
    ]);
    const refused = await a2Episode(record, optionsFor(fixture, seat)).catch((error: unknown) => error);
    expect(refused).toBeInstanceOf(A2EpisodeError);
    expect(refused).toMatchObject({ code: "rejoin_without_timer_snapshot", message: expect.stringContaining("no timer snapshot") });
  }, 60_000);
});

describe("a rejoin that is not this battle's channel is refused, and no Episode is written", () => {
  const fixture = readFixture(WITH_SENTCHOICE);
  const seat = seatOf(fixture);
  const [first, second] = fixture.frames[seat] as [string[], string[]];

  const mutate = (frame: string, from: string | RegExp, to: string): string => {
    const changed = frame.replace(from, to);
    expect(changed).not.toBe(frame);
    return changed;
  };
  /** The rejoin's frames with `edit` applied to the frame at `index`. */
  const rejoin = (index: number, edit: (frame: string) => string): string[][] => [first, second.map((f, i) => (i === index ? edit(f) : f))];
  const requestAt = second.findIndex((f) => f.includes("|request|"));

  async function refused(connections: string[][], message: RegExp): Promise<void> {
    const played = await play(fixture, connections);
    expect("error" in played.outcome).toBe(true);
    const error = (played.outcome as { error: unknown }).error;
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toMatch(message);
    // The bot forfeits what it cannot follow, and the fatal record is all that is handed over.
    expect(played.transports.flatMap((t) => t.sent)).toContain(`${fixture.room}|/forfeit`);
    const [record] = played.records as [A2BattleRecord];
    expect(record.fatal).not.toBeNull();
    await expect(a2Episode(record, optionsFor(fixture, seat))).rejects.toBeInstanceOf(A2EpisodeError);
  }

  it("a rejoin that does not replay the earlier room log", async () => {
    // The first frame of the rejoin holds `|init|battle` and the replay: change one line of it.
    await refused(rejoin(0, (frame) => mutate(frame, "|gametype|doubles", "|gametype|singles")), /replay the earlier channel's room log/);
  }, 60_000);

  it("a spliced request: the re-sent request keeps its body but not its rqid", async () => {
    await refused(rejoin(requestAt, (frame) => mutate(frame, /"rqid":(\d+)/, '"rqid":9999')), /another rqid/);
  }, 60_000);

  it("a rejoin on the wrong seat", async () => {
    const other = rivalOf(seat);
    await refused(rejoin(requestAt, (frame) => mutate(frame, `"id":"${seat}"`, `"id":"${other}"`)), new RegExp(`not the bot's seat ${seat}`));
  }, 60_000);

  it("the Episode writer itself refuses channels that do not join, without quoting them", async () => {
    const record = finished(await play(fixture, fixture.frames[seat]));
    const changed = structuredClone(record);
    const [, second_] = changed.channels as [PlayerChannel, PlayerChannel];
    second_.lines = second_.lines.map((line) => (line === "|gametype|doubles" ? "|gametype|singles" : line));
    const error = await a2Episode(changed, optionsFor(fixture, seat)).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(A2EpisodeError);
    expect((error as Error).message).toBe("the 2 connections' channels do not join into one continuous channel");
    const other = structuredClone(record);
    other.channels.reverse();
    await expect(a2Episode(other, optionsFor(fixture, seat))).rejects.toBeInstanceOf(A2EpisodeError);
  }, 60_000);
});
