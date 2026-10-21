import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  A2Channel,
  type A2BattleRecord,
  connectWebSocket,
  ProtocolReader,
  runA2Bot,
  type SequencedRequest,
  splitFrame,
  type Transport,
} from "@areneon/bot-client";
import {
  type Engine,
  classOfSets,
  enumerateCandidates,
  type PlayerChannel,
  playerChannel,
  type PokemonSet,
  resolveFormat,
  rightsFor,
  type RulesSnapshot,
  type SyntheticTeamHashes,
  teamHash,
  type VersionManifest,
} from "@areneon/showdown-bridge";
import { type ServerSetting, serverSettings } from "./config.ts";
import { launch, type LaunchOptions, type LocalServer } from "./launch.ts";

/**
 * The harness side of the local server (ADR-005 §5): a launch whose config
 * turns `logchallenges` on, so the server writes a finished challenge
 * battle's input log, and the synthetic bot-vs-bot recording built on it.
 *
 * The input log holds both teams and every choice: an omniscient evaluation
 * artifact, never an actor's input. It is for the A1/A2 equality test (#764)
 * only. Nothing here is used when a human spars with the bot.
 */

export type Seat = "p1" | "p2";
const SEATS: readonly Seat[] = ["p1", "p2"];

/** The settings of a harness server: the sparring ones, with `logchallenges` on. */
export function harnessSettings(port: number): ServerSetting[] {
  return serverSettings(port, true);
}

/**
 * `launch` with `logchallenges` on: the only caller that does. This module
 * (the `./harness` export) imports bot-client, a devDependency of this
 * package, so it must stay out of `index.ts`.
 */
export function launchHarness(options: Omit<LaunchOptions, "logChallenges">): Promise<LocalServer> {
  return launch({ ...options, logChallenges: true });
}

export class HarnessError extends Error {
  override name = "HarnessError";
}

/** A recording whose reconnect is not the planned one: the battle is random, so the recorder may try again. */
export class ReconnectNotPlannedError extends HarnessError {
  override name = "ReconnectNotPlannedError";
}

/** What the server wrote for one battle: only the fields a fixture keeps. */
export interface ServerInputLog {
  /** The server's lines of the sim's input log: `>start`, `>player`, then every choice. */
  lines: string[];
  /** Winner's user name; `""` or absent for a tie. */
  winner: string;
  turns: number;
}

export interface ReadLogOptions {
  /** How long to wait for the server to write the file (default 15 s). */
  timeoutMs?: number;
}

/**
 * Where the server writes a battle's log, relative to its install root: the
 * pinned `logBattle` (server/room-battle.ts) writes
 * `logs/<yyyy-mm>/<format>/<yyyy-mm-dd>/<replay id>.log.json`, dated by the wall
 * clock, so the directories are searched rather than computed.
 */
export function findServerLog(dir: string, format: string, room: string): string | undefined {
  // The file is named by the replay id: the room id without its `battle-`.
  const file = `${room.replace(/^battle-/, "")}.log.json`;
  const logs = join(dir, "logs");
  const children = (path: string): string[] => {
    try {
      return readdirSync(path);
    } catch {
      return [];
    }
  };
  for (const month of children(logs)) {
    for (const day of children(join(logs, month, format))) {
      if (children(join(logs, month, format, day)).includes(file)) return join(logs, month, format, day, file);
    }
  }
  return undefined;
}

/**
 * The input log the harness server wrote for `room`, once the battle has
 * ended. Only `inputLog`, `winner` and `turns` are read: the rest of the
 * file (the wall-clock `timestamp`, the room log) is dropped.
 */
export async function readServerInputLog(
  dir: string,
  format: string,
  room: string,
  options: ReadLogOptions = {},
): Promise<ServerInputLog> {
  const until = Date.now() + (options.timeoutMs ?? 15_000);
  let path = findServerLog(dir, format, room);
  while (path === undefined) {
    if (Date.now() >= until) throw new HarnessError(`the server wrote no input log for ${room} (is logchallenges on?)`);
    await new Promise((done) => setTimeout(done, 50));
    path = findServerLog(dir, format, room);
  }
  const data = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  const { inputLog, winner, turns } = data;
  if (!Array.isArray(inputLog) || !inputLog.every((line) => typeof line === "string") || typeof turns !== "number") {
    throw new HarnessError(`${path} is not a server battle log`);
  }
  return { lines: inputLog as string[], winner: typeof winner === "string" ? winner : "", turns };
}

/**
 * One recorded battle: the input log, both players' own channels, the teams,
 * the seed and the pin. `schema_version` 3 stores a seat's raw frames and
 * channels per connection (see `frames`); version 2, one connection per seat,
 * is `HarnessFixtureV2` and is read through `upgradeFixture`.
 */
export interface HarnessFixture {
  schema_version: 3;
  /** Proven from the two teams' hashes (`buildHarnessFixture` refuses a fixture that is not synthetic). */
  data_class: "synthetic";
  rights: "cc0-synthetic";
  license: "CC0-1.0";
  provenance: string;
  manifest: VersionManifest;
  format_id: string;
  /** The battle room both bots joined (the `>ROOMID` header of `frames`). */
  room: string;
  /** The seed the server drew for the battle (`>start`); the server takes no seed from a challenge. */
  seed: string;
  /** The names and avatars the server put on the `>player` lines: A1 must use the same to equal the channels. */
  players: Record<Seat, { name: string; avatar: string; team: string }>;
  /** Each seat's team as its bot sent it (`/utm`), nicknames and all. */
  teams: Record<Seat, PokemonSet[]>;
  /**
   * The teams as the server used them, unpacked from the input log's own
   * `>player` lines. The server may clear nicknames (its nickname filter
   * decides), which changes `|switch|` names. A1 plays these, not `teams`.
   */
  served_teams: Record<Seat, PokemonSet[]>;
  /** `null` for a tie. */
  winner: Seat | null;
  turns: number;
  /**
   * Each player's raw server frames for the battle room, exactly as its bot
   * received them from the WebSocket (header, lines, order and frame
   * boundaries), before any parsing. Only that player's own stream: the
   * server sends each client only what it may see. One list per connection
   * that received the room, in the order the bot opened them: a bot that
   * lost its connection and logged in again has a second list, which starts
   * at its own `|init|battle` (the server replays the room log to a joining
   * connection) and holds the re-sent request. Stored as received: nothing
   * joins or reorders the lists.
   */
  frames: Record<Seat, string[][]>;
  /**
   * Each player's own channels as its bot client built them from its
   * `frames`: one per connection, each from its own `|init|battle`
   * (`A2BattleRecord.channels`). Derived: `verifyHarnessFixture` replays
   * each connection's frames through a fresh `ProtocolReader` and
   * `A2Channel`, as the session does, and requires this exact result.
   */
  channels: Record<Seat, PlayerChannel[]>;
  /** The server's input log: both teams and every choice. Omniscient, never an actor's input. */
  input_log: { omniscient: true; lines: string[] };
}

/**
 * The fixtures recorded before reconnects were stored: one connection per
 * seat, so `frames` and `channels` hold that connection directly.
 */
export interface HarnessFixtureV2 extends Omit<HarnessFixture, "schema_version" | "frames" | "channels"> {
  schema_version: 2;
  frames: Record<Seat, string[]>;
  channels: Record<Seat, PlayerChannel>;
}

/**
 * A fixture as the current schema stores it. A version 2 fixture becomes the
 * version 3 one with a single connection per seat (its content is not
 * touched); any other version is refused.
 */
export function upgradeFixture(fixture: HarnessFixture | HarnessFixtureV2): HarnessFixture {
  const refuse = (why: string): never => {
    throw new HarnessError(`harness fixture refused: ${why}`);
  };
  if (fixture.schema_version === 3) {
    for (const seat of SEATS) {
      if (!Array.isArray(fixture.frames[seat]) || !fixture.frames[seat].every((connection) => Array.isArray(connection))) {
        refuse(`${seat} frames of a version 3 fixture are a list of connections, each a list of frames`);
      }
      if (!Array.isArray(fixture.channels[seat])) refuse(`${seat} channels of a version 3 fixture are a list, one per connection`);
    }
    return fixture;
  }
  if (fixture.schema_version !== 2) {
    return refuse(`schema_version ${String((fixture as { schema_version: unknown }).schema_version)} is not 2 or 3`);
  }
  for (const seat of SEATS) {
    if (!Array.isArray(fixture.frames[seat]) || Array.isArray(fixture.channels[seat])) {
      refuse(`${seat} frames of a version 2 fixture are one list of frames and its channel is one channel`);
    }
  }
  const { frames, channels, ...rest } = fixture;
  return {
    ...rest,
    schema_version: 3,
    frames: { p1: [frames.p1], p2: [frames.p2] },
    channels: { p1: [channels.p1], p2: [channels.p2] },
  };
}

interface PlayerLine {
  name: string;
  avatar: string;
  team: string;
}

const PLAYER_LINE = /^>player (p[12]) (\{.*\})$/;

/** The `>start` options and the two `>player` lines of an input log. */
export function parseInputLog(lines: readonly string[]): {
  start: { formatid: string; seed: string };
  players: Record<Seat, PlayerLine>;
} {
  const first = lines[0] ?? "";
  if (!first.startsWith(">start ")) throw new HarnessError("the input log does not start with >start");
  const start = JSON.parse(first.slice(">start ".length)) as { formatid: string; seed: string };
  const players: Partial<Record<Seat, PlayerLine>> = {};
  for (const line of lines.slice(1, 3)) {
    const match = PLAYER_LINE.exec(line);
    if (match === null) throw new HarnessError(`not a >player line: ${line.slice(0, 40)}`);
    players[match[1] as Seat] = JSON.parse(match[2] as string) as PlayerLine;
  }
  if (players.p1 === undefined || players.p2 === undefined) throw new HarnessError("the input log lacks a player");
  return { start, players: players as Record<Seat, PlayerLine> };
}

/**
 * A pure observer on a transport (the transport boundary of ADR-005 §1):
 * every frame is copied into `into`, as received and before any listener
 * parses it, then delivered unchanged. Sending, closing and close listeners
 * pass straight through.
 */
export function tapTransport(inner: Transport, into: string[]): Transport {
  return {
    send: (line) => inner.send(line),
    onFrame: (cb) =>
      inner.onFrame((frame) => {
        into.push(frame);
        cb(frame);
      }),
    onClose: (cb) => inner.onClose(cb),
    close: () => inner.close(),
  };
}

/** What a `|t:|<epoch seconds>` line's number becomes in a fixture: the server's wall clock is not kept. */
export const REDACTED_CLOCK = "|t:|0";

/**
 * `frame` with the number of every `|t:|` line replaced (the only edit a
 * stored frame gets): the server writes its wall clock there, and neither
 * `ProtocolReader` nor `A2Channel` reads it (`stripServerLines` drops the
 * line whatever its number). Lines, order and frame boundaries stay.
 */
export function redactClock(frame: string): string {
  return frame
    .split("\n")
    .map((line) => (line.startsWith("|t:|") ? REDACTED_CLOCK : line))
    .join("\n");
}

/**
 * The frames of `room` out of one bot's connections (each a list of every
 * frame that connection received, in order): the room's own frames only, the
 * lobby and global frames dropped, one list per connection that received the
 * room (a connection that never joined it, or only logged in, has none).
 * The lists keep the order the connections were opened in. A bot that
 * reconnected has several; none is an error: the bot never joined the room.
 */
export function roomFrames(connections: readonly (readonly string[])[], room: string): string[][] {
  const kept = connections
    .map((frames) => frames.filter((frame) => splitFrame(frame).room === room).map(redactClock))
    .filter((frames) => frames.length > 0);
  if (kept.length === 0) throw new HarnessError(`no connection received ${room}`);
  return kept;
}

/**
 * A bot's channel as its session builds it: the room's raw frames in order
 * through `ProtocolReader` (it must decode every request, all of them the
 * seat's own) and `A2Channel` (server lines dropped, re-sent requests left
 * out). `dropped` and `resent` are the adapter's own counts.
 */
export function channelFromFrames(
  frames: readonly string[],
  room: string,
  seat: Seat,
  data_class: PlayerChannel["data_class"],
): { channel: PlayerChannel; dropped: string[]; resent: number; requests: number } {
  const reader = new ProtocolReader();
  const adapter = new A2Channel({ room, player: seat, data_class });
  let requests = 0;
  for (const frame of frames) {
    for (const event of reader.read(frame)) {
      if (event.kind !== "request") continue;
      requests += 1;
      if (event.request.player !== seat) throw new HarnessError(`${seat}'s frames hold a request for ${event.request.player}`);
    }
    adapter.receive(frame);
  }
  const built = adapter.build();
  return { ...built, requests };
}

/**
 * Widens a known synthetic set by the played form of teams already in it. The
 * bot's validator edits a team in place (it fills the nickname, the gender of a
 * genderless species), so a stored fixture holds the played form; the pinned
 * validator produces it here, on a copy, so no rule of it is written by hand.
 * It cannot make a team synthetic: a team whose own hash is not in `known` is
 * refused (`HarnessError`), and only the played forms of listed teams are added.
 */
export function withPlayedForms(
  engine: Engine,
  formatId: string,
  known: SyntheticTeamHashes,
  teams: readonly (readonly PokemonSet[])[],
): Set<string> {
  const hashes = new Set(known);
  for (const team of teams) {
    if (!known.has(teamHash(team))) throw new HarnessError("a played form is added only for a team already known synthetic");
    const played = structuredClone(team) as PokemonSet[];
    engine.sim.TeamValidator.get(formatId).validateTeam(played);
    hashes.add(teamHash(played));
  }
  return hashes;
}

export interface FixtureInput {
  engine: Engine;
  snapshot: RulesSnapshot;
  format: string;
  provenance: string;
  log: ServerInputLog;
  /** The two bots' records of the battle, keyed by their seat. */
  records: Record<Seat, A2BattleRecord>;
  /** Every frame each bot's connections received, per seat (see `roomFrames`). */
  connections: Record<Seat, string[][]>;
  teams: Record<Seat, PokemonSet[]>;
  /** A label for each seat's team, e.g. its file under fixtures/teams. */
  teamLabels: Record<Seat, string>;
  /** The synthetic teams there are: both teams must be in it, or no fixture is built. */
  known_synthetic: SyntheticTeamHashes;
}

/** Build the fixture of a finished battle; it must pass `verifyHarnessFixture`. */
export async function buildHarnessFixture(input: FixtureInput): Promise<HarnessFixture> {
  const { engine, snapshot, format, log, records, connections, teams, teamLabels, known_synthetic } = input;
  // A fixture is committed CC0 data: its class is proven from the hashes, and only synthetic is built.
  if (SEATS.some((seat) => classOfSets(teams[seat], known_synthetic) !== "synthetic")) {
    throw new HarnessError("a fixture is built only from known synthetic teams");
  }
  const { start, players } = parseInputLog(log.lines);
  const channels = {} as Record<Seat, PlayerChannel[]>;
  const frames = {} as Record<Seat, string[][]>;
  let room = "";
  for (const seat of SEATS) {
    const record = records[seat];
    if (record.player !== seat || record.room.length === 0) {
      throw new HarnessError(`the ${seat} record is for ${String(record.player)}`);
    }
    // One channel per connection that joined the room; `verifyHarnessFixture` requires them to match the frames.
    channels[seat] = [...record.channels];
    frames[seat] = roomFrames(connections[seat], record.room);
    if (room !== "" && room !== record.room) throw new HarnessError("the bots did not play the same battle");
    room = record.room;
  }
  const served = (seat: Seat): PokemonSet[] => {
    const team = engine.sim.Teams.unpack(players[seat].team);
    if (team === null) throw new HarnessError(`${seat}'s team in the input log cannot be unpacked`);
    return team;
  };
  const winner = SEATS.find((seat) => players[seat].name === log.winner) ?? null;
  const fixture: HarnessFixture = {
    schema_version: 3,
    data_class: "synthetic",
    rights: rightsFor("synthetic") as "cc0-synthetic",
    license: "CC0-1.0",
    provenance: input.provenance,
    manifest: resolveFormat(engine, format, snapshot).manifest,
    format_id: format,
    room,
    seed: start.seed,
    players: {
      p1: { name: players.p1.name, avatar: players.p1.avatar, team: teamLabels.p1 },
      p2: { name: players.p2.name, avatar: players.p2.avatar, team: teamLabels.p2 },
    },
    teams,
    served_teams: { p1: served("p1"), p2: served("p2") },
    winner,
    turns: log.turns,
    frames,
    channels,
    input_log: { omniscient: true, lines: log.lines },
  };
  await verifyHarnessFixture(engine, fixture, known_synthetic);
  return fixture;
}

/** The sim run on an input log, wall-clock `|t:|` lines dropped. */
export interface Replay {
  /** Both sides' hidden information: an evaluation artifact. */
  omniscient: string[];
  channels: Record<Seat, string[]>;
}

/** Feed an input log to the pinned in-process sim, as `/importinputlog` would. */
export async function replayInputLog(engine: Engine, lines: readonly string[]): Promise<Replay> {
  const stream = new engine.sim.BattleStream();
  const streams = engine.sim.getPlayerStreams(stream);
  const replay: Replay = { omniscient: [], channels: { p1: [], p2: [] } };
  const collect = async (source: AsyncIterable<string>, into: string[]): Promise<void> => {
    for await (const chunk of source) {
      for (const line of chunk.split("\n")) if (!line.startsWith("|t:|")) into.push(line);
    }
  };
  const done = Promise.all([
    collect(streams.omniscient, replay.omniscient),
    collect(streams.p1, replay.channels.p1),
    collect(streams.p2, replay.channels.p2),
  ]);
  streams.omniscient.write(lines.join("\n"));
  streams.omniscient.writeEnd();
  await done;
  return replay;
}

/** The requests a connection's frames carry, as its `ProtocolReader` reads them (re-sends of the same request left out). */
function requestsIn(frames: readonly string[]): { rqid: number; hash: string }[] {
  const reader = new ProtocolReader();
  const requests: { rqid: number; hash: string }[] = [];
  for (const frame of frames) {
    for (const event of reader.read(frame)) {
      if (event.kind === "request" && !event.resend) {
        requests.push({ rqid: event.request.rqid as number, hash: event.request.request_hash });
      }
    }
  }
  return requests;
}

/**
 * What is wrong with `seat`'s raw frames, connection by connection: their
 * shape (non-empty strings of the fixture's room, headed, each connection's
 * first frame starting with `|init|battle`), their privacy (no `|showteam|`
 * or `|split|` line, no request but the seat's own, no `p<other>` secret
 * line: `A2Channel` refuses those too), that `channelFromFrames` rebuilds
 * exactly the stored channel of that connection (one channel per
 * connection), and that the requests go on across connections: a later
 * connection's first request is the previous one's last re-sent (same
 * `rqid`, same `request_hash`) or a newer one (higher `rqid`), never an
 * older one. A seat with one connection is reported as `p1`, one with
 * several as `p1 connection 2`.
 */
function frameProblems(fixture: HarnessFixture, seat: Seat): string[] {
  const connections = fixture.frames[seat];
  const channels = fixture.channels[seat];
  if (!Array.isArray(connections) || connections.length === 0) return [`${seat} frames are not a non-empty list of connections`];
  const problems: string[] = [];
  if (channels.length !== connections.length) {
    problems.push(`${seat} has ${channels.length} channels for ${connections.length} connections`);
  }
  let previous: { rqid: number; hash: string } | undefined;
  connections.forEach((frames, c) => {
    const who = connections.length === 1 ? seat : `${seat} connection ${c + 1}`;
    if (!Array.isArray(frames) || frames.length === 0 || !frames.every((frame) => typeof frame === "string")) {
      problems.push(`${who} frames are not a non-empty list of strings`);
      return;
    }
    const split = frames.map((frame) => splitFrame(frame));
    if (!frames.every((frame) => frame.startsWith(`>${fixture.room}\n`)) || split.some(({ room }) => room !== fixture.room)) {
      problems.push(`${who} frames are not all headed >${fixture.room}`);
    }
    if (split[0]?.lines[0] !== "|init|battle") problems.push(`${who} frames do not start with |init|battle`);
    const lines = split.flatMap(({ lines: l }) => l);
    if (lines.some((line) => line.startsWith("|showteam|") || line.startsWith("|split|"))) {
      problems.push(`${who} frames have a team sheet or a |split| marker`);
    }
    if (lines.some((line) => line.startsWith("|t:|") && line !== REDACTED_CLOCK)) {
      problems.push(`${who} frames have a wall-clock |t:| line`);
    }
    for (const line of lines.filter((l) => l.startsWith("|request|"))) {
      if (!line.includes(`"id":"${seat}"`)) problems.push(`${who} frames hold a request that is not its own`);
    }
    try {
      const replayed = channelFromFrames(frames, fixture.room, seat, fixture.data_class).channel;
      if (JSON.stringify(replayed) !== JSON.stringify(channels[c])) {
        problems.push(`${who} channel is not what its frames give through ProtocolReader and A2Channel`);
      }
      const requests = requestsIn(frames);
      const first = requests[0];
      if (first !== undefined && previous !== undefined) {
        if (first.rqid < previous.rqid || (first.rqid === previous.rqid && first.hash !== previous.hash)) {
          problems.push(`${who} starts with a request older than, or different from, the one before it`);
        }
      }
      previous = requests.at(-1) ?? previous;
    } catch (error) {
      problems.push(`${who} frames cannot be replayed: ${(error as Error).message}`);
    }
  });
  return problems;
}

/**
 * Check a fixture without a server. Refuses (listing every problem) unless:
 * it is synthetic with matching rights; it names this engine's commit; its
 * input log's seed, format, names, avatars and packed teams are the ones the
 * fixture states; each channel (one per connection) is that player's own
 * (`playerChannel`: no opponent SECRET line, no other player's request) and
 * has no `|showteam|`; `served_teams` is what the input log packs and
 * differs from `teams` at most in nicknames; each seat's raw `frames`, per
 * connection, are well formed, its own, go on across connections and
 * rebuild the stored `channels` (`frameProblems`); a version 2 fixture is
 * checked as its version 3 upgrade (`upgradeFixture`), any other version is
 * refused; and the pinned sim run on the input log gives
 * the fixture's `winner` and `turns` and the channels' `|player|` lines
 * (names and avatars) and `|win|` line. The other channel lines are NOT
 * compared with the replay: it is one batch, so it holds only the last
 * `|request|`; the step-by-step comparison is #764.
 */
export async function verifyHarnessFixture(
  engine: Engine,
  stored: HarnessFixture | HarnessFixtureV2,
  known_synthetic: SyntheticTeamHashes,
): Promise<void> {
  const fixture = upgradeFixture(stored);
  const problems: string[] = [];
  if (fixture.data_class !== "synthetic" || fixture.rights !== rightsFor(fixture.data_class)) {
    problems.push("not synthetic with cc0-synthetic rights");
  }
  for (const seat of SEATS) {
    if (classOfSets(fixture.teams[seat], known_synthetic) !== "synthetic") problems.push(`${seat} team is not a known synthetic team`);
  }
  if (fixture.manifest.showdown_commit !== engine.commit) problems.push("recorded for another Showdown commit");
  const { start, players } = parseInputLog(fixture.input_log.lines);
  if (start.seed !== fixture.seed || start.formatid !== fixture.format_id) problems.push("input log start differs");
  for (const seat of SEATS) {
    const { name, avatar } = fixture.players[seat];
    if (players[seat].name !== name || players[seat].avatar !== avatar) problems.push(`${seat} name or avatar differs`);
    const logged = engine.sim.Teams.unpack(players[seat].team);
    if (JSON.stringify(logged) !== JSON.stringify(fixture.served_teams[seat])) {
      problems.push(
        `${seat} served team differs from the input log: ${JSON.stringify(logged)} vs ${JSON.stringify(fixture.served_teams[seat])}`,
      );
    }
    // The sent team and the served one differ at most in nicknames.
    const core = (team: readonly PokemonSet[]): string =>
      JSON.stringify(team.map(({ species, item, ability, moves }) => [species, item, ability, moves]));
    if (core(fixture.teams[seat]) !== core(fixture.served_teams[seat])) {
      problems.push(`${seat} sent and served teams differ in more than nicknames`);
    }
    for (const own of fixture.channels[seat]) {
      const channel = playerChannel(own);
      if (channel.player !== seat || channel.data_class !== fixture.data_class) problems.push(`${seat} channel is not its own`);
      if (channel.lines.some((line) => line.startsWith("|showteam|"))) problems.push(`${seat} channel has a team sheet`);
    }
    problems.push(...frameProblems(fixture, seat));
  }
  const replay = await replayInputLog(engine, fixture.input_log.lines);
  const win = (lines: readonly string[]): string | undefined => lines.find((line) => line.startsWith("|win|"));
  const last = replay.omniscient.filter((line) => line.startsWith("|turn|")).at(-1);
  if (last !== `|turn|${fixture.turns}`) problems.push(`turns ${fixture.turns} differs from the replay's ${String(last)}`);
  const replayWinner = SEATS.find((seat) => replay.omniscient.includes(`|win|${fixture.players[seat].name}`)) ?? null;
  if (replayWinner !== fixture.winner) problems.push(`winner ${String(fixture.winner)} differs from the replay's ${String(replayWinner)}`);
  const header = (lines: readonly string[]): string[] => lines.filter((line) => line.startsWith("|player|"));
  for (const seat of SEATS) {
    // The last connection holds the end of the battle; every connection's replayed log holds the `|player|` lines.
    const mine = fixture.channels[seat].at(-1)?.lines ?? [];
    if (win(mine) !== win(replay.channels[seat])) problems.push(`${seat} channel and replay disagree on the winner`);
    for (const own of fixture.channels[seat]) {
      if (header(own.lines).join("\n") !== header(replay.channels[seat]).join("\n")) {
        problems.push(`${seat} channel and replay disagree on the |player| lines`);
      }
    }
  }
  if (problems.length > 0) throw new HarnessError(`harness fixture refused: ${problems.join("; ")}`);
}

/** A bot of the pair: the one that waits for the challenge, or the one that sends it. */
export type Role = "sparring" | "challenger";

/**
 * Where the recorder cuts one bot's connection, so the server re-sends the
 * request to the connection the bot opens next (ADR-005 §1). The cut is made
 * at the transport, like the tap, so `runA2Bot` and the session are the
 * ones of a sparring run: it sees a close and reconnects through its normal
 * path.
 */
export interface DropPlan {
  /** The bot whose first connection is cut. */
  who: Role;
  /**
   * The bot's own request that decides the cut: its `request`-th `|request|`
   * line (1-based, `wait` included, `|request|null` not counted) on that
   * connection. Team Preview is request 1.
   */
  request: number;
  /**
   * `before_choice`: cut when that request arrives, before the bot can
   * answer it, so the server holds no choice and re-sends the request alone.
   * `after_choice`: cut right after the bot sent its choice for it, so the
   * server holds the choice and re-sends the request with `|sentchoice|`.
   * Then the other bot's choice for the same request is held back until the
   * cut bot's next connection has received the request, so the server is
   * still waiting for it (otherwise the turn could resolve before the cut).
   */
  when: "before_choice" | "after_choice";
}

const CHOOSE = /^[^|]*\|\/choose /;
const isRequest = (line: string): boolean => line.startsWith("|request|") && line !== "|request|null";

/** Calls what it is given once it is opened, at once if it is open already. */
class Gate {
  #open = false;
  readonly #waiting: (() => void)[] = [];

  open(): void {
    this.#open = true;
    for (const run of this.#waiting.splice(0)) run();
  }

  then(run: () => void): void {
    if (this.#open) run();
    else this.#waiting.push(run);
  }
}

/**
 * The transports of a `DropPlan`: `wrap` gives each connection of each bot
 * its role in the plan. The wrappers only withhold (frames after the cut, the
 * other bot's one choice) and close; they never change a line, so the tap
 * outside them stores what the bot was given.
 *
 * - The cut bot's first connection counts its requests and, at the plan's
 *   point, closes the inner transport. From then on it delivers no frame and
 *   sends nothing: the bot has "lost" the connection, so what it was never
 *   shown is not in its frames and the server's replay on the next connection
 *   shows it.
 * - Its later connections open the gate when they have received a request.
 * - With `after_choice` the other bot's first connection holds back its
 *   choice for the plan's request until that gate opens.
 */
export class ConnectionDrop {
  readonly #plan: DropPlan;
  readonly #holdMs: number;
  readonly #rejoined = new Gate();
  #holdTimer: ReturnType<typeof setTimeout> | undefined;

  /**
   * `holdMs` bounds how long the other bot's choice is held (default 10 s): if the cut bot never
   * reconnects, the choice goes out when it is up, so the battle is not stuck (the recording is
   * then refused by `reconnectEvidence`'s check, not hung).
   */
  constructor(plan: DropPlan, holdMs = 10_000) {
    this.#plan = plan;
    this.#holdMs = holdMs;
  }

  #open(): void {
    clearTimeout(this.#holdTimer);
    this.#rejoined.open();
  }

  /** `inner` as the plan wants the `index`-th (0-based) connection `who` opens to behave. */
  wrap(who: Role, index: number, inner: Transport): Transport {
    if (who === this.#plan.who) return index === 0 ? this.#cut(inner) : this.#watch(inner);
    return this.#plan.when === "after_choice" && index === 0 ? this.#hold(inner) : inner;
  }

  #cut(inner: Transport): Transport {
    let requests = 0;
    let reached = false;
    let cut = false;
    const drop = (): void => {
      cut = true;
      inner.close();
    };
    return {
      send(line) {
        if (cut) return;
        inner.send(line);
        if (reached && CHOOSE.test(line)) drop();
      },
      onFrame: (cb) =>
        inner.onFrame((frame) => {
          if (cut) return;
          const before = requests;
          requests += splitFrame(frame).lines.filter(isRequest).length;
          const arrives = before < this.#plan.request && requests >= this.#plan.request;
          // Set first: a bot that answers at once sends its choice inside `cb`. Only `after_choice`
          // cuts in `send`; `before_choice` cuts below, once, however `cb` went.
          if (arrives && this.#plan.when === "after_choice") reached = true;
          cb(frame);
          if (arrives && this.#plan.when === "before_choice") drop();
        }),
      onClose: (cb) => inner.onClose(cb),
      close: () => inner.close(),
    };
  }

  #watch(inner: Transport): Transport {
    return {
      ...inner,
      onFrame: (cb) =>
        inner.onFrame((frame) => {
          cb(frame);
          if (splitFrame(frame).lines.some(isRequest)) this.#open();
        }),
    };
  }

  #hold(inner: Transport): Transport {
    let requests = 0;
    return {
      ...inner,
      send: (line) => {
        if (CHOOSE.test(line) && requests === this.#plan.request) {
          this.#holdTimer ??= setTimeout(() => this.#open(), this.#holdMs).unref();
          this.#rejoined.then(() => inner.send(line));
        } else {
          inner.send(line);
        }
      },
      onFrame: (cb) =>
        inner.onFrame((frame) => {
          requests += splitFrame(frame).lines.filter(isRequest).length;
          cb(frame);
        }),
    };
  }
}

export interface BotPair {
  sparring: A2BattleRecord;
  challenger: A2BattleRecord;
  /** Every frame each bot's connections received, in order (`tapTransport`). */
  connections: { sparring: string[][]; challenger: string[][] };
}

export interface BotVsBotOptions {
  engine: Engine;
  snapshot: RulesSnapshot;
  /** The harness server's WebSocket URL. */
  url: string;
  format: string;
  /** The sparring bot's and the challenger's user names. */
  names: { sparring: string; challenger: string };
  teams: { sparring: PokemonSet[]; challenger: PokemonSet[] };
  /** The synthetic teams there are (hashes): each bot proves its channels' class from it and the other's team. */
  known_synthetic: SyntheticTeamHashes;
  /** Opens a transport to `url`; default `connectWebSocket`. For tests. */
  connect?: (url: string) => Promise<Transport>;
  /** Cut one bot's first connection mid-battle so it reconnects (see `DropPlan`); none by default. */
  drop?: DropPlan;
}

/**
 * Two `runA2Bot` clients play one Bo1 on the server: the sparring bot waits
 * for the challenge, the challenger (harness-only `challenge` option) sends
 * it once the sparring bot is logged in. Both play the shared actor rule.
 */
export async function playBotVsBot(options: BotVsBotOptions): Promise<BotPair> {
  const { engine, snapshot, url, format, names, teams } = options;
  const open = options.connect ?? ((to: string) => connectWebSocket(to));
  // Each bot's connections, tapped at the transport: its frames as received.
  const connections = { sparring: [] as string[][], challenger: [] as string[][] };
  const drop = options.drop === undefined ? undefined : new ConnectionDrop(options.drop);
  const tapped = (who: keyof typeof connections) => async (to: string) => {
    const frames: string[] = [];
    const index = connections[who].push(frames) - 1;
    const transport = await open(to);
    return tapTransport(drop === undefined ? transport : drop.wrap(who, index, transport), frames);
  };
  // Each bot knows the other's team (the harness gave both), so a class can be proven. The validator edits
  // a bot's team in place, so the other bot is given a copy taken before either ran.
  const pristine = structuredClone(teams);
  const common = { engine, snapshot, url, format, known_synthetic: options.known_synthetic, mode: "harness" as const, battles: 1 };
  let loggedIn: () => void = () => {};
  const ready = new Promise<void>((done) => (loggedIn = done));
  const sparring = runA2Bot({
    ...common,
    name: names.sparring,
    challenger: names.challenger,
    team: teams.sparring,
    opponent_team: pristine.challenger,
    connect: tapped("sparring"),
    onLog(entry) {
      if (entry.event === "logged_in") loggedIn();
    },
  });
  await Promise.race([ready, sparring.then(() => Promise.reject(new HarnessError("the sparring bot ended before login")))]);
  const challenger = runA2Bot({
    ...common,
    name: names.challenger,
    challenger: names.sparring,
    challenge: names.sparring,
    team: teams.challenger,
    opponent_team: pristine.sparring,
    connect: tapped("challenger"),
  });
  const [a, b] = await Promise.all([sparring, challenger]);
  const [mine, theirs] = [a.battles[0], b.battles[0]];
  if (mine === undefined || theirs === undefined || mine.room !== theirs.room) {
    throw new HarnessError("the bots did not play the same battle");
  }
  return { sparring: mine, challenger: theirs, connections };
}

/** What a seat's second connection shows of a reconnect (see `reconnectEvidence`). */
export interface ReconnectEvidence {
  /** The re-sent request's phase and its place (`request_seq`) on the first connection. */
  phase: string;
  request_seq: number;
  /** Which decision of the seat it is: the non-wait requests of the first connection, the re-sent one included. */
  decision: number;
  /** The second connection's frames hold `|sentchoice|`: the server already held the bot's choice. */
  sentchoice: boolean;
  /**
   * The choice the `|sentchoice|` line holds, if any, in the bot's own notation (`move 1 1, move 1`;
   * the server's input log writes the same choice another way, `move moonblast +1, move earthquake`).
   */
  choice: string | undefined;
  /** The request as the second connection received it. */
  resent: SequencedRequest;
}

/**
 * Whether a seat's frames (per connection, as stored) show a real reconnect:
 * the second connection's first request is the first connection's last one,
 * sent again (same `rqid` and `request_hash`), after its own `|init|battle`.
 * `undefined` if not.
 */
export function reconnectEvidence(connections: readonly (readonly string[])[]): ReconnectEvidence | undefined {
  const read = (frames: readonly string[]): { requests: SequencedRequest[]; choice: string | undefined } => {
    const reader = new ProtocolReader();
    const events = frames.flatMap((frame) => reader.read(frame));
    return {
      requests: events.flatMap((event) => (event.kind === "request" && !event.resend ? [event.request] : [])),
      choice: events.flatMap((event) => (event.kind === "sentchoice" ? [event.choice] : []))[0],
    };
  };
  const [first, second] = connections.map(read);
  const last = first?.requests.at(-1);
  const again = second?.requests[0];
  if (last === undefined || again === undefined || last.rqid !== again.rqid || last.request_hash !== again.request_hash) {
    return undefined;
  }
  const { choice } = second as { choice: string | undefined };
  return {
    phase: last.phase,
    request_seq: last.request_seq,
    decision: (first as { requests: SequencedRequest[] }).requests.filter((request) => request.phase !== "wait").length,
    sentchoice: choice !== undefined,
    choice,
    resent: again,
  };
}

export interface RecordOptions extends Omit<BotVsBotOptions, "url"> {
  /** A server launched with `launchHarness`, from the runtime copy at `dir`. */
  server: LocalServer;
  dir: string;
  provenance: string;
  /** Labels for the two teams, e.g. their files under fixtures/teams. */
  teamLabels: { sparring: string; challenger: string };
  readLog?: ReadLogOptions;
}

/** Play one bot-vs-bot battle on the harness server and build its fixture. */
export async function recordHarnessBattle(options: RecordOptions): Promise<HarnessFixture> {
  const { server, dir, format, teams, teamLabels } = options;
  const url = `ws://127.0.0.1:${server.session.port}/showdown/websocket`;
  const pair = await playBotVsBot({ ...options, url });
  const log = await readServerInputLog(dir, format, pair.sparring.room, options.readLog);
  const sparringSeat = pair.sparring.player as Seat;
  const challengerSeat: Seat = sparringSeat === "p1" ? "p2" : "p1";
  const fixture = await buildHarnessFixture({
    engine: options.engine,
    snapshot: options.snapshot,
    format,
    provenance: options.provenance,
    log,
    records: { [sparringSeat]: pair.sparring, [challengerSeat]: pair.challenger } as Record<Seat, A2BattleRecord>,
    connections: {
      [sparringSeat]: pair.connections.sparring,
      [challengerSeat]: pair.connections.challenger,
    } as Record<Seat, string[][]>,
    teams: { [sparringSeat]: teams.sparring, [challengerSeat]: teams.challenger } as Record<Seat, PokemonSet[]>,
    teamLabels: { [sparringSeat]: teamLabels.sparring, [challengerSeat]: teamLabels.challenger } as Record<Seat, string>,
    known_synthetic: options.known_synthetic,
  });
  if (options.drop !== undefined) {
    // The recording is only worth keeping if the cut happened as planned.
    const seat = options.drop.who === "sparring" ? sparringSeat : challengerSeat;
    const evidence = reconnectEvidence(fixture.frames[seat]);
    const sentchoice = options.drop.when === "after_choice";
    // The held choice must be one the request offers (the bot's own: the first candidate of the actor rule).
    const offered = evidence !== undefined && enumerateCandidates(evidence.resent).candidates.some((c) => c.choice === evidence.choice);
    if (
      evidence === undefined ||
      evidence.phase === "wait" ||
      evidence.sentchoice !== sentchoice ||
      (sentchoice && !offered)
    ) {
      throw new ReconnectNotPlannedError(
        `the ${options.drop.who} bot's reconnect is not the planned one (${JSON.stringify(evidence && { ...evidence, resent: undefined })}): record again`,
      );
    }
  }
  return fixture;
}
