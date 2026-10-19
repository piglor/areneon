import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type A2BattleRecord, runA2Bot } from "@areneon/bot-client";
import {
  type Engine,
  type PlayerChannel,
  playerChannel,
  type PokemonSet,
  resolveFormat,
  rightsFor,
  type RulesSnapshot,
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

/** One recorded battle: the input log, both players' own channels, the teams, the seed and the pin. */
export interface HarnessFixture {
  schema_version: 1;
  data_class: "synthetic";
  rights: "cc0-synthetic";
  license: "CC0-1.0";
  provenance: string;
  manifest: VersionManifest;
  format_id: string;
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
  /** Each player's own channel as its bot client received it (the server's lines already dropped). */
  channels: Record<Seat, PlayerChannel>;
  /** The server's input log: both teams and every choice. Omniscient, never an actor's input. */
  input_log: { omniscient: true; lines: string[] };
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

export interface FixtureInput {
  engine: Engine;
  snapshot: RulesSnapshot;
  format: string;
  provenance: string;
  log: ServerInputLog;
  /** The two bots' records of the battle, keyed by their seat. */
  records: Record<Seat, A2BattleRecord>;
  teams: Record<Seat, PokemonSet[]>;
  /** A label for each seat's team, e.g. its file under fixtures/teams. */
  teamLabels: Record<Seat, string>;
}

/** Build the fixture of a finished battle; it must pass `verifyHarnessFixture`. */
export async function buildHarnessFixture(input: FixtureInput): Promise<HarnessFixture> {
  const { engine, snapshot, format, log, records, teams, teamLabels } = input;
  const { start, players } = parseInputLog(log.lines);
  const channels = {} as Record<Seat, PlayerChannel>;
  for (const seat of SEATS) {
    const record = records[seat];
    if (record.player !== seat || record.room.length === 0) {
      throw new HarnessError(`the ${seat} record is for ${String(record.player)}`);
    }
    if (record.channels.length !== 1) {
      throw new HarnessError(`${seat} has ${record.channels.length} channels: one connection per player is expected`);
    }
    channels[seat] = record.channels[0] as PlayerChannel;
  }
  const served = (seat: Seat): PokemonSet[] => {
    const team = engine.sim.Teams.unpack(players[seat].team);
    if (team === null) throw new HarnessError(`${seat}'s team in the input log cannot be unpacked`);
    return team;
  };
  const winner = SEATS.find((seat) => players[seat].name === log.winner) ?? null;
  const fixture: HarnessFixture = {
    schema_version: 1,
    data_class: "synthetic",
    rights: rightsFor("synthetic") as "cc0-synthetic",
    license: "CC0-1.0",
    provenance: input.provenance,
    manifest: resolveFormat(engine, format, snapshot).manifest,
    format_id: format,
    seed: start.seed,
    players: {
      p1: { name: players.p1.name, avatar: players.p1.avatar, team: teamLabels.p1 },
      p2: { name: players.p2.name, avatar: players.p2.avatar, team: teamLabels.p2 },
    },
    teams,
    served_teams: { p1: served("p1"), p2: served("p2") },
    winner,
    turns: log.turns,
    channels,
    input_log: { omniscient: true, lines: log.lines },
  };
  await verifyHarnessFixture(engine, fixture);
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

/**
 * Check a fixture without a server. Refuses (listing every problem) unless:
 * it is synthetic with matching rights; it names this engine's commit; its
 * input log's seed, format, names, avatars and packed teams are the ones the
 * fixture states; each channel is that player's own (`playerChannel`: no
 * opponent SECRET line, no other player's request) and has no
 * `|showteam|`; `served_teams` is what the input log packs and differs from
 * `teams` at most in nicknames; and the pinned sim run on the input log gives
 * the fixture's `winner` and `turns` and the channels' `|player|` lines
 * (names and avatars) and `|win|` line. The other channel lines are NOT
 * compared with the replay: it is one batch, so it holds only the last
 * `|request|`; the step-by-step comparison is #764.
 */
export async function verifyHarnessFixture(engine: Engine, fixture: HarnessFixture): Promise<void> {
  const problems: string[] = [];
  if (fixture.schema_version !== 1) problems.push(`schema_version ${String(fixture.schema_version)}`);
  if (fixture.data_class !== "synthetic" || fixture.rights !== rightsFor(fixture.data_class)) {
    problems.push("not synthetic with cc0-synthetic rights");
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
    const channel = playerChannel(fixture.channels[seat]);
    if (channel.player !== seat || channel.data_class !== "synthetic") problems.push(`${seat} channel is not its own`);
    if (channel.lines.some((line) => line.startsWith("|showteam|"))) problems.push(`${seat} channel has a team sheet`);
  }
  const replay = await replayInputLog(engine, fixture.input_log.lines);
  const win = (lines: readonly string[]): string | undefined => lines.find((line) => line.startsWith("|win|"));
  const last = replay.omniscient.filter((line) => line.startsWith("|turn|")).at(-1);
  if (last !== `|turn|${fixture.turns}`) problems.push(`turns ${fixture.turns} differs from the replay's ${String(last)}`);
  const replayWinner = SEATS.find((seat) => replay.omniscient.includes(`|win|${fixture.players[seat].name}`)) ?? null;
  if (replayWinner !== fixture.winner) problems.push(`winner ${String(fixture.winner)} differs from the replay's ${String(replayWinner)}`);
  const header = (lines: readonly string[]): string[] => lines.filter((line) => line.startsWith("|player|"));
  for (const seat of SEATS) {
    const mine = fixture.channels[seat].lines;
    if (win(mine) !== win(replay.channels[seat])) problems.push(`${seat} channel and replay disagree on the winner`);
    if (header(mine).join("\n") !== header(replay.channels[seat]).join("\n")) {
      problems.push(`${seat} channel and replay disagree on the |player| lines`);
    }
  }
  if (problems.length > 0) throw new HarnessError(`harness fixture refused: ${problems.join("; ")}`);
}

export interface BotPair {
  sparring: A2BattleRecord;
  challenger: A2BattleRecord;
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
}

/**
 * Two `runA2Bot` clients play one Bo1 on the server: the sparring bot waits
 * for the challenge, the challenger (harness-only `challenge` option) sends
 * it once the sparring bot is logged in. Both play the shared actor rule.
 */
export async function playBotVsBot(options: BotVsBotOptions): Promise<BotPair> {
  const { engine, snapshot, url, format, names, teams } = options;
  const common = { engine, snapshot, url, format, data_class: "synthetic" as const, battles: 1 };
  let loggedIn: () => void = () => {};
  const ready = new Promise<void>((done) => (loggedIn = done));
  const sparring = runA2Bot({
    ...common,
    name: names.sparring,
    challenger: names.challenger,
    team: teams.sparring,
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
  });
  const [a, b] = await Promise.all([sparring, challenger]);
  const [mine, theirs] = [a.battles[0], b.battles[0]];
  if (mine === undefined || theirs === undefined || mine.room !== theirs.room) {
    throw new HarnessError("the bots did not play the same battle");
  }
  return { sparring: mine, challenger: theirs };
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
  return await buildHarnessFixture({
    engine: options.engine,
    snapshot: options.snapshot,
    format,
    provenance: options.provenance,
    log,
    records: { [sparringSeat]: pair.sparring, [challengerSeat]: pair.challenger } as Record<Seat, A2BattleRecord>,
    teams: { [sparringSeat]: teams.sparring, [challengerSeat]: teams.challenger } as Record<Seat, PokemonSet[]>,
    teamLabels: { [sparringSeat]: teamLabels.sparring, [challengerSeat]: teamLabels.challenger } as Record<Seat, string>,
  });
}
