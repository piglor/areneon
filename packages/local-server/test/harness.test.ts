import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, loadRulesSnapshot, resolveFormat } from "@areneon/showdown-bridge";
import type { PokemonSet } from "@areneon/showdown-bridge";
import type { A2BattleRecord } from "@areneon/bot-client";
import { describe, expect, it } from "vitest";
import {
  buildHarnessFixture,
  channelFromFrames,
  ConnectionDrop,
  type DropPlan,
  findServerLog,
  type HarnessFixture,
  HarnessError,
  harnessSettings,
  parseInputLog,
  readServerInputLog,
  reconnectEvidence,
  REDACTED_CLOCK,
  redactClock,
  replayInputLog,
  roomFrames,
  tapTransport,
  upgradeFixture,
  verifyHarnessFixture,
} from "../src/harness.ts";
import { renderServerConfig, serverSettings } from "../src/index.ts";
import { FIXTURE_FORMAT, RECORDINGS, recordingProvenance } from "../src/recordings.ts";
import { COMMITTED, stored } from "./harness-fixtures.ts";
import { tempDir } from "./helpers.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const clone = (fixture: HarnessFixture): HarnessFixture => structuredClone(fixture);

describe("the harness config", () => {
  const value = (settings: ReturnType<typeof serverSettings>): unknown =>
    settings.find((setting) => setting.key === "logchallenges")?.value;

  it("turns logchallenges on, and nothing else differs from the sparring config", () => {
    expect(value(harnessSettings(8123))).toBe(true);
    const others = (settings: ReturnType<typeof serverSettings>) =>
      settings.filter((setting) => setting.key !== "logchallenges");
    expect(others(harnessSettings(8123))).toEqual(others(serverSettings(8123)));
    expect(harnessSettings(8123)).toHaveLength(serverSettings(8123).length);
  });

  it("is never the normal config: the sparring settings and config.js keep logchallenges off", () => {
    expect(value(serverSettings(8123))).toBe(false);
    expect(renderServerConfig(8123)).toContain("exports.logchallenges = false;");
    expect(renderServerConfig(8123)).not.toContain("exports.logchallenges = true;");
    expect(renderServerConfig(8123, true)).toContain("exports.logchallenges = true;");
  });
});

describe("the server's input log file", () => {
  const logFile = (dir: string, room: string, body: unknown): string => {
    const day = join(dir, "logs", "2026-10", FIXTURE_FORMAT, "2026-10-19");
    mkdirSync(day, { recursive: true });
    const path = join(day, `${room.replace(/^battle-/, "")}.log.json`);
    writeFileSync(path, typeof body === "string" ? body : JSON.stringify(body));
    return path;
  };

  it("is found by replay id under the dated directories, and only its fields are read", async () => {
    const dir = tempDir("log");
    const room = `battle-${FIXTURE_FORMAT}-7`;
    mkdirSync(join(dir, "logs", "chat"), { recursive: true });
    expect(findServerLog(dir, FIXTURE_FORMAT, room)).toBeUndefined();
    const path = logFile(dir, room, {
      inputLog: [">start {}", ">p1 team 1"],
      winner: "Someone",
      turns: 3,
      timestamp: "Mon Oct 19 2026",
      log: ["|j|x"],
    });
    logFile(dir, `battle-${FIXTURE_FORMAT}-8`, {});
    expect(findServerLog(dir, FIXTURE_FORMAT, room)).toBe(path);
    expect(await readServerInputLog(dir, FIXTURE_FORMAT, room)).toEqual({
      lines: [">start {}", ">p1 team 1"],
      winner: "Someone",
      turns: 3,
    });
  });

  it("reads a tie as an empty winner, and waits for a file that is written late", async () => {
    const dir = tempDir("late");
    const room = `battle-${FIXTURE_FORMAT}-3`;
    setTimeout(() => logFile(dir, room, { inputLog: [">start {}"], turns: 9 }), 120);
    expect(await readServerInputLog(dir, FIXTURE_FORMAT, room)).toEqual({ lines: [">start {}"], winner: "", turns: 9 });
  });

  it("refuses a missing file (logchallenges off) and a file that is not a battle log", async () => {
    const dir = tempDir("none");
    const room = `battle-${FIXTURE_FORMAT}-1`;
    await expect(readServerInputLog(dir, FIXTURE_FORMAT, room, { timeoutMs: 100 })).rejects.toThrow(/wrote no input log/);
    for (const body of [{ inputLog: "x", turns: 1 }, { inputLog: [1], turns: 1 }, { inputLog: [], turns: "1" }]) {
      logFile(dir, room, body);
      await expect(readServerInputLog(dir, FIXTURE_FORMAT, room)).rejects.toThrow(/not a server battle log/);
    }
  });
});

describe("parseInputLog", () => {
  const { fixture } = COMMITTED[0] as (typeof COMMITTED)[0];
  const lines = fixture.input_log.lines;

  it("reads the seed, the format and each player's name and avatar", () => {
    const { start, players } = parseInputLog(lines);
    expect(start).toEqual({ formatid: FIXTURE_FORMAT, seed: fixture.seed });
    expect(players.p1.name).toBe(fixture.players.p1.name);
    expect(players.p2.avatar).toBe(fixture.players.p2.avatar);
  });

  it("refuses a log without >start, with a bad player line or with a player missing", () => {
    expect(() => parseInputLog(lines.slice(1))).toThrow("does not start with >start");
    expect(() => parseInputLog([lines[0] as string, "x", "y"])).toThrow("not a >player line");
    const p1 = lines[1] as string;
    expect(() => parseInputLog([lines[0] as string, p1, p1])).toThrow("lacks a player");
  });
});

/** The foe's items, abilities and moves that appear in a channel's Team Preview. */
function foeSecretsInPreview(lines: readonly string[], foeTeam: HarnessFixture["teams"]["p1"]): string[] {
  const preview = lines.slice(0, lines.indexOf("|start")).join("\n");
  return foeTeam.flatMap((set) => [set.item, set.ability, ...set.moves]).filter((secret) => preview.includes(`|${secret}`) || preview.includes(`|X|${secret}`));
}

/** The seats that have more than one connection in a fixture. */
const reconnectSeats = (fixture: HarnessFixture): ("p1" | "p2")[] =>
  (["p1", "p2"] as const).filter((seat) => fixture.frames[seat].length > 1);

describe.each(COMMITTED)("the committed fixture $recording.file", ({ recording, fixture }) => {
  it("passes verification on the pinned engine, with no server", async () => {
    await expect(verifyHarnessFixture(engine, fixture)).resolves.toBeUndefined();
  });

  it("is synthetic CC0 for the pinned format, recorded for the documented bots and teams", () => {
    expect(fixture.data_class).toBe("synthetic");
    expect(fixture.rights).toBe("cc0-synthetic");
    expect(fixture.license).toBe("CC0-1.0");
    expect(fixture.provenance).toBe(recordingProvenance(engine.commit, recording.drop));
    expect(fixture.manifest).toEqual(resolveFormat(engine, FIXTURE_FORMAT, snapshot).manifest);
    expect(fixture.format_id).toBe(FIXTURE_FORMAT);
    const names = [fixture.players.p1.name, fixture.players.p2.name].sort();
    expect(names).toEqual([recording.names.challenger, recording.names.sparring].sort());
    expect(fixture.input_log.omniscient).toBe(true);
    expect(fixture.input_log.lines.some((line) => line.startsWith("|t:|"))).toBe(false);
    expect(fixture.seed).toMatch(/^sodium,[0-9a-f]{32}$/);
    // No wall clock anywhere: every |t:| line of every connection is |t:|0.
    expect(JSON.stringify(fixture.frames)).not.toMatch(/\|t:\|(?!0\b)/);
  });

  it("has a second connection for the bot the recording cut, and for no other", () => {
    const cut = recording.drop === undefined ? undefined : (fixture.players.p1.name === recording.names[recording.drop.who] ? "p1" : "p2");
    expect(reconnectSeats(fixture)).toEqual(cut === undefined ? [] : [cut]);
    for (const seat of ["p1", "p2"] as const) {
      expect(fixture.frames[seat]).toHaveLength(seat === cut ? 2 : 1);
      expect(fixture.channels[seat]).toHaveLength(fixture.frames[seat].length);
    }
    if (cut === undefined) return;
    // A real reconnect: the second connection starts at its own |init|battle and holds the request again.
    const evidence = reconnectEvidence(fixture.frames[cut]);
    expect(evidence).toBeDefined();
    expect(evidence?.phase).not.toBe("wait");
    expect(evidence?.sentchoice).toBe(recording.drop?.when === "after_choice");
    expect(fixture.frames[cut][1]?.[0]).toContain("\n|init|battle");
  });

  it("records the teams as sent and as served, which differ at most in nicknames", () => {
    for (const seat of ["p1", "p2"] as const) {
      const core = (team: typeof fixture.teams.p1) => team.map(({ species, item, ability, moves }) => [species, item, ability, moves]);
      expect(core(fixture.served_teams[seat])).toEqual(core(fixture.teams[seat]));
      expect(fixture.served_teams[seat]).toHaveLength(6);
    }
  });

  it("holds only each player's own view: no team sheet, own requests, and the foe's sheet nowhere in its preview", () => {
    for (const seat of ["p1", "p2"] as const) {
      const foe = seat === "p1" ? "p2" : "p1";
      for (const channel of fixture.channels[seat]) {
        expect(channel.player).toBe(seat);
        expect(channel.lines.some((line) => line.startsWith("|showteam|"))).toBe(false);
        for (const line of channel.lines.filter((l) => l.startsWith("|request|"))) expect(line).toContain(`"id":"${seat}"`);
        expect(foeSecretsInPreview(channel.lines, fixture.teams[foe])).toEqual([]);
      }
      // Negative control: the same scan finds a foe item put into the preview.
      const { lines } = fixture.channels[seat][0] as (typeof fixture.channels.p1)[0];
      const item = fixture.teams[foe][0]?.item as string;
      const tainted = [...lines.slice(0, 3), `|poke|${foe}|X|${item}`, ...lines.slice(3)];
      expect(foeSecretsInPreview(tainted, fixture.teams[foe])).toContain(item);
    }
  });

  it("would catch a secret: a foe line, a foe request or a team sheet put into a channel is refused", async () => {
    const foeLines = (fixture.channels.p2[0] as (typeof fixture.channels.p1)[0]).lines;
    const foeRequest = foeLines.find((l) => l.startsWith("|request|")) as string;
    expect(foeRequest).toContain('"id":"p2"');
    for (const secret of [foeRequest, "|split|p2", `|showteam|p2|${fixture.served_teams.p2[0]?.species}`]) {
      const copy = clone(fixture);
      (copy.channels.p1[0] as (typeof fixture.channels.p1)[0]).lines.splice(10, 0, secret);
      await expect(verifyHarnessFixture(engine, copy), secret.slice(0, 30)).rejects.toThrow();
    }
    // And the untouched fixture passes, so the refusals above are the secret's doing.
    await expect(verifyHarnessFixture(engine, fixture)).resolves.toBeUndefined();
  });

  it("refuses each way it can be wrong", async () => {
    const refused = async (change: (f: HarnessFixture) => void, reason: string | RegExp): Promise<void> => {
      const copy = clone(fixture);
      change(copy);
      await expect(verifyHarnessFixture(engine, copy)).rejects.toThrow(reason);
    };
    const channel = (f: HarnessFixture, seat: "p1" | "p2", c = 0) => f.channels[seat][c] as (typeof f.channels.p1)[0];
    await refused((f) => ((f as { schema_version: number }).schema_version = 1), "schema_version 1 is not 2 or 3");
    await refused((f) => ((f as { data_class: string }).data_class = "private"), "not synthetic");
    await refused((f) => ((f as { rights: string }).rights = "owner-private"), "not synthetic");
    await refused((f) => (f.manifest.showdown_commit = "0".repeat(40)), "another Showdown commit");
    await refused((f) => (f.seed = "sodium,00"), "input log start differs");
    await refused((f) => (f.format_id = "gen9ou"), "input log start differs");
    await refused((f) => (f.players.p1.name = "other"), "p1 name or avatar differs");
    await refused((f) => (f.players.p2.avatar = "0"), "p2 name or avatar differs");
    await refused((f) => (f.served_teams.p1 = f.served_teams.p1.map((set) => ({ ...set, name: "Nick" }))), "p1 served team differs from the input log");
    await refused((f) => (f.served_teams.p2[0] = { ...(f.served_teams.p2[0] as PokemonSet), item: "Nothing" }), "p2 served team differs");
    await refused((f) => (f.teams.p1[0] = { ...(f.teams.p1[0] as PokemonSet), item: "Nothing" }), "p1 sent and served teams differ in more than nicknames");
    await refused((f) => (f.turns += 1), /turns \d+ differs from the replay/);
    await refused((f) => (f.winner = f.winner === "p1" ? "p2" : "p1"), /winner p. differs from the replay/);
    await refused((f) => (f.winner = null), /winner null differs/);
    await refused((f) => (channel(f, "p2").player = "p1"), /p2 channel is not its own|holds a request for/);
    await refused((f) => channel(f, "p1").lines.push("|showteam|p2|x"), "p1 channel has a team sheet");
    await refused((f) => channel(f, "p1").lines.push("|split|p2"), /only the opponent may see/);
    await refused(
      (f) => {
        const last = f.channels.p1.length - 1;
        channel(f, "p1", last).lines = channel(f, "p1", last).lines.filter((line) => !line.startsWith("|win|"));
      },
      "p1 channel and replay disagree on the winner",
    );
    await refused(
      (f) => (channel(f, "p2").lines = channel(f, "p2").lines.map((l) => l.replace(/^(\|player\|p1\|[^|]*\|)\d+/, "$19999"))),
      "p2 channel and replay disagree on the |player| lines",
    );
    await refused((f) => (f.channels.p1 = []), "p1 has 0 channels for");
  });

  it("holds raw server frames of its own room that rebuild its channels, connection by connection", () => {
    for (const seat of ["p1", "p2"] as const) {
      fixture.frames[seat].forEach((frames, c) => {
        expect(frames.length).toBeGreaterThan(c === 0 ? 10 : 3);
        expect(frames.every((frame) => frame.startsWith(`>${fixture.room}\n`))).toBe(true);
        expect(frames[0]).toContain("|init|battle");
        const raw = frames.join("\n");
        // Server-only lines are there, as received; the adapter drops them from the channel.
        for (const kind of ["|j|", "|title|", REDACTED_CLOCK]) expect(raw).toContain(kind);
        expect(fixture.channels[seat][c]?.lines.some((line) => line.startsWith("|j|"))).toBe(false);
        // Only the seat's own stream: its requests, no team sheet, no split marker, no clock.
        expect(raw).not.toMatch(/\|showteam\||\|split\||\|t:\|(?!0\b)/);
        const built = channelFromFrames(frames, fixture.room, seat);
        expect(built.channel).toEqual(fixture.channels[seat][c]);
        expect(built.dropped).toEqual([]);
        expect(built.requests).toBe(fixture.channels[seat][c]?.lines.filter((line) => line.startsWith("|request|")).length);
      });
      expect(fixture.frames[seat][0]?.join("\n")).toContain("|inactive|");
    }
  });

  it("refuses frames that are malformed, not its own, clocked or not what the channel is", async () => {
    const refused = async (change: (f: HarnessFixture) => void, reason: string | RegExp): Promise<void> => {
      const copy = clone(fixture);
      change(copy);
      await expect(verifyHarnessFixture(engine, copy)).rejects.toThrow(reason);
    };
    const first = (f: HarnessFixture, seat: "p1" | "p2") => f.frames[seat][0] as string[];
    const at = (f: HarnessFixture, seat: "p1" | "p2", prefix: string): number =>
      first(f, seat).findIndex((frame) => frame.includes(`\n${prefix}`));
    // A seat with several connections is named with the connection; one with a single one by its seat.
    const name = (seat: "p1" | "p2", c = 0) => (fixture.frames[seat].length === 1 ? seat : `${seat} connection ${c + 1}`);
    await refused((f) => (f.frames.p1 = []), "p1 frames are not a non-empty list of connections");
    await refused((f) => (f.frames.p1[0] = []), `${name("p1")} frames are not a non-empty list of strings`);
    await refused((f) => (f.frames.p2[0] = [1 as never]), `${name("p2")} frames are not a non-empty list of strings`);
    await refused((f) => (f.room = "battle-other"), /frames are not all headed >battle-other/);
    await refused((f) => (first(f, "p1")[2] = "|inactive|no header"), `${name("p1")} frames are not all headed`);
    await refused(
      (f) => (first(f, "p1")[0] = (first(f, "p1")[0] as string).replace("|init|battle", "|title|x")),
      new RegExp(`${name("p1")} frames do not start with \\|init\\|battle`),
    );
    await refused((f) => first(f, "p2").push(`>${f.room}\n|showteam|p1|x`), `${name("p2")} frames have a team sheet or a |split| marker`);
    await refused((f) => first(f, "p1").push(`>${f.room}\n|split|p1`), `${name("p1")} frames have a team sheet or a |split| marker`);
    await refused(
      (f) => (first(f, "p1")[at(f, "p1", "|t:|")] = `>${f.room}\n|t:|1760000000`),
      `${name("p1")} frames have a wall-clock |t:| line`,
    );
    // A foe's request in the stream: not its own (and the adapter refuses it too).
    await refused((f) => {
      const foe = (first(f, "p2")[at(f, "p2", "|request|")] as string).split("\n")[1] as string;
      first(f, "p1").splice(at(f, "p1", "|request|"), 0, `>${f.room}\n${foe}`);
    }, new RegExp(`${name("p1")} frames hold a request that is not its own`));
    // The stored channel is derived: it must be what the frames give.
    await refused((f) => f.channels.p1[0]?.lines.splice(5, 1), `${name("p1")} channel is not what its frames give`);
    await refused((f) => first(f, "p2").splice(at(f, "p2", "|move|"), 1), `${name("p2")} channel is not what its frames give`);
    await refused((f) => (first(f, "p1")[4] = `>${f.room}\n|t:|0\n|gametype|singles`), `${name("p1")} channel is not what its frames give`);
    await refused((f) => (first(f, "p1")[7] = `>${f.room}\n|request|{`), new RegExp(`${name("p1")} frames cannot be replayed`));
    await refused((f) => first(f, "p1").push(`>${f.room}\n|init|`), new RegExp(`${name("p1")} frames cannot be replayed: .* is not a battle room`));
  });

  it("refuses a seat whose connections and channels do not pair up, or whose later connection is wrong", async () => {
    const refused = async (change: (f: HarnessFixture) => void, reason: string | RegExp): Promise<void> => {
      const copy = clone(fixture);
      change(copy);
      await expect(verifyHarnessFixture(engine, copy)).rejects.toThrow(reason);
    };
    // A second connection's frames without a channel for it: the stored channels are one short.
    const single = reconnectSeats(fixture).includes("p1") ? "p2" : "p1";
    await refused((f) => f.frames[single].push([...(f.frames[single][0] as string[])]), new RegExp(`${single} has 1 channels for 2 connections`));
    await refused((f) => f.channels[single].push(f.channels[single][0] as (typeof f.channels.p1)[0]), new RegExp(`${single} has 2 channels for 1 connections`));
    for (const seat of reconnectSeats(fixture)) {
      const second = (f: HarnessFixture) => f.frames[seat][1] as string[];
      await refused((f) => (second(f)[0] = (second(f)[0] as string).replace("\n|init|battle", "")), `${seat} connection 2 frames do not start with |init|battle`);
      await refused((f) => second(f).push(`>${f.room}\n|t:|1760000000`), `${seat} connection 2 frames have a wall-clock |t:| line`);
      await refused((f) => (f.channels[seat][1] = f.channels[seat][0] as (typeof f.channels.p1)[0]), `${seat} connection 2 channel is not what its frames give`);
      await refused((f) => second(f).splice(0, 1), /connection 2 frames (are not all headed|do not start)|cannot be replayed/);
      // The rejoin goes back to an older request, or reuses an rqid for another one.
      const older = (f: HarnessFixture, rqid: string) => {
        const frames = second(f);
        const i = frames.findIndex((frame) => frame.includes("|request|"));
        frames[i] = (frames[i] as string).replace(/"rqid":\d+/, rqid);
      };
      await refused((f) => older(f, '"rqid":1'), `${seat} connection 2 starts with a request older than, or different from, the one before it`);
      await refused((f) => {
        const frames = second(f);
        const i = frames.findIndex((frame) => frame.includes("|request|"));
        const before = reconnectEvidence(f.frames[seat]);
        expect(before).toBeDefined();
        // The same rqid, a request body that is not the same.
        frames[i] = (frames[i] as string).replace(/"condition":"[^"]*"/, '"condition":"1/1"');
      }, /connection 2 (starts with a request older than, or different from|channel is not what its frames give)/);
    }
  });

  it("replays on the pinned sim to the recorded result, the same every time", async () => {
    const a = await replayInputLog(engine, fixture.input_log.lines);
    const b = await replayInputLog(engine, fixture.input_log.lines);
    expect(a).toEqual(b);
    expect(a.omniscient.some((line) => line.startsWith("|t:|"))).toBe(false);
    const winner = fixture.winner === null ? "|tie" : `|win|${fixture.players[fixture.winner].name}`;
    expect(a.omniscient).toContain(winner);
    expect(a.omniscient.filter((line) => line.startsWith("|turn|")).at(-1)).toBe(`|turn|${fixture.turns}`);
  });
});

describe("schema versions", () => {
  const raw = stored(RECORDINGS[0]?.file as string);

  it("keeps the fixtures recorded before reconnects as they are: version 2, one connection per seat", () => {
    expect(raw.schema_version).toBe(2);
    expect((raw as { frames: { p1: string[] } }).frames.p1.every((frame) => typeof frame === "string")).toBe(true);
    const upgraded = upgradeFixture(raw);
    expect(upgraded.schema_version).toBe(3);
    expect(upgraded.frames.p1).toEqual([(raw as { frames: { p1: string[] } }).frames.p1]);
    expect(upgraded.channels.p2).toEqual([(raw as { channels: { p2: unknown } }).channels.p2]);
    const { schema_version: _a, frames: _b, channels: _c, ...rest } = upgraded;
    const { schema_version: _d, frames: _e, channels: _f, ...before } = raw;
    expect(rest).toEqual(before);
  });

  it("checks a version 2 fixture as its upgrade, and hands a version 3 one back as it is", async () => {
    await expect(verifyHarnessFixture(engine, raw)).resolves.toBeUndefined();
    const v3 = upgradeFixture(raw);
    expect(upgradeFixture(v3)).toBe(v3);
    const bad = structuredClone(raw) as { frames: { p1: string[] } };
    bad.frames.p1.splice(bad.frames.p1.findIndex((frame) => frame.includes("\n|move|")), 1);
    await expect(verifyHarnessFixture(engine, bad as never)).rejects.toThrow("p1 channel is not what its frames give");
  });

  it("refuses any other version, naming it", () => {
    for (const version of [1, 4, "3"]) {
      expect(() => upgradeFixture({ ...raw, schema_version: version } as never)).toThrow(
        new HarnessError(`harness fixture refused: schema_version ${String(version)} is not 2 or 3`),
      );
    }
  });
});

describe("buildHarnessFixture", () => {
  const { recording, fixture } = COMMITTED[0] as (typeof COMMITTED)[0];
  const [reconnect] = COMMITTED.filter((c) => c.recording.drop !== undefined) as typeof COMMITTED;
  const record = (seat: "p1" | "p2", over: Partial<A2BattleRecord> = {}, from = fixture): A2BattleRecord =>
    ({
      room: from.room,
      player: seat,
      result: "win",
      channels: from.channels[seat],
      ...over,
    }) as A2BattleRecord;
  const input = (
    over: { log?: { winner: string }; records?: Record<"p1" | "p2", A2BattleRecord>; connections?: Record<"p1" | "p2", string[][]> } = {},
    from = fixture,
    labels = { p1: recording.teams.challenger, p2: recording.teams.sparring },
  ) => ({
    engine,
    snapshot,
    format: FIXTURE_FORMAT,
    provenance: from.provenance,
    log: { lines: from.input_log.lines, winner: "", turns: from.turns, ...over.log },
    records: over.records ?? { p1: record("p1", {}, from), p2: record("p2", {}, from) },
    connections: over.connections ?? { p1: from.frames.p1, p2: from.frames.p2 },
    teams: from.teams,
    teamLabels: labels,
  });
  const winnerOf = (f: HarnessFixture) => (f.winner === null ? "" : f.players[f.winner].name);

  it("rebuilds the committed fixture from its parts", async () => {
    const built = await buildHarnessFixture(input({ log: { winner: winnerOf(fixture) } }));
    expect(built).toEqual({ ...fixture, players: { p1: { ...fixture.players.p1, team: built.players.p1.team }, p2: { ...fixture.players.p2, team: built.players.p2.team } } });
  });

  it("rebuilds a fixture with a reconnect: both of the bot's connections, and the channel of each", async () => {
    const { fixture: from, recording: made } = reconnect as (typeof COMMITTED)[0];
    // The server's lobby frames and the frames of other rooms come in between, on every connection.
    const noisy = (connections: string[][]): string[][] =>
      connections.map((frames) => ["|updateuser| guest|0|1", ">lobby\n|queryresponse|rooms|{}", ...frames, ">battle-other-9\n|init|battle"]);
    const built = await buildHarnessFixture(
      input(
        { log: { winner: winnerOf(from) }, connections: { p1: noisy(from.frames.p1), p2: noisy(from.frames.p2) } },
        from,
        { p1: made.teams.challenger, p2: made.teams.sparring },
      ),
    );
    expect(built.frames).toEqual(from.frames);
    expect(built.channels).toEqual(from.channels);
    expect(reconnectSeats(built)).toHaveLength(1);
  });

  it("builds from every frame the bots received: other rooms dropped, the clock redacted, the rest as received", async () => {
    const noisy = (connections: string[][]): string[][] => [
      [
        "|updateuser| guest|0|1",
        ">lobby\n|queryresponse|rooms|{}",
        ...(connections[0] as string[]).map((frame) => frame.replace(REDACTED_CLOCK, "|t:|1760000000")),
        ">battle-other-9\n|init|battle",
      ],
    ];
    const built = await buildHarnessFixture(
      input({ log: { winner: winnerOf(fixture) }, connections: { p1: noisy(fixture.frames.p1), p2: noisy(fixture.frames.p2) } }),
    );
    expect(built.frames).toEqual(fixture.frames);
  });

  it("refuses a battle whose bots are in different rooms", async () => {
    const records = { p1: record("p1"), p2: record("p2", { room: "battle-x" }) };
    const p2 = (fixture.frames.p2[0] as string[]).map((frame) => frame.replace(fixture.room, "battle-x"));
    await expect(buildHarnessFixture(input({ records, connections: { p1: fixture.frames.p1, p2: [p2] } }))).rejects.toThrow(
      "the bots did not play the same battle",
    );
  });

  it("refuses a record for the wrong seat or room, and channels that are not one per connection", async () => {
    await expect(buildHarnessFixture(input({ records: { p1: record("p2"), p2: record("p2") } }))).rejects.toThrow(
      "the p1 record is for p2",
    );
    await expect(buildHarnessFixture(input({ records: { p1: record("p1", { room: "" }), p2: record("p2") } }))).rejects.toThrow(
      "the p1 record is for p1",
    );
    const two = record("p2", { channels: [...fixture.channels.p2, ...fixture.channels.p2] });
    await expect(buildHarnessFixture(input({ records: { p1: record("p1"), p2: two } }))).rejects.toThrow(
      /p2 has 2 channels for 1 connections/,
    );
  });

  it("refuses a bot that never received the room", async () => {
    await expect(buildHarnessFixture(input({ connections: { p1: [[">lobby\n|x"]], p2: fixture.frames.p2 } }))).rejects.toThrow(
      `no connection received ${fixture.room}`,
    );
  });
});

describe("the transport tap", () => {
  const fake = () => {
    const sent: string[] = [];
    const listeners = { frame: [] as ((frame: string) => void)[], close: [] as ((code: number) => void)[] };
    let closed = 0;
    const inner = {
      send: (line: string) => void sent.push(line),
      onFrame: (cb: (frame: string) => void) => void listeners.frame.push(cb),
      onClose: (cb: (code: number) => void) => void listeners.close.push(cb),
      close: () => void (closed += 1),
    };
    return { inner, sent, listeners, closed: () => closed };
  };

  it("copies each frame as received, in order, and changes nothing else", () => {
    const { inner, sent, listeners, closed } = fake();
    const seen: string[] = [];
    const copy: string[] = [];
    const tap = tapTransport(inner, copy);
    tap.onFrame((frame) => seen.push(frame));
    const codes: number[] = [];
    tap.onClose((code) => codes.push(code));
    for (const frame of [">r\n|a|1\n|b|2", "|c|3", ">r\n|t:|1760000000"]) for (const cb of listeners.frame) cb(frame);
    for (const cb of listeners.close) cb(1000);
    tap.send("/join r");
    tap.close();
    expect(copy).toEqual([">r\n|a|1\n|b|2", "|c|3", ">r\n|t:|1760000000"]);
    expect(seen).toEqual(copy);
    expect([sent, codes, closed()]).toEqual([["/join r"], [1000], 1]);
  });

  it("copies a frame before the listener sees it, so a throwing listener loses nothing", () => {
    const { inner, listeners } = fake();
    const copy: string[] = [];
    tapTransport(inner, copy).onFrame(() => {
      throw new Error("boom");
    });
    expect(() => listeners.frame[0]?.("|x")).toThrow("boom");
    expect(copy).toEqual(["|x"]);
  });
});

describe("ConnectionDrop (the recorder's cut of one connection)", () => {
  const room = ">battle-x\n";
  const request = (n: number) => `${room}|request|{"n":${n}}\n|inactive|Time left`;
  const fake = () => {
    const sent: string[] = [];
    const delivered: string[] = [];
    const listeners = { frame: [] as ((frame: string) => void)[], close: [] as ((code: number) => void)[] };
    let closed = 0;
    const inner = {
      send: (line: string) => void sent.push(line),
      onFrame: (cb: (frame: string) => void) => void listeners.frame.push(cb),
      onClose: (cb: (code: number) => void) => void listeners.close.push(cb),
      close: () => void (closed += 1),
    };
    /** What the server sends: every listener of the transport sees the frame. */
    const push = (frame: string) => listeners.frame.forEach((cb) => cb(frame));
    return { inner, sent, delivered, listeners, push, closed: () => closed };
  };
  const choose = (n: number) => `battle-x|/choose move 1, move 2|${n}`;
  const plan = (when: DropPlan["when"], request = 2): DropPlan => ({ who: "sparring", request, when });

  it("cuts when the request arrives, before the bot can answer it, and then withholds everything", () => {
    const { inner, sent, listeners, push, closed } = fake();
    const wrapped = new ConnectionDrop(plan("before_choice")).wrap("sparring", 0, inner);
    const seen: string[] = [];
    wrapped.onFrame((frame) => seen.push(frame));
    const codes: number[] = [];
    wrapped.onClose((code) => codes.push(code));
    push(request(1));
    push(`${room}|request|null`);
    push(`${room}|j|x`);
    expect(closed()).toBe(0);
    push(`${room}|j|a\n|request|{"n":2}`);
    expect(closed()).toBe(1);
    // Delivered to the bot (it saw the request), then nothing more: not a frame, not a send.
    expect(seen.at(-1)).toContain('"n":2');
    push(request(3));
    wrapped.send(choose(3));
    expect(seen).toHaveLength(4);
    expect(sent).toEqual([]);
    // The close the transport reports still reaches the session, and close() still closes.
    for (const cb of listeners.close) cb(1000);
    wrapped.close();
    expect([codes, closed()]).toEqual([[1000], 2]);
  });

  it("cuts right after the bot sends its choice for the request, even one sent while the request is handled", () => {
    const { inner, sent, push, closed } = fake();
    const wrapped = new ConnectionDrop(plan("after_choice")).wrap("sparring", 0, inner);
    const seen: string[] = [];
    wrapped.onFrame((frame) => {
      seen.push(frame);
      // A bot that answers at once sends from inside the frame handler.
      if (frame.includes('"n":3')) wrapped.send(choose(9));
    });
    push(request(1));
    wrapped.send(choose(1)); // before the request: an ordinary choice
    wrapped.send("battle-x|/timer on");
    push(request(2));
    expect(closed()).toBe(0);
    wrapped.send("battle-x|/rejectopenteamsheets"); // not a choice
    expect(closed()).toBe(0);
    wrapped.send(choose(2));
    expect(sent).toEqual([choose(1), "battle-x|/timer on", "battle-x|/rejectopenteamsheets", choose(2)]);
    expect(closed()).toBe(1);
    push(request(3));
    expect(seen).toHaveLength(2);

    // The same when the choice is sent inside the handler of the request's frame.
    const again = fake();
    const quick = new ConnectionDrop(plan("after_choice", 1)).wrap("sparring", 0, again.inner);
    quick.onFrame(() => quick.send(choose(1)));
    again.push(request(1));
    expect(again.sent).toEqual([choose(1)]);
    expect(again.closed()).toBe(1);
  });

  it("leaves the other bot and the cut bot's later connections alone, but for the pause and the signal", () => {
    const drop = new ConnectionDrop(plan("before_choice"));
    const a = fake();
    expect(drop.wrap("challenger", 0, a.inner)).toBe(a.inner);
    expect(drop.wrap("challenger", 1, a.inner)).toBe(a.inner);
    const later = fake();
    const wrapped = drop.wrap("sparring", 1, later.inner);
    const seen: string[] = [];
    wrapped.onFrame((frame) => seen.push(frame));
    later.push(request(1));
    wrapped.send(choose(1));
    expect(seen).toHaveLength(1);
    expect(later.sent).toEqual([choose(1)]);
    expect(later.closed()).toBe(0);
  });

  it("holds the other bot's choice for the same request until the cut bot has its request again", () => {
    const drop = new ConnectionDrop(plan("after_choice"));
    const other = fake();
    const held = drop.wrap("challenger", 0, other.inner);
    held.onFrame(() => {});
    const codes: number[] = [];
    held.onClose((code) => codes.push(code));
    other.push(request(1));
    held.send(choose(1)); // an earlier request: sent at once
    other.push(request(2));
    held.send("battle-x|/timer on"); // not a choice
    held.send(choose(2)); // the request the cut bot answers: held
    held.send(choose(2)); // (a second one: held too, in order)
    expect(other.sent).toEqual([choose(1), "battle-x|/timer on"]);
    other.push(request(3));
    held.send(choose(3)); // a later request: not held
    expect(other.sent.at(-1)).toBe(choose(3));

    // The cut bot's next connection: a frame that is no request does not open the gate, a request does.
    const rejoin = fake();
    const watched = drop.wrap("sparring", 1, rejoin.inner);
    const order: string[] = [];
    watched.onFrame((frame) => order.push(frame.includes("|request|") ? "request seen" : "frame seen"));
    rejoin.push(`${room}|init|battle\n|title|x`);
    expect(other.sent).not.toContain(choose(2));
    rejoin.push(request(2));
    expect(other.sent.slice(-2)).toEqual([choose(2), choose(2)]);
    // The bot saw the request before the held choice went out.
    expect(order).toEqual(["frame seen", "request seen"]);
    // Once open, a late choice for that request is not held any more.
    const late = fake();
    const lateHeld = drop.wrap("challenger", 0, late.inner);
    lateHeld.onFrame(() => {});
    late.push(request(1));
    late.push(request(2));
    lateHeld.send(choose(2));
    expect(late.sent).toEqual([choose(2)]);
    held.close();
    for (const cb of other.listeners.close) cb(1006);
    expect([codes, other.closed()]).toEqual([[1006], 1]);
  });

  it("holds nothing for a bot that is cut before its choice", () => {
    const drop = new ConnectionDrop(plan("before_choice"));
    const other = fake();
    expect(drop.wrap("challenger", 0, other.inner)).toBe(other.inner);
  });
});

describe("reconnectEvidence", () => {
  const withDrop = COMMITTED.filter((c) => c.recording.drop !== undefined);

  it("sees a re-sent request after the second connection's own |init|battle, with or without |sentchoice|", () => {
    expect(withDrop.map((c) => c.recording.drop?.when).sort()).toEqual(["after_choice", "before_choice"]);
    for (const { recording, fixture } of withDrop) {
      const seat = reconnectSeats(fixture)[0] as "p1" | "p2";
      const evidence = reconnectEvidence(fixture.frames[seat]);
      expect(evidence?.sentchoice).toBe(recording.drop?.when === "after_choice");
      expect(evidence?.phase).not.toBe("wait");
      expect(evidence?.request_seq).toBeGreaterThan(1);
    }
  });

  it("sees none in a single connection, in connections that do not repeat a request, or in empty ones", () => {
    const { fixture } = COMMITTED[0] as (typeof COMMITTED)[0];
    const frames = fixture.frames.p1[0] as string[];
    expect(reconnectEvidence([frames])).toBeUndefined();
    expect(reconnectEvidence([frames, frames])).toBeUndefined();
    expect(reconnectEvidence([[], frames])).toBeUndefined();
    expect(reconnectEvidence([frames, [`>${fixture.room}\n|init|battle`]])).toBeUndefined();
    expect(reconnectEvidence([])).toBeUndefined();
  });
});

describe("redactClock and roomFrames", () => {
  it("replaces only the number of a |t:| line, keeping lines and frame boundaries", () => {
    expect(redactClock(">r\n|t:|1760000000\n|move|x|t:|5\n|t:|")).toBe(`>r\n${REDACTED_CLOCK}\n|move|x|t:|5\n${REDACTED_CLOCK}`);
    expect(redactClock("|j|x")).toBe("|j|x");
  });

  it("keeps the room's frames of each connection that received it, in the order the connections were opened", () => {
    const frames = [">r\n|init|battle", ">lobby\n|x", "|y", ">r\n|t:|1"];
    const kept = [">r\n|init|battle", `>r\n${REDACTED_CLOCK}`];
    expect(roomFrames([frames], "r")).toEqual([kept]);
    // A connection that never joined the room (only a login) is left out; several that did are kept apart.
    expect(roomFrames([["|z"], frames], "r")).toEqual([kept]);
    expect(roomFrames([frames, ["|z"], [">r\n|init|battle\n|a"]], "r")).toEqual([kept, [">r\n|init|battle\n|a"]]);
    expect(() => roomFrames([["|z"]], "r")).toThrow("no connection received r");
  });
});

describe("RECORDINGS", () => {
  it("names each fixture file and uses distinct bot names", () => {
    expect(new Set(RECORDINGS.map((r) => r.file)).size).toBe(RECORDINGS.length);
    const names = RECORDINGS.flatMap((r) => [r.names.sparring, r.names.challenger]);
    expect(new Set(names).size).toBe(names.length);
    for (const name of names) expect(name.length).toBeLessThanOrEqual(18);
  });
});
