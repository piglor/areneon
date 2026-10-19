import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, loadRulesSnapshot, resolveFormat } from "@areneon/showdown-bridge";
import type { PokemonSet } from "@areneon/showdown-bridge";
import type { A2BattleRecord } from "@areneon/bot-client";
import { describe, expect, it } from "vitest";
import {
  buildHarnessFixture,
  channelFromFrames,
  findServerLog,
  type HarnessFixture,
  harnessSettings,
  parseInputLog,
  readServerInputLog,
  REDACTED_CLOCK,
  redactClock,
  replayInputLog,
  roomFrames,
  tapTransport,
  verifyHarnessFixture,
} from "../src/harness.ts";
import { renderServerConfig, serverSettings } from "../src/index.ts";
import { FIXTURE_FORMAT, RECORDINGS, recordingProvenance } from "../src/recordings.ts";
import { COMMITTED } from "./harness-fixtures.ts";
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

describe.each(COMMITTED)("the committed fixture $recording.file", ({ recording, fixture }) => {
  it("passes verification on the pinned engine, with no server", async () => {
    await expect(verifyHarnessFixture(engine, fixture)).resolves.toBeUndefined();
  });

  it("is synthetic CC0 for the pinned format, recorded for the documented bots and teams", () => {
    expect(fixture.data_class).toBe("synthetic");
    expect(fixture.rights).toBe("cc0-synthetic");
    expect(fixture.license).toBe("CC0-1.0");
    expect(fixture.provenance).toBe(recordingProvenance(engine.commit));
    expect(fixture.manifest).toEqual(resolveFormat(engine, FIXTURE_FORMAT, snapshot).manifest);
    expect(fixture.format_id).toBe(FIXTURE_FORMAT);
    const names = [fixture.players.p1.name, fixture.players.p2.name].sort();
    expect(names).toEqual([recording.names.challenger, recording.names.sparring].sort());
    expect(fixture.input_log.omniscient).toBe(true);
    expect(fixture.input_log.lines.some((line) => line.startsWith("|t:|"))).toBe(false);
    expect(fixture.seed).toMatch(/^sodium,[0-9a-f]{32}$/);
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
      const { lines } = fixture.channels[seat];
      expect(fixture.channels[seat].player).toBe(seat);
      expect(lines.some((line) => line.startsWith("|showteam|"))).toBe(false);
      for (const line of lines.filter((l) => l.startsWith("|request|"))) expect(line).toContain(`"id":"${seat}"`);
      expect(foeSecretsInPreview(lines, fixture.teams[foe])).toEqual([]);
      // Negative control: the same scan finds a foe item put into the preview.
      const item = fixture.teams[foe][0]?.item as string;
      const tainted = [...lines.slice(0, 3), `|poke|${foe}|X|${item}`, ...lines.slice(3)];
      expect(foeSecretsInPreview(tainted, fixture.teams[foe])).toContain(item);
    }
  });

  it("would catch a secret: a foe line, a foe request or a team sheet put into a channel is refused", async () => {
    const foeLines = fixture.channels.p2.lines;
    const foeRequest = foeLines.find((l) => l.startsWith("|request|")) as string;
    expect(foeRequest).toContain('"id":"p2"');
    for (const secret of [foeRequest, "|split|p2", `|showteam|p2|${fixture.served_teams.p2[0]?.species}`]) {
      const copy = clone(fixture);
      copy.channels.p1.lines.splice(10, 0, secret);
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
    await refused((f) => ((f as { schema_version: number }).schema_version = 1), "schema_version 1");
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
    await refused((f) => (f.channels.p2.player = "p1"), /p2 channel is not its own|holds a request for/);
    await refused((f) => f.channels.p1.lines.push("|showteam|p2|x"), "p1 channel has a team sheet");
    await refused((f) => f.channels.p1.lines.push("|split|p2"), /only the opponent may see/);
    await refused(
      (f) => (f.channels.p1.lines = f.channels.p1.lines.filter((line) => !line.startsWith("|win|"))),
      "p1 channel and replay disagree on the winner",
    );
    await refused(
      (f) => (f.channels.p2.lines = f.channels.p2.lines.map((l) => l.replace(/^(\|player\|p1\|[^|]*\|)\d+/, "$19999"))),
      "p2 channel and replay disagree on the |player| lines",
    );
  });

  it("holds raw server frames of its own room that rebuild its channels", () => {
    for (const seat of ["p1", "p2"] as const) {
      const frames = fixture.frames[seat];
      expect(frames.length).toBeGreaterThan(10);
      expect(frames.every((frame) => frame.startsWith(`>${fixture.room}\n`))).toBe(true);
      expect(frames[0]).toContain("|init|battle");
      const raw = frames.join("\n");
      // Server-only lines are there, as received; the adapter drops them from the channel.
      for (const kind of ["|j|", "|inactive|", "|title|", REDACTED_CLOCK]) expect(raw).toContain(kind);
      expect(fixture.channels[seat].lines.some((line) => line.startsWith("|j|"))).toBe(false);
      // Only the seat's own stream: its requests, no team sheet, no split marker, no clock.
      expect(raw).not.toMatch(/\|showteam\||\|split\||\|t:\|(?!0\b)/);
      const built = channelFromFrames(frames, fixture.room, seat);
      expect(built.channel).toEqual(fixture.channels[seat]);
      expect(built.dropped).toEqual([]);
      expect(built.requests).toBe(fixture.channels[seat].lines.filter((line) => line.startsWith("|request|")).length);
    }
  });

  it("refuses frames that are malformed, not its own, clocked or not what the channel is", async () => {
    const refused = async (change: (f: HarnessFixture) => void, reason: string | RegExp): Promise<void> => {
      const copy = clone(fixture);
      change(copy);
      await expect(verifyHarnessFixture(engine, copy)).rejects.toThrow(reason);
    };
    const at = (f: HarnessFixture, seat: "p1" | "p2", prefix: string): number =>
      f.frames[seat].findIndex((frame) => frame.includes(`\n${prefix}`));
    await refused((f) => (f.frames.p1 = []), "p1 frames are not a non-empty list of strings");
    await refused((f) => (f.frames.p2 = [1 as never]), "p2 frames are not a non-empty list of strings");
    await refused((f) => (f.room = "battle-other"), /frames are not all headed >battle-other/);
    await refused((f) => (f.frames.p1[2] = "|inactive|no header"), "p1 frames are not all headed");
    await refused(
      (f) => (f.frames.p1[0] = (f.frames.p1[0] as string).replace("|init|battle", "|title|x")),
      /p1 frames do not start with \|init\|battle/,
    );
    await refused((f) => f.frames.p2.push(`>${f.room}\n|showteam|p1|x`), "p2 frames have a team sheet or a |split| marker");
    await refused((f) => f.frames.p1.push(`>${f.room}\n|split|p1`), "p1 frames have a team sheet or a |split| marker");
    await refused(
      (f) => (f.frames.p1[at(f, "p1", "|t:|")] = `>${f.room}\n|t:|1760000000`),
      "p1 frames have a wall-clock |t:| line",
    );
    // A foe's request in the stream: not its own (and the adapter refuses it too).
    await refused((f) => {
      const foe = (f.frames.p2[at(f, "p2", "|request|")] as string).split("\n")[1] as string;
      f.frames.p1.splice(at(f, "p1", "|request|"), 0, `>${f.room}\n${foe}`);
    }, /p1 frames hold a request that is not its own/);
    // The stored channel is derived: it must be what the frames give.
    await refused((f) => f.channels.p1.lines.splice(5, 1), "p1 channel is not what its frames give");
    await refused((f) => f.frames.p2.splice(at(f, "p2", "|move|"), 1), "p2 channel is not what its frames give");
    await refused((f) => (f.frames.p1[4] = `>${f.room}\n|t:|0\n|gametype|singles`), "p1 channel is not what its frames give");
    await refused((f) => (f.frames.p1[7] = `>${f.room}\n|request|{`), /p1 frames cannot be replayed/);
    await refused((f) => f.frames.p1.push(`>${f.room}\n|init|`), /p1 frames cannot be replayed: .* is not a battle room/);
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

describe("buildHarnessFixture", () => {
  const { recording, fixture } = COMMITTED[0] as (typeof COMMITTED)[0];
  const record = (seat: "p1" | "p2", over: Partial<A2BattleRecord> = {}): A2BattleRecord =>
    ({
      room: fixture.room,
      player: seat,
      result: "win",
      channels: [fixture.channels[seat]],
      ...over,
    }) as A2BattleRecord;
  const input = (
    over: { log?: { winner: string }; records?: Record<"p1" | "p2", A2BattleRecord>; connections?: Record<"p1" | "p2", string[][]> } = {},
  ) => ({
    engine,
    snapshot,
    format: FIXTURE_FORMAT,
    provenance: fixture.provenance,
    log: { lines: fixture.input_log.lines, winner: "", turns: fixture.turns, ...over.log },
    records: over.records ?? { p1: record("p1"), p2: record("p2") },
    connections: over.connections ?? { p1: [fixture.frames.p1], p2: [fixture.frames.p2] },
    teams: fixture.teams,
    teamLabels: { p1: recording.teams.challenger, p2: recording.teams.sparring },
  });

  it("rebuilds the committed fixture from its parts", async () => {
    const winner = fixture.winner === null ? "" : fixture.players[fixture.winner].name;
    const built = await buildHarnessFixture(input({ log: { winner } }));
    expect(built).toEqual({ ...fixture, players: { p1: { ...fixture.players.p1, team: built.players.p1.team }, p2: { ...fixture.players.p2, team: built.players.p2.team } } });
  });

  it("builds from every frame the bots received: other rooms dropped, the clock redacted, the rest as received", async () => {
    const winner = fixture.winner === null ? "" : fixture.players[fixture.winner].name;
    const noisy = (frames: string[]): string[][] => [
      [
        "|updateuser| guest|0|1",
        ">lobby\n|queryresponse|rooms|{}",
        ...frames.map((frame) => frame.replace(REDACTED_CLOCK, "|t:|1760000000")),
        ">battle-other-9\n|init|battle",
      ],
    ];
    const built = await buildHarnessFixture(input({ log: { winner }, connections: { p1: noisy(fixture.frames.p1), p2: noisy(fixture.frames.p2) } }));
    expect(built.frames).toEqual(fixture.frames);
  });

  it("refuses a battle whose bots are in different rooms", async () => {
    const records = { p1: record("p1"), p2: record("p2", { room: "battle-x" }) };
    const p2 = fixture.frames.p2.map((frame) => frame.replace(fixture.room, "battle-x"));
    await expect(buildHarnessFixture(input({ records, connections: { p1: [fixture.frames.p1], p2: [p2] } }))).rejects.toThrow(
      "the bots did not play the same battle",
    );
  });

  it("refuses a record for the wrong seat or room, and a player with several channels", async () => {
    await expect(buildHarnessFixture(input({ records: { p1: record("p2"), p2: record("p2") } }))).rejects.toThrow(
      "the p1 record is for p2",
    );
    await expect(buildHarnessFixture(input({ records: { p1: record("p1", { room: "" }), p2: record("p2") } }))).rejects.toThrow(
      "the p1 record is for p1",
    );
    const two = record("p2", { channels: [fixture.channels.p2, fixture.channels.p2] });
    await expect(buildHarnessFixture(input({ records: { p1: record("p1"), p2: two } }))).rejects.toThrow(
      "p2 has 2 channels",
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

describe("redactClock and roomFrames", () => {
  it("replaces only the number of a |t:| line, keeping lines and frame boundaries", () => {
    expect(redactClock(">r\n|t:|1760000000\n|move|x|t:|5\n|t:|")).toBe(`>r\n${REDACTED_CLOCK}\n|move|x|t:|5\n${REDACTED_CLOCK}`);
    expect(redactClock("|j|x")).toBe("|j|x");
  });

  it("keeps one connection's frames of the room, and refuses none or several", () => {
    const frames = [">r\n|init|battle", ">lobby\n|x", "|y", ">r\n|t:|1"];
    expect(roomFrames([frames], "r")).toEqual([">r\n|init|battle", `>r\n${REDACTED_CLOCK}`]);
    expect(roomFrames([["|z"], frames], "r")).toHaveLength(2);
    expect(() => roomFrames([["|z"]], "r")).toThrow("0 connections received r");
    expect(() => roomFrames([frames, frames], "r")).toThrow("2 connections received r");
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
