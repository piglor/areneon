import { readFileSync } from "node:fs";
import { join } from "node:path";
import { importPaste, loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { afterEach, describe, expect, it } from "vitest";
import {
  findServerLog,
  launchHarness,
  playBotVsBot,
  readServerInputLog,
  recordHarnessBattle,
  verifyHarnessFixture,
} from "../src/harness.ts";
import { createRuntimeCopy, launch, type LocalServer, SERVER_CONFIG_PATH } from "../src/index.ts";
import { FIXTURE_FORMAT, RECORDINGS, recordingProvenance } from "../src/recordings.ts";
import { A2_DIR, fixtureTeamHashes } from "./harness-fixtures.ts";
import { freePort, tempDir } from "./helpers.ts";

// The pinned server on 127.0.0.1 with the harness config, two bot clients
// playing one Bo1 (ADR-005 §5). Everything else about the fixtures is checked
// without a server in harness.test.ts.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const TEAMS = join(A2_DIR, "..", "..", "teams", FIXTURE_FORMAT);
const team = (file: string) => importPaste(engine, readFileSync(join(TEAMS, file), "utf8"));
const recording = RECORDINGS[0] as (typeof RECORDINGS)[0];

const servers: LocalServer[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.stop()));
});

async function start(harness: boolean): Promise<{ server: LocalServer; dir: string }> {
  const copy = createRuntimeCopy({ runtimeRoot: tempDir("harness-runtime") });
  const options = { dir: copy.dir, port: await freePort() };
  const server = await (harness ? launchHarness(options) : launch(options));
  servers.push(server);
  return { server, dir: copy.dir };
}

const bots = (server: LocalServer) => ({
  engine,
  snapshot,
  url: `ws://127.0.0.1:${server.session.port}/showdown/websocket`,
  format: FIXTURE_FORMAT,
  names: recording.names,
  teams: { sparring: team(recording.teams.sparring), challenger: team(recording.teams.challenger) },
  known_synthetic: fixtureTeamHashes(engine),
});

describe("the harness server", () => {
  it(
    "records a bot-vs-bot battle's input log and builds a fixture that verifies",
    async () => {
      const { server, dir } = await start(true);
      const config = readFileSync(join(dir, SERVER_CONFIG_PATH), "utf8");
      expect(config).toContain("exports.logchallenges = true;");
      expect(config).toContain('exports.bindaddress = "127.0.0.1";');

      const fixture = await recordHarnessBattle({
        ...bots(server),
        server,
        dir,
        provenance: recordingProvenance(engine.commit),
        teamLabels: recording.teams,
      });
      await verifyHarnessFixture(engine, fixture, fixtureTeamHashes(engine));
      const names = [fixture.players.p1.name, fixture.players.p2.name].sort();
      expect(names).toEqual([recording.names.challenger, recording.names.sparring].sort());
      // The challenger sits in the seat the server gave it, with that team.
      const challengerSeat = fixture.players.p1.name === recording.names.challenger ? "p1" : "p2";
      expect(fixture.players[challengerSeat].team).toBe(recording.teams.challenger);
      expect(fixture.winner === null || ["p1", "p2"].includes(fixture.winner)).toBe(true);
      expect(fixture.input_log.lines.length).toBeGreaterThan(8);
      expect(JSON.stringify(fixture)).not.toMatch(/\|t:\|(?!0\b)|"timestamp"/);
      // Each seat's raw frames are the room's, headed and in order, and rebuild its channel.
      for (const seat of ["p1", "p2"] as const) {
        // No reconnect without a drop plan: one connection per seat.
        expect(fixture.frames[seat]).toHaveLength(1);
        expect(fixture.frames[seat][0]?.every((frame) => frame.startsWith(`>${fixture.room}\n`))).toBe(true);
        expect(fixture.frames[seat][0]?.[0]).toContain("|init|battle");
      }
    },
    60_000,
  );

  it(
    "writes no input log under the normal config",
    async () => {
      const { server, dir } = await start(false);
      expect(readFileSync(join(dir, SERVER_CONFIG_PATH), "utf8")).toContain("exports.logchallenges = false;");
      const pair = await playBotVsBot(bots(server));
      expect(pair.sparring.room).toBe(pair.challenger.room);
      await expect(readServerInputLog(dir, FIXTURE_FORMAT, pair.sparring.room, { timeoutMs: 1_500 })).rejects.toThrow(
        /wrote no input log/,
      );
      expect(findServerLog(dir, FIXTURE_FORMAT, pair.sparring.room)).toBeUndefined();
    },
    60_000,
  );
});
