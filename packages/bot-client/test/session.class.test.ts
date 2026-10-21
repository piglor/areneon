import { loadEngine, loadRulesSnapshot, type PokemonSet, teamHash } from "@areneon/showdown-bridge";
import { type HarnessFixture, type Seat } from "@areneon/local-server/harness";
import { describe, expect, it, vi } from "vitest";
import { type A2BotOptions, type A2BattleRecord, memoryTransport, type MemoryTransport, runA2Bot, SessionError } from "../src/index.ts";
import { fixtureFiles, readFixture } from "./a1-a2-equality.ts";

// ADR-008 §5, ADR-009 (#824): the class of what a session records is proven
// from team hashes, never given. The session replays a recorded battle's
// frames; only the known set and the opponent's team change between cases.

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const fixture: HarnessFixture = readFixture(fixtureFiles().find((f) => readFixture(f).frames.p1.length === 1) as string);
const seat: Seat = "p1";
const rival: Seat = "p2";

async function classOf(extra: Partial<A2BotOptions>): Promise<string[]> {
  const transports: MemoryTransport[] = [];
  const records: A2BattleRecord[] = [];
  const run = runA2Bot({
    engine,
    snapshot,
    url: "ws://127.0.0.1:8000/showdown/websocket",
    name: fixture.players[seat].name,
    challenger: fixture.players[rival].name,
    format: fixture.format_id,
    team: structuredClone(fixture.teams[seat]),
    battles: 1,
    clock: () => 0,
    reconnect: { attempts: 1, delayMs: 5 },
    connect: async () => {
      const transport = memoryTransport();
      transports.push(transport);
      return transport;
    },
    onBattle: (record) => records.push(record),
    ...extra,
  });
  await vi.waitFor(() => expect(transports).toHaveLength(1));
  const t = transports[0] as MemoryTransport;
  t.push("|challstr|4|0123abcd");
  t.push(`|updateuser| ${fixture.players[seat].name}|1|1|{"blockChallenges":false}`);
  for (const frame of fixture.frames[seat][0] as string[]) if (!t.closed) t.push(frame);
  const result = await run;
  const record = (result.battles[0] ?? records[0]) as A2BattleRecord;
  return record.channels.map((channel) => channel.data_class);
}

const own = (): Set<string> => new Set([teamHash(fixture.teams[seat])]);
const both = (): Set<string> => new Set([teamHash(fixture.teams[seat]), teamHash(fixture.teams[rival])]);
const rivalTeam = (): PokemonSet[] => structuredClone(fixture.teams[rival]);

describe("the class of a session's channels is derived (#824)", () => {
  it("is synthetic only when the bot's team and the known opponent's team are both in the known set", async () => {
    expect(await classOf({ known_synthetic: both(), opponent_team: rivalTeam() })).toEqual(["synthetic"]);
  }, 60_000);

  it("is private when the opponent's team is unknown, even if the bot's own is synthetic", async () => {
    expect(await classOf({ known_synthetic: both() })).toEqual(["private"]);
  }, 60_000);

  it("is private when the opponent's team is not in the known set (negative control)", async () => {
    expect(await classOf({ known_synthetic: own(), opponent_team: rivalTeam() })).toEqual(["private"]);
  }, 60_000);

  it("is private when the bot's own team is not in the known set, and by default", async () => {
    const rivalOnly = new Set([teamHash(fixture.teams[rival])]);
    expect(await classOf({ known_synthetic: rivalOnly, opponent_team: rivalTeam() })).toEqual(["private"]);
    expect(await classOf({})).toEqual(["private"]);
  }, 60_000);

  it("refuses an opponent team in a live match: a human's team is never known", async () => {
    await expect(
      runA2Bot({
        engine,
        snapshot,
        url: "ws://127.0.0.1:8000/showdown/websocket",
        name: "Areneon-bot-x",
        challenger: "Human",
        format: fixture.format_id,
        team: structuredClone(fixture.teams[seat]),
        mode: "live_match",
        opponent_team: rivalTeam(),
      }),
    ).rejects.toThrow(SessionError);
  });
});
