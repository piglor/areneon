import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { type ChannelFixture, GOLDEN_CHANNELS, GOLDEN_DIR, syntheticTeam } from "../src/golden.ts";
import {
  continueChannel,
  ObservationError,
  observe,
  type PlayerChannel,
  playerChannel,
} from "../src/index.ts";

// A rejoin (ADR-008 §3, §4): the room log is replayed to the new connection
// with no |request| or |error| line, then the stored request is sent again.
// These tests cut the golden p1 channel into connections the way the server
// does and join them back.

const REQUEST = "|request|";
const fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, Object.keys(GOLDEN_CHANNELS)[0] as string), "utf8")) as ChannelFixture;
const isRequest = (line: string): boolean => line.startsWith(REQUEST);
// The in-process engine sends no rqid; the server numbers every request.
let n = 0;
const all = fixture.channel.lines.map((line) =>
  isRequest(line) ? REQUEST + JSON.stringify({ ...(JSON.parse(line.slice(REQUEST.length)) as object), rqid: ++n }) : line,
);
const at = (k: number): number => all.findIndex((line, i) => isRequest(line) && all.slice(0, i + 1).filter(isRequest).length === k);
const channel = (lines: string[], data_class: "synthetic" | "private" = "synthetic", player: "p1" | "p2" = "p1"): PlayerChannel =>
  playerChannel({ schema_version: 1, data_class, player, lines });
const replayOf = (end: number): string[] => all.slice(0, end).filter((line) => !isRequest(line));
const request = (k: number): string => all[at(k)] as string;

describe("continueChannel", () => {
  it("joins a rejoin that is only the stored request sent again: the continuous channel, a Team Preview anchor again", async () => {
    const earlier = channel(all.slice(0, at(3) + 1));
    const rejoin = channel([...replayOf(at(3)), request(3)]);
    const joined = continueChannel(earlier, rejoin);
    expect(joined.lines).toEqual(earlier.lines);
    expect(joined.lines.filter(isRequest)).toHaveLength(3);
  });

  it("builds an observation of a rejoin that equals the one of the continuous channel (full, with log and request_seq)", async () => {
    const earlier = channel(all.slice(0, at(2) + 1));
    // After the replay, request 2 sent again and then the battle goes on to request 5.
    const rejoin = channel([...replayOf(at(2)), request(2), ...all.slice(at(2) + 1, at(5) + 1)]);
    const team = { schema_version: 1 as const, data_class: "synthetic" as const, team: syntheticTeam() as never };
    const common = { team, manifest: fixture.manifest };
    // Alone, the rejoin channel has no Team Preview request: no observation.
    await expect(observe({ ...common, channel: rejoin })).rejects.toThrow(/continueChannel/);
    const joined = continueChannel(earlier, rejoin);
    expect(joined.lines).toEqual(all.slice(0, at(5) + 1));
    const expected = await observe({ ...common, channel: channel(all.slice(0, at(5) + 1)) });
    const got = await observe({ ...common, channel: joined });
    expect(got).toEqual(expected);
    expect(got.identity.request_seq).toBe(5);
  });

  it("keeps a first request of the rejoin that is a new one (another rqid) where the rejoin has it", () => {
    // The earlier connection ended at request 2; the rejoin's first request is request 3 (a new rqid).
    const earlier = channel(all.slice(0, at(2) + 1));
    const rejoin = channel([...replayOf(at(2)), ...all.slice(at(2) + 1, at(3) + 1)]);
    expect(continueChannel(earlier, rejoin).lines).toEqual(all.slice(0, at(3) + 1));
  });

  it("puts the player-alone |error| lines of the earlier channel back where they were", () => {
    const error = "|error|[Invalid choice] synthetic";
    const earlier = channel([...all.slice(0, at(2) + 1), error]);
    const rejoin = channel([...replayOf(at(2)), request(2)]);
    expect(continueChannel(earlier, rejoin).lines).toEqual([...all.slice(0, at(2) + 1), error]);
  });

  it("folds across several reconnects", () => {
    const first = channel(all.slice(0, at(2) + 1));
    const second = channel([...replayOf(at(2)), request(2), ...all.slice(at(2) + 1, at(4) + 1)]);
    const third = channel([...replayOf(at(4)), request(4)]);
    expect(continueChannel(continueChannel(first, second), third).lines).toEqual(all.slice(0, at(4) + 1));
  });

  it("is private when either channel is", () => {
    const earlier = channel(all.slice(0, at(2) + 1), "private");
    const rejoin = channel([...replayOf(at(2)), request(2)]);
    expect(continueChannel(earlier, rejoin).data_class).toBe("private");
    expect(continueChannel(channel(all.slice(0, at(2) + 1)), channel([...replayOf(at(2)), request(2)], "private")).data_class).toBe("private");
  });

  it("refuses another player's channel", () => {
    const earlier = channel(all.slice(0, at(2) + 1));
    const other = { ...channel([...replayOf(at(2)), request(2)]), player: "p2" } as PlayerChannel;
    expect(() => continueChannel(earlier, other)).toThrow(ObservationError);
  });

  it("refuses a rejoin that does not replay the earlier room log (a tampered or other battle's replay)", () => {
    const earlier = channel(all.slice(0, at(3) + 1));
    const lines = [...replayOf(at(3)), request(3)];
    const lost = lines.findIndex((line) => line.startsWith("|move|"));
    expect(lost).toBeGreaterThan(0);
    expect(() => continueChannel(earlier, channel(lines.filter((_, i) => i !== lost)))).toThrow(/does not replay/);
    expect(() => continueChannel(earlier, channel(lines.map((l, i) => (i === lost ? `${l}x` : l))))).toThrow(/does not replay/);
    // A replay shorter than what the player already saw.
    expect(() => continueChannel(earlier, channel([...replayOf(at(2)), request(2)]))).toThrow(/does not replay/);
  });

  it("adds nothing the player did not receive: every line of the result is one of the two channels'", () => {
    const earlier = channel(all.slice(0, at(2) + 1));
    const rejoin = channel([...replayOf(at(2)), request(2), ...all.slice(at(2) + 1, at(4) + 1)]);
    const seen = new Set([...earlier.lines, ...rejoin.lines]);
    expect(continueChannel(earlier, rejoin).lines.every((line) => seen.has(line))).toBe(true);
  });

  it("refuses a player-alone line of the rejoin that sits inside the room log the player already had", () => {
    const earlier = channel(all.slice(0, at(3) + 1));
    const replay = replayOf(at(3));
    const inside = [...replay.slice(0, 2), request(2), ...replay.slice(2), request(3)];
    expect(() => continueChannel(earlier, channel(inside))).toThrow(/inside the room log/);
    const error = [...replay.slice(0, 2), "|error|[Invalid choice] forged", ...replay.slice(2), request(3)];
    expect(() => continueChannel(earlier, channel(error))).toThrow(/inside the room log/);
  });

  it("refuses a stray old request placed before the re-sent one", () => {
    const earlier = channel(all.slice(0, at(3) + 1));
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), request(1), request(3)]))).toThrow(/rise strictly/);
  });

  it("refuses a re-sent request with the same body and another rqid, and one without an rqid", () => {
    const earlier = channel(all.slice(0, at(3) + 1));
    const other = REQUEST + JSON.stringify({ ...JSON.parse(request(3).slice(REQUEST.length)), rqid: 50 });
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), other]))).toThrow(/another rqid/);
    const none = REQUEST + JSON.stringify({ ...JSON.parse(request(3).slice(REQUEST.length)), rqid: undefined });
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), none]))).toThrow(/another rqid/);
  });

  it("refuses rqids that do not rise strictly across the joined requests", () => {
    const earlier = channel(all.slice(0, at(2) + 1));
    const back = REQUEST + JSON.stringify({ ...JSON.parse(request(3).slice(REQUEST.length)), rqid: 1 });
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), back]))).toThrow(/rise strictly/);
    const same = REQUEST + JSON.stringify({ ...JSON.parse(request(3).slice(REQUEST.length)), rqid: 2 });
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), same]))).toThrow(/rise strictly/);
  });

  it("refuses requests that carry no rqid at all", () => {
    const bare = (k: number): string => REQUEST + JSON.stringify({ ...JSON.parse(request(k).slice(REQUEST.length)), rqid: undefined });
    const earlier = channel([...all.slice(0, at(2)), bare(2)]);
    expect(() => continueChannel(earlier, channel([...replayOf(at(3)), request(3)]))).toThrow(/rise strictly/);
  });
});
