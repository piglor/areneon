import { readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import channelSchema from "@areneon/schema/contracts/v1/player-channel-log.schema.json" with { type: "json" };
import { describe, expect, it } from "vitest";
import {
  battleSummary,
  type BattleSummary,
  type CaptureProvenance,
  captureProvenance,
  type ChannelFixture,
  GOLDEN_CHANNELS,
  GOLDEN_PROVENANCE,
  GOLDEN_DIR,
  GOLDEN_FORMAT,
  GOLDEN_PHASES,
  GOLDEN_SUMMARY,
  goldenBattle,
  goldenChannels,
  goldenFixtures,
  syntheticTeam,
} from "../src/golden.ts";
import { type BattleRecord, loadEngine, replayActor, runBattle } from "../src/index.ts";

const committed = <T>(file: string): T =>
  JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as T;

describe("golden per-player channel fixtures (ADR-008 §5)", () => {
  const validChannel = ajvFor("contract").compile(channelSchema);

  it("match a fresh capture and are synthetic PlayerChannelLogs", async () => {
    const fresh = goldenChannels(await goldenBattle(loadEngine()));
    expect(Object.keys(fresh).sort()).toEqual(Object.keys(GOLDEN_CHANNELS).sort());
    for (const [file, fixture] of Object.entries(fresh)) {
      const stored = committed<ChannelFixture>(file);
      expect(stored, file).toEqual(fixture);
      expect(validChannel(stored.channel), JSON.stringify(validChannel.errors)).toBe(true);
      expect(stored.channel.data_class).toBe("synthetic");
    }
  });

  it("carry each player's own requests only, and no wall-clock lines", () => {
    for (const [file, player] of Object.entries(GOLDEN_CHANNELS)) {
      const { lines } = committed<ChannelFixture>(file).channel;
      const requests = lines.filter((l) => l.startsWith("|request|"));
      expect(requests.length).toBeGreaterThan(0);
      for (const request of requests) {
        expect((JSON.parse(request.slice("|request|".length)) as { side: { id: string } }).side.id).toBe(player);
      }
      expect(lines.some((l) => l.startsWith("|t:|"))).toBe(false);
    }
  });
});

describe("golden request fixtures", () => {
  // A Showdown pin bump that changes behaviour fails here; re-capture with
  // scripts/capture-golden.ts and review the diff.
  it("match a fresh capture from the pinned engine", async () => {
    const fresh = goldenFixtures(await goldenBattle(loadEngine()));
    expect(Object.keys(fresh).sort()).toEqual(Object.keys(GOLDEN_PHASES).sort());
    for (const [file, fixture] of Object.entries(fresh)) {
      expect(committed(file), file).toEqual(fixture);
    }
  });

  // Cross-process determinism: this process reproduces the log hash that
  // the capture run committed (same commit + seed + choices).
  it("reproduce the committed battle log hash in a fresh process", async () => {
    const summary = committed<BattleSummary>(GOLDEN_SUMMARY);
    expect(battleSummary(await goldenBattle(loadEngine()))).toEqual(summary);
  });

  it("replay the committed choices to the same log hash", async () => {
    const summary = committed<BattleSummary>(GOLDEN_SUMMARY);
    const team = syntheticTeam();
    const replayed = await runBattle({
      engine: loadEngine(),
      formatId: GOLDEN_FORMAT,
      seed: summary.seed,
      teams: { p1: team, p2: team },
      actors: { p1: replayActor(summary.choices, "p1"), p2: replayActor(summary.choices, "p2") },
    });
    expect(battleSummary(replayed).omniscient_log_sha256).toBe(summary.omniscient_log_sha256);
  });

  // Not compared with a fresh capture on purpose (see GOLDEN_PROVENANCE).
  it("record which rules snapshot they were captured under", async () => {
    const fresh = captureProvenance(await goldenBattle(loadEngine()));
    expect(committed<CaptureProvenance>(GOLDEN_PROVENANCE).rules_snapshot_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(fresh.rules_snapshot_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("embed the format rules record in every request fixture", async () => {
    const record = await goldenBattle(loadEngine());
    for (const file of Object.keys(GOLDEN_PHASES)) {
      expect(committed<{ format_rules: unknown }>(file).format_rules, file).toEqual(record.format_rules);
    }
  });

  it("refuse a battle missing a golden phase", () => {
    const record = { requests: { p1: [], p2: [] } } as unknown as BattleRecord;
    expect(() => goldenFixtures(record)).toThrow(/no p1 preview request/);
  });
});
