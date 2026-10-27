import { beforeAll, describe, expect, it } from "vitest";
import type { CandidateFeatures, FeatureRow } from "../src/candidate-features.ts";
import { addDominated, type BenchDecision, emptyDominated, episodeDecisions, estimateWallTime, progressDecisions, runBench, type Skipped, statsOf, summarise } from "../src/feature-bench.ts";
import type { FeatureResult } from "../src/feature-service.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Player, runBattle } from "../src/runner.ts";
import { corpus, engine } from "./damage-aware-support.ts";

const row = (slot: number, move: number, target: number, mean: number | null, mega = false): FeatureRow =>
  mean === null ? { slot, move, target, mega, kind: "unsupported", reason: "query_kind" } : { slot, move, target, mega, kind: "computed", mean_ppm: mean, max_ppm: mean, min_ppm: mean, ohko_rolls: 0 };
const features = (rows: FeatureRow[]): CandidateFeatures => ({ feature_set: "probe-damage/1", feature_set_hash: "h", rows });

describe("probe-dominated rate", () => {
  it("counts a chosen move whose mean is below half the best row of the same slot", () => {
    const count = emptyDominated();
    const f = features([row(0, 1, 1, 100_000), row(0, 2, 1, 400_000), row(0, 2, 2, 300_000), row(1, 1, 1, 0), row(1, 3, 1, null)]);
    addDominated(count, f, "move 1 1, move 1 1"); // slot 0: 100k < 400k/2 -> dominated; slot 1: best 0
    addDominated(count, f, "move 2 2, move 3 1"); // slot 0: 300k vs 400k not dominated; slot 1 unsupported row
    addDominated(count, f, "switch 3, pass"); // not moves
    addDominated(count, f, "move 4, move 1 2"); // slot 0 has no row for move 4; slot 1 move 1 target 2 has no row of that target (falls back to any)
    expect(count).toEqual({ evaluable: 2, dominated: 1, not_evaluable: { switch_or_pass: 2, no_damage_row: 1, unsupported_row: 1, best_zero: 2 } });
  });
  it("is empty for no choice parts it can score", () => {
    const count = emptyDominated();
    addDominated(count, features([]), "move 1 1, move 1 1");
    expect(count.evaluable).toBe(0);
  });
});

describe("statistics", () => {
  const result = (ms: number, rows: FeatureRow[], counts = { queries: 4, probe_runs: 1, unsupported_queries: 0 }): FeatureResult => ({ ms, counts, outcome: { kind: "features", features: features(rows) } });
  it("summarises latency and the rows", () => {
    expect(summarise([])).toBeNull();
    const l = summarise([10, 20, 30, 40, 1000]);
    expect(l).toMatchObject({ n: 5, median_ms: 30, p95_ms: 1000, p99_ms: 1000, max_ms: 1000 });
    const s = statsOf([result(10, [row(0, 1, 1, 5), row(0, 1, 2, null)]), { ms: 5, counts: { queries: 0, probe_runs: 0, unsupported_queries: 0 }, outcome: { kind: "error", code: "reader_gap" } }]);
    expect(s).toMatchObject({ decisions: 2, featured: 1, errors: { reader_gap: 1 }, rows: { total: 2, computed: 1, unsupported: { query_kind: 1 }, per_decision_mean_x100: 200 }, probe: { queries: 4, probe_runs: 1, memo_hit_ppm: 750_000 }, total_builder_ms: 15 });
    expect(statsOf([]).probe.memo_hit_ppm).toBe(0);
    expect(statsOf([]).rows.per_decision_mean_x100).toBe(0);
  });
  it("estimates the generation wall time against 1.5 times the control", () => {
    const e = estimateWallTime({ play_ms: 180_000, load_ms: 20_000, decisions: 10_000 }, 50, [1, 8]);
    expect(e.limit_ms).toBe(300_000);
    expect(e.builder_cpu_ms).toBe(1_000_000);
    expect(e.by_threads).toEqual([
      { threads: 1, added_ms: 1_000_000, total_ms: 1_200_000, ratio_x100: 600, met: false },
      { threads: 8, added_ms: 125_000, total_ms: 325_000, ratio_x100: 163, met: false },
    ]);
    expect(estimateWallTime({ play_ms: 180_000, load_ms: 20_000, decisions: 10_000 }, 10, [4]).by_threads[0]).toMatchObject({ met: true });
  });
});

describe("recorded decisions", () => {
  let record: Awaited<ReturnType<typeof runBattle>>;
  const battle = corpus[0] as never as { seed: string; teams: Record<Player, never[]>; actor_seeds: Record<Player, number> };
  beforeAll(async () => {
    record = await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams as never, actors: { p1: randomLegalActor(battle.actor_seeds.p1), p2: randomLegalActor(battle.actor_seeds.p2) }, maxTurns: 4 });
  }, 60_000);

  it("reads an Episode's and a progress game's move decisions and runs them through single and pooled builders", async () => {
    const skipped: Skipped = { not_move: 0, unreplayable_games: 0 };
    const mine = record.choices.filter((c) => c.player === "p1");
    const episode = { player: "p1", channel: record.channels.p1, team: { team: battle.teams.p1 }, decisions: mine.map((c) => ({ choice: c.choice, identity: { request_seq: c.request_seq } })) };
    const fromEpisode = episodeDecisions(episode, "e", "episode", skipped);
    expect(fromEpisode.length).toBeGreaterThan(0);
    expect(skipped.not_move).toBeGreaterThan(0); // the Team Preview request
    expect(fromEpisode.at(-1)?.task.final).toBe(true);
    expect(episodeDecisions({ ...episode, decisions: [{ choice: "x", identity: { request_seq: 999 } }] }, "e", "episode", skipped)).toEqual([]);

    const game = { seed: battle.seed, policies: { p1: "policy:aa", p2: "damage-aware@v1" }, teams: { p1: { team: battle.teams.p1 }, p2: { team: battle.teams.p2 } }, decisions: record.choices.map((c) => ({ player: c.player, request_seq: c.request_seq, chosen: c.choice })) } as never;
    const progress = await progressDecisions(engine, GOLDEN_FORMAT, 4, game, "g", (p) => (p.startsWith("policy:") ? "learned" : "baseline"), skipped);
    expect(new Set(progress.map((d) => d.seat))).toEqual(new Set(["learned", "baseline"]));
    // The replay's own channel for p1 is the recorded one: the same decisions as the Episode's.
    expect(progress.filter((d) => d.task.player === "p1").map((d) => d.chosen)).toEqual(fromEpisode.map((d) => d.chosen));
    const broken = await progressDecisions(engine, GOLDEN_FORMAT, 4, { ...(game as object), decisions: [] } as never, "g", () => "x", skipped);
    expect(broken).toEqual([]);
    expect(skipped.unreplayable_games).toBe(1);

    const few: BenchDecision[] = fromEpisode.slice(0, 3);
    const report = await runBench({ format_id: GOLDEN_FORMAT, data_class: "synthetic" }, few, [2]);
    expect(report.single_thread.featured).toBe(few.length);
    expect(report.pools[0]).toMatchObject({ size: 2, rows_equal_single: true });
    expect(report.dominated.episode).toBeDefined();
  }, 240_000); // replays on the engine, real probe in the single run and in a two-thread pool
});
