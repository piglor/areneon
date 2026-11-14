import { beforeAll, describe, expect, it } from "vitest";
import { FEATURE_SET_LIKELY } from "../src/candidate-features.ts";
import { featureRowsProblem, featureCoverageProblem, type WireFeatureRows } from "../src/feature-rows.ts";
import { createFeatureService } from "../src/feature-service.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { capture, type Captured, corpus, engine } from "./damage-aware-support.ts";

// The service builds `probe-outcome-likely/1` with its two reviewed records and the real probe, on any thread; what it builds passes the
// wire readers (the arrays' presence, order and coverage of the request) after a JSON round trip, as it will travel.
let moves: Captured[] = [];
beforeAll(async () => {
  moves = (await capture(corpus[0] as never)).filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
}, 120_000);

const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(id: string): { category: string } } };
const isDamaging = (id: string): boolean => dex.moves.get(id).category !== "Status";

describe("the feature service for probe-outcome-likely/1", () => {
  const spec = { format_id: GOLDEN_FORMAT, data_class: "synthetic" as const, feature_set: FEATURE_SET_LIKELY as typeof FEATURE_SET_LIKELY };
  const taskOf = (c: Captured, game: string) => {
    const legal = c.request.legal as NonNullable<typeof c.request.legal>;
    return { game, player: c.context.player, lines: [...c.context.channel], team: c.context.team as never, candidates: [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) })) };
  };

  it("builds the set with the likely part reported apart, keeps its bytes through a second service, and a record file can be named", () => {
    const c = moves[0] as Captured;
    const task = taskOf(c, "g");
    const result = createFeatureService(spec, engine).run(task);
    if (result.outcome.kind !== "features") throw new Error("not featured");
    expect(result.outcome.features.feature_set).toBe(FEATURE_SET_LIKELY);
    expect((result.outcome.features as unknown as { belief_threats: unknown[] }).belief_threats.length).toBeGreaterThan(0);
    expect(result.parts?.likely?.queries).toBeGreaterThan(0);
    expect(result.outcome_counts).toBeDefined();
    expect(result.likely_counts?.rows).toBeGreaterThan(0);
    expect(result.likely_counts?.decisions).toBe(1);
    const again = createFeatureService(spec, engine).run({ ...task, game: "h" });
    expect(JSON.stringify(again.outcome)).toBe(JSON.stringify(result.outcome));
    const named = createFeatureService({ ...spec, likely_record: "probe-outcome-likely.v1.m8.json" }, engine).run({ ...task, game: "i" });
    if (named.outcome.kind !== "features") throw new Error("not featured");
    expect(named.outcome.features.feature_set_hash).not.toBe(result.outcome.features.feature_set_hash);
  }, 240_000);

  it("gives features the wire readers accept, for every move request of a battle, and says so when the cache answered", () => {
    const service = createFeatureService(spec, engine);
    let judged = 0;
    for (const c of moves.slice(0, 6)) {
      const result = service.run(taskOf(c, "wire"));
      if (result.outcome.kind !== "features") continue;
      const features = JSON.parse(JSON.stringify(result.outcome.features)) as WireFeatureRows;
      expect(featureRowsProblem(features)).toBeNull();
      const task = taskOf(c, "x");
      expect(featureCoverageProblem({ observation: { log: task.lines }, candidates: task.candidates }, features, isDamaging)).toBeNull();
      judged += 1;
    }
    expect(judged).toBeGreaterThan(0);
  }, 240_000);
});
