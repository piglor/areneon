import { beforeAll, describe, expect, it } from "vitest";
import { FEATURE_SET_OUTCOME } from "../src/candidate-features.ts";
import { createFeatureService } from "../src/feature-service.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { capture, type Captured, corpus, engine } from "./damage-aware-support.ts";

// The service builds `probe-outcome/1` with its reviewed record and the mechanics config that lists `move_outcome`, with the real probe.
let moves: Captured[] = [];
beforeAll(async () => {
  moves = (await capture(corpus[0] as never)).filter((c) => c.request.phase === "move" && (c.request.legal?.choices.length ?? 0) >= 2);
}, 120_000);

describe("the feature service for probe-outcome/1", () => {
  it("builds the set, reports the outcomes' cost apart and keeps the same bytes through a second service", () => {
    const spec = { format_id: GOLDEN_FORMAT, data_class: "synthetic" as const, feature_set: FEATURE_SET_OUTCOME as typeof FEATURE_SET_OUTCOME };
    const c = moves[0] as Captured;
    const legal = c.request.legal as NonNullable<typeof c.request.legal>;
    const task = { game: "g", player: c.context.player, lines: [...c.context.channel], team: c.context.team as never, candidates: [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) })) };
    const result = createFeatureService(spec, engine).run(task);
    if (result.outcome.kind !== "features") throw new Error("not featured");
    expect(result.outcome.features.feature_set).toBe(FEATURE_SET_OUTCOME);
    expect((result.outcome.features as unknown as { outcomes: unknown[] }).outcomes.length).toBeGreaterThan(0);
    expect(result.parts?.outcomes?.queries).toBeGreaterThan(0);
    // the cost reductions are reported per task (#1353): rows skipped or folded, and the refusal and history_unknown causes
    expect(result.outcome_counts).toMatchObject({ skipped: expect.any(Number), folded: expect.any(Number), asked_all: expect.any(Number), causes: expect.any(Object) });
    expect((result.outcome_counts?.skipped ?? 0) + (result.outcome_counts?.folded ?? 0)).toBeGreaterThan(0);
    const again = createFeatureService(spec, engine).run({ ...task, game: "h" });
    expect(JSON.stringify(again.outcome)).toBe(JSON.stringify(result.outcome));
  }, 120_000);
});
