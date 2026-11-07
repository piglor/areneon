// The hashed selection of the search labels (ADR-026 erratum E2 decision 3): the thresholds below are n * 10^6 / 2^64 for the first
// 8 bytes n of SHA-256 over the JCS of {game, generation, player, purpose: "search-label/1", request_seq, run_key}, computed with Python's
// hashlib. A decision is selected exactly when the share (ppm) is above its threshold.
import { describe, expect, expectTypeOf, it } from "vitest";
import type { Episode } from "../src/episode.ts";
import type { BattleRecord } from "../src/runner.ts";
import { type LabelContext, labelEpisode, labelSelected } from "../src/search-labeller.ts";

const cases: [string, number, number, string, number, number][] = [
  ["run-a", 3, 0, "p1", 2, 740102.5826413984],
  ["run-a", 3, 0, "p1", 3, 202013.34931818166],
  ["run-a", 3, 1, "p2", 2, 325917.26952448906],
  ["run-b", 4, 7, "p1", 5, 485150.45514440007],
  ["run-a", 3, 0, "p2", 2, 131450.05509179665],
];

describe("labelSelected", () => {
  it.each(cases)("%s generation %i game %i %s request %i is selected from a share above %f ppm", (run_key, generation, game, player, request_seq, threshold) => {
    const key = { run_key, generation, game, player, request_seq };
    expect(labelSelected(key, Math.floor(threshold))).toBe(false);
    expect(labelSelected(key, Math.floor(threshold) + 1)).toBe(true);
    expect(labelSelected(key, 1_000_000)).toBe(true);
  });

  it("depends on every field of the key", () => {
    const base = { run_key: "run-a", generation: 3, game: 0, player: "p1", request_seq: 2 };
    expect(labelSelected(base, 740_103)).toBe(true);
    expect(labelSelected({ ...base, player: "p2" }, 740_103)).toBe(true);
    expect(labelSelected({ ...base, player: "p2" }, 131_450)).toBe(false);
    expect(labelSelected({ ...base, request_seq: 3 }, 202_014)).toBe(true);
    expect(labelSelected({ ...base, request_seq: 3 }, 202_013)).toBe(false);
  });

  it("selects about the share of a uniform sample", () => {
    let hits = 0;
    const total = 4000;
    for (let seq = 1; seq <= total; seq++) if (labelSelected({ run_key: "r", generation: 3, game: seq % 40, player: seq % 2 === 0 ? "p1" : "p2", request_seq: seq }, 250_000)) hits++;
    expect(hits).toBeGreaterThan(total * 0.2);
    expect(hits).toBeLessThan(total * 0.3);
  });
});

describe("what reaches a labeller", () => {
  it("is an Episode and the reviewed records and ports: no record, omniscient log, hidden verdict, seed or opponent team", () => {
    expectTypeOf<keyof LabelContext>().toEqualTypeOf<"engine" | "labels" | "client" | "run_key" | "generation" | "reply_model">();
    expectTypeOf<Parameters<typeof labelEpisode>[1]>().toEqualTypeOf<Episode>();
    expectTypeOf<BattleRecord>().not.toMatchTypeOf<Episode>();
    expectTypeOf<LabelContext>().not.toHaveProperty("seed");
    expectTypeOf<LabelContext>().not.toHaveProperty("omniscient_log");
    expectTypeOf<LabelContext>().not.toHaveProperty("hidden_verdicts");
    expect(true).toBe(true);
  });
});
