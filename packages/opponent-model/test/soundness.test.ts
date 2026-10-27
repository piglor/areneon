// Corpus-wide acceptance of the belief (ADR-025 sections 3 and 8, slice 2 part 4, finding N6): over every request of both
// seats of every committed battle (the golden battle, the random-play corpus, `unavailable/`, the Illusion battle and the
// single-feature battles) the belief exists, apart from the counted reader flags, and the opponent's REAL sets and REAL
// brought set are always possible under it. The real team is read by this harness only (`corpus.ts`'s `Truth`); `belief`
// is never given it. The run is repeated with a prior that is uniform over the vocabulary (constraints alone), which also
// yields the rank of the true value per field.

import { beforeAll, describe, expect, it } from "vitest";
import { BeliefError } from "../src/index.ts";
import { type Point, type Rank, beliefAt, games, idsOf, points, rankTable, ranks, skewedPrior, sound } from "./corpus.ts";
import { uniformPrior } from "./real.ts";

const everything = games().flatMap((g) => points(g));
/** The skewed prior runs on the first ten random-play battles and every other corpus (the uniform one on all of them). */
const subset = games({ random: 10 }).flatMap((g) => points(g));

interface Outcome {
  point: Point;
  error: BeliefError | null;
}

const cache = new Map<unknown, ReturnType<typeof compute>>();
const run = (prior: (ids: string[]) => ReturnType<typeof uniformPrior>) => {
  if (!cache.has(prior)) cache.set(prior, compute(prior, prior === uniformPrior ? everything : subset));
  return cache.get(prior) as ReturnType<typeof compute>;
};

function compute(prior: (ids: string[]) => ReturnType<typeof uniformPrior>, over: readonly Point[]) {
  const sounds: { point: Point; ok: boolean; assignments: number }[] = [];
  const errors: Outcome[] = [];
  const rankRows: { phase: "preview" | "battle"; rank: Rank }[] = [];
  for (const point of over) {
    try {
      const b = beliefAt(point, prior(idsOf(point)));
      if (point.truth === null) throw new Error("a belief exists at a state with an Illusion twin");
      const truth = { ...point, truth: point.truth };
      sounds.push({ point, ok: sound(b, truth).length > 0, assignments: b.assignments.length });
      for (const rank of ranks(b, truth)) rankRows.push({ phase: point.request === 0 ? "preview" : "battle", rank });
    } catch (e) {
      if (!(e instanceof BeliefError)) throw e;
      errors.push({ point, error: e });
    }
  }
  return { sounds, errors, rankRows };
}

describe("the committed corpora", () => {
  it("cover both seats of every battle, a long run of requests and the Illusion battle", () => {
    expect(new Set(everything.map((p) => p.game)).size).toBeGreaterThanOrEqual(45);
    expect(new Set(everything.map((p) => p.seat))).toEqual(new Set(["p1", "p2"]));
    expect(everything.length).toBeGreaterThan(2000);
    expect(everything.some((p) => p.game === "illusion")).toBe(true);
  });
});

describe.each([
  ["a skewed prior (named entries and a residual)", skewedPrior],
  ["a prior uniform over the vocabulary (constraints alone)", uniformPrior],
] as const)("soundness with %s", (_name, prior) => {
  const result = { get sounds() { return run(prior).sounds; }, get errors() { return run(prior).errors; } };

  it("has a belief at every request but the counted reader flags: a flagged channel or an Illusion twin", { timeout: 300_000 }, () => {
    if (prior === uniformPrior) expect(result.errors.length).toBeGreaterThan(0); // the corpus does hold such requests (b39's flagged line, an Illusion twin)
    for (const { point, error } of result.errors) {
      expect(error?.code, `${point.game} ${point.seat} request ${point.request}`).toBe("reader_flagged");
      expect(point.flags.length > 0 || point.state.opponent.roster.some((m) => m.phantom === true)).toBe(true);
    }
    // Every other request has a belief, and none of them is lost.
    expect(result.sounds.length + result.errors.length).toBeGreaterThan(prior === uniformPrior ? 2000 : 700);
    expect(result.errors.length).toBeLessThan((result.sounds.length + result.errors.length) / 20);
  });

  it("holds the opponent's real sets and real brought set possible at every request, both seats", { timeout: 300_000 }, () => {
    const unsound = result.sounds.filter((s) => !s.ok).map((s) => `${s.point.game} ${s.point.seat} request ${s.point.request}`);
    expect(unsound).toEqual([]);
    expect(new Set(result.sounds.map((s) => s.point.seat))).toEqual(new Set(["p1", "p2"]));
    // The Illusion battle is in it, and a disguise mixture (more than one assignment) too.
    expect(result.sounds.some((s) => s.point.game === "illusion" && s.assignments > 1)).toBe(true);
  });
});

describe("the true value's rank under the uniform prior (N6)", () => {
  const rankRows = [] as ReturnType<typeof compute>["rankRows"];
  const table = [] as ReturnType<typeof rankTable>;
  beforeAll(() => {
    rankRows.push(...run(uniformPrior).rankRows);
    table.push(...rankTable(rankRows));
  }, 300_000);

  it("has a row for every field in the preview and in battle, and ranks lie among the options", () => {
    expect(table.map((r) => `${r.phase}/${r.field}`)).toEqual(["preview/forme", "preview/ability", "preview/item", "preview/move", "battle/forme", "battle/ability", "battle/item", "battle/move"]);
    for (const { rank } of rankRows) {
      expect(rank.rank).toBeGreaterThanOrEqual(1);
      expect(rank.rank).toBeLessThanOrEqual(rank.options);
    }
    for (const row of table) {
      expect(row.n).toBeGreaterThan(100);
      expect(row.top1_ppm).toBeLessThanOrEqual(row.top3_ppm);
      expect(row.mean_rel_ppm).toBeGreaterThan(0);
      expect(row.mean_rel_ppm).toBeLessThanOrEqual(1_000_000);
    }
  });

  it("a flat belief ranks last: at Team Preview the uniform prior knows nothing, and in battle the revealed facts move the true value up", () => {
    const row = (phase: string, field: string) => table.find((r) => r.phase === phase && r.field === field)!;
    expect(row("preview", "move").mean_rel_ppm).toBeGreaterThan(900_000); // no information: the true move is near the bottom of the ranking
    expect(row("battle", "move").mean_rel_ppm).toBeLessThan(row("preview", "move").mean_rel_ppm);
  });
});
