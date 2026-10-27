// The evaluation-only prediction-quality harness (ADR-025 section 13, slice 3 part 3): the primitives and one
// slot against hand-worked figures, seeded games on the dev split through the real belief, port and v1 rule,
// the refusal counts, the outputs (counts and integers, nothing of a team) and the source scan.
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import type { SlotMarginal } from "@areneon/opponent-model";
import { describe, expect, it } from "vitest";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import {
  type Accumulator,
  type OracleGame,
  type Option,
  accumulator,
  calibrate,
  emptyBins,
  judge,
  oracleGames,
  probability,
  qualityContext,
  report,
  scoreGame,
  scoreSlot,
} from "../src/prediction-quality.ts";
import { engine, FORMAT } from "./support.ts";

const r = (num: number, den: number) => ({ num: BigInt(num), den: BigInt(den) });
const opts = (...list: [string | null, number, number][]): Option[] => list.map(([id, n, d]) => ({ id, p: r(n, d) }));

describe("the primitives, worked by hand", () => {
  it("turns an exact ratio into a probability", () => {
    expect(probability(r(1, 4))).toBe(0.25);
    expect(probability(r(1, 3))).toBeCloseTo(1 / 3, 11);
    expect(probability(r(1, 10 ** 15))).toBe(1e-12); // a tiny positive ratio stays above zero, so -ln stays finite
    expect(judge(opts(["a", 1, 1]), "a").loss_micro).toBe(0);
  });

  it("judges a distribution: top-1 is the expected hit under a uniform tie-break, log-loss is -ln p in micro-nats", () => {
    const d = opts(["a", 1, 2], ["b", 1, 4], [null, 1, 4]);
    expect(judge(d, "a")).toEqual({ hit_ppm: 1_000_000, loss_micro: 693_147 }); // -ln(1/2)
    expect(judge(d, "b")).toEqual({ hit_ppm: 0, loss_micro: 1_386_294 }); // -ln(1/4)
    expect(judge(d, null)).toEqual({ hit_ppm: 0, loss_micro: 1_386_294 });
    // two options tied at the top: either is the pick with probability 1/2
    expect(judge(opts(["a", 1, 2], ["b", 1, 2]), "b")).toEqual({ hit_ppm: 500_000, loss_micro: 693_147 });
    // three tied: floor(10^6 / 3)
    expect(judge(opts(["a", 1, 3], ["b", 1, 3], ["c", 1, 3]), "c").hit_ppm).toBe(333_333);
    // a true value with no mass, or not listed, has no loss and no hit
    expect(judge(opts(["a", 1, 1], ["b", 0, 1]), "b")).toEqual({ hit_ppm: 0, loss_micro: null });
    expect(judge(opts(["a", 1, 1]), "z")).toEqual({ hit_ppm: 0, loss_micro: null });
  });

  it("fills the reliability table: ten bins, the last one closed, zero-mass options left out", () => {
    const bins = emptyBins();
    calibrate(bins, opts(["a", 1, 2], ["b", 1, 10], ["c", 1, 1], ["d", 0, 1], ["e", 9, 10]), (id) => id === "a" || id === "c");
    expect(bins.map((b) => b.n)).toEqual([0, 1, 0, 0, 0, 1, 0, 0, 0, 2]);
    expect(bins[5]).toEqual({ n: 1, p_ppm: 500_000, true: 1 });
    expect(bins[9]).toEqual({ n: 2, p_ppm: 1_900_000, true: 1 }); // 0.9 and 1
    expect(bins[1]).toEqual({ n: 1, p_ppm: 100_000, true: 0 });
  });
});

describe("one slot, worked by hand", () => {
  // The belief: item ix 3/5, iy 2/5; ability a1 1/2, a2 1/2; moves ma (revealed, 1), mb 1/2, mc 1/4, md 1/4.
  // The prior alone: item ix 1/5, iy 1/5, none 3/5; ability a1 3/4, a2 1/4; moves ma 1/2, mb 1/2, mc 1/2, md 1/2 (inclusions).
  // The real set: item iy, ability a2, moves ma mb mc, ma revealed. v1 assumed: no item, ability a1.
  const slot = (value: Partial<SlotMarginal>) => value as SlotMarginal;
  const belief = slot({ items: [{ id: "ix", p: r(3, 5) }, { id: "iy", p: r(2, 5) }], abilities: [{ id: "a1", p: r(1, 2) }, { id: "a2", p: r(1, 2) }], moves: [{ id: "ma", p: r(1, 1) }, { id: "mb", p: r(1, 2) }, { id: "mc", p: r(1, 4) }, { id: "md", p: r(1, 4) }] });
  const prior = slot({ items: [{ id: "ix", p: r(1, 5) }, { id: "iy", p: r(1, 5) }, { id: null, p: r(3, 5) }], abilities: [{ id: "a1", p: r(3, 4) }, { id: "a2", p: r(1, 4) }], moves: [{ id: "ma", p: r(1, 2) }, { id: "mb", p: r(1, 2) }, { id: "mc", p: r(1, 2) }, { id: "md", p: r(1, 2) }] });
  const acc = accumulator();
  scoreSlot(acc, "battle", { belief, prior, v1: { item: null, ability: "a1" } }, { item: "iy", ability: "a2", moves: ["ma", "mb", "mc"] }, { mega: false, moves_revealed: ["ma"] });
  const rows = report(acc).rows;
  const row = (field: string, method: string) => rows.find((x) => x.field === field && x.method === method);

  it("scores the item: the belief's top-1 misses (ix), the prior's tied-free top-1 misses, v1 misses", () => {
    expect(row("item", "belief")).toEqual({ phase: "battle", field: "item", method: "belief", n: 1, top1_ppm: 0, log_loss_micro: 916_291, zero_mass: 0 }); // -ln(2/5)
    expect(row("item", "prior")).toMatchObject({ top1_ppm: 0, log_loss_micro: 1_609_438 }); // -ln(1/5); none is the top
    expect(row("item", "v1")).toEqual({ phase: "battle", field: "item", method: "v1", n: 1, top1_ppm: 0, log_loss_micro: null, zero_mass: null });
  });

  it("scores the ability: the belief ties a1 and a2 (expected hit 1/2), the prior and v1 pick a1", () => {
    expect(row("ability", "belief")).toMatchObject({ top1_ppm: 500_000, log_loss_micro: 693_147 });
    expect(row("ability", "prior")).toMatchObject({ top1_ppm: 0, log_loss_micro: 1_386_294 });
    expect(row("ability", "v1")).toMatchObject({ top1_ppm: 0 });
  });

  it("scores only the unrevealed moves: mb and mc of the real set against mb, mc, md", () => {
    // belief: top is mb (1/2) and it is real -> 1; loss is the mean of -ln(1/2) and -ln(1/4) = (693147 + 1386294) / 2
    expect(row("move", "belief")).toMatchObject({ n: 1, top1_ppm: 1_000_000, log_loss_micro: 1_039_721, zero_mass: 0 });
    // prior: mb, mc, md tie at 1/2: two of the three tied options are real -> floor(2 * 10^6 / 3); loss -ln(1/2)
    expect(row("move", "prior")).toMatchObject({ top1_ppm: 666_666, log_loss_micro: 693_147 });
    expect(row("move", "v1")).toBeUndefined();
  });

  it("does not judge the ability of an evolved Mega, and calibrates the belief's unrevealed move options", () => {
    const other = accumulator();
    scoreSlot(other, "battle", { belief, prior, v1: null }, { item: "iy", ability: "a2", moves: ["ma", "mb", "mc"] }, { mega: true, moves_revealed: ["ma"] });
    expect(report(other).rows.map((x) => `${x.field}/${x.method}`)).toEqual(["item/belief", "item/prior", "move/belief", "move/prior"]);
    const bins = report(acc).calibration.belief.move;
    expect(bins.map((b) => [b.n, b.true])).toEqual([[0, 0], [0, 0], [2, 1], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]); // mc, md at 1/4 (mc real); mb at 1/2 (real)
  });

  it("counts a true move the belief gave no mass, and leaves it out of the mean", () => {
    const zero = accumulator();
    const dead = slot({ ...belief, moves: [{ id: "mb", p: r(1, 1) }, { id: "mc", p: r(0, 1) }] });
    scoreSlot(zero, "battle", { belief: dead, prior, v1: null }, { item: "iy", ability: "a2", moves: ["mb", "mc"] }, { mega: false, moves_revealed: [] });
    expect(report(zero).rows.find((x) => x.field === "move" && x.method === "belief")).toMatchObject({ zero_mass: 1, log_loss_micro: 0 });
  });
});

const vocabulary = goldenTeamVocabulary();
const ctx = qualityContext(engine, vocabulary);
const played = await oracleGames(engine, FORMAT, 0, 3);
const run = (games: readonly OracleGame[], c = ctx): Accumulator => {
  const acc = accumulator();
  for (const g of games) scoreGame(c, g, acc);
  return acc;
};

describe("seeded games on the dev split", () => {
  const acc = run(played);
  const out = report(acc);
  const cell = (phase: string, field: string, method: string) => out.rows.find((x) => x.phase === phase && x.field === field && x.method === method);

  it("play the same teams and channels for the same run key", async () => {
    const again = await oracleGames(engine, FORMAT, 1, 1);
    expect(again[0]).toEqual(played[1]);
    expect(played.every((g) => g.channels.p1.length > 10 && g.teams.p2.length === 6)).toBe(true);
  });

  it("judge four brought Pokemon at every request, both seats, and the belief never misses the real value", () => {
    expect(out.games).toBe(3);
    expect(out.refusals).toEqual({});
    expect(out.no_truth).toBe(0);
    expect(out.scored.preview).toBe(6); // two seats a game
    expect(out.scored.battle).toBe(out.requests - 6);
    expect(cell("preview", "item", "belief")?.n).toBe(4 * 6);
    expect(cell("battle", "item", "belief")?.n).toBe(4 * out.scored.battle);
    expect(cell("battle", "item", "v1")?.n).toBe(4 * out.scored.battle);
    for (const row of out.rows) if (row.method !== "v1") expect(row.zero_mass, `${row.phase}/${row.field}/${row.method}`).toBe(0);
    expect(out.v1_failures).toBe(0);
  });

  it("have the belief equal to the prior alone at Team Preview, and the evidence help from the first move on", () => {
    for (const field of ["item", "ability", "move"]) {
      expect(cell("preview", field, "belief")).toEqual({ ...cell("preview", field, "prior"), method: "belief" });
    }
    expect(cell("battle", "item", "belief")?.log_loss_micro as number).toBeLessThan(cell("battle", "item", "prior")?.log_loss_micro as number);
    for (const row of out.rows) {
      expect(row.top1_ppm === null || (row.top1_ppm >= 0 && row.top1_ppm <= 1_000_000)).toBe(true);
    }
  });

  it("report the joint consistency of the sampled worlds against the product of the single rates", () => {
    const j = out.joint;
    expect(j.worlds).toBe(8 * (out.scored.preview + out.scored.battle));
    expect(j.pairs).toBeGreaterThan(0);
    expect(j.pairs).toBeLessThanOrEqual(4 * j.worlds);
    expect(acc.joint.both).toBeLessThanOrEqual(Math.min(acc.joint.item, acc.joint.moves));
    expect(j.joint_ppm).toBe(Math.floor((acc.joint.both * 1_000_000) / j.pairs));
    expect(j.product_ppm).toBe(Math.floor(((j.item_ppm as number) * (j.moves_ppm as number)) / 1_000_000));
    expect(j.redraw_ppm).toBe(Math.floor((j.redraws * 1_000_000) / j.worlds));
    expect(j.sample_refusals).toEqual({});
  });

  it("are deterministic: the same games give the same report", () => {
    expect(report(run(played))).toEqual(out);
  });

  it("read the real team: handing the opponent other items changes the item rows (a negative control for the oracle)", () => {
    const swapped = played.map((g) => ({ ...g, teams: { p1: g.teams.p1.map((s) => ({ ...s, item: "" })), p2: g.teams.p2.map((s) => ({ ...s, item: "" })) } }));
    const changed = report(run(swapped));
    expect(changed.rows.find((x) => x.phase === "battle" && x.field === "item" && x.method === "belief")).not.toEqual(cell("battle", "item", "belief"));
  });

  it("holds counts and integers only: no set, name or hash of a team is in the output", () => {
    const text = JSON.stringify(out);
    for (const g of played) {
      for (const seat of ["p1", "p2"] as const) {
        for (const set of g.teams[seat]) {
          for (const word of [set.species, set.item, set.ability, ...set.moves]) if (word.length > 3) expect(text.toLowerCase()).not.toContain(word.toLowerCase());
        }
      }
    }
    expect(text).not.toMatch(/[0-9a-f]{40}/);
    const leaves: unknown[] = [];
    const walk = (value: unknown): void => {
      if (value !== null && typeof value === "object") Object.values(value).forEach(walk);
      else leaves.push(value);
    };
    walk(out);
    expect(leaves.every((v) => v === null || (typeof v === "number" && Number.isInteger(v)) || ["preview", "battle", "item", "ability", "move", "belief", "prior", "v1"].includes(v as string))).toBe(true);
  });
});

describe("refusals and fallbacks are counted, not fixed", () => {
  it("counts a belief refusal by code and leaves those requests out of the rows", () => {
    // a line the reader cannot take, placed just after the Team Preview request: the belief then refuses the later requests of that seat
    // (the flags are those raised up to each request, as the actor sees them) and answers the Team Preview itself
    const game = played[0] as OracleGame;
    const at = game.channels.p1.findIndex((l) => l.startsWith("|request|"));
    const lines = [...game.channels.p1.slice(0, at + 1), "|zzunknownline|x", ...game.channels.p1.slice(at + 1)];
    const flagged: OracleGame = { ...game, channels: { p1: lines, p2: game.channels.p2 } };
    const acc = run([flagged]);
    const out = report(acc);
    expect(out.scored.preview).toBe(2);
    expect(out.refusals.reader_flagged).toBeGreaterThan(0);
    expect(out.scored.preview + out.scored.battle + Object.values(out.refusals).reduce((a, b) => a + b, 0) + out.no_truth).toBe(out.requests);
  });

  it("counts a roster that cannot be matched to the real team", () => {
    const g = played[0] as OracleGame;
    const wrong: OracleGame = { ...g, teams: { p1: g.teams.p1, p2: g.teams.p1.slice().reverse().map((s, i) => ({ ...s, species: (g.teams.p2[i] as { species: string }).species === "" ? s.species : "Pikachu" })) } };
    expect(report(run([wrong])).no_truth).toBeGreaterThan(0);
  });

  it("counts a world sampling that refuses, apart from the belief", () => {
    const port = { ...ctx.port, sample: () => { throw Object.assign(new Error("x"), { name: "BeliefError", code: "legality_exhausted" }); } };
    const out = report(run([played[0] as OracleGame], { ...ctx, port }));
    expect(out.joint.sample_refusals.legality_exhausted).toBe(out.scored.preview + out.scored.battle);
    expect(out.joint.pairs).toBe(0);
  });

  it("lets a defect that is not a refusal through", () => {
    const port = { ...ctx.port, sample: () => { throw new Error("broken"); } };
    expect(() => run([played[0] as OracleGame], { ...ctx, port })).toThrow("broken");
  });

  it("counts a Pokemon v1 cannot build a set for", () => {
    const v1 = { ...ctx.v1, vocabulary: { ...vocabulary, roster: [] } };
    expect(report(run([played[0] as OracleGame], { ...ctx, v1 })).v1_failures).toBeGreaterThan(0);
  });
});

describe("the harness stays outside every actor", () => {
  const root = fileURLToPath(new URL("../..", import.meta.url));
  const sources = (dir: string): string[] =>
    !existsSync(dir)
      ? []
      : readdirSync(dir).flatMap((name) => {
          const path = join(dir, name);
          return statSync(path).isDirectory() ? sources(path) : /\.(ts|cjs|mjs)$/.test(name) ? [path] : [];
        });

  it("is imported by no source file of any package, and not exported from a package index", () => {
    const importers = readdirSync(root)
      .flatMap((pkg) => sources(join(root, pkg, "src")))
      .filter((file) => readFileSync(file, "utf8").includes("prediction-quality") && !file.endsWith("prediction-quality.ts"));
    expect(importers.map((f) => relative(root, f))).toEqual([]);
  });

  it("imports no actor module", () => {
    const text = readFileSync(join(root, "team-dataset", "src", "prediction-quality.ts"), "utf8");
    const imports = [...text.matchAll(/from "([^"]+)"/g)].map((m) => m[1]);
    expect(imports.filter((i) => /actor/.test(i as string))).toEqual([]);
  });
});
