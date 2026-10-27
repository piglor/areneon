// The exact seeded sampler (ADR-025 section 4): frequencies against the DP's exact marginals on a case
// whose Item Clause total exceeds 2^64, the same belief and stream giving the same world, every world
// respecting the revealed facts and Item Clause, the legality redraws, the spread port and the type test.

import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { BeliefError, type BeliefErrorCode, type OpponentBelief, type OpponentWorld, type SetLegality, type WorldSet, belief, createSampler, prepared } from "../src/index.ts";
import { jcs } from "../src/canonical.ts";
import { worldStream } from "../src/draw.ts";
import { toHex } from "../src/sha256.ts";
import { alpha, betaFormes, hidingConfig, inert, input, prior, seen, config, vocabulary } from "./support.ts";

type Forme = UsagePrior["species"][number]["formes"][number];

const accept: SetLegality = { legal: () => true };
const sampler = createSampler({ legality: accept });
const STREAM = Uint8Array.from({ length: 32 }, (_, i) => i * 3 + 1);
const SW = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;
const refusal = (run: () => unknown): BeliefErrorCode => {
  try {
    run();
  } catch (e) {
    if (e instanceof BeliefError) return e.code;
    throw e;
  }
  throw new Error("expected a BeliefError");
};
const value = (r: { num: bigint; den: bigint }): number => Number(r.num) / Number(r.den);
const streamOf = (i: number): Uint8Array => worldStream(new Uint8Array(32), i);

// --- a case with large weights: the Item Clause total is far above 2^64 -------------------------------

const spreads = (): Forme["spreads"] => ({
  denominator: 1000003,
  entries: [
    { nature: "hardy", stat_points: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, weight: 700001 },
    { nature: "adamant", stat_points: { hp: 1, atk: 2, def: 3, spa: 4, spd: 5, spe: 6 }, weight: 300002 },
  ],
  residual: 0,
  unmapped: 0,
});
const items = (ix: number, iy: number, none: number): Forme["items"] => ({ denominator: ix + iy + none, entries: [{ id: "ix", weight: ix }, { id: "iy", weight: iy }], no_item: none, residual: 0, unmapped: 0 });

const rich = (f: Forme, ix: number, iy: number, none: number): Forme => ({ ...f, items: items(ix, iy, none), spreads: spreads() });

function bigPrior(): UsagePrior {
  const species = (id: string, formes: Forme[]) => ({ id, denominator: formes.reduce((s, f) => s + f.weight, 0), formes: formes as [Forme, ...Forme[]] });
  const [beta, mega] = betaFormes();
  return prior((p) => {
    p.species = [
      species("alpha", [rich(alpha(), 1200007, 900011, 800003)]),
      species("beta", [rich(beta, 600011, 500009, 300007), mega]),
      species("delta", [rich(inert("delta", "d1", "dm"), 1100003, 700001, 400009)]),
      species("epsilon", [rich(inert("epsilon", "ill", "em"), 900001, 1300009, 500029)]),
      species("gamma", [rich({ ...inert("gamma", "g1", "gm"), moves: { denominator: 4, slots: 4, entries: [1, 2, 3, 4].map((k) => ({ id: `gm${k}`, inclusion: 4 })), empty_slots: 0, residual: 0, unmapped: 0 } }, 1000003, 1000033, 100003)]),
      species("zeta", [rich(inert("zeta", "z1", "zm"), 800011, 800017, 800021)]),
    ] as UsagePrior["species"];
  });
}

const big = (more: readonly string[] = []): OpponentBelief => belief(input(more, { prior: bigPrior() }));

/** The tolerance of a frequency estimate of probability p over n trials: five standard errors. */
const near = (observed: number, p: number, n: number): void => {
  const tolerance = 5 * Math.sqrt((p * (1 - p)) / n) + 1e-9;
  expect(Math.abs(observed - p), `observed ${observed} against ${p} over ${n}`).toBeLessThanOrEqual(tolerance);
};

describe("the frequencies of sample() against the DP's exact marginals, with a total above 2^64", () => {
  const b = big();
  const N = 6000;
  const worlds = Array.from({ length: N }, (_, i) => sampler.sample(b, streamOf(i)).world);

  it("the case is built so that the Item Clause total exceeds 2^64 (and so does a draw below it need more than 64 bits)", () => {
    expect(b.assignments[0]!.normaliser).toBeGreaterThan(2n ** 64n);
    expect(b.disguise.total).toBeGreaterThan(2n ** 64n);
  });

  it("brought: exactly the picked size, each slot with the belief's p_brought", () => {
    for (const w of worlds) expect(w.slots.filter((s) => s.brought)).toHaveLength(b.brought.picked);
    for (const s of b.slots) near(worlds.filter((w) => w.slots[s.index]!.brought).length / N, value(s.p_brought), N);
  });

  it("items (Item Clause joint), abilities, spreads, formes and moves follow the belief's marginals", () => {
    for (const s of b.slots) {
      const sets = worlds.flatMap((w) => (w.slots[s.index]!.set === null ? [] : [w.slots[s.index]!.set as WorldSet]));
      const n = sets.length;
      for (const m of s.items) near(sets.filter((x) => x.item === m.id).length / n, value(m.p), n);
      for (const m of s.abilities) near(sets.filter((x) => x.ability === m.id).length / n, value(m.p), n);
      for (const m of s.spreads) near(sets.filter((x) => `${x.nature}:${Object.values(x.stat_points).join("-")}` === m.id).length / n, value(m.p), n);
      for (const m of s.formes) near(sets.filter((x) => x.forme === m.id).length / n, value(m.p), n);
      for (const m of s.moves) near(sets.filter((x) => x.moves.includes(m.id)).length / n, value(m.p), n);
    }
  });

  it("no item is held twice across the six slots in any world (checked on the brought ones: the rule across all six is the DP's)", () => {
    for (const w of worlds) {
      const held = w.slots.flatMap((s) => (s.set?.item == null ? [] : [s.set.item]));
      expect(new Set(held).size).toBe(held.length);
    }
  });

  it("the joint of two slots that share items matches the DP: P(alpha holds ix and gamma holds iy) among worlds with both brought", () => {
    const both = worlds.filter((w) => w.slots[0]!.brought && w.slots[2]!.brought);
    const joint = both.filter((w) => w.slots[0]!.set!.item === "ix" && w.slots[2]!.set!.item === "iy").length / both.length;
    // The exact joint from the cells: sum of the DP's weight of the two assignments over the normaliser, by brute force on the integer cells.
    const a = b.assignments[0]!;
    const w = (i: number, item: string | null) => a.slots[i]!.cells.filter((c) => c.item === item).reduce((t, c) => t + c.weight, 0n);
    const itemsOf = ["betite", "ix", "iy", null] as const;
    let total = 0n;
    let hit = 0n;
    const walk = (i: number, held: (string | null)[], weight: bigint): void => {
      if (i === 6) {
        total += weight;
        if (held[0] === "ix" && held[2] === "iy") hit += weight;
        return;
      }
      for (const item of itemsOf) {
        if (item !== null && held.includes(item)) continue;
        walk(i + 1, [...held, item], weight * w(i, item));
      }
    };
    walk(0, [], 1n);
    expect(total).toBe(a.normaliser);
    near(joint, Number((hit * 1_000_000_000n) / total) / 1e9, both.length);
  });
});

describe("the tolerance has teeth", () => {
  it("a frequency of the wrong probability is flagged (negative control)", () => {
    expect(() => near(0.4, 0.3, 6000)).toThrow();
    expect(() => near(0.3, 0.3, 6000)).not.toThrow();
  });
});

describe("the same belief and stream give the same world", () => {
  const b = big();

  it("sample is a function of the belief and the stream, and a replay gives the same worlds", () => {
    const first = sampler.sample(b, STREAM);
    expect(jcs(JSON.parse(JSON.stringify(first)))).toBe(jcs(JSON.parse(JSON.stringify(sampler.sample(b, STREAM)))));
    expect(sampler.sample(big(), STREAM)).toEqual(first);
    expect(createSampler({ legality: accept }).sample(b, STREAM)).toEqual(first);
    expect(JSON.stringify(sampler.sample(b, streamOf(5)))).not.toBe(JSON.stringify(first));
  });

  it("sampleWorlds(belief, stream, n) is world i = sample(belief, SHA-256(JCS({stream, index: i, purpose}))), each weighing 1/n", () => {
    const out = sampler.sampleWorlds(b, STREAM, 8);
    expect(out.worlds).toHaveLength(8);
    out.worlds.forEach((w, i) => {
      const own = createHash("sha256").update(jcs({ stream: toHex(STREAM), index: i, purpose: "opponent-world-index/1" })).digest();
      expect(w.index).toBe(i);
      expect(w.weight).toEqual({ num: 1n, den: 8n });
      expect({ world: w.world, redraws: w.redraws }).toEqual(sampler.sample(b, Uint8Array.from(own)));
    });
    expect(out.worlds.map((w) => w.weight)).toEqual(Array.from({ length: 8 }, () => ({ num: 1n, den: 8n })));
    // A consumer that redraws asks for more worlds: the first eight do not change.
    expect(sampler.sampleWorlds(b, STREAM, 12).worlds.slice(0, 8).map((w) => w.world)).toEqual(out.worlds.map((w) => w.world));
    expect(out.redraws).toBe(0);
  });

  it("refuses a stream that is not 32 bytes and a count that is not a positive integer", () => {
    expect(() => sampler.sample(b, new Uint8Array(31))).toThrow(RangeError);
    for (const n of [0, -1, 1.5, Number.NaN]) expect(() => sampler.sampleWorlds(b, STREAM, n)).toThrow(RangeError);
  });

  it("builds the tables once per belief", () => {
    expect(prepared(b)).toBe(prepared(b));
    expect(prepared(big())).not.toBe(prepared(b));
  });
});

describe("every world respects the revealed facts", () => {
  it("a revealed move, item and ability, a settled brought Pokemon and a Mega Evolution", () => {
    const lines = [
      SW("a", "Alp", "Alpha"),
      "|move|p2a: Alp|Mb",
      "|-item|p2a: Alp|Ix",
      "|-ability|p2a: Alp|A2",
      SW("b", "Gam", "Gamma"),
      "|move|p2b: Gam|Gm2",
    ];
    const b = big(lines);
    for (let i = 0; i < 300; i += 1) {
      const { world } = sampler.sample(b, streamOf(i));
      const a = world.slots[0]!;
      const g = world.slots[2]!;
      expect(a.brought && g.brought).toBe(true);
      expect(a.set).toMatchObject({ species: "alpha", forme: "alpha", item: "ix", ability: "a2", level: 50, gender: "M", shown: { name: "Alp", species: "alpha" } });
      expect(a.set!.moves).toContain("mb");
      expect(a.set!.moves).toContain("ma");
      expect(g.set!.moves).toContain("gm2");
      expect(g.set!.item).not.toBe("ix");
      expect(world.slots.filter((s) => s.brought)).toHaveLength(4);
    }
  });

  it("a Mega Evolution fixes forme and stone; a stone holder not yet evolved keeps the pre-Evolution ability", () => {
    const evolved = big([SW("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]);
    for (let i = 0; i < 100; i += 1) {
      const set = sampler.sample(evolved, streamOf(i)).world.slots[1]!.set!;
      expect(set).toMatchObject({ species: "beta", forme: "betamega", item: "betite", ability: "bmega" });
    }
    const before = big([SW("a", "Bet", "Beta")]);
    const sets = Array.from({ length: 400 }, (_, i) => sampler.sample(before, streamOf(i)).world.slots[1]!.set!);
    const stone = sets.filter((s) => s.forme === "betamega");
    expect(stone.length).toBeGreaterThan(0);
    for (const s of stone) expect(["b1", "b2"]).toContain(s.ability);
    for (const s of sets.filter((x) => x.forme === "beta")) expect(s.item).not.toBe("betite");
  });

  it("the shown identifiers of a Pokemon not yet seen are the preview's", () => {
    const slot = sampler.sample(big([SW("a", "Alp", "Alpha")]), STREAM).world.slots.find((s) => s.brought && s.index !== 0)!;
    const set = slot.set!;
    expect(set.shown).toEqual({ name: null, species: set.species, level: 50, gender: "M" });
    expect(set.level).toBe(50);
  });

  it("falls back to the format's level when the channel shows none", () => {
    const b = big();
    const noLevel: OpponentBelief = { ...b, slots: b.slots.map((s) => ({ ...s, shown: { ...s.shown, level: null } })), sampling: { ...b.sampling, level: 77 } };
    for (const s of sampler.sample(noLevel, STREAM).world.slots) if (s.set !== null) expect(s.set.level).toBe(77);
  });
});

describe("a belief without Item Clause draws the items independently", () => {
  it("two slots can hold the same item, and the item marginals are the slots' own", () => {
    const b = belief(input([], { prior: bigPrior(), vocabulary: { ...vocabulary(), rules: { adjust_level: 50, min_team_size: 6, max_team_size: 6, picked_team_size: 4 } } }));
    expect(b.itemClause.enforced).toBe(false);
    const worlds = Array.from({ length: 3000 }, (_, i) => sampler.sample(b, streamOf(i)).world);
    expect(worlds.some((w) => w.slots.filter((s) => s.set?.item === "ix").length > 1)).toBe(true);
    const alphaSets = worlds.flatMap((w) => (w.slots[0]!.set === null ? [] : [w.slots[0]!.set]));
    for (const m of b.slots[0]!.items) near(alphaSets.filter((x) => x.item === m.id).length / alphaSets.length, value(m.p), alphaSets.length);
  });
});

describe("the disguise assignment is drawn first", () => {
  const lines = [SW("a", "Alp", "Alpha"), "|-item|p2a: Alp|Ix"];
  const b = belief(input(lines, { prior: bigPrior(), config: hidingConfig }));

  it("each assignment is drawn with its weight, and the position's evidence goes to the slot it is assigned to", () => {
    expect(b.assignments).toHaveLength(2);
    const N = 3000;
    let moved = 0;
    for (let i = 0; i < N; i += 1) {
      const { world } = sampler.sample(b, streamOf(i));
      const alphaItem = world.slots[0]!.set?.item;
      const epsilon = world.slots[4]!;
      // Item Clause: the item shown is held by the position's own slot (identity) or by the hiding slot (moved), never both.
      const isMoved = epsilon.set?.item === "ix";
      if (isMoved) {
        expect(epsilon.brought).toBe(true);
        expect(alphaItem).not.toBe("ix");
        moved += 1;
      } else {
        expect(world.slots[0]!.brought).toBe(true);
        expect(alphaItem).toBe("ix");
      }
    }
    const share = b.assignments.find((a) => a.moved.length > 0)!.p;
    near(moved / N, value(share), N);
  });
});

describe("legality: a world with an illegal set is redrawn from the next counter block", () => {
  const b = big();

  it("redraws until the port accepts, returns the count and is deterministic", () => {
    // Refuse any world in which a brought slot holds the nature adamant.
    const picky: SetLegality = { legal: (set) => set.nature !== "adamant" };
    const s = createSampler({ legality: picky });
    const counts = Array.from({ length: 200 }, (_, i) => s.sample(b, streamOf(i)));
    expect(counts.some((c) => c.redraws > 0)).toBe(true);
    for (const c of counts) for (const slot of c.world.slots) expect(slot.set === null || slot.set.nature === "hardy").toBe(true);
    expect(s.sample(b, streamOf(3))).toEqual(counts[3]);
    const total = s.sampleWorlds(b, STREAM, 20);
    expect(total.redraws).toBe(total.worlds.reduce((t, w) => t + w.redraws, 0));
  });

  it("each redraw is a fresh world: the sets the port sees differ between attempts", () => {
    const seenSets: string[] = [];
    const s = createSampler({ legality: { legal: (set) => { seenSets.push(JSON.stringify(set)); return seenSets.length > 4; } } });
    const out = s.sample(b, STREAM);
    expect(out.redraws).toBeGreaterThanOrEqual(1);
    expect(new Set(seenSets).size).toBeGreaterThan(1);
  });

  it("gives up after max_legality_redraws redraws with legality_redraws_exhausted", () => {
    let calls = 0;
    const never = createSampler({ legality: { legal: () => { calls += 1; return false; } } });
    const limited: OpponentBelief = belief(input([], { prior: bigPrior(), config: { ...config, max_legality_redraws: 3 } }));
    expect(refusal(() => never.sample(limited, STREAM))).toBe("legality_redraws_exhausted");
    expect(calls).toBe(4);
    const none = belief(input([], { prior: bigPrior(), config: { ...config, max_legality_redraws: 0 } }));
    calls = 0;
    expect(refusal(() => never.sample(none, STREAM))).toBe("legality_redraws_exhausted");
    expect(calls).toBe(1);
  });
});

describe("the spread port", () => {
  const withOther = (): OpponentBelief =>
    belief(input([], { prior: prior((p) => {
      p.species[0]!.formes[0]!.spreads = { denominator: 2, entries: [], residual: 2, unmapped: 0 };
    }) }));

  it("an other-spread draw without a SpreadSampler is spread_sampler_required", () => {
    const b = withOther();
    expect(refusal(() => Array.from({ length: 40 }, (_, i) => sampler.sample(b, streamOf(i))))).toBe("spread_sampler_required");
  });

  it("with the port, its nature and Stat Points are used and its draws come from the world's stream", () => {
    const b = withOther();
    const port = createSampler({
      legality: accept,
      spreads: { other: (species, forme, below) => ({ nature: below(2n) === 0n ? "calm" : "bold", stat_points: { hp: Number(below(33n)), atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }) },
    });
    const sets = Array.from({ length: 200 }, (_, i) => port.sample(b, streamOf(i)).world.slots[0]!.set).filter((s): s is WorldSet => s !== null);
    expect(sets.length).toBeGreaterThan(50);
    expect(new Set(sets.map((s) => s.nature))).toEqual(new Set(["calm", "bold"]));
    expect(new Set(sets.map((s) => s.stat_points.hp)).size).toBeGreaterThan(10);
    expect(port.sample(b, streamOf(1))).toEqual(port.sample(b, streamOf(1)));
  });
});

describe("the type test: sample takes a belief and a stream, nothing else", () => {
  it("has no parameter for the opponent's team, the omniscient log, hidden verdicts, a battle record or the seed", () => {
    const b = big();
    // @ts-expect-error: the opponent's real team is not an input
    sampler.sample(b, STREAM, { team: [] });
    // @ts-expect-error: nor a seed
    sampler.sample(b, STREAM, "sodium,0");
    // @ts-expect-error: the stream is bytes, not a seed string
    expect(() => sampler.sample(b, "sodium,0")).toThrow();
    // @ts-expect-error: a belief input is not a belief
    expect(() => sampler.sample(input(), STREAM)).toThrow();
    const parameters: Parameters<typeof sampler.sample> = [b, STREAM];
    expect(parameters).toHaveLength(2);
    const sample: (belief: OpponentBelief, stream: Uint8Array) => { world: OpponentWorld; redraws: number } = sampler.sample;
    expect(typeof sample).toBe("function");
  });

  it("the seen channel of a hidden-information canary: a world never exposes more than the sets", () => {
    const { opponent } = seen([SW("a", "Alp", "Alpha")]);
    expect(opponent.roster).toHaveLength(6);
    const keys = Object.keys(sampler.sample(big(), STREAM));
    expect(keys.sort()).toEqual(["redraws", "world"]);
  });
});
