// The Illusion-aware belief (ADR-025 section 3, N-B): hand-written channels for every branch of the
// disguise mixture. `epsilon` can hide its identity (its ability `ill` is on the config's list); a
// second hiding member, `zeta`, is made by giving it the same ability in the vocabulary.

import { describe, expect, it } from "vitest";
import { BeliefError, type BeliefErrorCode, belief, topWorlds } from "../src/index.ts";
import type { BeliefInput, Marginal, OpponentBelief } from "../src/index.ts";
import { ONE, add, ratio } from "../src/ratio.ts";
import type { Ratio } from "../src/ratio.ts";
import { hidingConfig, inert, input, prior, seen, text, vocabulary } from "./support.ts";

const p = (num: bigint, den = 1n) => ratio(num, den);
const slot = (b: OpponentBelief, species: string) => b.slots.find((s) => s.species === species)!;
const get = <T>(list: readonly Marginal<T>[], id: T) => list.find((m) => m.id === id)?.p;
const refusal = (run: () => unknown): BeliefErrorCode => {
  try {
    run();
  } catch (e) {
    if (e instanceof BeliefError) return e.code;
    throw e;
  }
  throw new Error("expected a BeliefError");
};

const ALPHA = 0;
const GAMMA = 2;
const EPSILON = 4;
const ZETA = 5;
const SW = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;
const alphaIn = SW("a", "Alp", "Alpha");
const gammaIn = SW("b", "Gam", "Gamma");

/** The vocabulary in which `zeta` can hide too. */
const twoHiding = () => {
  const v = vocabulary();
  return { ...v, roster: v.roster.map((r) => (r.id === "zeta" ? { ...r, abilities: [...r.abilities, "ill"] as [string, ...string[]] } : r)) };
};

/** A usage prior in which epsilon holds the item `ix` with probability 1 / (1 + none) (by default it never holds one). */
const epsilonWithItem = (none = 1) =>
  prior((pr) => {
    const e = inert("epsilon", "ill", "em");
    pr.species[3] = { id: "epsilon", denominator: 1, formes: [{ ...e, items: { denominator: 1 + none, entries: [{ id: "ix", weight: 1 }], no_item: none, residual: 0, unmapped: 0 } }] };
  });

const run = (more: readonly string[], change: Partial<BeliefInput> = {}) => belief(input(more, { config: hidingConfig, ...change }));

describe("no position shown yet: nothing to assign", () => {
  const b = run([]);

  it("an identity-hiding species among the six no longer refuses; the belief is the plain one", () => {
    expect(b.assignments).toHaveLength(1);
    expect(b.disguise).toMatchObject({ hiding: [EPSILON], unsettled: [EPSILON], positions: [] });
    expect(b.assignments[0]).toMatchObject({ moved: [], required: [] });
    expect(text(b.slots)).toBe(text(belief(input()).slots));
  });
});

describe("one unsettled shown position", () => {
  const b = run([alphaIn]);

  it("is either itself or the hiding member: three worlds of structure, two assignments, equal weight", () => {
    expect(b.disguise).toMatchObject({ hiding: [EPSILON], unsettled: [EPSILON], positions: [ALPHA] });
    expect(b.assignments.map((a) => a.moved)).toEqual([[], [{ position: ALPHA, slot: EPSILON }]]);
    expect(b.assignments.map((a) => a.required)).toEqual([[ALPHA], [EPSILON]]);
    // Nothing is known of the Pokemon, so the two explanations are exactly as likely.
    expect(b.assignments.map((a) => a.p)).toEqual([p(1n, 2n), p(1n, 2n)]);
    expect(b.disguise.total).toBe(b.assignments.reduce((s, a) => s + a.weight, 0n));
    expect(b.constraints).toContain("disguise:assignments=2");
  });

  it("the brought-set shortcut is off: neither the shown nor the hiding Pokemon is certain", () => {
    expect(b.brought).toEqual({ picked: 4, settled: [], subsets: 10n });
    // alpha: in the four for sure when it is itself, else with the rest of the roster.
    expect(slot(b, "alpha").p_brought).toEqual(add(p(1n, 2n), p(3n, 10n)));
    expect(slot(b, "epsilon").p_brought).toEqual(add(p(1n, 2n), p(3n, 10n)));
    expect(slot(b, "alpha").settled).toBe(false);
    expect(slot(b, "gamma").p_brought).toEqual(p(3n, 5n));
  });

  it("each assignment carries its own slot beliefs and says whose evidence it is", () => {
    const moved = b.assignments[1]!;
    expect(moved.slots[EPSILON]!.constraints).toContain("disguise_of:0");
    expect(moved.slots[ALPHA]!.constraints).toContain("disguised_by:4");
    expect(moved.slots[EPSILON]!.settled).toBe(true);
    expect(moved.slots[ALPHA]!.settled).toBe(false);
    // the details shown (forme, level, gender) stay with the position
    expect(moved.slots[ALPHA]!.constraints).toEqual(expect.arrayContaining(["forme:alpha", "level:50", "gender:M"]));
    expect(b.assignments[0]!.slots[ALPHA]!.settled).toBe(true);
    expect(slot(b, "alpha").constraints).toContain("species:alpha");
    expect(slot(b, "alpha").constraints).not.toContain("settled");
  });

  it("the mixture marginals are the weighted sum of the assignments' marginals", () => {
    const [a0, a1] = b.assignments;
    const mix = (pick: (s: OpponentBelief["slots"][number]) => readonly Marginal<string | null>[], id: string | null): Ratio => {
      let sum = p(0n);
      for (const a of [a0!, a1!]) sum = add(sum, ratioMul(a.p, get(pick(a.slots[GAMMA]!), id) ?? p(0n)));
      return sum;
    };
    for (const m of slot(b, "gamma").items) expect(m.p).toEqual(mix((s) => s.items, m.id));
  });

  it("the plain belief of a team without a hiding species is unchanged by the config", () => {
    const plain = belief(input([alphaIn]));
    expect(plain.assignments).toHaveLength(1);
    expect(plain.brought).toEqual({ picked: 4, settled: [ALPHA], subsets: 10n });
    expect(plain.disguise.positions).toEqual([]);
  });
});

/** The moves of `alpha` in the test prior, for a forme that holds no item. */
const alphaMoves = () => prior().species[0]!.formes[0]!.moves;

const ratioMul = (a: Ratio, b: Ratio): Ratio => ratio(a.num * b.num, a.den * b.den);

describe("evidence that only one explanation allows", () => {
  it("a move the shown species cannot have is the hiding member's: one assignment, the move is certain there", () => {
    const b = run([alphaIn, "|move|p2a: Alp|Em1"]);
    expect(b.assignments).toHaveLength(1);
    expect(b.assignments[0]!.moved).toEqual([{ position: ALPHA, slot: EPSILON }]);
    expect(get(slot(b, "epsilon").moves, "em1")).toEqual(p(1n));
    expect(slot(b, "epsilon").p_brought).toEqual(p(1n));
    expect(slot(b, "epsilon").settled).toBe(true);
    // alpha has no evidence: its moves are the prior's, and it is in the four only by the roster's odds
    expect(get(slot(b, "alpha").moves, "ma")).toEqual(p(1n));
    expect(slot(b, "alpha").p_brought).toEqual(p(3n, 5n));
    expect(slot(b, "epsilon").constraints).toContain("move:em1");
    expect(b.constraints).not.toContain("disguise:assignments=1");
  });

  it("a move only the shown species can hold rules the disguise out", () => {
    // `ma` is alpha's (always in its set); epsilon cannot hold it, so only alpha explains it.
    const b = run([alphaIn, "|move|p2a: Alp|Ma"]);
    expect(b.assignments.map((a) => a.moved)).toEqual([[]]);
  });

  it("a Mega Evolution cannot be a disguise", () => {
    const b = run([SW("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]);
    expect(b.assignments.map((a) => a.moved)).toEqual([[]]);
    expect(b.brought.settled).toEqual([1]);
  });

  it("a revealed item goes to the member that can hold it", () => {
    const b = run([alphaIn, "|-item|p2a: Alp|Ix"], { prior: epsilonWithItem() });
    // Both explain it. Worked out on paper (the item prior of alpha is ix 5, iy 2, none 3 of 10; gamma's ix 1, iy 2, none 1 of 4;
    // the rest hold none; the Item Clause forbids a second ix and a second iy): shown as itself, alpha = ix (1/2), epsilon none
    // (1/2), gamma iy or none (3/4): 3/16. Disguised, epsilon = ix (1/2), alpha and gamma take iy and none without repeating iy:
    // alpha iy and gamma none 1/5 x 1/4, alpha none and gamma iy 3/10 x 1/2, both none 3/10 x 1/4 = 11/40, times 1/2: 11/80.
    expect(b.assignments.map((a) => a.moved)).toEqual([[], [{ position: ALPHA, slot: EPSILON }]]);
    expect(get(b.assignments[1]!.slots[EPSILON]!.items, "ix")).toEqual(p(1n));
    expect(get(b.assignments[0]!.slots[EPSILON]!.items, "ix")).toBeUndefined();
    expect(b.assignments.map((a) => a.p)).toEqual([p(15n, 26n), p(11n, 26n)]);
  });

  it("explanations are weighed by how likely each member is to hold the item", () => {
    // epsilon holds ix a quarter of the time: 1/2 x 3/4 x 3/4 = 9/32 against 1/4 x 11/40 = 11/160, that is 45 : 11.
    const b = run([alphaIn, "|-item|p2a: Alp|Ix"], { prior: epsilonWithItem(3) });
    expect(b.assignments.map((a) => a.p)).toEqual([p(45n, 56n), p(11n, 56n)]);
    // the probability that epsilon holds ix is the share of the disguised explanation
    expect(get(slot(b, "epsilon").items, "ix")).toEqual(p(11n, 56n));
  });
});

describe("Hall's condition", () => {
  it("two positions that only the one hiding member can explain have no assignment", () => {
    const more = [alphaIn, gammaIn, "|move|p2a: Alp|Em1", "|move|p2b: Gam|Em2"];
    expect(refusal(() => run(more))).toBe("evidence_has_no_support");
  });

  it("with two hiding members the same two positions are matched, in the one way that works", () => {
    const more = [alphaIn, gammaIn, "|move|p2a: Alp|Em1", "|move|p2b: Gam|Zm1"];
    const b = run(more, { vocabulary: twoHiding() });
    expect(b.assignments).toHaveLength(1);
    expect(b.assignments[0]!.moved).toEqual([{ position: ALPHA, slot: EPSILON }, { position: GAMMA, slot: ZETA }]);
    expect(b.assignments[0]!.required).toEqual([EPSILON, ZETA]);
    expect(b.brought.subsets).toBe(6n);
  });

  it("each hiding member is used at most once: three positions' worth of choices are counted exactly", () => {
    // Two shown positions, two hiding members: the identity, 2 x 2 single disguises, and 2 matchings of both.
    const b = run([alphaIn, gammaIn], { vocabulary: twoHiding() });
    expect(b.assignments).toHaveLength(7);
    const used = b.assignments.map((a) => a.moved.map((m) => m.slot));
    expect(used.every((slots) => new Set(slots).size === slots.length)).toBe(true);
    expect(b.assignments.reduce((s, a) => add(s, a.p), p(0n))).toEqual(ONE);
    expect(b.assignments[0]!.moved).toEqual([]);
  });

  it("the second hiding member can hold a move the first cannot", () => {
    const b = run([alphaIn, "|move|p2a: Alp|Zm1"], { vocabulary: twoHiding() });
    expect(b.assignments.map((a) => a.moved)).toEqual([[{ position: ALPHA, slot: ZETA }]]);
  });
});

describe("the reviewed bound on assignments", () => {
  const more = [alphaIn, gammaIn];

  it("above max_disguise_assignments the belief refuses with its own code", () => {
    expect(run(more).assignments).toHaveLength(3);
    expect(refusal(() => run(more, { config: { ...hidingConfig, max_disguise_assignments: 2 } }))).toBe("disguise_assignments_exceeded");
    expect(run(more, { config: { ...hidingConfig, max_disguise_assignments: 3 } }).assignments).toHaveLength(3);
  });

  it("a refusal carries no hidden information: a fixed message and no position", () => {
    try {
      run(more, { config: { ...hidingConfig, max_disguise_assignments: 1 } });
      throw new Error("expected a refusal");
    } catch (e) {
      expect(e).toBeInstanceOf(BeliefError);
      expect((e as BeliefError).slot).toBeNull();
      expect((e as BeliefError).message).toBe("more disguise assignments than the reviewed bound");
    }
  });
});

describe("a replace settles the disguise", () => {
  const before = [alphaIn, "|move|p2a: Alp|Em1"];
  const replace = "|replace|p2a: Eps|Epsilon, L50, M";

  it("after |replace| the reader's settled state applies and the belief has one assignment", () => {
    const after = run([...before, replace]);
    expect(after.assignments).toHaveLength(1);
    expect(after.disguise).toMatchObject({ unsettled: [], positions: [] });
    expect(after.brought.settled).toEqual([EPSILON]);
    expect(get(slot(after, "epsilon").moves, "em1")).toEqual(p(1n));
    // the shown species is no longer in play
    expect(slot(after, "alpha").settled).toBe(false);
    expect(slot(after, "alpha").p_brought).toEqual(p(3n, 5n));
  });

  it("equals the belief of the same facts shown on the member from the start", () => {
    const after = run([...before, replace]);
    const direct = run([SW("a", "Eps", "Epsilon"), "|move|p2a: Eps|Em1"]);
    for (const s of ["alpha", "epsilon", "gamma"]) {
      expect(text(slot(after, s).moves)).toBe(text(slot(direct, s).moves));
      expect(text(slot(after, s).items)).toBe(text(slot(direct, s).items));
      expect(slot(after, s).p_brought).toEqual(slot(direct, s).p_brought);
    }
  });

  it("a late replace: moves and an item shown for several turns before it are still the member's", () => {
    const turns = [alphaIn, "|move|p2a: Alp|Em1", "|-item|p2a: Alp|Ix", "|turn|2", "|move|p2a: Alp|Em2", "|turn|3", "|move|p2a: Alp|Em3"];
    const prior = epsilonWithItem();
    const mid = run(turns, { prior });
    // Alpha cannot hold the em moves: the disguise is the only explanation, and ix is epsilon's.
    expect(mid.assignments).toHaveLength(1);
    expect(mid.assignments[0]!.moved).toEqual([{ position: ALPHA, slot: EPSILON }]);
    expect(get(slot(mid, "epsilon").items, "ix")).toEqual(p(1n));
    expect(get(slot(mid, "epsilon").moves, "em3")).toEqual(p(1n));
    const late = run([...turns, replace], { prior });
    expect(late.assignments).toHaveLength(1);
    expect(late.assignments[0]!.moved).toEqual([]);
    expect(text(slot(late, "epsilon").moves)).toBe(text(slot(mid, "epsilon").moves));
    expect(text(slot(late, "epsilon").items)).toBe(text(slot(mid, "epsilon").items));
    // the shown species lost the item and the moves it never had
    expect(get(slot(late, "alpha").items, "ix")).not.toEqual(p(1n));
  });

  it("without the mixture the same channel has no world: the item cannot be the shown species'", () => {
    const narrow = prior((pr) => {
      const e = epsilonWithItem().species[3]!;
      pr.species[3] = e;
      pr.species[0] = { id: "alpha", denominator: 1, formes: [{ ...inert("alpha", "a1", "ma"), moves: alphaMoves() }] };
    });
    const more = [alphaIn, "|-item|p2a: Alp|Ix"];
    expect(refusal(() => belief(input(more, { prior: narrow })))).toBe("evidence_has_no_support");
    const b = run(more, { prior: narrow });
    expect(b.assignments.map((a) => a.moved)).toEqual([[{ position: ALPHA, slot: EPSILON }]]);
    expect(get(slot(b, "epsilon").items, "ix")).toEqual(p(1n));
  });

  it("evidence that only the shown species explains keeps the identity until a replace says otherwise", () => {
    const mid = run([alphaIn, "|move|p2a: Alp|Ma", "|-item|p2a: Alp|Ix"], { prior: epsilonWithItem() });
    expect(mid.assignments.map((a) => a.moved)).toEqual([[]]);
  });
});

describe("the mixture as a whole", () => {
  it("is a pure function of its input", () => {
    const i = input([alphaIn, gammaIn], { config: hidingConfig });
    expect(text(belief(i))).toBe(text(belief(i)));
  });

  it("its worlds add up to one and name the assignment they are in", () => {
    const b = run([alphaIn]);
    const all = topWorlds(b, 100000);
    expect(all.reduce((s, w) => add(s, w.p), p(0n))).toEqual(ONE);
    expect(new Set(all.map((w) => w.assignment))).toEqual(new Set([0, 1]));
    for (let k = 1; k < all.length; k += 1) expect(all[k]!.p.num * all[k - 1]!.p.den).toBeLessThanOrEqual(all[k - 1]!.p.num * all[k]!.p.den);
  });

  it("a reader flag still refuses", () => {
    const flagged = seen([alphaIn, "|-nonsense|p2a: X"]);
    expect(refusal(() => run([], { opponent: flagged.opponent, flags: flagged.flags }))).toBe("reader_flagged");
  });

  it("with the Item Clause absent the assignments still mix exactly", () => {
    const open = { ...vocabulary(), rules: { adjust_level: 50, min_team_size: 6, max_team_size: 6, picked_team_size: 4 } };
    const b = run([alphaIn], { vocabulary: open });
    expect(b.assignments.map((a) => a.p)).toEqual([p(1n, 2n), p(1n, 2n)]);
  });
});
