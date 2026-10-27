import { describe, expect, it } from "vitest";
import { BELIEF_FORMAT, BeliefError, type BeliefErrorCode, belief, topWorlds } from "../src/index.ts";
import type { Marginal, OpponentBelief } from "../src/index.ts";
import { ratio } from "../src/ratio.ts";
import { alpha, betaFormes, gamma, hidingConfig, inert, input, prior, seen, text, vocabulary } from "./support.ts";

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

// Switch-ins of the opponent's Pokemon, as the opponent's lines of player p1's own channel.
const SWITCH = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;

describe("the belief over an unseen team (Team Preview only)", () => {
  const b = belief(input());

  it("is tagged with the format and carries the worked-out Item Clause normaliser", () => {
    expect(b.format).toBe(BELIEF_FORMAT);
    // alpha 5/2/3 (ix, iy, none), gamma 1/2/1, beta 3 none + 1 stone, three inert slots: (10*4 - 5*1 - 2*2) * 4 = 31 * 4.
    expect(b.itemClause).toEqual({ enforced: true, normaliser: 124n });
    expect(b.constraints).toEqual(["item_clause:1", "brought:picked=4"]);
  });

  it("Item Clause moves the item marginals of the two slots that share items", () => {
    // Without the rule alpha would hold ix half the time; the rule takes the worlds in which gamma also holds it.
    expect(slot(b, "alpha").items).toEqual([{ id: null, p: p(12n, 31n) }, { id: "ix", p: p(15n, 31n) }, { id: "iy", p: p(4n, 31n) }]);
    expect(slot(b, "gamma").items).toEqual([{ id: null, p: p(10n, 31n) }, { id: "ix", p: p(5n, 31n) }, { id: "iy", p: p(16n, 31n) }]);
  });

  it("the fields given a forme: abilities with the residual expanded, moves by the conditional Poisson design", () => {
    const a = slot(b, "alpha");
    expect(a.abilities).toEqual([{ id: "a1", p: p(3n, 4n) }, { id: "a2", p: p(1n, 4n) }]);
    expect(a.moves).toEqual([{ id: "ma", p: p(1n) }, { id: "mb", p: p(1n, 2n) }, { id: "mc", p: p(1n, 2n) }, { id: "md", p: p(1n, 2n) }]);
    expect(a.mode).toEqual({ forme: "alpha", ability: "a1", item: "ix" });
    expect(a.formes[0]?.moves.outcomes.map((o) => o.size)).toEqual([2, 3]);
  });

  it("brought four: every previewed Pokemon is in the four with probability 4/6, over C(6,4) subsets", () => {
    expect(b.brought).toEqual({ picked: 4, settled: [], subsets: 15n });
    for (const s of b.slots) expect(s.p_brought).toEqual(p(2n, 3n));
  });

  it("inert slots hold no item and carry their whole pool", () => {
    const d = slot(b, "delta");
    expect(d.items).toEqual([{ id: null, p: p(1n) }]);
    expect(d.moves.map((m) => m.id)).toEqual(["dm1", "dm2", "dm3", "dm4"]);
    expect(d.spreads).toEqual([{ id: "hardy:0-0-0-0-0-0", p: p(1n) }]);
  });

  it("is a pure function: the same input gives the same belief, and the input is not changed", () => {
    const frozen = input();
    const before = text(frozen);
    expect(text(belief(frozen))).toBe(text(b));
    expect(text(frozen)).toBe(before);
  });

  it("without an Item Clause in the rule table the slots are independent", () => {
    const rules = vocabulary();
    const open = belief(input([], { vocabulary: { ...rules, rules: { min_team_size: 6, max_team_size: 6, picked_team_size: 4 } } }));
    expect(open.itemClause.enforced).toBe(false);
    expect(slot(open, "alpha").items).toEqual([{ id: null, p: p(3n, 10n) }, { id: "ix", p: p(1n, 2n) }, { id: "iy", p: p(1n, 5n) }]);
    expect(open.constraints[0]).toBe("item_clause:absent");
  });
});

describe("hard constraints from what the reader shows", () => {
  it("a revealed item is removed from every other slot, and a switch-in settles the brought set", () => {
    const b = belief(input([SWITCH("a", "Gam", "Gamma"), "|-item|p2a: Gam|Ix"]));
    expect(slot(b, "gamma").items).toEqual([{ id: "ix", p: p(1n) }]);
    expect(slot(b, "alpha").items).toEqual([{ id: null, p: p(3n, 5n) }, { id: "iy", p: p(2n, 5n) }]);
    expect(b.brought).toEqual({ picked: 4, settled: [2], subsets: 10n });
    expect(slot(b, "gamma").p_brought).toEqual(p(1n));
    expect(slot(b, "alpha").p_brought).toEqual(p(3n, 5n));
    expect(slot(b, "gamma").constraints).toEqual(["species:gamma", "forme:gamma", "level:50", "gender:M", "settled", "item:ix"]);
  });

  it("a chosen move is in the set, and a move called by another effect is not evidence", () => {
    const base = [SWITCH("a", "Alp", "Alpha")];
    const b = belief(input([...base, "|move|p2a: Alp|Mb"]));
    const a = slot(b, "alpha");
    expect(get(a.moves, "mb")).toEqual(p(1n));
    // size 2 -> 1/3 of the other two, size 3 -> 1/2 each, weighted 1/3 and 2/3.
    expect(get(a.moves, "mc")).toEqual(p(1n, 3n));
    expect(a.constraints).toContain("move:mb");
    const called = belief(input([...base, "|move|p2a: Alp|Mb|[from]Copycat"]));
    expect(text(slot(called, "alpha").moves)).toBe(text(slot(belief(input(base)), "alpha").moves));
  });

  it("a revealed ability outside the truncated list never empties the belief", () => {
    const b = belief(input([SWITCH("a", "Alp", "Alpha"), "|-ability|p2a: Alp|A2"]));
    expect(slot(b, "alpha").abilities).toEqual([{ id: "a2", p: p(1n) }]);
    // the item and the moves are untouched by the ability
    expect(get(slot(b, "alpha").items, "ix")).toEqual(p(15n, 31n));
  });

  it("a revealed item outside the truncated list never empties the belief", () => {
    // gamma's list names ix and iy only; betite takes the residual 1 of 5.
    const wide = prior((pr) => {
      pr.species[4] = { id: "gamma", denominator: 1, formes: [gamma((f) => (f.items = { denominator: 5, entries: [{ id: "ix", weight: 1 }, { id: "iy", weight: 2 }], no_item: 1, residual: 1, unmapped: 0 }))] };
    });
    const b = belief(input([SWITCH("a", "Gam", "Gamma"), "|-item|p2a: Gam|Betite"], { prior: wide }));
    expect(slot(b, "gamma").items).toEqual([{ id: "betite", p: p(1n) }]);
    // the stone is taken: the Mega forme of beta cannot hold it
    expect(slot(b, "beta").formes.map((f) => [f.id, f.p])).toEqual([["beta", p(1n)]]);
    // without a residual the same fact has no support at all
    expect(refusal(() => belief(input([SWITCH("a", "Gam", "Gamma"), "|-item|p2a: Gam|Betite"])))).toBe("evidence_has_no_support");
  });

  it("an item or ability the vocabulary does not know adds no constraint", () => {
    const base = belief(input([SWITCH("a", "Alp", "Alpha")]));
    const b = belief(input([SWITCH("a", "Alp", "Alpha"), "|-item|p2a: Alp|Unknownium", "|-ability|p2a: Alp|Mystery", "|move|p2a: Alp|Struggle"]));
    expect(text(slot(b, "alpha").items)).toBe(text(slot(base, "alpha").items));
    expect(text(slot(b, "alpha").moves)).toBe(text(slot(base, "alpha").moves));
    expect(slot(b, "alpha").constraints).toEqual(expect.arrayContaining(["item_skipped:unknownium", "ability_skipped:mystery", "move_skipped:struggle"]));
  });

  it("a Pokemon under Transform shows another's moves, so they add no constraint", () => {
    const b = belief(input([SWITCH("a", "Alp", "Alpha"), "|-transform|p2a: Alp|p2a: Alp", "|move|p2a: Alp|Mb"]));
    expect(slot(b, "alpha").constraints).toContain("moves_skipped:transform");
    expect(slot(b, "alpha").constraints).not.toContain("move:mb");
  });

  it("two settled holders of one item have no world", () => {
    const lines = [SWITCH("a", "Alp", "Alpha"), SWITCH("b", "Gam", "Gamma"), "|-item|p2a: Alp|Ix", "|-item|p2b: Gam|Ix"];
    expect(refusal(() => belief(input(lines)))).toBe("evidence_has_no_support");
  });

  it("more Pokemon in play than the picked size is refused", () => {
    const names = ["alpha", "beta", "gamma", "delta", "zeta"] as const;
    const lines = names.map((s, k) => (k < 2 ? SWITCH(k === 0 ? "a" : "b", s, s[0]!.toUpperCase() + s.slice(1)) : `|switch|p2${k % 2 === 0 ? "a" : "b"}: N${k}|${s[0]!.toUpperCase() + s.slice(1)}, L50, M|100/100`));
    expect(refusal(() => belief(input(lines)))).toBe("brought_exceeds_picked");
  });
});

describe("the Mega test (B1)", () => {
  const stoneHolder = (more: readonly string[] = [], change: Parameters<typeof prior>[0] = () => undefined) => belief(input(more, { prior: prior(change) }));

  it("at Team Preview a stone holder is a base-species ability list and the stone counts", () => {
    const b = stoneHolder();
    const beta = slot(b, "beta");
    expect(beta.formes.map((f) => [f.id, f.mega, f.p])).toEqual([["beta", false, p(3n, 4n)], ["betamega", true, p(1n, 4n)]]);
    // Only the base species' abilities: the Mega ability is not an option before the Evolution.
    expect(beta.abilities).toEqual([{ id: "b1", p: p(3n, 4n) }, { id: "b2", p: p(1n, 4n) }]);
    expect(beta.items).toEqual([{ id: null, p: p(3n, 4n) }, { id: "betite", p: p(1n, 4n) }]);
    // The stone counts for Item Clause: when another slot is known to hold it, the Mega forme is out.
    const wide = prior((pr) => {
      pr.species[4] = { id: "gamma", denominator: 1, formes: [gamma((f) => (f.items = { denominator: 5, entries: [{ id: "ix", weight: 1 }, { id: "iy", weight: 2 }], no_item: 1, residual: 1, unmapped: 0 }))] };
    });
    const taken = belief(input([SWITCH("a", "Gam", "Gamma"), "|-item|p2a: Gam|Betite"], { prior: wide }));
    expect(slot(taken, "beta").formes.map((f) => f.id)).toEqual(["beta"]);
    // ... and the stone share falls where it competes: alpha-gamma 41 ways with gamma free, 31 without the stone, beta none 3 and stone 1: 31 / (3 * 41 + 31).
    const competing = belief(input([], { prior: wide }));
    expect(get(slot(competing, "beta").items, "betite")).toEqual(p(31n, 154n));
  });

  it("after the switch-in the shown forme is still the preview species and the Mega ability is not an option", () => {
    const b = stoneHolder([SWITCH("a", "Bet", "Beta")]);
    expect(slot(b, "beta").formes.map((f) => f.id)).toEqual(["beta", "betamega"]);
    expect(slot(b, "beta").abilities.map((a) => a.id)).toEqual(["b1", "b2"]);
  });

  it("a revealed pre-Evolution ability fits both formes and leaves their odds alone", () => {
    const b = stoneHolder([SWITCH("a", "Bet", "Beta"), "|-ability|p2a: Bet|B2"]);
    expect(slot(b, "beta").abilities).toEqual([{ id: "b2", p: p(1n) }]);
    expect(slot(b, "beta").formes.map((f) => [f.id, f.p])).toEqual([["beta", p(3n, 4n)], ["betamega", p(1n, 4n)]]);
  });

  it("after the Evolution the forme and stone are fixed and the Mega ability applies", () => {
    const evolve = [SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"];
    const b = stoneHolder(evolve);
    const beta = slot(b, "beta");
    expect(beta.formes.map((f) => [f.id, f.p])).toEqual([["betamega", p(1n)]]);
    expect(beta.abilities).toEqual([{ id: "bmega", p: p(1n) }]);
    expect(beta.items).toEqual([{ id: "betite", p: p(1n) }]);
    expect(beta.constraints).toEqual(expect.arrayContaining(["forme:betamega", "mega:betite"]));
    // An ability shown before the Evolution is still the reader's last ability: it says nothing of the Mega ability.
    const earlier = stoneHolder([SWITCH("a", "Bet", "Beta"), "|-ability|p2a: Bet|B2", ...evolve.slice(1)]);
    expect(slot(earlier, "beta").abilities).toEqual([{ id: "bmega", p: p(1n) }]);
    expect(slot(earlier, "beta").constraints).toContain("ability_ambiguous:b2");
    // An ability shown after it is the Mega ability.
    const later = stoneHolder([...evolve, "|-ability|p2a: Bet|Bmega"]);
    expect(slot(later, "beta").constraints).toContain("ability:bmega");
  });

  it("a Mega Evolution with another stone, or into a forme the prior lacks, is refused", () => {
    expect(refusal(() => stoneHolder([SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Gammite"]))).toBe("forme_not_in_prior");
    expect(refusal(() => stoneHolder([SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"], (pr) => (pr.species[1] = { id: "beta", denominator: 1, formes: [betaFormes()[0]] })))).toBe("forme_not_in_prior");
  });

  it("a revealed stone on an unevolved holder (Frisk) makes it the Mega forme", () => {
    const wide = prior((pr) => {
      pr.species[1] = { id: "beta", denominator: 4, formes: betaFormes() };
    });
    const b = belief(input([SWITCH("a", "Bet", "Beta"), "|-item|p2a: Bet|Betite"], { prior: wide }));
    expect(slot(b, "beta").formes.map((f) => [f.id, f.p])).toEqual([["betamega", p(1n)]]);
  });
});

describe("refusals (typed, counted by the consumer)", () => {
  it("a reader-flagged line, or a phantom entry", () => {
    const flagged = seen(["|-nonsense|p2a: X"]);
    expect(flagged.flags).toHaveLength(1);
    expect(refusal(() => belief(input([], { flags: flagged.flags })))).toBe("reader_flagged");
    const s = seen();
    const phantom = { ...s.opponent, roster: [{ ...s.opponent.roster[0]!, phantom: true }, ...s.opponent.roster.slice(1)] };
    expect(refusal(() => belief(input([], { opponent: phantom })))).toBe("reader_flagged");
  });

  it("a species that can hide its identity, until the Illusion-aware belief exists", () => {
    // The configured ability is on epsilon's roster entry; without it the same team has a belief.
    expect(belief(input()).slots).toHaveLength(6);
    expect(refusal(() => belief(input([], { config: hidingConfig })))).toBe("identity_hiding_species");
  });

  it("another format, an Item Clause the filter does not know, a team of another size, a species without usage data", () => {
    expect(refusal(() => belief(input([], { vocabulary: { ...vocabulary(), format_id: "other" } })))).toBe("format_mismatch");
    expect(refusal(() => belief(input([], { vocabulary: vocabulary({ item_clause: 2 }) })))).toBe("item_clause_unsupported");
    expect(refusal(() => belief(input([], { vocabulary: vocabulary({ picked_team_size: 7 }) })))).toBe("team_size");
    expect(refusal(() => belief(input([], { vocabulary: vocabulary({ picked_team_size: 0 }) })))).toBe("team_size");
    expect(refusal(() => belief(input([], { vocabulary: vocabulary({ min_team_size: 7, max_team_size: 8 }) })))).toBe("team_size");
    expect(refusal(() => belief(input([], { prior: prior((pr) => (pr.species = pr.species.filter((s) => s.id !== "zeta") as typeof pr.species)) })))).toBe("species_not_in_prior");
  });

  it("a forme change in battle the prior has no entry for adds no forme constraint", () => {
    // The preview hides the forme (Alpha-*); the switch-in shows one the prior has no entry for.
    const s = seen([`|switch|p2a: Alp|Alpha-X, L50, M|100/100`], { species: { alphax: { base: "alpha", types: [] } } }, ["alpha"]);
    const b = belief(input([], { opponent: s.opponent, flags: s.flags }));
    expect(slot(b, "alpha").formes.map((f) => f.id)).toEqual(["alpha"]);
    expect(slot(b, "alpha").constraints).toContain("forme_unmatched:alphax");
  });

  it("finds the prior entry by the shown forme when the reader's base species is not a preview species", () => {
    const s = seen([], { species: { delta: { base: "deltabase", types: [] } } });
    expect(s.opponent.roster[3]?.base_species).toBe("deltabase");
    expect(slot(belief(input([], { opponent: s.opponent, flags: s.flags })), "delta").formes.map((f) => f.id)).toEqual(["delta"]);
  });

  it("a Mega Evolution without a stone name in the reader's state is explained by any Mega forme of the shown forme", () => {
    const s = seen([SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]);
    const bare = { ...s.opponent, roster: s.opponent.roster.map((m) => (m.mega ? { ...m, mega_stone: null } : m)) };
    const b = belief(input([], { opponent: bare, flags: s.flags }));
    expect(slot(b, "beta").constraints).toContain("mega:unknown");
    expect(slot(b, "beta").formes.map((f) => f.id)).toEqual(["betamega"]);
  });

  it("a prior list with no mass at all", () => {
    const hollow = prior((pr) => {
      pr.species[0] = { id: "alpha", denominator: 1, formes: [alpha((f) => (f.items = { denominator: 4, entries: [], no_item: 0, residual: 0, unmapped: 4 }))] };
    });
    expect(refusal(() => belief(input([], { prior: hollow })))).toBe("empty_distribution");
  });

  it("a forme the usage prior lists but the vocabulary lacks", () => {
    const stray = prior((pr) => {
      pr.species[2] = { id: "delta", denominator: 1, formes: [inert("deltax", "d1", "dm")] };
    });
    expect(refusal(() => belief(input([], { prior: stray })))).toBe("forme_not_in_prior");
  });

  it("carries its code and the roster position", () => {
    try {
      belief(input([], { prior: prior((pr) => (pr.species = pr.species.filter((s) => s.id !== "gamma") as typeof pr.species)) }));
    } catch (e) {
      expect(e).toBeInstanceOf(BeliefError);
      expect((e as BeliefError).slot).toBe(2);
      expect((e as BeliefError).name).toBe("BeliefError");
    }
  });
});

describe("top-K joint worlds (diagnostic)", () => {
  it("lists the most likely (forme, item) combinations best first with exact probabilities", () => {
    const b = belief(input());
    const top = topWorlds(b, 3);
    expect(top).toHaveLength(3);
    // Best: alpha ix (5), gamma iy (2), beta forme none (3), inert 1 each: 30 / 124.
    expect(top[0]!.p).toEqual(p(30n, 124n));
    expect(top[0]!.slots.map((s) => s.item)).toEqual(["ix", null, "iy", null, null, null]);
    for (let k = 1; k < top.length; k += 1) expect(top[k]!.p.num * top[k - 1]!.p.den).toBeLessThanOrEqual(top[k - 1]!.p.num * top[k]!.p.den);
    // The worlds never repeat an item, and every probability is at most the belief's total of 1.
    const all = topWorlds(b, 1000);
    const total = all.reduce((s, w) => s + Number(w.p.num) / Number(w.p.den), 0);
    expect(total).toBeCloseTo(1, 12);
    expect(all.every((w) => new Set(w.slots.filter((s) => s.item !== null).map((s) => s.item)).size === w.slots.filter((s) => s.item !== null).length)).toBe(true);
  });

  it("without the clause equal items may repeat", () => {
    const open = belief(input([], { vocabulary: { ...vocabulary(), rules: { min_team_size: 6, max_team_size: 6, picked_team_size: 4 } } }));
    expect(topWorlds(open, 1)[0]!.slots.map((s) => s.item)).toEqual(["ix", null, "iy", null, null, null]);
    expect(topWorlds(open, 0)).toEqual([]);
  });
});
