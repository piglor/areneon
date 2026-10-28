import { describe, expect, it } from "vitest";
import { BeliefError, belief, trapExclusions, TrappingCheckError } from "../src/index.ts";
import type { BeliefInput, OpponentBelief, TrapCandidate, TrapLevel, TrappingCheck, TrapSlot } from "../src/index.ts";
import { ratio } from "../src/ratio.ts";
import { hidingConfig, input, text } from "./support.ts";

// A stand-in for the bridge's engine check: `alpha` with ability a1 and `gamma` with g1 would show maybeTrapped,
// every other candidate shows nothing. The bridge's own test runs the real engine.
const calls: TrapCandidate[] = [];
const check = (observed: readonly TrapLevel[], shows: (c: TrapCandidate) => readonly TrapLevel[] = (c) => [(c.species === "alpha" && c.ability === "a1") || (c.species === "gamma" && c.ability === "g1") ? 1 : 0]): TrappingCheck => ({
  observed,
  shows: (c) => {
    calls.push(c);
    return shows(c);
  },
});

const SWITCH = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;
const A = SWITCH("a", "Alp", "Alpha");
const G = SWITCH("b", "Gam", "Gamma");
const withTrap = (more: readonly string[], trapping: TrappingCheck | null | undefined, change: Partial<BeliefInput> = {}) => belief(input(more, { trapping, ...change }));
const abilities = (b: OpponentBelief, species: string) => b.slots.find((s) => s.species === species)!.abilities;
const refusal = (run: () => unknown) => {
  try {
    run();
  } catch (e) {
    if (e instanceof BeliefError) return e.code;
    throw e;
  }
  throw new Error("expected a BeliefError");
};

describe("the own request's trapping flags as hard constraints (ticket #1127)", () => {
  it("no flags, an absent port or an empty observation leave the belief exactly as it was", () => {
    const base = text(belief(input([A])).slots);
    for (const trapping of [undefined, null, check([])]) expect(text(withTrap([A], trapping).slots)).toBe(base);
    expect(withTrap([A], check([])).constraints).toEqual(belief(input([A])).constraints);
  });

  it("upper bound: an unseen flag excludes the abilities that would have shown one, and scales the forme's weight by the share left", () => {
    const b = withTrap([A], check([0]));
    expect(abilities(b, "alpha")).toEqual([{ id: "a2", p: ratio(1n, 1n) }]);
    expect(b.slots.find((s) => s.species === "alpha")!.constraints).toContain("trapping_excluded:a1");
    expect(b.constraints).toContain("trapping:upper");
    // the others are untouched
    expect(text(b.slots.find((s) => s.species === "delta"))).toBe(text(belief(input([A])).slots.find((s) => s.species === "delta")));
  });

  it("lower bound: a seen maybeTrapped with exactly one foe that could show it requires that ability", () => {
    const b = withTrap([A], check([1]));
    expect(abilities(b, "alpha")).toEqual([{ id: "a1", p: ratio(1n, 1n) }]);
    expect(b.slots.find((s) => s.species === "alpha")!.constraints).toContain("trapping_required:a2");
    expect(b.constraints).toContain("trapping:lower");
  });

  it("with two foes that could show it the disjunction is left out (the belief stays wider)", () => {
    const b = withTrap([A, G], check([1]));
    expect(abilities(b, "alpha")).toEqual(belief(input([A, G])).slots.find((s) => s.species === "alpha")!.abilities);
    expect(b.constraints).not.toContain("trapping:lower");
  });

  it("with no foe that could show it the flag has another cause and adds nothing", () => {
    const b = withTrap([A], check([1], () => [0]));
    expect(text(b.slots)).toBe(text(belief(input([A])).slots));
  });

  it("a foe whose ability is not its species' own is no evidence and blocks the lower bound", () => {
    const received = [A, G, "|-ability|p2b: Gam|G1|[from] ability: Trace"];
    const b = withTrap(received, check([1]));
    expect(b.constraints).not.toContain("trapping:lower");
    // the upper bound still holds for the foe whose ability is known
    expect(abilities(withTrap(received, check([0])), "alpha")).toEqual([{ id: "a2", p: ratio(1n, 1n) }]);
  });

  it("evidence the flags contradict leaves no world", () => {
    expect(refusal(() => withTrap([A, "|-ability|p2a: Alp|A1"], check([0])))).toBe("evidence_has_no_support");
    expect(refusal(() => withTrap([A, "|-ability|p2a: Alp|A2"], check([1])))).toBe("evidence_has_no_support");
    // a revealed ability the flag allows is kept
    expect(abilities(withTrap([A, "|-ability|p2a: Alp|A1"], check([1])), "alpha")).toEqual([{ id: "a1", p: ratio(1n, 1n) }]);
  });

  it("a trapped: true level (a working trap on an earlier active) is excluded by a lower observation and required by a higher one", () => {
    const hidden = (c: TrapCandidate): readonly TrapLevel[] => [c.species === "alpha" && c.ability === "a1" ? 2 : 0];
    for (const observed of [0, 1] as const) expect(abilities(withTrap([A], check([observed], hidden)), "alpha")).toEqual([{ id: "a2", p: ratio(1n, 1n) }]);
    expect(abilities(withTrap([A], check([2], hidden)), "alpha")).toEqual([{ id: "a1", p: ratio(1n, 1n) }]);
    expect(abilities(withTrap([A], check([2], () => [0])), "alpha")).toEqual(belief(input([A])).slots.find((s) => s.species === "alpha")!.abilities);
  });

  it("only Pokemon in play are asked, as the species they are shown as", () => {
    calls.length = 0;
    withTrap([A], check([0]));
    expect(calls.map((c) => `${c.species}:${c.ability}`)).toEqual(["alpha:a1", "alpha:a2"]);
    calls.length = 0;
    withTrap([A, "|faint|p2a: Alp"], check([0]));
    expect(calls).toEqual([]);
  });

  it("a Mega Evolved Pokemon is asked as its Mega forme with the stone; before the Evolution as the preview species", () => {
    const beta = SWITCH("a", "Bet", "Beta");
    calls.length = 0;
    withTrap([beta], check([0], () => [0]));
    expect(calls.map((c) => [c.species, c.ability, c.item])).toEqual([["beta", "b1", null], ["beta", "b2", null]]);
    calls.length = 0;
    withTrap([beta, "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"], check([0], () => [0]));
    expect(calls.map((c) => [c.species, c.ability, c.item])).toEqual([["betamega", "bmega", "betite"]]);
  });

  it("under a possible disguise the check is not used", () => {
    calls.length = 0;
    const b = belief(input([A], { config: hidingConfig, trapping: check([0]) }));
    expect(calls).toEqual([]);
    expect(b.disguise.positions).toEqual([0]);
    expect(b.constraints.some((c) => c.startsWith("trapping"))).toBe(false);
  });

  it("an answer of another shape than the observation is a defect", () => {
    expect(() => withTrap([A], check([0], () => [0, 0]))).toThrow(TrappingCheckError);
    expect(() => withTrap([A], check([0], () => [3 as TrapLevel]))).toThrow(TrappingCheckError);
    expect(() => withTrap([A], check([3 as TrapLevel]))).toThrow(TrappingCheckError);
  });

  it("slots the vocabulary cannot place are skipped and block the lower bound", () => {
    const slot = (over: Partial<TrapSlot>): TrapSlot => ({ index: 0, active: true, species: "alpha", eligible: true, abilities: ["a1", "a2"], item: null, ...over });
    const c = check([1]);
    expect(trapExclusions(c, [slot({}), slot({ index: 1, abilities: null })]).constraints).toEqual(["trapping:upper"]);
    expect(trapExclusions(c, [slot({ species: null })]).excluded.size).toBe(0);
    expect(trapExclusions(c, [slot({ active: false })]).excluded.size).toBe(0);
    expect(trapExclusions(c, [slot({})]).excluded.get(0)).toEqual(new Set(["a2"]));
  });
});
