// The N6 cases of ADR-025 (review finding, slice 2 part 4): Trick and Switcheroo, Knock Off, berry consumption and Mega
// Evolution, as hand-written channels on the hand-made vocabulary. In each the TRUE set (written in the test, never given
// to the belief) stays possible at every step, and an item received from another Pokemon adds item evidence only where the
// reader records the transfer: a received item is flagged by the reader and the belief reads no item from it.

import { describe, expect, it } from "vitest";
import { belief } from "../src/index.ts";
import type { OpponentBelief } from "../src/index.ts";
import { input, text } from "./support.ts";

const SWITCH = "|switch|p2a: Alp|Alpha, L50, M|100/100";
const items = (b: OpponentBelief, species = "alpha") => b.slots.find((s) => s.species === species)!.items;
const possible = (b: OpponentBelief, species: string, item: string | null) => items(b, species).some((m) => m.id === item && m.p.num > 0n);
const at = (lines: readonly string[]) => belief(input([SWITCH, ...lines]));
const bare = belief(input([SWITCH]));

describe("Trick and Switcheroo (item swaps)", () => {
  // The true alpha holds ix. Trick gives it iy, the item of the Pokemon it swapped with.
  const TRICK = "|-item|p2a: Alp|Iy|[from] move: Trick";
  const SWITCHEROO = "|-item|p2a: Alp|Iy|[from] move: Switcheroo";

  it("the received item is no evidence about alpha's own: the belief is the one without the line, and the true item stays possible", () => {
    for (const line of [TRICK, SWITCHEROO]) {
      const b = at([line]);
      expect(possible(b, "alpha", "ix")).toBe(true);
      expect(text(b.slots)).toBe(text(bare.slots));
    }
  });

  it("a later reveal on the same Pokemon is still the received item's: the taint is sticky, so nothing is concluded", () => {
    const b = at([TRICK, "|-damage|p2a: Alp|90/100|[from] item: Iy"]);
    expect(possible(b, "alpha", "ix")).toBe(true);
  });

  it("an item shown before the swap is evidence and stays the Pokemon's own while no swap is recorded", () => {
    const b = at(["|-damage|p2a: Alp|90/100|[from] item: Ix"]);
    expect(items(b)).toEqual([{ id: "ix", p: { num: 1n, den: 1n } }]);
  });
});

describe("Knock Off and berry consumption (item loss)", () => {
  it("a knocked-off item leaves the reader with no item: the belief is as wide as without the line, and the true item stays possible", () => {
    const b = at(["|-enditem|p2a: Alp|Ix|[from] move: Knock Off|[of] p1a: Foe"]);
    expect(possible(b, "alpha", "ix")).toBe(true);
    expect(possible(b, "alpha", "iy")).toBe(true);
    expect(text(b.slots)).toBe(text(bare.slots));
  });

  it("a berry eaten is gone from the reader too; an item revealed first and then consumed keeps the true set possible", () => {
    const b = at(["|-heal|p2a: Alp|100/100|[from] item: Iy", "|-enditem|p2a: Alp|Iy|[eat]"]);
    expect(possible(b, "alpha", "iy")).toBe(true);
    expect(possible(b, "alpha", "ix")).toBe(true);
  });

  it("an item lost is still never no evidence against the others: gamma, not in play, keeps its own list", () => {
    const b = at(["|-enditem|p2a: Alp|Ix|[from] move: Knock Off|[of] p1a: Foe"]);
    expect(items(b, "gamma")).toEqual(items(bare, "gamma"));
  });
});

describe("Mega Evolution (B1) keeps the true set possible at each stage", () => {
  const BET = "|switch|p2a: Bet|Beta, L50, M|100/100";
  const stages = [
    ["Team Preview", []],
    ["after the switch-in", [BET]],
    ["after the Evolution", [BET, "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]],
  ] as const;

  it.each(stages)("%s: the true holder of the stone (as the Mega forme) and its ability have a probability above zero", (_name, lines) => {
    const b = belief(input(lines));
    const beta = b.slots.find((s) => s.species === "beta")!;
    expect(beta.formes.some((f) => f.id === "betamega" && f.p.num > 0n)).toBe(true);
    expect(beta.items.some((m) => m.id === "betite" && m.p.num > 0n)).toBe(true);
    // before the Evolution the ability is the base species'; after it, the Mega forme's
    const abilities = beta.abilities.map((m) => m.id);
    expect(abilities.some((a) => a === (lines.length === 3 ? "bmega" : "b1"))).toBe(true);
  });
});
