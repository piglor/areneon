import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { BELIEF_ERROR_CODES, BeliefError, belief, OPPONENT_SHEET_FORMAT, opponentSheet, refusedSheet, roundPpm, SHEET_K, SHEET_REFUSAL_CODES, sheetFeatureSetHash, sheetRefusalOf } from "../src/index.ts";
import { digest } from "../src/canonical.ts";
import { ratio } from "../src/ratio.ts";
import { hidingConfig, input, seen } from "./support.ts";

const HASH = "a".repeat(64);
/** SHA-256 of the canonical JSON of the block of each case (computed by this code once, reviewed, then pinned). */
const GOLDEN: Record<string, string> = {
  preview: "d5606086f63163dfbddbd522bd3919acda7b8577f1af1623dff5b8c5277e2284",
  revealed: "debbf85b16fb3f8a33a1b3737c899e525160a0b9f19985b94d0e1f011e24197e",
  mega: "0c40da77533e07b62f3569b32b53505b0a882254f287f31297b8c8943e08c50f",
  disguise: "f9bc8390592d357226f8f27d85f89072571e4ddb6faffc2a4c0b2b51f581b2a9",
};
const SWITCH = (letter: "a" | "b", name: string, species: string) => `|switch|p2${letter}: ${name}|${species}, L50, M|100/100`;

/** The block of a channel, from the belief and the reader's opponent half the belief was computed from. */
function sheetOf(more: readonly string[] = [], change: Parameters<typeof input>[1] = {}) {
  const i = input(more, change);
  return opponentSheet(belief(i), i.opponent, HASH);
}
const slot = (block: ReturnType<typeof sheetOf>, species: string) => block.slots.find((s) => s.species === species)!;

describe("rounding: half up of the exact rational, each entry alone", () => {
  it("rounds the exact fraction", () => {
    expect(roundPpm(ratio(1n, 3n))).toBe(333333);
    expect(roundPpm(ratio(2n, 3n))).toBe(666667);
    expect(roundPpm(ratio(1n, 2000000n))).toBe(1); // exactly 0.5 ppm: half up
    expect(roundPpm(ratio(1n, 4000000n))).toBe(0);
    expect(roundPpm(ratio(1n))).toBe(1_000_000);
    expect(roundPpm(ratio(0n))).toBe(0);
  });
});

describe("the block of an unseen team (Team Preview only)", () => {
  const block = sheetOf();

  it("is a belief block with the feature set hash and one slot per preview entry, in preview order", () => {
    expect(block.source).toEqual({ kind: "belief", hash: HASH });
    expect(block.slots.map((s) => s.species)).toEqual(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]);
  });

  it("brought_ppm is 4/6 rounded half up, predicted", () => {
    for (const s of block.slots) expect(s).toMatchObject({ brought_ppm: 666667, brought_source: "predicted" });
  });

  it("items are the Item-Clause-aware marginals (worked out in belief.test.ts), sorted by ppm then id, null last among equals", () => {
    expect(slot(block, "alpha").item).toEqual({ source: "predicted", entries: [{ id: "ix", ppm: 483871 }, { id: null, ppm: 387097 }, { id: "iy", ppm: 129032 }], other_ppm: 0 });
    expect(slot(block, "gamma").item).toEqual({ source: "predicted", entries: [{ id: "iy", ppm: 516129 }, { id: null, ppm: 322581 }, { id: "ix", ppm: 161290 }], other_ppm: 0 });
    expect(slot(block, "delta").item).toEqual({ source: "predicted", entries: [{ id: null, ppm: 1_000_000 }], other_ppm: 0 });
  });

  it("abilities and the nature from the spread marginals; moves as inclusions, slots_known 0", () => {
    const a = slot(block, "alpha");
    expect(a.ability).toEqual({ source: "predicted", entries: [{ id: "a1", ppm: 750000 }, { id: "a2", ppm: 250000 }], other_ppm: 0 });
    expect(a.nature).toEqual({ source: "predicted", entries: [{ id: "hardy", ppm: 1_000_000 }], other_ppm: 0 });
    expect(a.moves).toEqual({ source: "predicted", entries: [{ id: "ma", inclusion_ppm: 1_000_000 }, { id: "mb", inclusion_ppm: 500000 }, { id: "mc", inclusion_ppm: 500000 }, { id: "md", inclusion_ppm: 500000 }], slots_known: 0 });
  });

  it("is a pure function: the same input gives the same block", () => {
    expect(JSON.stringify(sheetOf())).toBe(JSON.stringify(block));
  });
});

describe("revealed facts are 1,000,000 ppm with source revealed", () => {
  it("a shown item, a shown ability, a brought Pokemon", () => {
    const gamma = slot(sheetOf([SWITCH("a", "Gam", "Gamma"), "|-item|p2a: Gam|Ix"]), "gamma");
    expect(gamma).toMatchObject({ brought_ppm: 1_000_000, brought_source: "revealed", item: { source: "revealed", entries: [{ id: "ix", ppm: 1_000_000 }], other_ppm: 0 } });
    const alpha = slot(sheetOf([SWITCH("a", "Alp", "Alpha"), "|-ability|p2a: Alp|A2"]), "alpha");
    expect(alpha.ability).toEqual({ source: "revealed", entries: [{ id: "a2", ppm: 1_000_000 }], other_ppm: 0 });
  });

  it("a revealed move is counted in slots_known and held at 1,000,000; a move called by another effect is not", () => {
    const moves = slot(sheetOf([SWITCH("a", "Alp", "Alpha"), "|move|p2a: Alp|Mb"]), "alpha").moves;
    expect(moves.slots_known).toBe(1);
    expect(moves.entries.slice(0, 2)).toEqual([{ id: "ma", inclusion_ppm: 1_000_000 }, { id: "mb", inclusion_ppm: 1_000_000 }]);
    expect(slot(sheetOf([SWITCH("a", "Alp", "Alpha"), "|move|p2a: Alp|Mb|[from]Copycat"]), "alpha").moves.slots_known).toBe(0);
  });

  it("an evolved Mega holds its stone as a revealed item", () => {
    const more = [SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"];
    expect(slot(sheetOf(more), "beta").item).toEqual({ source: "revealed", entries: [{ id: "betite", ppm: 1_000_000 }], other_ppm: 0 });
  });
});

describe("the cut to k, the residual and the nature that no named spread backs", () => {
  it("keeps the top k only and puts the rest in other_ppm (max(0, 10^6 - sum))", () => {
    const i = input();
    const cut = opponentSheet(belief(i), i.opponent, HASH, { item: 2, ability: 1, nature: 3, moves: 2 });
    const a = slot(cut, "alpha");
    expect(a.item).toEqual({ source: "predicted", entries: [{ id: "ix", ppm: 483871 }, { id: null, ppm: 387097 }], other_ppm: 129032 });
    expect(a.ability).toEqual({ source: "predicted", entries: [{ id: "a1", ppm: 750000 }], other_ppm: 250000 });
    expect(a.moves.entries).toEqual([{ id: "ma", inclusion_ppm: 1_000_000 }, { id: "mb", inclusion_ppm: 500000 }]);
  });

  it("a nature field with no named spread is unknown, with no entries", () => {
    const i = input();
    const b = belief(i);
    const noSpreads = { ...b, slots: b.slots.map((s) => ({ ...s, spreads: [{ id: "other", p: ratio(1n) }] })) };
    expect(opponentSheet(noSpreads, i.opponent, HASH).slots[0]!.nature).toEqual({ source: "unknown", entries: [], other_ppm: 0 });
  });

  it("the default k values are the reviewed ones", () => {
    expect(SHEET_K).toEqual({ item: 4, ability: 3, nature: 3, moves: 8 });
  });
});

describe("ties, merged natures and empty fields, on a hand-made belief", () => {
  const half = ratio(1n, 2n);
  const fabricated = (change: (slot: Record<string, unknown>) => void, revealedMoves: string[] = []) => {
    const i = input();
    const b = belief(i);
    const slots = b.slots.map((s, k) => {
      const copy: Record<string, unknown> = { ...s };
      if (k === 0) change(copy);
      return copy;
    });
    const opponent = { ...i.opponent, roster: i.opponent.roster.map((m, k) => (k === 0 ? { ...m, moves_revealed: revealedMoves } : m)) };
    return opponentSheet({ ...b, slots } as never, opponent, HASH).slots[0]!;
  };

  it("equal ppm sorts by id, a null item id last, whatever the order of the belief's list", () => {
    const items = [{ id: "iy", p: ratio(1n, 3n) }, { id: null, p: ratio(1n, 3n) }, { id: "ix", p: ratio(1n, 3n) }];
    expect(fabricated((s) => (s.items = items)).item.entries.map((e) => e.id)).toEqual(["ix", "iy", null]);
    expect(fabricated((s) => (s.items = [...items].reverse())).item.entries.map((e) => e.id)).toEqual(["ix", "iy", null]);
    expect(fabricated((s) => (s.items = [items[1], items[0]])).item.entries.map((e) => e.id)).toEqual(["iy", null]);
    expect(fabricated((s) => (s.items = [items[0], items[1]])).item.entries.map((e) => e.id)).toEqual(["iy", null]);
  });

  it("spreads of one nature are summed; the other-spread entry names no nature", () => {
    const spreads = [{ id: "adamant:0-0-0-0-0-0", p: half }, { id: "adamant:1-0-0-0-0-0", p: ratio(1n, 4n) }, { id: "other", p: ratio(1n, 4n) }];
    expect(fabricated((s) => (s.spreads = spreads)).nature).toEqual({ source: "predicted", entries: [{ id: "adamant", ppm: 750000 }], other_ppm: 250000 });
  });

  it("an item or move list with no support is unknown, and four revealed moves make the move field revealed", () => {
    const empty = fabricated((s) => ((s.items = []), (s.moves = [])));
    expect(empty.item).toEqual({ source: "unknown", entries: [], other_ppm: 0 });
    expect(empty.moves).toEqual({ source: "unknown", entries: [], slots_known: 0 });
    const four = ["ma", "mb", "mc", "md"];
    const full = fabricated((s) => (s.moves = four.map((id) => ({ id, p: ratio(1n) }))), four);
    expect(full.moves).toMatchObject({ source: "revealed", slots_known: 4 });
  });
});

describe("a zero-probability option does not stop a fact from being revealed", () => {
  it("is filtered before the single-entry test", () => {
    const i = input();
    const b = belief(i);
    const slots = b.slots.map((s, k) => (k === 0 ? { ...s, items: [{ id: "ix", p: ratio(1n) }, { id: "iy", p: ratio(0n) }] } : s));
    const roster = i.opponent.roster.map((m, k) => (k === 0 ? { ...m, item: "ix", item_known: true } : m));
    const block = opponentSheet({ ...b, slots } as never, { ...i.opponent, roster }, HASH);
    expect(block.slots[0]!.item).toEqual({ source: "revealed", entries: [{ id: "ix", ppm: 1_000_000 }], other_ppm: 0 });
  });
});

describe("an unsettled identity-hiding member with a shown position: the block is unknown with a fixed code", () => {
  const i = input([SWITCH("a", "Alp", "Alpha")], { config: hidingConfig });
  const block = opponentSheet(belief(i), i.opponent, HASH);

  it("refuses with disguise_unsettled and is unknown everywhere", () => {
    expect(block.source).toEqual({ kind: "belief", hash: HASH, refusal: "disguise_unsettled" });
    for (const s of block.slots) expect(s).toMatchObject({ brought_ppm: 0, brought_source: "unknown", item: { source: "unknown", entries: [], other_ppm: 0 }, moves: { source: "unknown", entries: [], slots_known: 0 } });
    expect(block.slots.map((s) => s.species)).toEqual(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]);
  });

  it("with nothing shown there is nothing to assign and the block is the plain one", () => {
    const quiet = input([], { config: hidingConfig });
    expect(JSON.stringify(opponentSheet(belief(quiet), quiet.opponent, HASH))).toBe(JSON.stringify(sheetOf()));
  });
});

describe("refusals", () => {
  it("refusedSheet is unknown everywhere with the species of the roster", () => {
    const block = refusedSheet(seen().opponent, "reader_flagged", HASH);
    expect(block.slots).toHaveLength(6);
    expect(block.source.refusal).toBe("reader_flagged");
  });

  it("the codes a refused block can carry are the belief's, minus the defects of the run, plus disguise_unsettled", () => {
    const defects = ["format_mismatch", "item_clause_unsupported", "spread_sampler_required"];
    expect([...SHEET_REFUSAL_CODES].sort()).toEqual([...BELIEF_ERROR_CODES.filter((c) => !defects.includes(c)), "disguise_unsettled"].sort());
    expect(sheetRefusalOf(new BeliefError("evidence_has_no_support", "x"))).toBe("evidence_has_no_support");
    for (const c of defects) expect(sheetRefusalOf(new BeliefError(c as never, "x"))).toBeNull();
    expect(sheetRefusalOf(new Error("x"))).toBeNull();
  });

  it("a belief of another roster size is a programming error", () => {
    const i = input();
    expect(() => opponentSheet(belief(i), { ...i.opponent, roster: i.opponent.roster.slice(0, 5) }, HASH)).toThrow(RangeError);
  });
});

describe("the feature set hash", () => {
  it("names the belief identity, the k values, the rounding and the builder format", () => {
    const h = sheetFeatureSetHash(HASH);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
    expect(sheetFeatureSetHash("b".repeat(64))).not.toBe(h);
    expect(sheetFeatureSetHash(HASH, { ...SHEET_K, item: 5 })).not.toBe(h);
    expect(() => sheetFeatureSetHash("zz")).toThrow(RangeError);
    expect(OPPONENT_SHEET_FORMAT).toBe("opponent-sheet-builder/1");
  });
});

describe("the builder reads the belief and the reader's opponent half only", () => {
  it("imports no sampler, draw or top-worlds module", () => {
    const code = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "src", "sheet.ts"), "utf8");
    for (const m of code.matchAll(/^import\s[^;]*?from\s+"([^"]+)";/gms)) expect(m[1], m[0]).not.toMatch(/sampler|draw|top-worlds|sha256/);
  });
});

describe("golden digests: a change of how a value is computed fails here and needs an OPPONENT_SHEET_FORMAT bump", () => {
  const cases: Record<string, () => unknown> = {
    preview: () => sheetOf(),
    revealed: () => sheetOf([SWITCH("a", "Gam", "Gamma"), SWITCH("b", "Alp", "Alpha"), "|-item|p2a: Gam|Ix", "|-ability|p2b: Alp|A2", "|move|p2b: Alp|Mb"]),
    mega: () => sheetOf([SWITCH("a", "Bet", "Beta"), "|detailschange|p2a: Bet|Beta-Mega, L50, M", "|-mega|p2a: Bet|Beta|Betite"]),
    disguise: () => sheetOf([SWITCH("a", "Alp", "Alpha")], { config: hidingConfig }),
  };
  it.each(Object.entries(GOLDEN))("%s", (name, expected) => {
    expect(digest((cases[name] as () => unknown)()), name).toBe(expected);
  });
});
