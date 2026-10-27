// Carry-overs of ADR-025 slice 2 parts 1 to 3 (#1124): the vocabulary hash check, the package's JCS against the
// bridge's, the stat ids against the bridge's, an item or ability the Pokemon received (the reader's new flags),
// and the narrowed catch of the disguise mixture.

import { describe, expect, it } from "vitest";
import { STAT_IDS as BRIDGE_STAT_IDS } from "../../showdown-bridge/src/engine.ts";
import { jcs as bridgeJcs } from "../../showdown-bridge/src/canonical-json.ts";
import { BeliefError, STAT_IDS, belief } from "../src/index.ts";
import { jcs } from "../src/canonical.ts";
import { slotEvidence } from "../src/evidence.ts";
import { hidingConfig, input, prior, seen, vocabulary, HEX } from "./support.ts";

const SWITCH = "|switch|p2a: Alp|Alpha, L50, M|100/100";
const first = (lines: readonly string[]) => slotEvidence(seen([SWITCH, ...lines]).opponent.roster[0]!);
const code = (run: () => unknown): string => {
  try {
    run();
  } catch (e) {
    if (e instanceof BeliefError) return e.code;
    throw e;
  }
  return "none";
};

describe("the prior's vocabulary hash", () => {
  it("must equal the vocabulary's: a prior derived against another team vocabulary is refused", () => {
    expect(code(() => belief(input([], { vocabulary: { ...vocabulary(), vocabulary_hash: "1".repeat(64) } })))).toBe("format_mismatch");
    expect(code(() => belief(input([], { prior: prior((p) => void (p.vocabulary_hash = "2".repeat(64))) })))).toBe("format_mismatch");
    expect(code(() => belief(input([], { vocabulary: { ...vocabulary(), vocabulary_hash: HEX } })))).toBe("none");
  });
});

describe("the package's own helpers against the bridge's", () => {
  it("the package's JCS equals the bridge's on strings, integers, nesting and key order", () => {
    const values: unknown[] = [null, true, 0, -7, 9007199254740991, "a \"\\\n😀", [], {}, { b: 1, a: [1, { z: null, y: "x" }], "é": 2, "10": 3, "9": 4 }];
    for (const v of values) expect(jcs(v)).toBe(bridgeJcs(v as never));
  });

  it("the stat ids are the bridge's", () => {
    expect([...STAT_IDS]).toEqual([...BRIDGE_STAT_IDS]);
  });
});

describe("an item or ability a Pokemon received is not its own (the reader's flags)", () => {
  it("Trick, Thief, Pickpocket and Recycle give an item that adds no item evidence; Frisk shows the target's own", () => {
    expect(first(["|-item|p2a: Alp|Ix|[from] move: Trick"]).item).toBeNull();
    expect(first(["|-item|p2a: Alp|Ix|[from] move: Thief|[of] p1a: Foe"]).item).toBeNull();
    expect(first(["|-item|p2a: Alp|Ix|[from] ability: Pickpocket|[of] p1a: Foe"]).item).toBeNull();
    expect(first(["|-item|p2a: Alp|Ix|[from] ability: Frisk|[of] p1a: Foe|[identify]"]).item).toBe("ix");
    expect(first(["|-item|p2a: Alp|Ix"]).item).toBe("ix");
  });

  it("the taint is sticky: a later own-looking item reveal does not make the received item the Pokemon's own", () => {
    expect(first(["|-item|p2a: Alp|Ix|[from] move: Trick", "|-damage|p2a: Alp|90/100|[from] item: Ix"]).item).toBeNull();
  });

  it("Trace, Mummy and a move that sets an ability give an ability that adds no ability evidence", () => {
    expect(first(["|-ability|p2a: Alp|A1|[from] ability: Trace|[of] p1a: Foe"]).ability).toBeNull();
    expect(first(["|-ability|p2a: Alp|A1|[from] ability: Mummy|[of] p1a: Foe"]).ability).toBeNull();
    expect(first(["|-ability|p2a: Alp|A1|[from] move: Entrainment"]).ability).toBeNull();
    expect(first(["|-ability|p2a: Alp|A1"]).ability).toBe("a1");
  });

  it("the belief keeps the true world possible after a Trick that gave the Pokemon an item it does not hold", () => {
    // alpha's prior items are ix, iy or none; "iy" received by Trick must not exclude an own item "ix".
    const b = belief(input([SWITCH, "|-item|p2a: Alp|Iy|[from] move: Trick"]));
    const alpha = b.slots.find((s) => s.species === "alpha")!;
    expect(alpha.items.some((m) => m.id === "ix" && m.p.num > 0n)).toBe(true);
    expect(alpha.constraints.some((c) => c.startsWith("item:"))).toBe(false);
  });
});

describe("the disguise mixture drops an assignment only for the refusals it knows", () => {
  const mixed = (change: (p: ReturnType<typeof prior>) => void) => input(["|switch|p2a: Eps|Epsilon, L50, M|100/100", "|move|p2a: Eps|Em1"], { prior: prior(change), config: hidingConfig });

  it("a prior list with no mass is the belief's error, not a dropped assignment", () => {
    const run = () => belief(mixed((p) => void (p.species.find((s) => s.id === "alpha")!.formes[0]!.items = { denominator: 1, entries: [], no_item: 0, residual: 0, unmapped: 1 })));
    expect(code(run)).toBe("empty_distribution");
  });

  it("an error that is not a BeliefError propagates from the mixture", () => {
    const broken = mixed((p) => void ((p.species.find((s) => s.id === "alpha")!.formes[0] as { moves: unknown }).moves = undefined));
    expect(() => belief(broken)).toThrow(TypeError);
  });
});
