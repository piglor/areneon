import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { EVIDENCE_FIELDS, IGNORED_FIELDS, belief, slotEvidence } from "../src/index.ts";
import { input, seen, text } from "./support.ts";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const SWITCH = "|switch|p2a: Alp|Alpha, L50, M|100/100";
const first = (lines: readonly string[]) => slotEvidence(seen([SWITCH, ...lines]).opponent.roster[0]!);

describe("the evidence allowlist is pinned to the reader's line grammar", () => {
  it("every field of the reader's Pokemon is either read or deliberately ignored", () => {
    const keys = Object.keys(seen([SWITCH]).opponent.roster[0]!).sort();
    const classified = [...EVIDENCE_FIELDS.filter((f) => f !== "phantom"), ...IGNORED_FIELDS].sort();
    // `phantom` exists only on the reader's twin entries, which the belief refuses.
    expect(classified).toEqual(keys);
    expect(new Set([...EVIDENCE_FIELDS, ...IGNORED_FIELDS]).size).toBe(EVIDENCE_FIELDS.length + IGNORED_FIELDS.length);
  });

  it("the belief reads the reader's Pokemon only through those fields", () => {
    const read = new Set<string>();
    for (const file of ["evidence.ts", "belief.ts"]) for (const m of readFileSync(join(SRC, file), "utf8").matchAll(/\bmon\.([a-z_]+)\b/g)) read.add(m[1] as string);
    expect([...read].filter((f) => !(EVIDENCE_FIELDS as readonly string[]).includes(f))).toEqual([]);
    // and the one other read of a Pokemon, the twin flag
    expect(readFileSync(join(SRC, "belief.ts"), "utf8")).toContain("m.phantom === true");
  });

  it("lines of the allowlist set the evidence the ticket names", () => {
    expect(first([])).toMatchObject({ settled: true, moves: [], item: null, ability: null, mega: false, shown: "alpha", level: 50, gender: "M" });
    expect(first(["|move|p2a: Alp|Mb"]).moves).toEqual(["mb"]);
    expect(first(["|-item|p2a: Alp|Ix"]).item).toBe("ix");
    expect(first(["|-ability|p2a: Alp|A1"]).ability).toBe("a1");
    expect(first(["|-damage|p2a: Alp|40/100|[from] item: Ix"]).item).toBe("ix"); // an item revealed by its own effect
    expect(first(["|detailschange|p2a: Alp|Alpha-Mega, L50, M", "|-mega|p2a: Alp|Alpha|Alphite"])).toMatchObject({ mega: true, megaStone: "alphite", shown: "alphamega" });
    expect(first(["|move|p2a: Alp|Ma", "|move|p2a: Alp|Mb"]).moves).toEqual(["ma", "mb"]);
  });

  it("a move called by another effect, a consumed item and a transformed Pokemon add no move or item evidence", () => {
    expect(first(["|move|p2a: Alp|Mb|[from]Copycat"]).moves).toEqual([]);
    expect(first(["|-item|p2a: Alp|Ix", "|-enditem|p2a: Alp|Ix"]).item).toBeNull();
    expect(first(["|-transform|p2a: Alp|p2a: Alp", "|move|p2a: Alp|Mb"])).toMatchObject({ moves: [], transformed: true });
  });

  it("lines outside the allowlist leave the belief as it was", () => {
    const base = text(belief(input([SWITCH])));
    for (const line of ["|-damage|p2a: Alp|40/100", "|-status|p2a: Alp|par", "|-boost|p2a: Alp|atk|1", "|-start|p2a: Alp|confusion", "|-weather|RainDance", "|-sidestart|p2: Bob|Reflect", "|-fieldstart|move: Trick Room", "|turn|2"]) {
      expect(text(belief(input([SWITCH, line]))), line).toBe(base);
    }
  });

  it("a faint keeps the Pokemon in the brought set", () => {
    expect(first(["|faint|p2a: Alp"]).settled).toBe(true);
  });

  it("the trapping input is the port and nothing else of the own side: the constraint reads only `observed` and `shows`", () => {
    const source = readFileSync(join(SRC, "trapping.ts"), "utf8");
    const read = new Set([...source.matchAll(/\bcheck\.([a-z_]+)\b/g)].map((m) => m[1]));
    expect([...read].sort()).toEqual(["observed", "shows"]);
    // and the only Pokemon fields it reads are the ones the allowlist names (evidence.ts: `trapEvidence`)
    expect(source).not.toMatch(/\bmon\./);
  });
});
