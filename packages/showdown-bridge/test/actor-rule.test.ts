import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { type ChoiceCase, loadCorpus } from "../src/choice-corpus.ts";
import { actorRule, NoCandidateError } from "../src/index.ts";

const set = (choices: string[], pending: string[]) => ({ choices, pending });

/** A corpus case's candidates split as `legalChoiceSet` splits them, order kept. */
const fromCase = (file: string) => {
  const { candidates } = loadCorpus()[file] as ChoiceCase;
  return {
    canonical: candidates.map((c) => c.choice),
    legal: set(
      candidates.filter((c) => !c.pending).map((c) => c.choice),
      candidates.filter((c) => c.pending).map((c) => c.choice),
    ),
  };
};

describe("actorRule (the shared A1/A2 actor rule, ADR-005 §1)", () => {
  it("plays the first non-pending candidate", () => {
    expect(actorRule(set(["move 1 1, move 2", "move 2, move 2"], []))).toBe("move 1 1, move 2");
  });

  it("plays a single candidate, pending or not", () => {
    expect(actorRule(set(["switch 3, pass"], []))).toBe("switch 3, pass");
    expect(actorRule(set([], ["switch 3, pass"]))).toBe("switch 3, pass");
  });

  it("plays the first pending candidate when every candidate is pending", () => {
    expect(actorRule(set([], ["move 1, switch 3", "move 1, switch 4"]))).toBe("move 1, switch 3");
  });

  it("prefers a non-pending candidate that comes after pending ones in canonical order", () => {
    // Imprison makes every move pending; the switches after them are not.
    const { canonical, legal } = fromCase("maybe-disabled.json");
    const choice = actorRule(legal);
    expect(choice).toBe(legal.choices[0]);
    expect(canonical.indexOf(choice)).toBeGreaterThan(canonical.indexOf(legal.pending[0] as string));
  });

  it("plays a non-pending candidate that comes before pending ones in canonical order", () => {
    // A possible Shadow Tag makes the switches pending; the moves before them are not.
    const { canonical, legal } = fromCase("maybe-trapped.json");
    const choice = actorRule(legal);
    expect(choice).toBe(canonical[0]);
    expect(choice).toBe(legal.choices[0]);
    expect(legal.pending.length).toBeGreaterThan(0);
  });

  it("refuses an empty set with a typed error", () => {
    expect(() => actorRule(set([], []))).toThrow(NoCandidateError);
    expect(() => actorRule(set([], []))).toThrow(/no candidate/);
  });

  it("is pure: it imports only types", () => {
    const source = readFileSync(new URL("../src/actor-rule.ts", import.meta.url), "utf8");
    const imports = source.split("\n").filter((line) => /^\s*(import|export .* from)\b/.test(line));
    expect(imports).toEqual([
      'import type { LegalChoiceSet } from "@areneon/schema/types/v1/legal-choice-set.ts";',
    ]);
    expect(source).not.toMatch(/\brequire\s*\(|\bimport\s*\(/);
  });
});
