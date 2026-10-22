import { describe, expect, it } from "vitest";
import { ChoiceParseError, parseJointChoice } from "../src/damage-aware-choice.ts";
import { loadCorpus } from "../src/choice-corpus.ts";

describe("parseJointChoice", () => {
  it("splits the slot parts of a joint choice", () => {
    expect(parseJointChoice("move 1 2, switch 3")).toEqual([{ kind: "move", index: 1, target: 2, mega: false }, { kind: "switch", to: 3 }]);
    expect(parseJointChoice("move 2 -1 mega, pass")).toEqual([{ kind: "move", index: 2, target: -1, mega: true }, { kind: "pass" }]);
    expect(parseJointChoice("move 3, move 4 mega")).toEqual([{ kind: "move", index: 3, target: null, mega: false }, { kind: "move", index: 4, target: null, mega: true }]);
  });
  it.each(["team 1234", "move", "move 0", "move 1 x", "move 1 1 2", "switch", "switch 0", "pass 1", "move 1 0", ""])("refuses %j", (choice) => {
    expect(() => parseJointChoice(choice)).toThrow(ChoiceParseError);
  });
  it("takes every battle candidate of the choice corpus", () => {
    let parsed = 0;
    for (const entry of Object.values(loadCorpus())) {
      for (const candidate of entry.candidates) {
        if (candidate.choice.startsWith("team ")) continue;
        expect(parseJointChoice(candidate.choice).length, candidate.choice).toBeGreaterThan(0);
        parsed++;
      }
    }
    expect(parsed).toBeGreaterThan(100);
  });
});
