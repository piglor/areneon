// A joint choice string split into its slot parts (ADR-006 syntax: `move 1 2 mega, switch 3`).
// The scorer needs only which move a slot plays at which target; anything it does not know fails
// closed, as the enumerator does.

export class ChoiceParseError extends Error {
  override name = "ChoiceParseError";
}

export type SlotPart =
  | { kind: "pass" }
  | { kind: "switch"; to: number }
  | { kind: "move"; index: number; target: number | null; mega: boolean };

const positive = (text: string | undefined): number => {
  if (text === undefined || !/^[1-9][0-9]*$/.test(text)) throw new ChoiceParseError("a choice index is a positive integer");
  return Number(text);
};

function parseSlot(text: string): SlotPart {
  const words = text.split(" ");
  if (words[0] === "pass" && words.length === 1) return { kind: "pass" };
  if (words[0] === "switch" && words.length === 2) return { kind: "switch", to: positive(words[1]) };
  if (words[0] === "move") {
    const rest = words.slice(2);
    const mega = rest[rest.length - 1] === "mega";
    if (mega) rest.pop();
    if (rest.length > 1) throw new ChoiceParseError("a move names at most one target");
    const target = rest[0];
    if (target !== undefined && !/^-?[1-9][0-9]*$/.test(target)) throw new ChoiceParseError("a move target is a signed location");
    return { kind: "move", index: positive(words[1]), target: target === undefined ? null : Number(target), mega };
  }
  throw new ChoiceParseError("a slot part is pass, switch or move");
}

/** The slot parts of a battle choice, in slot order. Team Preview choices (`team ...`) are refused: they are not scored. */
export function parseJointChoice(choice: string): SlotPart[] {
  return choice.split(", ").map(parseSlot);
}
