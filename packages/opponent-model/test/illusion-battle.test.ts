// The committed Illusion battle (fixtures/showdown/<format>/illusion/battle.json) through the belief
// (ADR-025 section 3 and 8, N-B, R11): a Zoroark on both sides whose disguise breaks. The test reads the
// opponent's REAL team as an oracle only (the belief never sees it): at every state of both seats'
// channels, before and after each |replace|, the true world has weight above zero. With a prior uniform
// over the vocabulary the constraints are the evidence alone.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { interpretChannel, toId } from "@areneon/public-state";
import { describe, expect, it } from "vitest";
import { belief } from "../src/index.ts";
import type { DisguiseAssignment, OpponentBelief } from "../src/index.ts";
import { DIR, realVocabulary, uniformPrior } from "./real.ts";
import { config } from "./support.ts";

interface Set6 {
  species: string;
  item: string;
  ability: string;
  moves: string[];
}
interface Battle {
  teams: Record<"p1" | "p2", Set6[]>;
  channels: Record<"p1" | "p2", { lines: string[] }>;
}
const battle = JSON.parse(readFileSync(join(DIR, "illusion", "battle.json"), "utf8")) as Battle;
const hiding = { ...config, identity_hiding_abilities: ["illusion"] };
const ZOROARK = 5;
const PIKACHU = 3;

/** Whether `assignment` can hold the true team: every slot's belief gives the true forme, ability, item and moves a probability above zero. */
function holds(assignment: DisguiseAssignment, team: readonly Set6[]): boolean {
  return team.every((set, i) => {
    const s = assignment.slots[i]!;
    return s.formes.some((f) => f.id === toId(set.species)) && s.abilities.some((m) => m.id === toId(set.ability)) && s.items.some((m) => m.id === toId(set.item)) && set.moves.every((mv) => s.moves.some((m) => m.id === toId(mv)));
  });
}

const sameMoved = (a: DisguiseAssignment, moved: readonly { position: number; slot: number }[]): boolean => JSON.stringify(a.moved) === JSON.stringify(moved);

/**
 * Both seats' states of a channel (every line), with the belief at each and the oracle's verdict. `window` is the line range
 * in which the opponent's Zoroark is shown as its disguise; the true assignment there moves the Pikachu position to it.
 */
function check(lines: readonly string[], seat: "p1" | "p2", window: readonly [number, number]) {
  const run = interpretChannel(seat, lines);
  expect(run.flags).toEqual([]);
  const team = battle.teams[seat === "p1" ? "p2" : "p1"];
  const seenBeliefs: { index: number; belief: OpponentBelief }[] = [];
  for (const frame of run.frames) {
    if (frame.index < 0 || frame.state.opponent.roster.length !== 6) continue;
    const ids = frame.state.opponent.roster.map((m) => m.base_species);
    const b = belief({ prior: uniformPrior(ids), vocabulary: realVocabulary, config: hiding, opponent: frame.state.opponent, flags: [] });
    seenBeliefs.push({ index: frame.index, belief: b });
    const inWindow = frame.index >= window[0] && frame.index < window[1];
    const truth = b.assignments.filter((a) => holds(a, team));
    // The true world has weight above zero, in the disguise's assignment while it stands, in the identity once it is settled.
    expect(truth.length, `line ${frame.index}`).toBeGreaterThan(0);
    const expected = inWindow ? [{ position: PIKACHU, slot: ZOROARK }] : [];
    expect(truth.some((a) => sameMoved(a, expected)), `line ${frame.index} (${inWindow ? "disguised" : "settled or unshown"})`).toBe(true);
    expect(truth.every((a) => a.p.num > 0n)).toBe(true);
  }
  return { run, beliefs: seenBeliefs };
}

describe("the true world keeps weight above zero on the committed Illusion battle", () => {
  it("seat p1 (the opponent's Zoroark shows as Pikachu from line 68 and is replaced at line 73)", () => {
    const { beliefs } = check(battle.channels.p1.lines, "p1", [68, 73]);
    const before = beliefs.find((x) => x.index === 72)!.belief;
    expect(before.assignments.length).toBeGreaterThan(1);
    expect(before.disguise.unsettled).toEqual([ZOROARK]);
    expect(before.disguise.positions).toContain(PIKACHU);
    const after = beliefs.find((x) => x.index === 73)!.belief;
    expect(after.disguise.unsettled).toEqual([]);
    expect(after.assignments).toHaveLength(1);
    expect(after.brought.settled).toContain(ZOROARK);
  });

  it("seat p2 (the opponent's Zoroark shows as Pikachu from line 28 and is replaced at line 37)", () => {
    const { beliefs } = check(battle.channels.p2.lines, "p2", [28, 37]);
    const request = beliefs.find((x) => x.index === 33)!.belief;
    expect(request.assignments.length).toBeGreaterThan(1);
    expect(request.disguise.positions).toEqual([PIKACHU, 4]);
    expect(beliefs.find((x) => x.index === 37)!.belief.assignments).toHaveLength(1);
  });

  it("a late replace: the disguised Pokemon uses a move shown as Pikachu for a turn before the replace", () => {
    // Seat p2's channel with the replace moved from after the first hit (line 37) to just before the faint (line 54). Zoroark
    // flinches, is hit and uses Spite (a move Pikachu cannot have) under its disguise, all across the turn boundary.
    const lines = battle.channels.p2.lines;
    const moved = lines.flatMap((line, i): string[] => {
      if (i === 37 || i === 38) return [];
      if (i === 54) return [lines[37] as string, lines[38] as string, line];
      return [i > 38 && i < 54 ? line.replace("p1a: Zoroark", "p1a: Pikachu") : line];
    });
    const spiteLine = moved.findIndex((l) => l.includes("|Spite|"));
    const replaceLine = moved.findIndex((l) => l.startsWith("|replace|"));
    expect(spiteLine).toBeGreaterThan(0);
    expect(replaceLine).toBeGreaterThan(spiteLine + 1);
    const { beliefs } = check(moved, "p2", [28, replaceLine]);
    // Between the move and the replace the move is the only explanation: Spite is Zoroark's, so one assignment.
    const mid = beliefs.find((x) => x.index === spiteLine)!.belief;
    expect(mid.assignments).toHaveLength(1);
    expect(mid.assignments[0]!.moved).toEqual([{ position: PIKACHU, slot: ZOROARK }]);
    expect(mid.slots[ZOROARK]!.moves.find((m) => m.id === "spite")!.p).toEqual({ num: 1n, den: 1n });
    // Before the move both explanations stand; after the replace the member is settled and the move is its own.
    expect(beliefs.find((x) => x.index === spiteLine - 1)!.belief.assignments.length).toBeGreaterThan(1);
    const after = beliefs.find((x) => x.index === replaceLine)!.belief;
    expect(after.assignments).toHaveLength(1);
    expect(after.slots[ZOROARK]!.moves.find((m) => m.id === "spite")!.p).toEqual({ num: 1n, den: 1n });
  });
});
