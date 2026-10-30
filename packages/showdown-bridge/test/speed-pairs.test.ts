// Observed orders of same-priority actions and the reasons an order is not isolated (ticket #1133, ADR-025 section 7):
// every reviewed rule of the committed config makes a pair non-isolated, on hand-written channels.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { interpretChannel } from "@areneon/public-state";
import { describe, expect, it } from "vitest";
import { readBeliefConfig } from "../src/belief-config.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type PairContext, extractPairs } from "../src/speed-pairs.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const vocabulary = JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT, "team-vocabulary.json"), "utf8")) as TeamVocabulary;
const config = readBeliefConfig(JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", GOLDEN_FORMAT, "belief.speed-v1.json"), "utf8")), vocabulary);
const rules = (config.speed_evidence as NonNullable<typeof config.speed_evidence>).non_isolating;

/** The golden battle as the second player saw it (turn 1: p2b, p1b, p1a move; turn 2: p2b, p1b, p2a move). */
const golden = (JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", GOLDEN_FORMAT, "channels", "p2.json"), "utf8")) as { channel: { lines: string[] } }).channel.lines;
const at = (line: string): number => golden.findIndex((l) => l.startsWith(line));
const cap = (id: string): string => id.charAt(0).toUpperCase() + id.slice(1);
const use = (who: string, move: string, target: string, ...tags: string[]): string => ["", "move", who, move, target, ...tags].join("|");

const stub = (over: Partial<PairContext> = {}): PairContext => ({
  rules,
  priorityOf: () => 0,
  abilitiesOf: () => [],
  mayHide: () => false,
  declarable: () => true,
  unmodelledOwn: () => false,
  ...over,
});
/** The golden channel with `extra` inserted before the line starting with `before`. */
const pairs = (before: string, extra: readonly string[], ctx: PairContext = stub()) => {
  const lines = [...golden];
  lines.splice(at(before), 0, ...extra);
  return extractPairs(interpretChannel("p2", lines).frames, ctx);
};
const turn2 = (list: ReturnType<typeof pairs>) => list.filter((p) => p.turn === 2);
const reasons = (list: ReturnType<typeof pairs>): (string | null)[] => turn2(list).map((p) => p.reason);

describe("the isolated pairs of the golden battle", () => {
  const list = extractPairs(interpretChannel("p2", golden).frames, stub());

  it("are the own and opponent Pokemon that each used one move in a turn, with who acted first", () => {
    // turn 1: own Charizard (p2b) before the foe's Charizard (p1b) and Venusaur (p1a); turn 2: own Charizard, then the foe's, then own Blastoise (p2a)
    expect(list.filter((p) => p.turn === 1).map((p) => [p.own.letter, p.foe.letter, p.first])).toEqual([["b", "b", "own"], ["b", "a", "own"]]);
    expect(turn2(list).map((p) => [p.own.letter, p.own.move, p.foe.letter, p.foe.move, p.first, p.reason])).toEqual([
      ["b", "flamethrower", "b", "flamethrower", "own", null],
      ["a", "surf", "b", "flamethrower", "foe", null],
    ]);
  });

  it("carry the state just before the later action", () => {
    const [pair] = turn2(list);
    expect(pair?.state.turn).toBe(2);
    expect(pair?.state.opponent.roster[pair?.foe.index as number]?.hp_num).toBe(100);
  });
});

describe("an order is not isolated when", () => {
  const FOE = "p1b: Charizard";
  const OWN = "p2b: Charizard";
  it("the priorities differ or a move is unknown to the engine", () => {
    expect(reasons(pairs("|turn|3", [], stub({ priorityOf: (m) => (m === "surf" ? 1 : 0) })))).toEqual([null, "priority"]);
    expect(reasons(pairs("|turn|3", [], stub({ priorityOf: () => null })))).toEqual(["priority", "priority"]);
  });

  it("a Pokemon used a second move line of its own in the turn (one a move called through [from] is no action)", () => {
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [use(OWN, "Protect", OWN)]))).toEqual(["repeated_actor", "repeated_actor", null]);
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [use(OWN, "Protect", OWN, "[from]move: Metronome")]))).toEqual([null, null]);
  });

  it("a stat stage, a status or an item changed before both actions (the reader's state shows it)", () => {
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-unboost|${FOE}|spe|1`]))).toEqual([null, "changed_state"]);
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-status|${FOE}|par`]))).toEqual([null, "changed_state"]);
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-weather|RainDance`]))).toEqual([null, "changed_state"]);
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-sidestart|p1: ?|move: Tailwind`]))).toEqual([null, "changed_state"]);
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-enditem|${FOE}|Leftovers`]))).toEqual([null, "changed_state"]);
  });

  it("a change and its undoing happened before both (the order was decided under the change)", () => {
    expect(reasons(pairs("|move|p2a: Blastoise|Surf", [`|-boost|${FOE}|spe|2`, `|-unboost|${FOE}|spe|2`]))).toEqual([null, "changed_state"]);
  });

  it("a volatile is on a combatant, or the opponent's item or ability is not its own", () => {
    expect(reasons(pairs("|turn|2", [`|-start|${FOE}|confusion`]))).toEqual(["volatile", "volatile"]);
    expect(reasons(pairs("|turn|2", [`|-item|${FOE}|Leftovers|[from] move: Trick|[of] ${OWN}`]))).toEqual(["changed_mon", "changed_mon"]);
    expect(reasons(pairs("|turn|2", [`|-ability|${FOE}|Static|[from] ability: Trace|[of] ${OWN}`]))).toEqual(["changed_mon", "changed_mon"]);
  });

  it("the own Pokemon holds an unmodelled item or ability", () => {
    expect(reasons(pairs("|turn|3", [], stub({ unmodelledOwn: (m) => m.name === "Charizard" })))).toEqual(["unmodelled_own", null]);
  });

  it.each(rules.moves)("the move %s was used earlier this turn, by anyone", (move) => {
    const listed = [use("p2a: Blastoise", cap(move), FOE)];
    expect(reasons(pairs("|move|p1b: Charizard|Flamethrower|p2a: Blastoise", listed))[0]).toBe("listed_move");
    // a turn later it no longer is, for a move whose effect lasts a turn
    const later = pairs("|turn|3", [use("p2a: Blastoise", cap(move), FOE)]).filter((p) => p.turn === 2);
    expect(later.map((p) => p.reason)).not.toContain("listed_move");
  });

  it.each(rules.lasting_moves)("the lasting move %s was used earlier in the channel, in any turn", (move) => {
    expect(reasons(pairs("|turn|2", [use(FOE, cap(move), OWN)]))[0]).toBe("lasting_move");
  });

  it.each(rules.abilities)("the ability %s is held by a Pokemon in play besides the two acting ones", (ability) => {
    const held = stub({ abilitiesOf: (mon) => (mon.name === "Blastoise" && mon.own ? [ability] : []) });
    expect(reasons(pairs("|turn|3", [], held))[0]).toBe("listed_ability");
    // the acting Pokemon's own ability is a candidate's field, not a bystander's
    expect(reasons(pairs("|turn|3", [], stub({ abilitiesOf: (mon) => (mon.name === "Charizard" ? [ability] : []) })))[0]).toBeNull();
  });

  it("an unrevealed identity-hiding member of the preview exists, or the field is one the probe cannot state", () => {
    expect(reasons(pairs("|turn|3", [], stub({ mayHide: (m) => m.base_species === "gengar" })))[0]).toBe("possible_disguise");
    expect(reasons(pairs("|turn|3", [], stub({ declarable: () => false })))[0]).toBe("undeclared_field");
  });
});
