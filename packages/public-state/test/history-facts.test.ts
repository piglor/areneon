import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  HISTORY_FACTS_FORMAT,
  HISTORY_FACTS_PATH,
  HistoryFactsError,
  HistoryFactsReader,
  type HistoryFacts,
  type HistoryFactsData,
  type PokemonHistory,
  UNSETTLED,
  historyFacts,
  historyFactsAtRequests,
  historyFactsSha256,
  ownHistoryChannel,
  readerSha256,
  worldFactsSha256,
} from "../src/index.ts";

// Hand-written lines for every history fact, reset and refusal (ADR-022 erratum E-22.5.2). The data is invented
// to the shape the bridge reads from the pinned engine; the live-engine parity test holds the facts equal to the
// engine's own fields on every committed battle.

const data: HistoryFactsData = {
  stall_moves: ["protect", "detect", "quickguard", "wideguard"],
  ally_side_moves: ["quickguard", "wideguard"],
  stall_counter: { start: 3, factor: 3, max: 27 },
  action_sources: ["move: Round", "ability: Dancer"],
  external_action_sources: ["ability: Dancer"],
  item_use_sources: ["move: Fling"],
  eat_tags: ["eat", "weaken"],
  steal_eat_tags: ["stealeat"],
  gem_tags: ["gem"],
  eaten_berries: ["oranberry"],
  last_item_restorers: ["move: Recycle"],
  last_item_takers: ["ability: Pickup"],
  direct_end_items: ["airballoon"],
  null_cant_reasons: ["recharge"],
  protect_breaker_moves: ["feint"],
  slot_conditions: [
    { id: "futuremove", moves: ["futuresight"], residual_turns: 3 },
    { id: "wish", moves: ["wish"], residual_turns: 2 },
  ],
  volatile_not_copied: ["torment"],
  shed_tail_copies: ["substitute"],
};

const HEAD = [
  "|player|p1|Ann|1|",
  "|player|p2|Bob|2|",
  "|start",
  "|switch|p1a: Alpha|Alpha, L50|100/100",
  "|switch|p1b: Beta|Beta, L50|100/100",
  "|switch|p2a: Gamma|Gamma, L50|100/100",
  "|switch|p2b: Delta|Delta, L50|100/100",
];
const request = (moves: { id: string; maxpp?: number }[], second = moves): string =>
  `|request|${JSON.stringify({ active: [{ moves }, { moves: second }], side: { pokemon: [] } })}`;
const REQUEST = request([{ id: "tackle", maxpp: 10 }, { id: "protect", maxpp: 10 }, { id: "struggle" }]);
const TURN = (n: number): string[] => ["|", `|turn|${n}`];
const END = ["|", "|upkeep"];

const run = (lines: string[], player: "p1" | "p2" = "p1"): HistoryFacts => historyFacts({ channel: ownHistoryChannel(player, lines), data });
const mon = (facts: HistoryFacts, side: "own" | "opponent", letter: string): Extract<PokemonHistory, { unknown: false }> => {
  const found = facts[side].active[letter];
  if (found === null || found === undefined || found.unknown) throw new Error("no known Pokemon");
  return found;
};
const code = (fn: () => unknown): string => {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(HistoryFactsError);
    return (error as HistoryFactsError).code;
  }
  return "none";
};

describe("identity and input", () => {
  it("has a format id and a file hash of its own, and leaves the reader's and the world facts' bytes alone", () => {
    expect(HISTORY_FACTS_FORMAT).toBe("history-facts/1");
    expect(historyFactsSha256()).toBe(createHash("sha256").update(readFileSync(HISTORY_FACTS_PATH)).digest("hex"));
    expect(historyFactsSha256()).not.toBe(readerSha256());
    expect(historyFactsSha256()).not.toBe(worldFactsSha256());
  });

  it("imports nothing and names no game move, item or ability (only protocol tokens)", () => {
    const source = readFileSync(HISTORY_FACTS_PATH, "utf8");
    expect(source).not.toMatch(/^\s*import\s/m);
    expect(source).not.toMatch(/require\(/);
    const code = source.replace(/\/\/.*$/gm, "").replace(/\/\*[\s\S]*?\*\//g, "");
    const literals = [...code.matchAll(/"([^"\n]*)"|`([^`\n]*)`/g)].map((m) => (m[1] ?? m[2] ?? "").toLowerCase());
    // Every string literal of the module must be a protocol wire token, an error code or a structural word listed here. A new
    // literal (a game id such as a move, item or ability, or a `[from]` text) fails until a reviewer adds it on purpose; game
    // facts come from `HistoryFactsData`. `stall` is the engine's condition id of the Baton Pass / Shed Tail copy table,
    // `recoil` and `confusion` are the `[from]` wire tokens of the engine's own damage sources, `slot` a wire word, `}` a fragment of the error-message template.
    const accepted = new Set([
      "history-facts/1", "p1", "p2", "bad_player", "secret_marker", "malformed_line", "unattributable_line", "stall_unresolved",
      "historyfactserror", "|", "|split", "|split|", "unsettled", "ok", "failed",
      "unknown", "none", "", "request", "move", "cant", "switch", "drag", "faint", "turn", "upkeep", "replace", "swap", "win", "tie",
      "-damage", "-heal", "-boost", "-unboost", "-status", "-start", "-sidestart", "-fieldstart", "-weather", "-singleturn",
      "-singlemove", "-curestatus", "-setboost", "-item", "-ability", "-clearallboost", "-clearboost", "-clearnegativeboost",
      "-clearpositiveboost", "-copyboost", "-swapboost", "-invertboost", "-cureteam", "-sethp", "-transform", "-formechange",
      "-prepare", "-mustrecharge", "-sideend", "-fieldend", "-end", "-enditem", "pending_slot", "?", "-fail", "-notarget", "-miss",
      "-immune", "-activate", "from", "string", "number", "baton pass", "shed tail", "slot", "of", "move: ", "|request|",
      "stall", "recoil", "confusion", "}",
    ]);
    const unexpected = literals.filter((l) => !accepted.has(l) && !l.includes("${"));
    expect(unexpected).toEqual([]);
    for (const id of ["protect", "fling", "recycle", "pickup", "harvest", "wish", "futuresight", "feint", "quickguard", "wideguard", "dancer", "round", "berry", "tackle", "gem", "stealeat"]) {
      expect(literals.filter((l) => l.includes(id)), id).toEqual([]);
    }
    // the only move names are the wire's `[from]` texts of the two copying switches
    expect(literals).toContain("baton pass");
  });

  it("takes only the branded own channel", () => {
    const channel = ownHistoryChannel("p1", HEAD);
    expect(channel.player).toBe("p1");
    expect(() =>
      historyFacts({
        // @ts-expect-error: the omniscient log is not an OwnHistoryChannel
        channel: { player: "p1", lines: HEAD },
        data,
      }),
    ).not.toThrow();
    // @ts-expect-error: a team may not be passed
    historyFacts({ channel, data, team: [] });
    // @ts-expect-error: a seed may not be passed
    historyFacts({ channel, data, seed: "x" });
  });

  it("refuses a bad player, a line that is no protocol line and a split marker", () => {
    expect(code(() => ownHistoryChannel("p3", HEAD))).toBe("bad_player");
    expect(code(() => ownHistoryChannel("p1", ["move"]))).toBe("malformed_line");
    expect(code(() => ownHistoryChannel("p1", ["|split|p1"]))).toBe("secret_marker");
    expect(code(() => ownHistoryChannel("p1", ["|split"]))).toBe("secret_marker");
  });
});

describe("move actions", () => {
  it("counts own move lines, cant lines and the confusion self-hit, not called moves", () => {
    const facts = run([
      ...HEAD,
      ...TURN(1),
      "|move|p1a: Alpha|Tackle|p2a: Gamma",
      "|-damage|p2a: Gamma|80/100",
      "|move|p1b: Beta|Metronome|p1b: Beta",
      "|move|p1b: Beta|Tackle|p2a: Gamma|[from] move: Metronome",
      "|cant|p2a: Gamma|par",
      "|-activate|p2b: Delta|confusion",
      "|-damage|p2b: Delta|90/100|[from] confusion",
      ...END,
    ]);
    expect(mon(facts, "own", "a").move_actions).toBe(1);
    expect(mon(facts, "own", "b").move_actions).toBe(1);
    expect(mon(facts, "opponent", "a").move_actions).toBe(1);
    expect(mon(facts, "opponent", "b").move_actions).toBe(1);
  });

  it("counts an action of its own that another effect names (a listed source), and a sleep-usable move after a cant once", () => {
    const facts = run([
      ...HEAD,
      ...TURN(1),
      "|move|p1a: Alpha|Tackle|p2a: Gamma",
      "|move|p1b: Beta|Round|p2a: Gamma|[from] move: Round",
      "|cant|p2b: Delta|slp",
      "|move|p2b: Delta|Snore|p1a: Alpha",
      ...END,
    ]);
    expect(mon(facts, "own", "b").move_actions).toBe(1);
    expect(mon(facts, "opponent", "b").move_actions).toBe(1);
  });

  it("resets at a switch-in", () => {
    const facts = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Tackle|p2a: Gamma", ...END, ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|90/100"]);
    expect(mon(facts, "own", "a").move_actions).toBe(0);
  });
});

describe("the stall counter", () => {
  const protect = (n: number, user = "p1a: Alpha"): string[] => [...TURN(n), `|move|${user}|Protect|${user}`, `|-singleturn|${user}|Protect`, ...END];

  it("follows consecutive successful uses to the ceiling and ends with the turn after", () => {
    expect(mon(run([...HEAD, ...protect(1)]), "own", "a").stall_counter).toBe(3); // after the residual of its own turn the volatile has one turn left
    const first = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect"]);
    expect(mon(first, "own", "a").stall_counter).toBe(3);
    const second = run([...HEAD, ...protect(1), ...TURN(2), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect"]);
    expect(mon(second, "own", "a").stall_counter).toBe(9);
    const next = run([...HEAD, ...protect(1), ...TURN(2)]);
    expect(mon(next, "own", "a").stall_counter).toBe(3); // the volatile lasts through the turn after a use
    const gone = run([...HEAD, ...protect(1), ...TURN(2), ...END, ...TURN(3)]);
    expect(mon(gone, "own", "a").stall_counter).toBeNull();
    const capped = run([...HEAD, ...protect(1), ...protect(2), ...protect(3), ...protect(4), ...TURN(5), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect"]);
    expect(mon(capped, "own", "a").stall_counter).toBe(27);
  });

  it("a failed use ends the chain, a called one counts, and a breaker deletes the volatile", () => {
    const failed = run([...HEAD, ...protect(1), ...TURN(2), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-fail|p1a: Alpha"]);
    expect(mon(failed, "own", "a").stall_counter).toBeNull();
    const called = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Metronome|p1a: Alpha", "|move|p1a: Alpha|Protect|p1a: Alpha|[from] move: Metronome", "|-singleturn|p1a: Alpha|Protect"]);
    expect(mon(called, "own", "a").stall_counter).toBe(3);
    const calledAlone = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Protect|p1a: Alpha|[from] move: Metronome", "|-singleturn|p1a: Alpha|Protect"]);
    expect(mon(calledAlone, "own", "a").stall_counter).toBe(3);
    const broken = run([...HEAD, ...protect(1), ...TURN(2), "|move|p2a: Gamma|Feint|p1a: Alpha", "|-activate|p1a: Alpha|move: Feint"]);
    expect(mon(broken, "own", "a").stall_counter).toBeNull();
    const other = run([...HEAD, ...protect(1), ...TURN(2), "|move|p2a: Gamma|Tackle|p1a: Alpha", "|-activate|p1a: Alpha|move: Tackle", "|-activate|p1b: Beta|ability: Static"]);
    expect(mon(other, "own", "a").stall_counter).toBe(3);
  });

  it("refuses an instructed stall move, which prints nothing (#1377; a known gap to the engine, kept so the file hash stays)", () => {
    const lines = [...HEAD, ...TURN(1), "|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect", "|move|p2a: Gamma|Instruct|p1a: Alpha", "|-singleturn|p1a: Alpha|move: Instruct|[of] p2a: Gamma", "|move|p1a: Alpha|Protect|p1a: Alpha", "|move|p2b: Delta|Tackle|p1b: Beta", ...END];
    expect(code(() => run(lines))).toBe("stall_unresolved");
  });

  it("a side-condition stall move whose condition is already up prints nothing and still succeeds", () => {
    const facts = run([
      ...HEAD,
      ...TURN(1),
      "|move|p1a: Alpha|Wide Guard|p1a: Alpha",
      "|-singleturn|p1a: Alpha|Wide Guard",
      "|move|p1b: Beta|Wide Guard|p1b: Beta",
      "|move|p2a: Gamma|Tackle|p1a: Alpha",
    ]);
    expect(mon(facts, "own", "b").stall_counter).toBe(3);
  });

  it("a stalling move with no outcome line is a typed refusal, and the facts before it stand", () => {
    const lines = [...HEAD, ...TURN(1), "|move|p1a: Alpha|Protect|p1a: Alpha", "|move|p2a: Gamma|Tackle|p1a: Alpha"];
    expect(code(() => run(lines))).toBe("stall_unresolved");
    expect(code(() => run(lines.slice(0, -1)))).toBe("stall_unresolved");
    const points = historyFactsAtRequests({ channel: ownHistoryChannel("p1", [...HEAD, REQUEST, ...lines.slice(HEAD.length), REQUEST]), data });
    expect(points.map((p) => p.refusal)).toEqual([null, "stall_unresolved"]);
    expect(points[0]?.facts?.turn).toBe(0);
  });

  it("a Baton Pass hands the counter over (the copy table decides), a Shed Tail does not", () => {
    const passed = run([...HEAD, ...protect(1), ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Baton Pass"]);
    expect(mon(passed, "own", "a").stall_counter).toBe(3);
    const shed = run([...HEAD, ...protect(1), ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Shed Tail"]);
    expect(mon(shed, "own", "a").stall_counter).toBeNull();
    const refused = run([...HEAD, ...protect(1), ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|100/100"]);
    expect(mon(refused, "own", "a").stall_counter).toBeNull();
    const noCopy = historyFacts({ channel: ownHistoryChannel("p1", [...HEAD, ...protect(1), ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Baton Pass"]), data: { ...data, volatile_not_copied: ["stall"], shed_tail_copies: ["stall"] } });
    expect(mon(noCopy, "own", "a").stall_counter).toBeNull();
    const shedCopies = historyFacts({ channel: ownHistoryChannel("p1", [...HEAD, ...protect(1), ...TURN(2), "|switch|p1a: Alpha|Alpha, L50|100/100|[from] Shed Tail"]), data: { ...data, shed_tail_copies: ["stall"] } });
    expect(mon(shedCopies, "own", "a").stall_counter).toBe(3);
  });
});

describe("hits, hurt and attackers", () => {
  const hit = (target = "p2a: Gamma"): string[] => [`|move|p1a: Alpha|Tackle|${target}`, `|-damage|${target}|70/100`];

  it("counts one hit per damage line of a move, notes the attacker and the hurt, and resets by the turn", () => {
    const facts = run([
      ...HEAD,
      ...TURN(1),
      "|move|p1a: Alpha|Double Hit|p2a: Gamma",
      "|-damage|p2a: Gamma|80/100",
      "|-damage|p2a: Gamma|60/100",
      "|-hitcount|p2a: Gamma|2",
      "|move|p1b: Beta|Tackle|p2a: Gamma",
      "|-damage|p2a: Gamma|50/100",
    ]);
    const gamma = mon(facts, "opponent", "a");
    expect(gamma.times_attacked).toBe(3);
    expect(gamma.hurt_this_turn).toBe(true);
    expect(gamma.attackers_this_turn).toEqual(["own:a", "own:b"]);
    const later = run([...HEAD, ...TURN(1), ...hit(), ...END, ...TURN(2)]);
    expect(mon(later, "opponent", "a").times_attacked).toBe(1);
    expect(mon(later, "opponent", "a").hurt_this_turn).toBe(false);
    expect(mon(later, "opponent", "a").attackers_this_turn).toEqual([]);
  });

  it("keeps hurt through turn 1 but not through a later turn, and resets everything at a switch", () => {
    const early = run([...HEAD, "|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|70/100", ...TURN(1)]);
    expect(mon(early, "opponent", "a").hurt_this_turn).toBe(true);
    expect(mon(early, "opponent", "a").attackers_this_turn).toEqual([]);
    const out = run([...HEAD, ...TURN(1), ...hit(), ...END, ...TURN(2), "|switch|p2a: Gamma|Gamma, L50|70/100"]);
    expect(mon(out, "opponent", "a").times_attacked).toBe(0);
  });

  it("a Substitute's damage and a user's own cost are no hit and no hurt; confusion and a drain are hurt; recoil is direct", () => {
    const facts = run([
      ...HEAD,
      ...TURN(1),
      "|move|p1a: Alpha|Substitute|p1a: Alpha",
      "|-start|p1a: Alpha|Substitute",
      "|-damage|p1a: Alpha|75/100",
      "|move|p2a: Gamma|Tackle|p1a: Alpha",
      "|-activate|p1a: Alpha|move: Substitute|[damage]",
      "|move|p2b: Delta|Struggle|p1b: Beta",
      "|-damage|p1b: Beta|90/100",
      "|-damage|p2b: Delta|90/100|[from] recoil",
      "|-activate|p1b: Beta|confusion",
      "|-damage|p1b: Beta|80/100|[from] confusion",
    ]);
    const alpha = mon(facts, "own", "a");
    expect(alpha.times_attacked).toBe(0);
    expect(alpha.hurt_this_turn).toBe(false);
    expect(mon(facts, "opponent", "b").hurt_this_turn).toBe(false);
    expect(mon(facts, "own", "b").times_attacked).toBe(1);
    expect(mon(facts, "own", "b").hurt_this_turn).toBe(true);
    expect(mon(facts, "own", "b").move_actions).toBe(1);
  });

  it("a Transform copies the target's count and clears the own moves used", () => {
    const facts = run([
      ...HEAD,
      request([{ id: "transform", maxpp: 10 }]),
      ...TURN(1),
      ...hit("p1b: Beta"),
      "|move|p1a: Alpha|Transform|p1b: Beta",
      "|-transform|p1a: Alpha|p1b: Beta|[from] move: Transform",
    ]);
    expect(mon(facts, "own", "a").times_attacked).toBe(1);
    expect(mon(facts, "own", "a").moves_used).toEqual([]);
    expect(code(() => run([...HEAD, "|-transform|p1a: Alpha|p2: Nobody"]))).toBe("unattributable_line");
  });
});

describe("the last move's result", () => {
  const result = (action: string[]): boolean | null => mon(run([...HEAD, ...TURN(1), ...action, ...END, ...TURN(2)]), "own", "a").last_move_failed;

  it("is failed after a cant, a miss or an immunity and not after a hit or a self-targeted success", () => {
    expect(result(["|cant|p1a: Alpha|par"])).toBe(true);
    expect(result(["|cant|p1a: Alpha|recharge"])).toBe(false);
    expect(result(["|move|p1a: Alpha|Tackle|p2a: Gamma|[miss]", "|-miss|p1a: Alpha|p2a: Gamma"])).toBe(true);
    expect(result(["|move|p1a: Alpha|Thunder Wave|p2a: Gamma", "|-immune|p2a: Gamma"])).toBe(true);
    expect(result(["|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|50/100"])).toBe(false);
    expect(result(["|move|p1a: Alpha|Swords Dance|p1a: Alpha", "|-boost|p1a: Alpha|atk|2"])).toBe(false);
    expect(result(["|-activate|p1a: Alpha|confusion", "|-damage|p1a: Alpha|90/100|[from] confusion"])).toBe(true);
  });

  it("is null when the lines cannot settle it", () => {
    expect(result(["|move|p1a: Alpha|Growl|p2a: Gamma", "|-fail|p1a: Alpha"])).toBeNull();
    expect(result(["|move|p1a: Alpha|Growl|p2a: Gamma", "|-unboost|p2a: Gamma|atk|0"])).toBeNull();
    expect(result(["|move|p1a: Alpha|Growl|p2a: Gamma"])).toBeNull();
    expect(result(["|move|p1a: Alpha|Metronome|p1a: Alpha", "|move|p1a: Alpha|Tackle|p2a: Gamma|[from] move: Metronome"])).toBeNull();
    // a reflected move overwrites the reflector's own last result
    const reflected = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|90/100", "|move|p2b: Delta|Growl|p1a: Alpha", "|move|p1a: Alpha|Growl|p2b: Delta|[from] ability: Magic Bounce", ...END, ...TURN(2)]);
    expect(mon(reflected, "own", "a").last_move_failed).toBeNull();
  });

  it("is reset by a switch and a Pokemon that did not act did not fail", () => {
    expect(result([])).toBe(false);
  });
});

describe("stat changes this turn", () => {
  it("flags a boost and an unboost, resets at a turn but not at turn 1, and at a switch", () => {
    const facts = run([...HEAD, "|-unboost|p1a: Alpha|atk|1", ...TURN(1), "|-boost|p1b: Beta|spe|1"]);
    expect(mon(facts, "own", "a").stats_lowered_this_turn).toBe(true);
    expect(mon(facts, "own", "b").stats_raised_this_turn).toBe(true);
    const later = run([...HEAD, "|-unboost|p1a: Alpha|atk|1", ...TURN(1), ...END, ...TURN(2)]);
    expect(mon(later, "own", "a").stats_lowered_this_turn).toBe(false);
    const out = run([...HEAD, "|-unboost|p1a: Alpha|atk|1", "|switch|p1a: Alpha|Alpha, L50|100/100"]);
    expect(mon(out, "own", "a").stats_lowered_this_turn).toBe(false);
    expect(run([...HEAD, "|-boost|p2: Nobody|atk|1"]).turn).toBe(0);
  });
});

describe("berries and the last item", () => {
  const roster = (lines: string[], side: "own" | "opponent" = "own", name = "Alpha") => run([...HEAD, ...TURN(1), ...lines])[side].roster[name];

  it("an eaten berry sets both facts, a plain use or a gem only the last item", () => {
    expect(roster(["|-enditem|p1a: Alpha|Sitrus Berry|[eat]"])).toEqual({ ate_berry: true, last_item: "sitrusberry" });
    expect(roster(["|-enditem|p1a: Alpha|Chilan Berry|[weaken]"])).toEqual({ ate_berry: true, last_item: "chilanberry" });
    expect(roster(["|-enditem|p1a: Alpha|White Herb"])).toEqual({ ate_berry: false, last_item: "whiteherb" });
    expect(roster(["|-enditem|p1a: Alpha|Normal Gem|[from] gem"])).toEqual({ ate_berry: false, last_item: "normalgem" });
    expect(roster(["|-enditem|p1a: Alpha|Air Balloon"])).toEqual({ ate_berry: false, last_item: null });
  });

  it("a theft or a removal sets neither, a stolen berry is eaten by the thief, a thrown item is the thrower's last item", () => {
    expect(roster(["|-enditem|p1a: Alpha|Leftovers|[from] move: Knock Off|[of] p2a: Gamma"])).toEqual({ ate_berry: false, last_item: null });
    expect(roster(["|-enditem|p1a: Alpha|Oran Berry|[from] stealeat|[move] Bug Bite|[of] p2a: Gamma"], "opponent", "Gamma")).toEqual({ ate_berry: true, last_item: null });
    expect(roster(["|-enditem|p1a: Alpha|Oran Berry|[from] stealeat|[move] Bug Bite|[of] p3a: Nobody"])).toEqual({ ate_berry: false, last_item: null });
    const thrown = ["|move|p1a: Alpha|Fling|p2a: Gamma", "|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling"];
    expect(roster(thrown)).toEqual({ ate_berry: false, last_item: "oranberry" });
    expect(roster(thrown, "opponent", "Gamma")).toEqual({ ate_berry: true, last_item: null });
    expect(roster(["|move|p1a: Alpha|Fling|p2a: Gamma", "|-enditem|p1a: Alpha|Leppa Berry|[from] move: Fling"], "opponent", "Gamma")).toEqual({ ate_berry: false, last_item: null });
    expect(roster(["|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling"], "opponent", "Gamma")).toEqual({ ate_berry: false, last_item: null });
    expect(roster(["|move|p1a: Alpha|Fling|p2: Fainted", "|-enditem|p1a: Alpha|Oran Berry|[from] move: Fling"], "opponent", "Gamma")).toEqual({ ate_berry: false, last_item: null });
  });

  it("Recycle and Harvest give the item back (the last item is cleared); Pickup takes an unnamed Pokemon's", () => {
    expect(roster(["|-enditem|p1a: Alpha|Sitrus Berry|[eat]", "|-item|p1a: Alpha|Sitrus Berry|[from] move: Recycle"])).toEqual({ ate_berry: true, last_item: null });
    expect(roster(["|-item|p1a: Alpha|Sitrus Berry|[from] ability: Static"])).toEqual({ ate_berry: false, last_item: null });
    expect(roster(["|-item|p1a: Alpha|Sitrus Berry"])).toEqual({ ate_berry: false, last_item: null });
    const taken = ["|-enditem|p1a: Alpha|Sitrus Berry|[eat]", "|-item|p2a: Gamma|Sitrus Berry|[from] ability: Pickup"];
    expect(roster(taken)?.last_item).toBe(UNSETTLED);
    expect(roster(["|-enditem|p1a: Alpha|Sitrus Berry|[eat]", "|-item|p2a: Gamma|Lum Berry|[from] ability: Pickup"])?.last_item).toBe("sitrusberry");
  });
});

describe("own moves used", () => {
  it("lists the own moves the own move lines used, from the request's PP-carrying moves, by position", () => {
    const facts = run([
      ...HEAD,
      REQUEST,
      ...TURN(1),
      "|move|p1a: Alpha|Tackle|p2a: Gamma",
      "|move|p1b: Beta|Protect|p1b: Beta",
      "|-singleturn|p1b: Beta|Protect",
      "|move|p2a: Gamma|Tackle|p1a: Alpha",
      "|move|p1b: Beta|Struggle|p2a: Gamma",
      "|move|p1b: Beta|Round|p2a: Gamma|[from] move: Round",
      ...END,
    ]);
    expect(mon(facts, "own", "a").moves_used).toEqual(["tackle"]);
    expect(mon(facts, "own", "b").moves_used).toEqual(["protect"]);
    expect(mon(facts, "opponent", "a").moves_used).toBeNull();
  });

  it("a PP-deducting action source counts, an external one does not, and a switch or a swap moves the request with the position", () => {
    const own = request([{ id: "round", maxpp: 10 }]);
    const used = (lines: string[]): string[] => mon(run([...HEAD, own, ...TURN(1), ...lines]), "own", "a").moves_used as string[];
    expect(used(["|move|p1a: Alpha|Round|p2a: Gamma|[from] move: Round"])).toEqual(["round"]);
    expect(used(["|move|p1a: Alpha|Round|p2a: Gamma|[from] ability: Dancer"])).toEqual([]);
    const swapped = run([...HEAD, request([{ id: "tackle", maxpp: 10 }], [{ id: "protect", maxpp: 10 }]), ...TURN(1), "|swap|p1a: Alpha|1", "|move|p1a: Beta|Protect|p1a: Beta", "|-singleturn|p1a: Beta|Protect"]);
    expect(mon(swapped, "own", "a").moves_used).toEqual(["protect"]);
    expect(mon(swapped, "own", "b").moves_used).toEqual([]);
    const half = run([...HEAD, request([{ id: "tackle", maxpp: 10 }], []), ...TURN(1), "|swap|p1a: Alpha|1", "|swap|p1a: Beta|1"]);
    expect(mon(half, "own", "a").moves_used).toEqual([]);
    const gone = run([...HEAD, own, ...TURN(1), "|switch|p1a: Alpha|Alpha, L50|100/100", "|move|p1a: Alpha|Round|p2a: Gamma"]);
    expect(mon(gone, "own", "a").moves_used).toEqual([]);
    const noList = run([...HEAD, "|request|{}", "|request|not json", "|request|null", `|request|${JSON.stringify({ active: [{}, { moves: [{ id: 3 }] }] })}`, ...TURN(1), "|move|p1a: Alpha|Tackle|p2a: Gamma"]);
    expect(mon(noList, "own", "a").moves_used).toEqual([]);
  });
});

describe("faints, the field's last move, pending slots and volatiles", () => {
  it("counts a side's faints and keeps the position empty", () => {
    const facts = run([...HEAD, ...TURN(1), "|faint|p2a: Gamma", "|faint|p1b: Beta"]);
    expect(facts.opponent.total_fainted).toBe(1);
    expect(facts.own.total_fainted).toBe(1);
    expect(facts.opponent.active.a).toBeNull();
    expect(facts.field_last_move).toBeNull();
    expect(code(() => run([...HEAD, "|faint|p2c: Nobody"]))).toBe("unattributable_line");
  });

  it("holds the latest move line's id as the field's last move, whatever its tag", () => {
    expect(run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Tackle|p2a: Gamma", "|move|p2a: Gamma|Growl|p1a: Alpha|[from] move: Metronome"]).field_last_move).toBe("growl");
  });

  it("a Wish lasts two turn residuals from its start and a Future Sight three; a failure cancels it; a landing is a hit of its user", () => {
    const wish = [...HEAD, ...TURN(1), "|move|p1a: Alpha|Wish|p1a: Alpha", ...END];
    expect(run([...wish, ...TURN(2)]).own.pending_slot).toEqual({ a: { wish: { started_turn: 1, after_residual: false, remaining: 1 } } });
    expect(run([...wish, ...TURN(2), ...END]).own.pending_slot).toEqual({});
    expect(run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Wish|p1a: Alpha", "|-fail|p1a: Alpha"]).own.pending_slot).toEqual({});
    expect(run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Wish|p1a: Alpha", "|-fail|p1b: Beta"]).own.pending_slot).toEqual({});
    const future = [...HEAD, ...TURN(1), "|move|p1a: Alpha|Future Sight|p2a: Gamma", "|-start|p1a: Alpha|move: Future Sight", ...END];
    expect(run([...future, ...TURN(2), ...END, ...TURN(3)]).opponent.pending_slot.a?.futuremove?.remaining).toBe(1);
    const landed = run([...future, ...TURN(2), ...END, ...TURN(3), "|", "|-end|p2a: Gamma|move: Future Sight", "|-damage|p2a: Gamma|60/100", ...END, ...TURN(4)]);
    expect(mon(landed, "opponent", "a").times_attacked).toBe(1);
    expect(mon(landed, "opponent", "a").attackers_this_turn).toEqual([]);
    expect(landed.opponent.pending_slot).toEqual({});
    const midTurn = run([...future, ...TURN(2), ...END, ...TURN(3), "|", "|-end|p2a: Gamma|move: Future Sight", "|-damage|p2a: Gamma|60/100"]);
    expect(mon(midTurn, "opponent", "a").attackers_this_turn).toEqual(["own:a"]);
    // the user left play: the attacker's position is not the channel's to place
    const left = run([...future, ...TURN(2), "|switch|p1a: Zeta|Zeta, L50|100/100", ...END, ...TURN(3), "|", "|-end|p2a: Gamma|move: Future Sight", "|-damage|p2a: Gamma|60/100"]);
    expect(mon(left, "opponent", "a").attackers_this_turn).toBeNull();
    expect(mon(left, "opponent", "a").times_attacked).toBe(1);
    expect(run([...HEAD, ...TURN(1), "|-end|p2a: Gamma|move: Future Sight"]).turn).toBe(1);
    expect(run([...HEAD, ...TURN(1), "|-end|p2a: Gamma|Stockpile"]).turn).toBe(1);
    // a target named without a position (fainted) is placed by where it last stood; a second use after the first lapsed starts again
    const fainted = run([...HEAD, ...TURN(1), "|faint|p2a: Gamma", "|move|p1a: Alpha|Future Sight|p2: Gamma", ...END]);
    expect(Object.keys(fainted.opponent.pending_slot)).toEqual(["a"]);
    const again = run([...wish, ...TURN(2), ...END, ...TURN(3), "|move|p1a: Alpha|Wish|p1a: Alpha", ...END, ...TURN(4)]);
    expect(again.own.pending_slot.a?.wish?.started_turn).toBe(3);
    const early = run([...HEAD, "|move|p1a: Alpha|Wish|p1a: Alpha", ...TURN(1)]);
    expect(early.own.pending_slot.a?.wish).toEqual({ started_turn: 0, after_residual: true, remaining: 2 });
  });

  it("dates volatiles from their start line and forgets them at the end", () => {
    const facts = run([...HEAD, ...TURN(1), "|-start|p1a: Alpha|move: Taunt", ...END, ...TURN(2), "|-start|p1a: Alpha|confusion", "|-end|p1a: Alpha|confusion", "|-start|p1b: Beta|typechange|Fire"]);
    expect(mon(facts, "own", "a").volatile_started).toEqual({ taunt: { started_turn: 1, after_residual: false } });
    expect(mon(facts, "own", "b").volatile_started.typechange).toEqual({ started_turn: 2, after_residual: false });
    expect(code(() => run([...HEAD, "|-start|p1a: Alpha|"]))).toBe("malformed_line");
    expect(run([...HEAD, "|-start|p1c: Nobody|confusion", "|-end|p1c: Nobody|confusion"]).turn).toBe(0);
    expect(mon(run([...HEAD, "|-start|p1a: Alpha|move: Taunt", "|-start|p1a: Alpha|move: Taunt"]), "own", "a").volatile_started.taunt?.after_residual).toBe(true);
  });
});

describe("an Illusion", () => {
  it("makes the Pokemon at its position unknown from the replace until it leaves play", () => {
    const lines = [...HEAD, ...TURN(1), "|move|p2a: Gamma|Tackle|p1a: Alpha", "|-damage|p1a: Alpha|90/100", "|replace|p2a: Zorua|Zorua, L50", "|-end|p2a: Zorua|Illusion"];
    const facts = run(lines);
    expect(facts.opponent.active.a).toEqual({ name: "Zorua", unknown: true });
    expect(facts.opponent.roster.Gamma).toEqual({ ate_berry: null, last_item: null });
    expect(facts.opponent.roster.Zorua).toEqual({ ate_berry: null, last_item: null });
    const out = run([...lines, "|", "|upkeep", ...TURN(2), "|switch|p2a: Gamma|Gamma, L50|100/100"]);
    expect(mon(out, "opponent", "a").name).toBe("Gamma");
    expect(code(() => run([...HEAD, "|replace|p2c: Nobody|Nobody, L50"]))).toBe("unattributable_line");
    expect(code(() => run([...HEAD, "|replace|p2a: Zorua|Zorua, L50"].slice(0, 6).concat(["|replace|p2a: Zorua|Zorua, L50"])))).toBe("none");
  });
});

describe("refusals", () => {
  it("a line it cannot place is a typed refusal that quotes nothing, sticky from that line on", () => {
    expect(code(() => run([...HEAD, "|move|p3a: Nobody|Tackle|p2a: Gamma"]))).toBe("unattributable_line");
    expect(code(() => run([...HEAD, "|move|p1a: Alpha||p2a: Gamma"]))).toBe("malformed_line");
    expect(code(() => run([...HEAD, "|switch|p1: Alpha|Alpha, L50|100/100"]))).toBe("unattributable_line");
    expect(code(() => run([...HEAD, "|swap|p1a: Alpha|7"]))).toBe("unattributable_line");
    expect(code(() => run([...HEAD, "|turn|x"]))).toBe("malformed_line");
    expect(code(() => run([...HEAD, ...TURN(3), "|turn|2"]))).toBe("malformed_line");
    expect(code(() => run([...HEAD, "|-enditem|nobody|Leftovers"]))).toBe("unattributable_line");
    const reader = new HistoryFactsReader("p1", data);
    for (const line of HEAD) reader.feed(line);
    reader.feed("|faint|p2c: Nobody");
    reader.feed("|move|p1a: Alpha|Tackle|p2a: Gamma");
    const error = (() => {
      try {
        reader.facts();
      } catch (e) {
        return e as HistoryFactsError;
      }
      return null;
    })();
    expect(error?.message).toBe(`history-facts: unattributable_line at line ${HEAD.length}`);
    expect(error?.line).toBe(HEAD.length);
    expect(new HistoryFactsError("bad_player", null).message).toBe("history-facts: bad_player");
  });

  it("reads the same channel from the other seat", () => {
    const facts = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Tackle|p2a: Gamma", "|-damage|p2a: Gamma|50/100"], "p2");
    expect(mon(facts, "own", "a").times_attacked).toBe(1);
    expect(mon(facts, "opponent", "a").move_actions).toBe(1);
  });
});

describe("odd input", () => {
  const bare = (line: string): string => {
    const { ally_side_moves: _ally, external_action_sources: _external, ...required } = data;
    const reader = new HistoryFactsReader("p1", required);
    for (const l of HEAD) reader.feed(l);
    reader.feed(line);
    reader.feed("|turn|1"); // a refusal is sticky, a fine line goes on
    try {
      reader.facts();
      return "none";
    } catch (error) {
      return (error as HistoryFactsError).code;
    }
  };

  it("takes lines with missing fields without a throw: they are no-ops or typed refusals", () => {
    expect(bare("nonsense")).toBe("malformed_line");
    for (const line of ["|request", "|cant", "|-fail", "|-notarget", "|-miss", "|-immune", "|-boost", "|-unboost", "|-item", "|-transform", "|-activate", "|-damage", "|upkeep", "|win|Ann", "|-weather|none"]) {
      expect(["none", "unattributable_line"], line).toContain(bare(line));
    }
    for (const line of ["|switch", "|swap", "|replace", "|faint", "|move", "|-enditem", "|-start", "|-end"]) {
      expect(["malformed_line", "unattributable_line"], line).toContain(bare(line));
    }
    expect(bare("|turn")).toBe("malformed_line");
  });

  it("a condition id outside the data has no span, and an action named without a target still counts", () => {
    const facts = run([...HEAD, ...TURN(1), "|move|p1a: Alpha|Wish", "|move|p1b: Beta|Tackle|", "|-boost|p1a: Alpha|atk|0", "|move|p1a: Alpha|Growl|p1a: Alpha", "|-unboost|p1a: Alpha|atk|0"]);
    expect(mon(facts, "own", "a").move_actions).toBe(2);
    expect(facts.own.pending_slot).toEqual({});
  });
});
