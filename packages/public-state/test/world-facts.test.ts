import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  INTERPRETER_PATH,
  WORLD_FACTS_FORMAT,
  WORLD_FACTS_PATH,
  WorldFactsError,
  WorldFactsReader,
  type WorldFactsData,
  ownChannel,
  readerSha256,
  worldFacts,
  worldFactsAtRequests,
  worldFactsSha256,
} from "../src/index.ts";

const data: WorldFactsData = {
  stall_moves: ["protect", "detect", "endure", "quickguard", "spikyshield"],
  terrains: ["electricterrain", "grassyterrain"],
  swap_side_conditions: ["reflect", "stealthrock", "tailwind"],
  toxic_stage_max: 15,
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

const run = (player: "p1" | "p2", lines: string[]) => worldFacts({ channel: ownChannel(player, lines), data });
const code = (fn: () => unknown): string => {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(WorldFactsError);
    return (error as WorldFactsError).code;
  }
  return "none";
};

describe("identity", () => {
  it("has a format id and a file hash of its own, and leaves the reader's bytes alone", () => {
    expect(WORLD_FACTS_FORMAT).toBe("world-facts/1");
    expect(worldFactsSha256()).toBe(createHash("sha256").update(readFileSync(WORLD_FACTS_PATH)).digest("hex"));
    expect(worldFactsSha256()).not.toBe(readerSha256());
    expect(readerSha256()).toBe(createHash("sha256").update(readFileSync(INTERPRETER_PATH)).digest("hex"));
  });

  it("imports nothing", () => {
    const source = readFileSync(WORLD_FACTS_PATH, "utf8");
    expect(source).not.toMatch(/^\s*import\s/m);
    expect(source).not.toMatch(/require\(/);
  });
});

describe("the input", () => {
  it("refuses a bad player, a line that is no protocol line and a split marker", () => {
    expect(code(() => ownChannel("p3", []))).toBe("bad_player");
    expect(code(() => ownChannel("p1", ["switch"]))).toBe("malformed_line");
    expect(code(() => ownChannel("p1", ["|split|p1"]))).toBe("secret_marker");
    expect(code(() => ownChannel("p1", ["|split"]))).toBe("secret_marker");
    expect(ownChannel("p2", HEAD).player).toBe("p2");
  });

  it("is typed so that an omniscient log, a team or a seed cannot be passed", () => {
    const neverRun = (omniscient: string[] & { readonly __brand: "OmniscientLog" }): void => {
      // @ts-expect-error: the omniscient log is not an OwnChannel
      void worldFacts({ channel: omniscient, data });
      // @ts-expect-error: nor is a bare array of lines
      void worldFacts({ channel: HEAD, data });
      // @ts-expect-error: no team may be passed
      void worldFacts({ channel: ownChannel("p1", HEAD), data, team: [] });
      // @ts-expect-error: no seed may be passed
      void worldFacts({ channel: ownChannel("p1", HEAD), data, seed: "sodium,00" });
    };
    expect(typeof neverRun).toBe("function");
  });
});

describe("turns since switch-in", () => {
  it("counts the turn lines since the switch, from either seat", () => {
    const lines = [...HEAD, "|turn|1", "|", "|switch|p1b: Zed|Zed, L50|100/100", "|upkeep", "|turn|2", "|upkeep", "|turn|3"];
    const own = run("p1", lines);
    expect(own.turn).toBe(3);
    expect(own.own.active.a?.active_turns).toBe(3);
    expect(own.own.active.b).toMatchObject({ name: "Zed", active_turns: 2 });
    expect(run("p2", lines).opponent.active.b).toMatchObject({ name: "Zed", active_turns: 2 });
    expect(run("p1", HEAD).own.active.a?.active_turns).toBe(0);
  });

  it("follows a swap of positions and a drag", () => {
    const lines = [...HEAD, "|turn|1", "|turn|2", "|swap|p1a: Alpha|1|[from] move: Ally Switch", "|drag|p2a: Eps|Eps, L50|100/100", "|turn|3"];
    const f = run("p1", lines);
    expect(f.own.active.a).toMatchObject({ name: "Beta", active_turns: 3 });
    expect(f.own.active.b).toMatchObject({ name: "Alpha", active_turns: 3 });
    expect(f.opponent.active.a).toMatchObject({ name: "Eps", active_turns: 1 });
  });

  it("refuses a swap with an unknown position and a switch with no position", () => {
    expect(code(() => run("p1", [...HEAD, "|swap|p1a: Alpha|5"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|swap|p1a: Alpha|x"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|swap|p1: Alpha|1"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|switch|p1: Alpha|Alpha, L50|100/100"]))).toBe("unattributable_line");
  });
});

describe("last move", () => {
  it("is the last move the Pokemon chose, not one another effect made it use", () => {
    const lines = [
      ...HEAD,
      "|turn|1",
      "|move|p1a: Alpha|Thunder Wave|p2a: Gamma",
      "|move|p1a: Alpha|Metronome|p1a: Alpha",
      "|move|p1a: Alpha|Tackle|p2a: Gamma|[from] move: Metronome",
      "|move|p2a: Gamma|Thunder Wave|p1a: Alpha|[from] ability: Magic Bounce",
      "|move|p2b: Delta|Outrage|p1b: Beta",
      "|move|p2b: Delta|Outrage|p1b: Beta|[from]lockedmove",
    ];
    const f = run("p1", lines);
    expect(f.own.active.a?.last_move).toBe("metronome");
    expect(f.opponent.active.a?.last_move).toBeNull();
    expect(f.opponent.active.b?.last_move).toBe("outrage");
  });

  it("resets on a switch-in and refuses a move from nowhere", () => {
    const lines = [...HEAD, "|move|p1a: Alpha|Tackle|p2a: Gamma", "|switch|p1a: Zed|Zed, L50|100/100"];
    expect(run("p1", lines).own.active.a?.last_move).toBeNull();
    expect(code(() => run("p1", ["|move|p1a: Alpha|Tackle|p2a: Gamma"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|move|p1: Alpha|Tackle|p2a: Gamma"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|move|p1a: Alpha||p2a: Gamma"]))).toBe("malformed_line");
  });
});

describe("Protect streak", () => {
  const use = (t: number) => [`|turn|${t}`, "|move|p1a: Alpha|Protect|p1a: Alpha", "|-singleturn|p1a: Alpha|Protect", "|upkeep"];

  it("counts consecutive successful uses", () => {
    expect(run("p1", [...HEAD, ...use(1)]).own.active.a?.protect_streak).toBe(1);
    expect(run("p1", [...HEAD, ...use(1), ...use(2), ...use(3)]).own.active.a?.protect_streak).toBe(3);
    expect(run("p1", [...HEAD, ...use(1), ...use(2), "|turn|3"]).own.active.a?.protect_streak).toBe(2);
  });

  it("ends when a turn passes without a use, on a failed use, on a switch and on a faint", () => {
    expect(run("p1", [...HEAD, ...use(1), "|turn|2", "|turn|3"]).own.active.a?.protect_streak).toBe(0);
    expect(run("p1", [...HEAD, ...use(1), ...use(3)]).own.active.a?.protect_streak).toBe(1);
    const failed = [...HEAD, ...use(1), "|turn|2", "|move|p1a: Alpha|Protect|p1a: Alpha", "|-fail|p1a: Alpha", "|upkeep", "|turn|3"];
    expect(run("p1", failed).own.active.a?.protect_streak).toBe(0);
    expect(run("p1", [...HEAD, ...use(1), "|switch|p1a: Zed|Zed, L50|100/100"]).own.active.a?.protect_streak).toBe(0);
  });

  it("reads the opponent and the other stall moves (Endure, Quick Guard)", () => {
    const lines = [...HEAD, "|turn|1", "|move|p2a: Gamma|Quick Guard|p2a: Gamma", "|-singleturn|p2a: Gamma|Quick Guard", "|move|p2b: Delta|Endure|p2b: Delta", "|-singleturn|p2b: Delta|move: Endure"];
    const f = run("p1", lines);
    expect(f.opponent.active.a?.protect_streak).toBe(1);
    expect(f.opponent.active.b?.protect_streak).toBe(1);
  });

  it("refuses a stall move whose outcome is not shown", () => {
    const open = [...HEAD, "|turn|1", "|move|p1a: Alpha|Protect|p1a: Alpha"];
    expect(code(() => run("p1", open))).toBe("stall_unresolved");
    expect(code(() => run("p1", [...open, "|move|p1b: Beta|Tackle|p2a: Gamma"]))).toBe("stall_unresolved");
    // another Pokemon's singleturn line, and a line of a kind that is no action, do not resolve or end it
    expect(run("p1", [...open, "|-activate|p1a: Alpha|something", "|-singleturn|p1a: Alpha|Protect"]).own.active.a?.protect_streak).toBe(1);
    expect(code(() => run("p1", [...open, "|-singleturn|p1b: Beta|Protect"]))).toBe("stall_unresolved");
  });
});

describe("faints", () => {
  it("counts per side and says whether the side lost one this or last turn", () => {
    const lines = [...HEAD, "|turn|1", "|faint|p2a: Gamma", "|upkeep", "|switch|p2a: Eps|Eps, L50|100/100", "|turn|2", "|faint|p1b: Beta"];
    const f = run("p1", lines);
    expect(f.own).toMatchObject({ total_fainted: 1, fainted_this_turn: true, fainted_last_turn: false });
    expect(f.opponent).toMatchObject({ total_fainted: 1, fainted_this_turn: false, fainted_last_turn: true });
    expect(f.own.active.b).toBeNull();
    expect(code(() => run("p1", [...HEAD, "|faint|p1: Beta"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|faint|p1c: Beta"]))).toBe("unattributable_line");
  });
});

describe("conditions", () => {
  it("records the start turn of weather, terrain, field and side conditions and counts layers", () => {
    const lines = [
      ...HEAD,
      "|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Gamma",
      "|turn|1",
      "|-weather|SunnyDay|[upkeep]",
      "|-fieldstart|move: Electric Terrain",
      "|-fieldstart|move: Trick Room|[of] p1a: Alpha",
      "|-sidestart|p2: Bob|move: Spikes",
      "|turn|2",
      "|-sidestart|p2: Bob|Spikes",
      "|-sidestart|p1: Ann|move: Tailwind",
      "|turn|3",
    ];
    const f = run("p1", lines);
    expect(f.weather).toEqual({ id: "sunnyday", started_turn: 0 });
    expect(f.terrain).toEqual({ id: "electricterrain", started_turn: 1 });
    expect(f.field_conditions).toEqual({ trickroom: { started_turn: 1 } });
    expect(f.opponent.conditions).toEqual({ spikes: { started_turn: 1, layers: 2 } });
    expect(f.own.conditions).toEqual({ tailwind: { started_turn: 2, layers: 1 } });
    expect(f.world_facts_format).toBe(WORLD_FACTS_FORMAT);
  });

  it("ends them and swaps the side conditions", () => {
    const lines = [
      ...HEAD,
      "|-weather|RainDance",
      "|-fieldstart|move: Grassy Terrain",
      "|-fieldstart|move: Gravity",
      "|-sidestart|p1: Ann|move: Reflect",
      "|-sidestart|p2: Bob|move: Stealth Rock",
    ];
    const swapped = run("p1", [...lines, "|-sidestart|p1: Ann|Mystery", "|-swapsideconditions"]);
    expect(Object.keys(swapped.own.conditions)).toEqual(["mystery", "stealthrock"]);
    expect(Object.keys(swapped.opponent.conditions)).toEqual(["reflect"]);
    const ended = [...lines, "|-weather|none", "|-fieldend|move: Gravity", "|-fieldend|move: Electric Terrain", "|-sideend|p1: Ann|Reflect|[from] move: Defog"];
    const f = run("p1", ended);
    expect(f.weather).toBeNull();
    expect(f.terrain).toEqual({ id: "grassyterrain", started_turn: 0 });
    expect(f.field_conditions).toEqual({});
    expect(f.own.conditions).toEqual({});
    expect(run("p1", [...ended, "|-fieldend|move: Grassy Terrain"]).terrain).toBeNull();
  });

  it("refuses a condition line with no name or a side it cannot attribute", () => {
    expect(code(() => run("p1", [...HEAD, "|-fieldstart|"]))).toBe("malformed_line");
    expect(code(() => run("p1", [...HEAD, "|-sidestart|p1: Ann|"]))).toBe("malformed_line");
    expect(code(() => run("p1", [...HEAD, "|-sidestart|Ann|Spikes"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|-sideend|p3: Ann|Spikes"]))).toBe("unattributable_line");
    expect(code(() => run("p1", [...HEAD, "|turn|x"]))).toBe("malformed_line");
    expect(code(() => run("p1", [...HEAD, "|turn|2", "|turn|1"]))).toBe("malformed_line");
  });
});

describe("Toxic stage", () => {
  const toxic = [...HEAD, "|turn|1", "|-status|p1a: Alpha|tox"];
  const tick = (turn: number) => ["|", "|-damage|p1a: Alpha|90/100|[from] psn", "|upkeep", `|turn|${turn}`];

  it("counts the residual lines since the status", () => {
    expect(run("p1", toxic).own.active.a?.toxic_ticks).toBe(0);
    expect(run("p1", [...toxic, ...tick(2), ...tick(3)]).own.active.a?.toxic_ticks).toBe(2);
    expect(run("p1", [...toxic, ...tick(2)]).own.active.b?.toxic_ticks).toBeNull();
    expect(run("p1", [...HEAD, "|-status|p1a: Alpha|brn", "|-damage|p1a: Alpha|90/100|[from] brn"]).own.active.a?.toxic_ticks).toBeNull();
  });

  it("stops at the data's stage maximum", () => {
    const many = Array.from({ length: 17 }, (_, i) => tick(i + 2)).flat();
    expect(run("p1", [...toxic, ...many]).own.active.a?.toxic_ticks).toBe(15);
    const low = worldFacts({ channel: ownChannel("p1", [...toxic, ...many]), data: { ...data, toxic_stage_max: 3 } });
    expect(low.own.active.a?.toxic_ticks).toBe(3);
  });

  it("restarts on a switch-in and ends on a cure", () => {
    const back = [...toxic, ...tick(2), "|switch|p1a: Zed|Zed, L50|100/100", "|switch|p1a: Alpha|Alpha, L50|100/100"];
    expect(run("p1", back).own.active.a?.toxic_ticks).toBe(0);
    expect(run("p1", [...toxic, "|-curestatus|p1a: Alpha|tox"]).own.active.a?.toxic_ticks).toBeNull();
    expect(run("p1", [...toxic, "|-cureteam|p1a: Beta|[from] move: Heal Bell"]).own.active.a?.toxic_ticks).toBeNull();
    expect(run("p1", [...toxic, "|faint|p1a: Alpha"]).own.active.a).toBeNull();
  });

  it("counts a Poison Heal line and reads the opponent's status", () => {
    const heal = [...toxic, "|", "|-heal|p1a: Alpha|100/100|[from] ability: Poison Heal", "|upkeep"];
    expect(run("p1", heal).own.active.a?.toxic_ticks).toBe(1);
    const theirs = [...HEAD, "|-status|p2b: Delta|tox", "|", "|-damage|p2b: Delta|94/100|[from] psn", "|upkeep"];
    expect(run("p1", theirs).opponent.active.b?.toxic_ticks).toBe(1);
  });

  it("refuses when a tick was not shown for a Pokemon that had the status before the residual", () => {
    expect(code(() => run("p1", [...toxic, "|", "|upkeep"]))).toBe("toxic_tick_unobserved");
    // applied in the residual (an item): no tick is due that turn
    expect(run("p1", [...HEAD, "|turn|1", "|", "|-status|p1a: Alpha|tox|[from] item: Toxic Orb", "|upkeep"]).own.active.a?.toxic_ticks).toBe(0);
    // a Pokemon that fainted from the tick is not expected to tick again
    expect(run("p1", [...toxic, "|", "|-damage|p1a: Alpha|0 fnt|[from] psn", "|faint|p1a: Alpha", "|upkeep"]).own.active.a).toBeNull();
  });
});

describe("refusals", () => {
  it("refuses an Illusion replace, from that line on, and keeps the facts before it", () => {
    const lines = [...HEAD, "|turn|1", "|replace|p1a: Alpha|Zoroark, L50", "|turn|2"];
    expect(code(() => run("p1", lines))).toBe("illusion_replace");
    const found = worldFactsAtRequests({ channel: ownChannel("p1", [...HEAD, "|turn|1", "|request|{}", "|replace|p1a: Alpha|Zoroark, L50", "|turn|2", "|request|{}"]), data });
    expect(found.map((r) => [r.line, r.refusal])).toEqual([
      [8, null],
      [11, "illusion_replace"],
    ]);
    expect(found[0]?.facts?.turn).toBe(1);
    expect(found[1]?.facts).toBeNull();
  });

  it("keeps the first refusal and is thrown with a fixed message that quotes no line", () => {
    const reader = new WorldFactsReader("p1", data);
    for (const line of [...HEAD, "|faint|p1: Nobody", "|replace|p1a: Alpha|X"]) reader.feed(line);
    expect(() => reader.facts()).toThrow(/^world-facts: unattributable_line at line 7$/);
    const bare = new WorldFactsError("bad_player", null);
    expect(bare.message).toBe("world-facts: bad_player");
    expect(bare.line).toBeNull();
    const odd = new WorldFactsReader("p1", data);
    odd.feed("no bar");
    expect(() => odd.facts()).toThrow(WorldFactsError);
  });

  it("ignores lines it does not read, request and error lines among them", () => {
    const lines = [...HEAD, "|request|{}", "|error|[Unavailable choice] x", "|-boost|p1a: Alpha|atk|1", "|win|Ann"];
    expect(run("p1", lines).own.active.a?.name).toBe("Alpha");
  });
});
