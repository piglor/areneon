import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_DIR, GOLDEN_FORMAT, type RequestFixture } from "../src/golden.ts";
import {
  type Candidate,
  decodeRequest,
  enumerateCandidates,
  EnumerationError,
  type JsonObject,
  loadEngine,
} from "../src/index.ts";

/** A golden request, deep-copied so a test can edit it. */
function golden(file: string): JsonObject {
  const fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as RequestFixture;
  return structuredClone(fixture.request);
}

// Edits reach deep into engine JSON (`r.active[0].moves[1]`); typing every
// path would bury what each case changes.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Edit = (request: { [key: string]: any }) => void;

/** Decode a golden request, then edit what the enumerator sees. */
function enumerate(file: string, edit: Edit = () => {}): Candidate[] {
  const decoded = decodeRequest(JSON.stringify(golden(file)));
  edit(decoded.request);
  return enumerateCandidates(decoded).candidates;
}

const choices = (candidates: Candidate[]): string[] => candidates.map((c) => c.choice);
/** The slot-1 or slot-2 part of each joint choice, deduplicated in order. */
const part = (candidates: Candidate[], slot: 0 | 1): string[] => [
  ...new Set(candidates.map((c) => c.choice.split(", ")[slot] as string)),
];

describe("Team Preview", () => {
  it("offers every ordered pick of 4 from 6, lexicographically", () => {
    const candidates = enumerate("preview.request.json");
    expect(candidates).toHaveLength(360);
    expect(candidates.slice(0, 3).map((c) => c.choice)).toEqual([
      "team 1234",
      "team 1235",
      "team 1236",
    ]);
    expect(candidates.at(-1)?.choice).toBe("team 6543");
    expect(candidates.every((c) => !c.pending)).toBe(true);
    expect(new Set(choices(candidates)).size).toBe(360);
  });

  it("fails closed on teams of 10 or more, which the engine reads comma-separated", () => {
    expect(() =>
      enumerate("preview.request.json", (r) => {
        r.side.pokemon = [...r.side.pokemon, ...r.side.pokemon.slice(0, 4)];
      }),
    ).toThrow(EnumerationError);
  });

  it.each([0, 7, 2.5, "4", null])("fails closed on maxChosenTeamSize %j", (size) => {
    expect(() => enumerate("preview.request.json", (r) => (r.maxChosenTeamSize = size))).toThrow(
      EnumerationError,
    );
  });
});

describe("move requests", () => {
  // Golden turn 1: Venusaur (Giga Drain, Sludge Bomb, Protect, Sleep Powder)
  // and Charizard (Flamethrower, Air Slash, Protect, Heat Wave); bench 3, 4.
  it("orders moves, then targets 1, 2, ally, self, then switches", () => {
    const candidates = enumerate("turn.request.json");
    expect(part(candidates, 0)).toEqual([
      "move 1 1",
      "move 1 2",
      "move 1 -2",
      "move 2 1",
      "move 2 2",
      "move 2 -2",
      "move 3",
      "move 4 1",
      "move 4 2",
      "move 4 -2",
      "switch 3",
      "switch 4",
    ]);
    expect(part(candidates, 1)).toEqual([
      "move 1 1",
      "move 1 2",
      "move 1 -1",
      "move 2 1",
      "move 2 2",
      "move 2 -1",
      "move 3",
      "move 4",
      "switch 3",
      "switch 4",
    ]);
    // 12 x 10 pairs, minus the two that switch the same bench Pokémon in twice.
    expect(candidates).toHaveLength(118);
    expect(choices(candidates)).not.toContain("switch 3, switch 3");
    expect(choices(candidates)).toContain("switch 3, switch 4");
    expect(choices(candidates)[0]).toBe("move 1 1, move 1 1");
    expect(candidates.every((c) => !c.pending)).toBe(true);
  });

  // Every move target type of the engine (`MoveTarget`, sim/dex-moves.ts):
  // what the slot-1 Pokémon may name. Self is -1, its ally -2.
  const TARGETS: Record<string, string[]> = {
    normal: ["move 1 1", "move 1 2", "move 1 -2"],
    any: ["move 1 1", "move 1 2", "move 1 -2"],
    adjacentFoe: ["move 1 1", "move 1 2"],
    adjacentAlly: ["move 1 -2"],
    adjacentAllyOrSelf: ["move 1 -2", "move 1 -1"],
    ...Object.fromEntries(
      [
        "all",
        "allAdjacent",
        "allAdjacentFoes",
        "allies",
        "allySide",
        "allyTeam",
        "foeSide",
        "randomNormal",
        "scripted",
        "self",
      ].map((t) => [t, ["move 1"]]),
    ),
  };

  it.each(Object.entries(TARGETS))("target type %s", (target, expected) => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.active[0].moves = [{ ...r.active[0].moves[0], target }];
    });
    expect(part(candidates, 0).filter((c) => c.startsWith("move"))).toEqual(expected);
  });

  it("covers every target type in the format's move pool", () => {
    interface ModDex {
      mod(id: string): {
        moves: { all(): { target: string; isNonstandard: string | null }[] };
      };
    }
    const engine = loadEngine();
    const dex = engine.sim.Dex as unknown as ModDex;
    const { mod } = engine.sim.Dex.formats.get(GOLDEN_FORMAT);
    const pool = new Set(
      dex
        .mod(mod)
        .moves.all()
        .filter((m) => m.isNonstandard === null)
        .map((m) => m.target),
    );
    expect(pool.size).toBeGreaterThan(5);
    expect([...pool].filter((t) => !(t in TARGETS))).toEqual([]);
  });

  it("sends a locked move without a target", () => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.active[0].moves = [{ move: "Outrage", id: "outrage" }];
    });
    expect(part(candidates, 0)).toEqual(["move 1", "switch 3", "switch 4"]);
  });

  it("skips disabled moves", () => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.active[0].moves[0].disabled = true;
      r.active[0].moves[1].disabled = "Taunt";
    });
    expect(part(candidates, 0).filter((c) => c.startsWith("move"))).toEqual([
      "move 3",
      "move 4 1",
      "move 4 2",
      "move 4 -2",
    ]);
  });

  it("offers Mega Evolution after each plain move, never on both slots", () => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.active[0].canMegaEvo = true;
      r.active[1].canMegaEvoY = true;
    });
    expect(part(candidates, 0).slice(0, 3)).toEqual(["move 1 1", "move 1 1 mega", "move 1 2"]);
    expect(part(candidates, 1)).toContain("move 4 megay");
    expect(choices(candidates)).toContain("move 1 1 mega, move 1 1");
    expect(choices(candidates)).toContain("move 1 1, move 1 1 megay");
    expect(choices(candidates).filter((c) => /mega.*, .*mega/.test(c))).toEqual([]);
  });

  it("orders events none, mega, megax, megay, terastallize; one Tera per side", () => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.active[0].moves = [r.active[0].moves[2]];
      Object.assign(r.active[0], {
        canMegaEvo: true,
        canMegaEvoX: true,
        canMegaEvoY: true,
      });
      r.active[0].canTerastallize = "Grass";
      r.active[1].canTerastallize = "Fire";
    });
    expect(part(candidates, 0)).toEqual([
      "move 1",
      "move 1 mega",
      "move 1 megax",
      "move 1 megay",
      "move 1 terastallize",
      "switch 3",
      "switch 4",
    ]);
    expect(choices(candidates).filter((c) => /terastallize.*, .*terastallize/.test(c))).toEqual([]);
    expect(choices(candidates)).toContain("move 1 mega, move 1 1 terastallize");
  });

  it("offers no switch out of a visibly trapped slot", () => {
    const candidates = enumerate("turn.request.json", (r) => (r.active[0].trapped = true));
    expect(part(candidates, 0).filter((c) => c.startsWith("switch"))).toEqual([]);
    expect(part(candidates, 1)).toContain("switch 3");
  });

  it("passes for a fainted active with no replacement", () => {
    const candidates = enumerate("turn.request.json", (r) => {
      r.side.pokemon[0].condition = "0 fnt";
      r.side.pokemon[2].condition = "0 fnt";
      r.side.pokemon[3].condition = "0 fnt";
    });
    expect(part(candidates, 0)).toEqual(["pass"]);
    expect(part(candidates, 1)).toEqual([
      "move 1 1",
      "move 1 2",
      "move 1 -1",
      "move 2 1",
      "move 2 2",
      "move 2 -1",
      "move 3",
      "move 4",
    ]);
  });

  it("passes for a commanding Pokémon", () => {
    const candidates = enumerate("turn.request.json", (r) => (r.side.pokemon[1].commanding = true));
    expect(part(candidates, 1)).toEqual(["pass"]);
  });
});

describe("hidden information makes candidates pending (ADR-006 rule 5)", () => {
  it("maybeTrapped: switches of that slot only", () => {
    const candidates = enumerate("turn.request.json", (r) => (r.active[0].maybeTrapped = true));
    const pending = candidates.filter((c) => c.pending).map((c) => c.choice);
    expect(pending.length).toBeGreaterThan(0);
    expect(pending.every((c) => c.startsWith("switch"))).toBe(true);
    expect(choices(candidates)).toContain("switch 3, switch 4");
    expect(candidates.find((c) => c.choice === "move 1 1, switch 3")?.pending).toBe(false);
  });

  it.each(["maybeDisabled", "maybeLocked"])("%s: moves of that slot", (flag) => {
    const candidates = enumerate("turn.request.json", (r) => (r.active[1][flag] = true));
    for (const c of candidates) {
      expect(c.pending, c.choice).toBe(c.choice.split(", ")[1]?.startsWith("move"));
    }
  });
});

describe("forced switches", () => {
  // Golden forced switch: slot 1 fainted, slot 2 active, bench 3 and 4.
  it("switches the forced slot in and passes the other", () => {
    expect(choices(enumerate("force-switch.request.json"))).toEqual([
      "switch 3, pass",
      "switch 4, pass",
    ]);
  });

  it("two forced slots: two different bench Pokémon", () => {
    const candidates = enumerate("force-switch.request.json", (r) => {
      r.forceSwitch = [true, true];
      r.side.pokemon[1].condition = "0 fnt";
    });
    expect(choices(candidates)).toEqual(["switch 3, switch 4", "switch 4, switch 3"]);
  });

  it("fewer bench Pokémon than forced slots: one switches, one passes", () => {
    const candidates = enumerate("force-switch.request.json", (r) => {
      r.forceSwitch = [true, true];
      r.side.pokemon[1].condition = "0 fnt";
      r.side.pokemon[3].condition = "0 fnt";
    });
    expect(choices(candidates)).toEqual(["switch 3, pass", "pass, switch 3"]);
  });

  it.each([
    ["one slot", [true]],
    ["a non-boolean flag", [true, 1]],
  ])("fails closed on %s", (_label, flags) => {
    expect(() => enumerate("force-switch.request.json", (r) => (r.forceSwitch = flags))).toThrow(
      EnumerationError,
    );
  });
});

describe("Revival Blessing (the engine's `reviving` request)", () => {
  /** The golden forced switch, edited into a reviving request: Venusaur is the reviver. */
  const revive =
    (fainted: number[], flags = [true, false]): Edit =>
    (r) => {
      r.forceSwitch = flags;
      r.side.pokemon[0].condition = "50/159";
      r.side.pokemon[0].reviving = true;
      for (const i of fainted) r.side.pokemon[i].condition = "0 fnt";
    };

  it("offers every fainted Pokémon, never a living one, and no pass while a bench Pokémon lives", () => {
    const withLiving: Edit = (r) => {
      revive([2, 3])(r);
      r.side.pokemon.push({ ...r.side.pokemon[1], condition: "100/100" });
    };
    expect(choices(enumerate("force-switch.request.json", withLiving))).toEqual([
      "switch 3, pass",
      "switch 4, pass",
    ]);
  });

  it("includes an active slot's fainted Pokémon as a target", () => {
    expect(choices(enumerate("force-switch.request.json", revive([1, 3])))).toEqual([
      "switch 2, pass",
      "switch 4, pass",
    ]);
  });

  it("lets the slot pass only when the engine has more flagged slots than switch-ins", () => {
    // Bench has no living Pokémon: forcedSwitches 0, forcedPasses 1 (sim/side.ts clearChoice).
    expect(choices(enumerate("force-switch.request.json", revive([1, 2, 3])))).toEqual([
      "switch 2, pass",
      "switch 3, pass",
      "switch 4, pass",
      "pass, pass",
    ]);
    expect(choices(enumerate("force-switch.request.json", revive([2, 3])))).toContain("pass, pass");
  });

  it("marks no candidate pending: nothing about it is hidden", () => {
    const candidates = enumerate("force-switch.request.json", revive([2, 3]));
    expect(candidates.every((c) => !c.pending)).toBe(true);
  });

  it.each([
    ["a reviving Pokémon on a move request", "turn.request.json", (r: Parameters<Edit>[0]) => (r.side.pokemon[2].reviving = true)],
    [
      "a reviving Pokémon that is not in a flagged slot",
      "force-switch.request.json",
      (r: Parameters<Edit>[0]) => {
        revive([2])(r);
        r.side.pokemon[0].reviving = false;
        r.side.pokemon[2].reviving = true;
      },
    ],
    [
      "two reviving Pokémon",
      "force-switch.request.json",
      (r: Parameters<Edit>[0]) => {
        revive([3], [true, true])(r);
        r.side.pokemon[1].reviving = true;
      },
    ],
    [
      "a second reviving Pokémon in an unflagged slot",
      "force-switch.request.json",
      (r: Parameters<Edit>[0]) => {
        revive([3])(r);
        r.side.pokemon[1].reviving = true;
      },
    ],
    [
      "a reviver with no fainted Pokémon to revive",
      "force-switch.request.json",
      (r: Parameters<Edit>[0]) => {
        revive([])(r);
        r.side.pokemon[1].condition = "100/157";
      },
    ],
    [
      "a reviver next to a forced switch that is not a revival",
      "force-switch.request.json",
      (r: Parameters<Edit>[0]) => revive([2, 3], [true, true])(r),
    ],
  ])("fails closed on %s", (_label, file, edit) => {
    expect(() => enumerate(file, edit)).toThrow(EnumerationError);
  });
});

describe("wait", () => {
  it("offers nothing", () => {
    const request = golden("turn.request.json");
    delete request.active;
    request.wait = true;
    expect(enumerateCandidates(decodeRequest(JSON.stringify(request))).candidates).toEqual([]);
  });
});

describe("fails closed on what it does not understand (ADR-006 rule 3)", () => {
  it.each<[string, Edit]>([
    ["Dynamax", (r) => (r.active[0].canDynamax = true)],
    ["Z-Moves", (r) => (r.active[0].canZMove = [null])],
    ["Ultra Burst", (r) => (r.active[0].canUltraBurst = true)],
    ["Max Moves", (r) => (r.active[0].maxMoves = { maxMoves: [] })],
    ["a multi battle ally", (r) => (r.ally = r.side)],
    ["an unknown request field", (r) => (r.newField = 1)],
    ["an unknown side field", (r) => (r.side.newField = 1)],
    ["an unknown Pokémon field", (r) => (r.side.pokemon[0].newField = 1)],
    ["an unknown move field", (r) => (r.active[0].moves[0].newField = 1)],
    ["Revival Blessing", (r) => (r.side.pokemon[2].reviving = true)],
    ["an unknown move target", (r) => (r.active[0].moves[0].target = "everyone")],
    ["a non-string move target", (r) => (r.active[0].moves[0].target = 3)],
    ["a singles request", (r) => r.active.pop()],
    ["a null active slot", (r) => (r.active[1] = null)],
    ["an active slot without a Pokémon", (r) => (r.side.pokemon = [r.side.pokemon[0]])],
    ["moves that are not a list", (r) => (r.active[0].moves = {})],
  ])("%s", (_label, edit) => {
    expect(() => enumerate("turn.request.json", edit)).toThrow(EnumerationError);
  });
});
