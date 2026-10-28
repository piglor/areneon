// One simulated turn of a search on a world (ADR-026 section 2): the player's view of the engine's output, the engine seed's
// effect, refusals, and that the world itself is never changed. The leaf values the search reads off the result.
import { beforeAll, describe, expect, it } from "vitest";
import { type Engine, loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { enumerateCandidates } from "../src/choices.ts";
import { decodeRequest } from "../src/request.ts";
import { legalReplies } from "../src/reply-model.ts";
import { hpKoDifferential, outcomeScore } from "../src/search-leaf.ts";
import { activeRequest, cloneWorld, foeSwitchChoice, playChoices, playerView } from "../src/search-turn.ts";
import { createWorldBuilder, type ScenarioWorld } from "../src/world-builder.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { SEARCH_HP_ROUNDING } from "../src/world-sampler.ts";
import { loadWorldWriters } from "../src/world-writers.ts";
import { cases } from "./world-builder-support.ts";

const SEED = `sodium,${"ab".repeat(16)}`;
let engine: Engine;
let world: ScenarioWorld;
beforeAll(() => {
  engine = loadEngine();
  const builder = createWorldBuilder({ engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: SEARCH_HP_ROUNDING, writers: loadWorldWriters(engine, GOLDEN_FORMAT) });
  for (const c of cases()) {
    const built = builder.build(c.input);
    if (built.kind === "world") {
      world = built.world;
      return;
    }
  }
  throw new Error("no world could be built from the committed channels");
});

const firstOwn = (w: ScenarioWorld): string => {
  const request = activeRequest(cloneWorld(engine, JSON.stringify(w.battle.toJSON()), SEED), w.seat) as object;
  return (enumerateCandidates(decodeRequest(JSON.stringify(request))).candidates.find((c) => !c.pending) as { choice: string }).choice;
};

describe("playerView: the engine's own split rule", () => {
  const text = ["|split|p1", "|-damage|p1a: A|10/50", "|-damage|p1a: A|20/100", "|split|p2", "|-damage|p2a: B|7/40", "|-damage|p2a: B|17/100", "|t:|123", "|move|p1a: A|Tackle|p2a: B", "|split|p1", "", "|only-shared"].join("\n");
  it("gives each seat the secret half of its own lines and the shared half of the other's, and drops timestamps and empty halves", () => {
    expect(playerView(text, 1)).toEqual(["|-damage|p1a: A|10/50", "|-damage|p2a: B|17/100", "|move|p1a: A|Tackle|p2a: B"]);
    expect(playerView(text, 2)).toEqual(["|-damage|p1a: A|20/100", "|-damage|p2a: B|7/40", "|move|p1a: A|Tackle|p2a: B", "|only-shared"]);
  });
});

describe("a turn on a world", () => {
  it("returns the seat's lines and its next request, and leaves the world as it was", () => {
    const before = JSON.stringify(world.battle.toJSON());
    const [own] = [firstOwn(world)];
    const foe = legalReplies(engine, world)[0] as string;
    const turn = playChoices(cloneWorld(engine, before, SEED), world.seat, { own, foe });
    expect(turn.refused).toBeNull();
    expect(turn.lines.some((line) => line.startsWith("|move|"))).toBe(true);
    expect(turn.lines.every((line) => line.startsWith("|") && !line.startsWith("|split") && !line.startsWith("|t:|"))).toBe(true);
    expect(turn.request === null || turn.lines.at(-1) === `|request|${turn.request}` || turn.ended !== null).toBe(true);
    expect(JSON.stringify(world.battle.toJSON())).toBe(before);
  });

  it("gives the same lines for the same seed and choices, and shows the opponent only as a share of 100", () => {
    const text = JSON.stringify(world.battle.toJSON());
    const own = firstOwn(world);
    const foe = legalReplies(engine, world)[0] as string;
    const run = () => playChoices(cloneWorld(engine, text, SEED), world.seat, { own, foe });
    expect(run().lines).toEqual(run().lines);
    const foeSide = world.seat === "p1" ? "p2" : "p1";
    for (const line of run().lines) {
      const damage = new RegExp(`^\\|-damage\\|${foeSide}[ab]: [^|]+\\|(\\d+)/(\\d+)`).exec(line);
      if (damage !== null) expect(damage[2]).toBe("100");
    }
  });

  it("refuses a choice the engine refuses, and reports a thrown engine error as a refusal", () => {
    const text = JSON.stringify(world.battle.toJSON());
    const foe = legalReplies(engine, world)[0] as string;
    expect(playChoices(cloneWorld(engine, text, SEED), world.seat, { own: "move 9 9, move 9 9", foe }).refused).toMatch(/error|refused/);
    expect(playChoices(cloneWorld(engine, text, SEED), world.seat, { own: firstOwn(world), foe: "move 9 9, move 9 9" }).refused).not.toBeNull();
    const clone = cloneWorld(engine, text, SEED);
    (clone as unknown as { choose: () => never }).choose = () => {
      throw new Error("boom");
    };
    expect(playChoices(clone, world.seat, { own: "x", foe: null }).refused).toBe("the engine threw: boom");
  });

  it("maps a finished battle to the seat's outcome", () => {
    const text = JSON.stringify(world.battle.toJSON());
    const outcome = (winner: string | undefined) => {
      const clone = cloneWorld(engine, text, SEED) as unknown as { ended: boolean; winner?: string };
      clone.ended = true;
      if (winner !== undefined) clone.winner = winner;
      return playChoices(clone as never, world.seat, { own: null, foe: null }).ended;
    };
    const other = world.seat === "p1" ? "p2" : "p1";
    expect([outcome(world.seat), outcome(other), outcome(""), outcome(undefined)]).toEqual(["own", "foe", "tie", "tie"]);
  });

  it("takes the opponent's first non-pending choice in canonical order when it must choose alone", () => {
    const clone = cloneWorld(engine, JSON.stringify(world.battle.toJSON()), SEED);
    expect(foeSwitchChoice(clone, world.seat)).toBe(legalReplies(engine, world)[0]);
  });

  it("reads a request from the clone", () => {
    const clone = cloneWorld(engine, JSON.stringify(world.battle.toJSON()), SEED);
    expect(activeRequest(clone, world.seat)).not.toBeNull();
    expect(activeRequest(clone, "p9")).toBeNull();
  });
});

describe("the leaf values", () => {
  it("scores a finished game 1, 1/2 or 0 in ppm", () => {
    expect([outcomeScore("own"), outcomeScore("tie"), outcomeScore("foe")]).toEqual([1_000_000, 500_000, 0]);
  });

  it("is level on an untouched world and moves with hit points and knock-outs, in integers", () => {
    const clone = cloneWorld(engine, JSON.stringify(world.battle.toJSON()), SEED);
    expect(hpKoDifferential(clone, world.seat)).toBe(500_000);
    const sides = clone.sides as unknown as { id: string; pokemon: { hp: number; maxhp: number; fainted: boolean }[] }[];
    const foe = sides.find((side) => side.id !== world.seat) as (typeof sides)[number];
    // The opponent loses half the hit points of one Pokemon of four: 1/8 of its share, so a quarter of that for the leaf.
    const mon = foe.pokemon[0] as (typeof foe.pokemon)[number];
    mon.hp = Math.floor(mon.maxhp / 2);
    const hurt = hpKoDifferential(clone, world.seat);
    expect(hurt).toBeGreaterThan(500_000);
    expect(Number.isInteger(hurt)).toBe(true);
    mon.hp = 0;
    mon.fainted = true;
    expect(hpKoDifferential(clone, world.seat)).toBeGreaterThan(hurt);
    expect(hpKoDifferential(clone, world.seat === "p1" ? "p2" : "p1")).toBeLessThan(500_000);
  });
});
