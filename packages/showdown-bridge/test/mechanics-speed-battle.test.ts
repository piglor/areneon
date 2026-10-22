import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig } from "../src/mechanics-config.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { readMechanicsQuery } from "../src/mechanics-query.ts";
import { checkScenario, startScenario } from "../src/mechanics-scenario.ts";
import { exportPaste, importPaste } from "../src/teams.ts";
import { fullSpeedQuery, SPEED_FIXTURE, type SpeedCase } from "./mechanics-speed-cases.ts";

// ADR-016 slice 2, cross-check: for an untied case the probe's order equals
// the order of the `|move|` lines of a real battle turn of the engine, run on
// the same state. The real turn goes through the engine's own queue (`sort`,
// the speed recomputation after a Mega Evolution, the PRNG that breaks ties);
// the probe goes through `resolveAction` and `comparePriority`. The state is
// put on the real battle by this test's own code.

let engine: Engine;
let config: MechanicsConfig;
let scenarioId: string;
beforeAll(() => {
  engine = loadEngine();
  config = loadMechanicsConfig(GOLDEN_FORMAT);
  scenarioId = checkScenario(engine, config);
});

interface RealMon {
  boosts: Record<string, number>;
  status: string;
  setStatus(status: string): unknown;
}
interface RealBattle {
  log: string[];
  field: { weather: string; terrain: string; setWeather(id: string, source: RealMon): unknown; clearWeather(): unknown; addPseudoWeather(id: string, source: RealMon): unknown };
  sides: { pokemon: RealMon[]; sideConditions: Record<string, unknown>; addSideCondition(id: string, source: RealMon): unknown }[];
  choose(slot: string, input: string): boolean;
  dex: { moves: { get(id: string): { target: string } } };
}

const NEEDS_TARGET = ["normal", "any", "adjacentFoe"];
const toId = (name: string): string => name.toLowerCase().replace(/[^a-z0-9]/g, "");

/** `move <id> [1]` for a declared action, the Protect every set holds for any other active Pokemon. */
function choice(battle: RealBattle, side: number, actions: { side: number; position: number; move: string | null; mega: boolean }[]): string {
  return [0, 1]
    .map((position) => {
      const action = actions.find((candidate) => candidate.side === side && candidate.position === position);
      if (action === undefined) return "move protect";
      const target = NEEDS_TARGET.includes(battle.dex.moves.get(action.move as string).target) ? " 1" : "";
      return `move ${action.move}${target}${action.mega ? " mega" : ""}`;
    })
    .join(", ");
}

// Lines that change a Speed, a priority or a field condition the order reads.
const STATE_LINES = /^\|-(boost|unboost|setboost|copyboost|clearboost|clearallboost|clearpositiveboost|clearnegativeboost|invertboost|swapboost|status|curestatus|cureteam|weather|fieldstart|fieldend|sidestart|sideend|transform|formechange|item|enditem|activate|ability)\|/;

/**
 * The action indices in the order of the turn's first `|move|` or `|cant|`
 * line of each acting Pokemon, the order of the `|-mega|` lines, and whether a
 * line that can change a Speed came between the turn's first and last action.
 */
function observed(battle: RealBattle, actions: { side: number; position: number }[]): { order: number[]; megas: number[]; changedMidTurn: boolean } {
  const slot = (ident: string): number => actions.findIndex((action) => `p${action.side + 1}${"ab"[action.position]}` === ident.slice(0, 3));
  const first = (kinds: string[]): number[] => {
    const seen: number[] = [];
    for (const line of battle.log) {
      const parts = line.split("|");
      const index = kinds.includes(parts[1] as string) ? slot(parts[2] as string) : -1;
      if (index >= 0 && !seen.includes(index)) seen.push(index);
    }
    return seen;
  };
  const turn = battle.log.slice(battle.log.findIndex((line) => line.startsWith("|turn|")));
  const acts = turn.flatMap((line, position) => (/^\|(move|cant)\|/.test(line) ? [position] : []));
  const changedMidTurn = turn.slice(acts[0] ?? 0, acts[acts.length - 1] ?? 0).some((line) => STATE_LINES.test(line));
  return { order: first(["move", "cant"]), megas: first(["-mega"]), changedMidTurn };
}

/** Put the declared state on a real battle with calls of this file's own. */
function declareOnRealBattle(battle: RealBattle, query: ReturnType<typeof fullSpeedQuery>["query"] & { kind: "speed_order" }): void {
  const lead = battle.sides[0]?.pokemon[0] as RealMon;
  if (query.state.weather === null) battle.field.clearWeather();
  else battle.field.setWeather(query.state.weather, lead);
  if (query.state.trick_room) battle.field.addPseudoWeather("trickroom", lead);
  for (const [index, side] of query.state.sides.entries()) if (side.tailwind) (battle.sides[index] as RealBattle["sides"][0]).addSideCondition("tailwind", lead);
  for (const [index, combatants] of query.sides.entries()) {
    for (const [slot, combatant] of combatants.entries()) {
      const pokemon = (battle.sides[index] as RealBattle["sides"][0]).pokemon[slot] as RealMon;
      if (combatant.status !== null) pokemon.setStatus(combatant.status);
      Object.assign(pokemon.boosts, combatant.boosts);
    }
  }
}

/** The execution order of the probe's answer, one entry per action: Evolutions leave the order, each move keeps its place. */
const moveOrder = (groups: number[][]): number[] => {
  const flat = groups.flat();
  return flat.filter((index, position) => flat.lastIndexOf(index) === position);
};

describe("the probe's order is the real turn's order, case by case", () => {
  const real = SPEED_FIXTURE.cases.filter((speedCase) => speedCase.real_battle);
  it("has real cases for priority, Tailwind, Trick Room, paralysis, Scarf, weather and Mega", () => {
    expect(real.map((speedCase) => speedCase.id)).toEqual(
      expect.arrayContaining(["priority-brackets", "tailwind-doubles-speed", "trick-room-reverses", "trick-room-speed-over-1809-acts-last", "paralysis-halves-speed", "choice-scarf-multiplies-by-1-5", "declared-sun-doubles-chlorophyll", "mega-becomes-faster-and-moves-first", "two-megas-in-speed-order"]),
    );
  });

  it.each(real.map((speedCase) => [speedCase.id, speedCase] as const))("%s", (_id, speedCase: SpeedCase) => {
    const parsed = fullSpeedQuery(SPEED_FIXTURE, speedCase);
    if (parsed.query.kind !== "speed_order") throw new Error("not a speed_order query");
    const query = parsed.query;
    const sets = query.sides.map((side) => side.map((combatant) => importPaste(engine, combatant.set)[0] as PokemonSet));
    const team = (side: PokemonSet[]): PokemonSet[] => (side.length === 1 ? [side[0] as PokemonSet, side[0] as PokemonSet] : side);
    const battle = startScenario(engine, scenarioId, { p1: team(sets[0] as PokemonSet[]), p2: team(sets[1] as PokemonSet[]) }) as unknown as RealBattle;
    declareOnRealBattle(battle, query);
    expect(battle.choose("p1", choice(battle, 0, query.actions))).toBe(true);
    expect(battle.choose("p2", choice(battle, 1, query.actions))).toBe(true);
    const turn = observed(battle, query.actions);

    const { result } = probeMechanics({ engine, config }, parsed);
    if (result.kind !== "computed" || result.answer.kind !== "speed_order") throw new Error("not computed");
    // a Pokemon knocked out earlier in the turn does not act: compare the ones that did
    const expected = moveOrder((speedCase.expected as { groups: number[][] }).groups);
    expect(turn.order.length).toBeGreaterThanOrEqual(2);
    expect(turn.order).toEqual(moveOrder(result.answer.groups).filter((action) => turn.order.includes(action)));
    expect(turn.order).toEqual(expected.filter((action) => turn.order.includes(action)));
    // the Evolutions come first, in the order of their groups
    const evolving = query.actions.flatMap((action, index) => (action.mega ? [index] : []));
    expect(turn.megas).toEqual(result.answer.groups.flat().filter((action, position, all) => evolving.includes(action) && all.indexOf(action) === position));
  });
});

describe("the probe's order is the real turn's order on the committed fixture teams, with the state the real battle starts in", () => {
  const records = readFileSync(join(TEAMS_DIR, "datasets", "fixtures-v1", "teams-dev.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((line) => importPaste(loadEngine(), (JSON.parse(line) as { paste: string }).paste));

  it("agrees on every pair whose turn has no exact tie", () => {
    let compared = 0;
    let skipped = 0;
    for (let index = 0; index + 1 < records.length && compared < 150; index += 1) {
      const sets = [(records[index] as PokemonSet[]).slice(0, 2), (records[index + 1] as PokemonSet[]).slice(0, 2)];
      const battle = startScenario(engine, scenarioId, { p1: sets[0] as PokemonSet[], p2: sets[1] as PokemonSet[] }) as unknown as RealBattle;
      // the state the real battle starts in, declared as the player would see it
      const side = (index: number) => ({ aurora_veil: false, light_screen: false, reflect: false, tailwind: Boolean((battle.sides[index] as RealBattle["sides"][0]).sideConditions.tailwind) });
      const query = readMechanicsQuery({
        schema_version: 1,
        data_class: "synthetic",
        format_id: GOLDEN_FORMAT,
        query: {
          kind: "speed_order",
          sides: sets.map((side, sideIndex) =>
            side.map((set, slot) => ({
              set: exportPaste(engine, [set]),
              boosts: Object.fromEntries(["atk", "def", "spa", "spd", "spe"].map((stat) => [stat, ((battle.sides[sideIndex] as RealBattle["sides"][0]).pokemon[slot] as RealMon).boosts[stat]])),
              status: null,
              hp: null,
              mega: false,
            })),
          ),
          actions: sets.flatMap((side, sideIndex) => side.map((set, position) => ({ side: sideIndex, position, move: toId(set.moves[0] as string), mega: false }))),
          state: { weather: battle.field.weather || null, terrain: battle.field.terrain || null, trick_room: false, sides: [side(0), side(1)] },
        },
      });
      if (query.query.kind !== "speed_order") throw new Error("unreachable");
      let outcome;
      try {
        outcome = probeMechanics({ engine, config }, query);
      } catch (error) {
        if (!(error instanceof MechanicsProbeError)) throw error;
        skipped += 1;
        continue;
      }
      const { result } = outcome;
      if (result.kind !== "computed" || result.answer.kind !== "speed_order" || result.answer.groups.some((group) => group.length > 1)) {
        skipped += 1;
        continue;
      }
      const accepted = battle.choose("p1", choice(battle, 0, query.query.actions)) && battle.choose("p2", choice(battle, 1, query.query.actions));
      if (!accepted) {
        skipped += 1;
        continue;
      }
      const turn = observed(battle, query.query.actions);
      if (turn.changedMidTurn) {
        skipped += 1; // the engine changed a Speed or a field condition during the turn: the probe's basis is the turn's start
        continue;
      }
      // a Pokemon knocked out earlier in the turn does not act: compare the ones that did
      expect(turn.order, `pair ${index}`).toEqual(moveOrder(result.answer.groups).filter((action) => turn.order.includes(action)));
      expect(turn.order.length).toBeGreaterThanOrEqual(2);
      compared += 1;
    }
    expect(compared, `${skipped} pairs skipped (tie, unsupported, refused choice or a mid-turn change)`).toBeGreaterThanOrEqual(20);
  });
});
