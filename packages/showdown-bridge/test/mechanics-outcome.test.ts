/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { Engine, PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT, TEAMS_DIR } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadMechanicsConfig, type MechanicsConfig, withMoveOutcome } from "../src/mechanics-config.ts";
import { declaredOutcomeScenario, OUTCOME_FORMAT } from "../src/mechanics-outcome.ts";
import { MechanicsProbeError, probeMechanics } from "../src/mechanics-probe.ts";
import { type MechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { MECHANICS_ADAPTER_FORMAT, readMechanicsResult } from "../src/mechanics-result.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { SPEED_FIXTURE } from "./mechanics-speed-cases.ts";
import { exportPaste, importPaste } from "../src/teams.ts";

// ADR-022 erratum E-22.5.3, slice H2: the `move_outcome` kind and the declared history, against the pinned engine. Each case is worked
// from the engine's own data (the move's `onTry`, `onPrepareHit`, `accuracy`, the stall condition's counter): the probe names no move.

let engine: Engine;
let config: MechanicsConfig;
let scenario: string;
beforeAll(() => {
  engine = loadEngine();
  config = withMoveOutcome(loadMechanicsConfig(GOLDEN_FORMAT));
  scenario = checkScenario(engine, config);
});

/** One Pokemon of a committed synthetic team as the engine's canonical PokePaste. */
const setOf = (file: string, index: number): string => exportPaste(engine, [importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"))[index] as PokemonSet]);
const canonical = (text: string): string => exportPaste(engine, engine.sim.Teams.import(text) as PokemonSet[]);

// Venusaur (own), Snorlax, Tyranitar and Garchomp (foes): the committed legal teams.
const VENUSAUR = () => setOf("legal-synthetic.txt", 0);
const SNORLAX = () => setOf("legal-synthetic.txt", 4);
const TYRANITAR = () => setOf("legal-opponent.txt", 0);
const GARCHOMP = () => setOf("legal-opponent.txt", 1);

const ZERO = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const NO_CONDITIONS = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const FIELD = { weather: null, terrain: null, trick_room: false, sides: [NO_CONDITIONS, NO_CONDITIONS] };
const known = (value: unknown) => ({ kind: "known", value });
const unknown = (a: unknown, b: unknown) => ({ kind: "unknown", ends: [a, b] });
const combatant = (set: string, history?: unknown) => ({ set, boosts: ZERO, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });
const move = (side: number, id: string | null) => ({ side, position: 0, move: id, mega: false });

interface Case {
  own?: string;
  foe?: string;
  actions: ReturnType<typeof move>[];
  target?: { side: number; position: number };
  own_history?: unknown;
  foe_history?: unknown;
  history?: unknown;
}
const query = (c: Case) =>
  readMechanicsQuery({
    schema_version: 1,
    data_class: "synthetic",
    format_id: GOLDEN_FORMAT,
    query: { kind: "move_outcome", sides: [[combatant(c.own ?? VENUSAUR(), c.own_history)], [combatant(c.foe ?? TYRANITAR(), c.foe_history)]], actions: c.actions, asked: 0, target: c.target ?? { side: 1, position: 0 }, state: FIELD, ...(c.history === undefined ? {} : { history: c.history }) },
  });
const run = (c: Case, cfg: MechanicsConfig = config) => probeMechanics({ engine, config: cfg }, query(c));
const answer = (c: Case) => {
  const { result, diagnostics } = run(c);
  expect(diagnostics).toEqual([]);
  expect(readMechanicsResult(result)).toEqual(result);
  if (result.kind !== "computed" || result.answer.kind !== "move_outcome") throw new Error(`not computed: ${JSON.stringify(result)}`);
  const { precondition, precondition_ppm, hit_ppm } = result.answer;
  return { precondition, precondition_ppm, hit_ppm };
};
const PASSES = { precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 1_000_000 };
const FAILS = { precondition: "fails", precondition_ppm: 0, hit_ppm: 0 };
const tackle = move(1, "tackle");

describe("the contract of the kind", () => {
  it("is in scope only when the config lists it: the committed config keeps its bytes and its hash", () => {
    const committed = loadMechanicsConfig(GOLDEN_FORMAT);
    expect(committed.scope.query_kinds.map((entry) => entry.kind)).toEqual(["damage", "speed_order", "stats"]);
    expect(run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 } }, committed).result).toMatchObject({ kind: "unsupported", reason: "query_kind", position: null });
    expect(config.scope.query_kinds.map((entry) => entry.kind)).toEqual(["damage", "move_outcome", "speed_order", "stats"]);
    expect(withMoveOutcome(config)).toBe(config);
  });

  it("answers with the adapter format that did not move, the config's identity and no gap row", () => {
    const { result } = run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 } });
    expect(MECHANICS_ADAPTER_FORMAT).toBe(2);
    expect(OUTCOME_FORMAT).toBe(1);
    expect(result).toMatchObject({ kind: "computed", adapter_format: 2, gap_ids: [] });
  });

  it("runs the Mega Evolution of the asked action first and names the Mega row", () => {
    const { result } = run({ own: SPEED_FIXTURE.sets.gengar as string, foe: SPEED_FIXTURE.sets.garchomp_fast as string, actions: [{ ...move(0, "shadowball"), mega: true }, move(1, "earthquake")] });
    // the foe's spread move is resolved by the engine's own random-target step, answered with the first candidate
    expect(result).toMatchObject({ kind: "computed", gap_ids: ["mega-evolution"], answer: { kind: "move_outcome", precondition: "passes", hit_ppm: 1_000_000 } });
    expect(readMechanicsResult(result)).toEqual(result);
  });

  it("gives the same JCS bytes on a rerun", () => {
    const one = (): string => jcs(run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { stall_counter: known(3) } }).result as unknown as Json);
    expect(one()).toBe(one());
  });
});

describe("Protect: the stall counter and the action still to come", () => {
  const protect = (own_history?: unknown, others = [tackle]) => ({ actions: [move(0, "protect"), ...others], target: { side: 0, position: 0 }, ...(own_history === undefined ? {} : { own_history }) });

  it("works with no stall counter while an action is queued after it", () => {
    expect(answer(protect())).toEqual(PASSES);
  });

  it("is the engine's own chance on the second use (1/3) and the third (1/9)", () => {
    expect(answer(protect({ stall_counter: known(3) }))).toEqual({ precondition: "chance", precondition_ppm: 333_333, hit_ppm: 1_000_000 });
    expect(answer(protect({ stall_counter: known(9) }))).toEqual({ precondition: "chance", precondition_ppm: 111_111, hit_ppm: 1_000_000 });
  });

  it("an exact 0 counter is no stall condition", () => {
    expect(answer(protect({ stall_counter: known(0) }))).toEqual(PASSES);
  });

  it("is aimed at the user whatever target the query names: a move of its own side has no foe target", () => {
    expect(answer({ actions: [move(0, "protect"), tackle], target: { side: 1, position: 0 } })).toEqual(PASSES);
  });

  it("Wide Guard, a move of a side, has no accuracy step: its first hit is its hit step; it reads the queue and, in this engine's data, not the stall counter", () => {
    const guard = (own_history?: unknown, others = [tackle]) => ({ actions: [move(0, "wideguard"), ...others], target: { side: 0, position: 0 }, ...(own_history === undefined ? {} : { own_history }) });
    expect(answer(guard())).toEqual(PASSES);
    expect(answer(guard({ stall_counter: known(3) }))).toEqual(PASSES);
    expect(answer(guard(undefined, []))).toEqual(FAILS);
  });

  it("fails when no action remains after it", () => {
    expect(answer(protect(undefined, []))).toEqual(FAILS);
    // an action that went before it leaves nothing queued either: the foe's switch comes first
    expect(answer(protect(undefined, [move(1, null)]))).toEqual(FAILS);
  });
});

describe("moves that read the history", () => {
  it("First Impression fails after the first move action and works on the first", () => {
    const fi = (move_actions: number) => answer({ actions: [move(0, "firstimpression")], own_history: { move_actions: known(move_actions) } });
    expect(fi(0)).toEqual(PASSES);
    expect(fi(1)).toEqual(FAILS);
  });

  it("Belch fails until a berry was eaten, then has its own accuracy", () => {
    const belch = (ate: boolean) => answer({ actions: [move(0, "belch")], own_history: { ate_berry: known(ate) } });
    expect(belch(false)).toEqual(FAILS);
    expect(belch(true)).toEqual({ precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 900_000 });
  });

  it("Last Resort fails while a move of the set is unused and works once every other one was used", () => {
    const set = canonical("Kangaskhan @ Leftovers\nAbility: Early Bird\nLevel: 50\nEVs: 32 HP / 32 Atk / 2 Def\nAdamant Nature\n- Last Resort\n- Protect\n");
    const resort = (used: string[]) => answer({ own: set, actions: [move(0, "lastresort")], own_history: { moves_used: known(used) } });
    expect(resort([])).toEqual(FAILS);
    expect(resort(["protect"])).toEqual(PASSES);
  });

  it("Spit Up fails with no Stockpile and works with two layers", () => {
    const spit = (volatiles: unknown[]) => answer({ actions: [move(0, "spitup")], own_history: { volatiles: known(volatiles) }, history: { turn: 3 } });
    expect(spit([])).toEqual(FAILS);
    expect(spit([{ id: "stockpile", started_turn: 2, after_residual: false, starts: 2 }])).toEqual(PASSES);
  });

  it("Sucker Punch works against a damaging branch and fails against a status move or a switch", () => {
    const punch = (foe: ReturnType<typeof move>) => answer({ actions: [move(0, "suckerpunch"), foe] });
    expect(punch(move(1, "tackle"))).toEqual(PASSES);
    expect(punch(move(1, "swordsdance"))).toEqual(FAILS);
    expect(punch(move(1, null))).toEqual(FAILS);
  });

  it("is the move's own accuracy: a chance of hitting, and a target the engine removes never gets a hit step", () => {
    expect(answer({ actions: [move(0, "thunder")] })).toEqual({ precondition: "passes", precondition_ppm: 1_000_000, hit_ppm: 700_000 });
    expect(answer({ actions: [move(0, "thunderbolt")], foe: GARCHOMP() })).toEqual(FAILS);
  });
});

describe("the two ends rule", () => {
  it("keeps equal answers: a fact the move does not read cannot change the outcome", () => {
    const { result } = run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { times_attacked: unknown(0, 4) } });
    expect(result).toMatchObject({ kind: "computed", answer: { kind: "move_outcome", precondition: "passes" } });
  });

  it("makes different answers unsupported with history_unknown", () => {
    const { result } = run({ actions: [move(0, "belch")], own_history: { ate_berry: unknown(false, true) } });
    expect(result).toMatchObject({ kind: "unsupported", reason: "history_unknown", position: null });
    expect(readMechanicsResult(result)).toEqual(result);
  });

  it("asks a fact of the damage kind at both ends too", () => {
    const damage = readMechanicsQuery({ schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "damage", attacker: combatant(SNORLAX(), { times_attacked: unknown(0, 2) }), defender: combatant(TYRANITAR()), move: "ragefist", state: FIELD } });
    expect(probeMechanics({ engine, config }, damage).result).toMatchObject({ kind: "unsupported", reason: "history_unknown" });
  });
});

describe("refused states and unsupported draws", () => {
  it("refuses a declared history the engine does not take, as a probe error and not a crash", () => {
    expect(() => run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { volatiles: known([{ id: "notacondition", started_turn: 1, after_residual: false }]) }, history: { turn: 2 } })).toThrow(MechanicsProbeError);
    expect(() => run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { last_item: known("notanitem") } })).toThrow(MechanicsProbeError);
    expect(() => run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { moves_used: known(["tackle"]) } })).toThrow(MechanicsProbeError);
    expect(() => run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, history: { field_last_move: known("notamove") } })).toThrow(MechanicsProbeError);
    expect(() => run({ actions: [move(0, "notamove")], target: { side: 0, position: 0 } })).toThrow(MechanicsProbeError);
    expect(() => run({ actions: [move(0, "protect")], target: { side: 0, position: 0 }, history: { turn: 2, sides: [{ pending_slots: known([{ position: 0, id: "notacondition", started_turn: 1, after_residual: false }]) }, {}] } })).toThrow(MechanicsProbeError);
  });

  it("is unsupported when the asked action ties in the engine's order with another (the engine would shuffle them)", () => {
    const { result } = run({ own: VENUSAUR(), foe: VENUSAUR(), actions: [move(0, "protect"), move(1, "protect")], target: { side: 0, position: 0 } });
    expect(result).toMatchObject({ kind: "unsupported", reason: "random_effect", position: 1 });
  });

  it("is unsupported when a declared volatile draws its length: the position is the Pokemon being declared", () => {
    const { result } = run({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, own_history: { volatiles: known([{ id: "confusion", started_turn: 1, after_residual: false }]) }, history: { turn: 2 } });
    expect(result).toMatchObject({ kind: "unsupported", reason: "random_effect", position: 0 });
  });

  it("is unsupported when the engine draws anything but a chance before the hit step", () => {
    // Present picks its power with the engine's `random(10)` while the move is modified, before any hit step
    const { result } = run({ actions: [move(0, "present")] });
    expect(result).toMatchObject({ kind: "unsupported", reason: "random_effect", position: 0 });
  });
});

describe("the declared history is in the engine's fields", () => {
  const history = (own: unknown, battle?: unknown) => ({ own_history: own, ...(battle === undefined ? {} : { history: battle }) });
  const built = (own: unknown, battle?: unknown, foe?: unknown) => {
    const q = query({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, ...history(own, battle), ...(foe === undefined ? {} : { foe_history: foe }) }).query as Extract<MechanicsQuery["query"], { kind: "move_outcome" }>;
    return q;
  };
  const fields = (own: unknown, battle?: unknown, foe?: unknown): any => {
    const q = built(own, battle, foe);
    const sets = [q.sides[0][0]?.set, q.sides[1][0]?.set].map((paste) => importPaste(engine, paste as string)[0] as PokemonSet);
    // the probe's own resolution of the query's history
    return { q, sets };
  };

  it("declares every fact through the engine's own field", async () => {
    const { resolveHistory } = await import("../src/mechanics-history.ts");
    const { q, sets } = fields(
      { move_actions: known(2), stall_counter: known(9), times_attacked: known(3), hurt_this_turn: known(true), attacked_by: known([{ side: 1, position: 0, damage: 40 }]), last_move_failed: known(true), stats_raised_this_turn: known(true), stats_lowered_this_turn: known(true), ate_berry: known(true), last_item: known("sitrusberry"), moves_used: known(["protect"]), volatiles: known([{ id: "taunt", started_turn: 2, after_residual: false }]) },
      { turn: 3, field_last_move: known("tackle"), sides: [{ total_fainted: known(2) }, { pending_slots: known([{ position: 0, id: "wish", started_turn: 3, after_residual: false }]) }] },
    );
    const resolved = resolveHistory(q as MechanicsQuery["query"], new Set()).history;
    const battle: any = declaredOutcomeScenario(engine, scenario, q, sets, { resolved: resolved as never, stall: "stall" });
    const mon = battle.sides[0].pokemon[0];
    expect(battle.turn).toBe(3);
    expect(mon.activeMoveActions).toBe(2);
    expect(mon.volatiles.stall.counter).toBe(9);
    expect(mon.timesAttacked).toBe(3);
    expect(mon.hurtThisTurn).not.toBeNull();
    expect(mon.attackedBy.map((a: any) => [a.slot, a.damage, a.thisTurn])).toEqual([["p2a", 40, true]]);
    expect(mon.moveLastTurnResult).toBe(false);
    expect(mon.statsRaisedThisTurn && mon.statsLoweredThisTurn && mon.ateBerry).toBe(true);
    expect(mon.lastItem).toBe("sitrusberry");
    expect(mon.moveSlots.filter((slot: any) => slot.used).map((slot: any) => slot.id)).toEqual(["protect"]);
    // the engine's own length of Taunt (3, one more for a target that had already moved: the default) less the turns since it began (begun at turn 2, now turn 3)
    expect(mon.volatiles.taunt.duration).toBe(3);
    expect(battle.sides[0].totalFainted).toBe(2);
    expect(battle.lastMove.id).toBe("tackle");
    // the slot condition began at turn 3: the engine's own start handler read that turn
    expect(battle.sides[1].slotConditions[0].wish.startingTurn).toBe(2);
    expect(battle.sides[1].slotConditions[0].wish.hp).toBeGreaterThan(0);
  });

  it("dates a volatile as the engine's start handler reads it: a target that had not moved gets Taunt's base length", async () => {
    const { resolveHistory } = await import("../src/mechanics-history.ts");
    const { q, sets } = fields({ volatiles: known([{ id: "taunt", started_turn: 2, after_residual: false, moved_before: false }]) }, { turn: 3 });
    const resolved = resolveHistory(q as MechanicsQuery["query"], new Set()).history;
    const battle: any = declaredOutcomeScenario(engine, scenario, q, sets, { resolved: resolved as never, stall: "stall" });
    expect(battle.sides[0].pokemon[0].volatiles.taunt.duration).toBe(2);
    // a volatile that has run out is a refusal
    const over = resolveHistory(fields({ volatiles: known([{ id: "taunt", started_turn: 1, after_residual: false, moved_before: false }]) }, { turn: 9 }).q as MechanicsQuery["query"], new Set()).history;
    expect(() => declaredOutcomeScenario(engine, scenario, q, sets, { resolved: over as never, stall: "stall" })).toThrow(MechanicsProbeError);
  });

  it("builds the scenario with no history at all as the declared state alone", () => {
    const q = built(undefined);
    const sets = [q.sides[0][0]?.set, q.sides[1][0]?.set].map((paste) => importPaste(engine, paste as string)[0] as PokemonSet);
    const battle: any = declaredOutcomeScenario(engine, scenario, q, sets, null);
    expect(battle.sides[0].pokemon[0].timesAttacked).toBe(0);
  });

  it("changes nothing when the history declares nothing", () => {
    const base = answer({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 } });
    expect(answer({ actions: [move(0, "protect"), tackle], target: { side: 0, position: 0 }, history: {} })).toEqual(base);
  });
});
