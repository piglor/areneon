import { describe, expect, it } from "vitest";
import { historyProblems, historyVariants, resolveHistory } from "../src/mechanics-history.ts";
import type { MechanicsQuery } from "../src/mechanics-query.ts";

// ADR-022 erratum E-22.5.3, slice H2: the declared history of a query and the two ends rule, without an engine.
// What the engine does with a declared fact is tested in `mechanics-outcome.test.ts` and `mechanics-history-live.test.ts`.

const boosts = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const field = { weather: null, terrain: null, trick_room: false, sides: [side, side] };
const known = (value: unknown) => ({ kind: "known", value });
const unknown = (a: unknown, b: unknown) => ({ kind: "unknown", ends: [a, b] });
const combatant = (history?: unknown) => ({ set: "x", boosts, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });
const outcome = (own?: unknown, history?: unknown, foe?: unknown): MechanicsQuery["query"] =>
  ({
    kind: "move_outcome",
    sides: [[combatant(own)], [combatant(foe)]],
    actions: [{ side: 0, position: 0, move: "protect", mega: false }],
    asked: 0,
    target: { side: 0, position: 0 },
    state: field,
    ...(history === undefined ? {} : { history }),
  }) as unknown as MechanicsQuery["query"];

describe("resolving a query's history", () => {
  it("is null for a query that declares none, whatever the kind", () => {
    expect(historyVariants(outcome())).toEqual([null]);
    expect(historyVariants({ kind: "stats", set: "x" })).toEqual([null]);
    expect(historyVariants({ kind: "damage", attacker: combatant(), defender: combatant(), move: "tackle", state: field } as unknown as MechanicsQuery["query"])).toEqual([null]);
  });

  it("resolves known facts of both Pokemon and the battle, and places each by the position a draw would report", () => {
    const query = outcome({ times_attacked: known(2), ate_berry: known(true) }, { turn: 4, field_last_move: known("tackle"), sides: [{ total_fainted: known(1) }, { pending_slots: known([{ position: 0, id: "wish", started_turn: 3, after_residual: false }]) }] }, { stall_counter: known(3), last_item: known(null) });
    const { history, unknowns } = resolveHistory(query, new Set());
    expect(unknowns).toEqual([]);
    expect(history).toEqual({
      turn: 4,
      field_last_move: "tackle",
      sides: [{ total_fainted: 1 }, { pending_slots: [{ position: 0, id: "wish", started_turn: 3, after_residual: false }] }],
      combatants: [
        { side: 0, position: 0, place: 0, facts: { times_attacked: 2, ate_berry: true } },
        { side: 1, position: 0, place: 1, facts: { stall_counter: 3, last_item: null } },
      ],
    });
    expect(historyVariants(query)).toEqual([history]);
  });

  it("counts the Pokemon of a speed order side by side and the attacker and defender of a damage query as 0 and 1", () => {
    const speed = { kind: "speed_order", sides: [[combatant(), combatant({ move_actions: known(1) })], [combatant({ move_actions: known(2) })]], actions: [], state: field } as unknown as MechanicsQuery["query"];
    expect(resolveHistory(speed, new Set()).history?.combatants.map((c) => [c.side, c.position, c.place])).toEqual([
      [0, 1, 1],
      [1, 0, 2],
    ]);
    const damage = { kind: "damage", attacker: combatant({ times_attacked: known(1) }), defender: combatant({ hurt_this_turn: known(true) }), move: "tackle", state: field } as unknown as MechanicsQuery["query"];
    expect(resolveHistory(damage, new Set()).history?.combatants.map((c) => [c.side, c.position, c.place])).toEqual([
      [0, 0, 0],
      [1, 0, 1],
    ]);
  });

  it("is a history even when it declares only the turn or nothing", () => {
    expect(resolveHistory(outcome(undefined, { turn: 2 }), new Set()).history).toEqual({ turn: 2, sides: [{}, {}], combatants: [] });
    expect(resolveHistory(outcome(undefined, {}), new Set()).history).toEqual({ sides: [{}, {}], combatants: [] });
  });
});

describe("the two ends rule", () => {
  it("asks once with every unknown fact at its first end, then once per unknown fact whose ends differ with that fact alone at its second end", () => {
    const query = outcome({ times_attacked: unknown(0, 4), last_move_failed: unknown(false, true), ate_berry: unknown(true, true) }, { field_last_move: unknown(null, "tackle") });
    const variants = historyVariants(query);
    // the unknown facts are met in order: field_last_move, times_attacked, last_move_failed, ate_berry (equal ends: no variant), then the corner
    expect(variants).toHaveLength(5);
    const facts = (index: number) => variants[index]?.combatants[0]?.facts;
    expect(variants[0]?.field_last_move).toBeNull();
    expect(facts(0)).toEqual({ times_attacked: 0, last_move_failed: false, ate_berry: true });
    expect(variants[1]?.field_last_move).toBe("tackle");
    expect(facts(1)).toEqual(facts(0));
    expect(facts(2)).toEqual({ times_attacked: 4, last_move_failed: false, ate_berry: true });
    expect(facts(3)).toEqual({ times_attacked: 0, last_move_failed: true, ate_berry: true });
    expect(variants[2]?.field_last_move).toBeNull();
  });

  it("adds the all-second-ends corner when two or more facts differ, so a joint effect is asked", () => {
    const query = outcome({ times_attacked: unknown(0, 4), last_move_failed: unknown(false, true), ate_berry: unknown(true, true) });
    const variants = historyVariants(query);
    expect(variants).toHaveLength(4);
    expect(variants[3]?.combatants[0]?.facts).toEqual({ times_attacked: 4, last_move_failed: true, ate_berry: true });
    // one differing fact: its single variant is already the corner
    expect(historyVariants(outcome({ times_attacked: unknown(0, 4) }))).toHaveLength(2);
  });

  it("takes whole lists as the ends of a fact", () => {
    const query = outcome({ attacked_by: unknown([], [{ side: 1, position: 0, damage: 10 }]) });
    const [first, second] = historyVariants(query);
    expect(first?.combatants[0]?.facts.attacked_by).toEqual([]);
    expect(second?.combatants[0]?.facts.attacked_by).toEqual([{ side: 1, position: 0, damage: 10 }]);
  });

  it("gives equal-ended facts no variant of their own", () => {
    expect(historyVariants(outcome({ times_attacked: unknown(2, 2) }))).toHaveLength(1);
  });
});

describe("what the profile cannot say of a history", () => {
  it("has no problem with a consistent history", () => {
    const query = outcome({ attacked_by: known([{ side: 1, position: 0, damage: 5 }]), volatiles: known([{ id: "taunt", started_turn: 2, after_residual: false }]) }, { turn: 3 });
    expect(historyProblems(query)).toEqual([]);
    expect(historyProblems({ kind: "stats", set: "x" })).toEqual([]);
  });

  it("wants a turn to date a volatile or a slot condition, in a known fact or at either end of an unknown one", () => {
    const volatile = { id: "taunt", started_turn: 2, after_residual: false };
    expect(historyProblems(outcome({ volatiles: known([volatile]) }))).toEqual(["/history needs a turn to date its volatiles and slot conditions"]);
    expect(historyProblems(outcome({ volatiles: unknown([], [volatile]) }))).toHaveLength(1);
    expect(historyProblems(outcome({ volatiles: known([]) }))).toEqual([]);
    expect(historyProblems(outcome(undefined, { sides: [{ pending_slots: known([{ position: 0, id: "wish", started_turn: 1, after_residual: false }]) }, {}] }))).toHaveLength(1);
    expect(historyProblems(outcome(undefined, { turn: 2, sides: [{ pending_slots: known([{ position: 0, id: "wish", started_turn: 1, after_residual: false }]) }, {}] }))).toEqual([]);
  });

  it("names declared Pokemon only: an attacker and a slot position must exist in the sides", () => {
    expect(historyProblems(outcome({ attacked_by: known([{ side: 1, position: 1, damage: 5 }]) }))).toHaveLength(1);
    expect(historyProblems(outcome({ attacked_by: unknown([], [{ side: 0, position: 1, damage: 5 }]) }))).toHaveLength(1);
    expect(historyProblems(outcome(undefined, { turn: 1, sides: [{}, { pending_slots: known([{ position: 1, id: "wish", started_turn: 1, after_residual: false }]) }] }))).toHaveLength(1);
    // a damage query has one declared Pokemon a side
    const damage = { kind: "damage", attacker: combatant({ attacked_by: known([{ side: 1, position: 0, damage: 5 }]) }), defender: combatant({ attacked_by: known([{ side: 0, position: 1, damage: 5 }]) }), move: "tackle", state: field } as unknown as MechanicsQuery["query"];
    expect(historyProblems(damage)).toHaveLength(1);
  });
});
