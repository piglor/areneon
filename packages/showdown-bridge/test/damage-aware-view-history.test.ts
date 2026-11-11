import { describe, expect, it } from "vitest";
import { declaresIds, unknownVolatiles, withoutVolatiles } from "../src/damage-aware-view.ts";
import type { MechanicsQuery } from "../src/mechanics-query.ts";
import type { MechanicsResult } from "../src/mechanics-result.ts";

// The helpers a view uses when the engine cannot be given a declared fact (engine-free): which facts name engine ids and what is left when they are removed.
const fighter = (history?: Record<string, unknown>) => ({ set: "x", boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, status: null, hp: null, mega: false, ...(history === undefined ? {} : { history }) });
const state = { weather: null, terrain: null, trick_room: false, sides: [{}, {}] };
const known = (value: unknown) => ({ kind: "known", value });
const full = { volatiles: known([{ id: "taunt", started_turn: 1, after_residual: false }]), moves_used: known(["tackle"]), last_item: known("x"), move_actions: known(1) };
const battle = { turn: 3, field_last_move: known("tackle"), sides: [{ total_fainted: known(1), pending_slots: known([]) }, { total_fainted: known(0) }] };
const damage = { kind: "damage", attacker: fighter(full), defender: fighter(), move: "tackle", state, history: battle } as unknown as MechanicsQuery["query"];
const speed = { kind: "speed_order", sides: [[fighter(full)], [fighter(full)]], actions: [], state, history: battle } as unknown as MechanicsQuery["query"];
const outcome = { ...speed, kind: "move_outcome" } as unknown as MechanicsQuery["query"];
const wrap = (query: MechanicsQuery["query"]) => ({ schema_version: 1, data_class: "synthetic", format_id: "f", query }) as MechanicsQuery;

describe("facts that name engine ids", () => {
  it("are found in every kind that has a history and in none that has not", () => {
    for (const query of [damage, speed, outcome]) expect(declaresIds(wrap(query))).toBe(true);
    expect(declaresIds(wrap({ kind: "stats", set: "x" } as never))).toBe(false);
    const counts = { kind: "damage", attacker: fighter({ move_actions: known(1) }), defender: fighter(), move: "tackle", state } as unknown as MechanicsQuery["query"];
    expect(declaresIds(wrap(counts))).toBe(false); // counts and flags name no engine id
    expect(declaresIds(wrap({ ...counts, history: { turn: 1, sides: [{ total_fainted: known(1) }, {}] } } as never))).toBe(false);
    expect(declaresIds(wrap({ ...counts, history: { sides: [{ pending_slots: known([]) }, {}] } } as never))).toBe(true);
    expect(declaresIds(wrap({ ...counts, history: { field_last_move: known(null) } } as never))).toBe(true);
    expect(declaresIds(wrap({ ...counts, history: {} } as never))).toBe(false);
  });
  it("are removed from the query and the rest of the history stays", () => {
    for (const query of [damage, speed, outcome]) {
      const stripped = withoutVolatiles(wrap(query)).query as unknown as { history: Record<string, unknown>; attacker?: { history: Record<string, unknown> }; sides?: { history: Record<string, unknown> }[][] };
      const holders = stripped.attacker !== undefined ? [stripped.attacker] : (stripped.sides ?? []).flat();
      for (const h of holders) expect(Object.keys(h.history)).toEqual(["move_actions"]);
      expect(Object.keys(stripped.history)).toEqual(["sides"]);
      expect(declaresIds(wrap(stripped as never))).toBe(false);
    }
    // a query without history, or of another kind, is a copy
    expect(withoutVolatiles(wrap({ kind: "stats", set: "x" } as never))).toEqual(wrap({ kind: "stats", set: "x" } as never));
    expect(withoutVolatiles(wrap({ kind: "damage", attacker: fighter(), defender: fighter(), move: "m", state } as never)).query).toMatchObject({ move: "m" });
  });
  it("make a computed answer history_unknown and leave any other result as it is", () => {
    const computed = { schema_version: 1, kind: "computed", data_class: "synthetic", query_hash: "h", gap_ids: [], answer: { kind: "stats" } } as unknown as MechanicsResult;
    const unknown = unknownVolatiles(computed);
    expect(unknown).toMatchObject({ kind: "unsupported", reason: "history_unknown", position: null, query_hash: "h" });
    expect("answer" in unknown).toBe(false);
    const other = { kind: "unsupported", reason: "random_effect", position: 1 } as unknown as MechanicsResult;
    expect(unknownVolatiles(other)).toBe(other);
  });
});
