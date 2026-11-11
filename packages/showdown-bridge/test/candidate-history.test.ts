import { describe, expect, it } from "vitest";
import { type HistoryFacts, type PokemonHistory, UNSETTLED, type WorldFacts } from "@areneon/public-state";
import { historyDeclarer, type HistoryExists } from "../src/candidate-history.ts";
import type { Fighter } from "../src/damage-aware-view.ts";
import { type OutcomeConfig, readOutcomeConfig } from "../src/outcome-config.ts";

// The declarations of the history facts, engine-free: hand-built facts and fighters, every branch of the two-ends rule.

const record: OutcomeConfig = readOutcomeConfig({ schema_version: 1, format_id: "f", assumed_branch: "assumed-set-filler", max_branches: 5, unknown_count_end: 20, unknown_item: "sitrusberry", unknown_move: "tackle", unknown_volatiles: ["stall", "stockpile"], volatiles: "declared" });
const exists: HistoryExists = { condition: (id) => ["taunt", "bide", "wish", "stockpile", "protosynthesis", "perishsong"].includes(id), stacking: (id) => id === "stockpile" };

const stint = (change: Partial<Extract<PokemonHistory, { unknown: false }>> = {}): PokemonHistory => ({
  name: "A",
  unknown: false,
  move_actions: 2,
  stall_counter: null,
  times_attacked: 1,
  hurt_this_turn: false,
  attackers_this_turn: [],
  last_move_failed: false,
  stats_raised_this_turn: false,
  stats_lowered_this_turn: true,
  moves_used: ["tackle"],
  volatile_started: {},
  ...change,
});
const side = (change: Partial<HistoryFacts["own"]> = {}): HistoryFacts["own"] => ({ total_fainted: 0, active: {}, pending_slot: {}, roster: {}, ...change });
const facts = (own: Partial<HistoryFacts["own"]>, opponent: Partial<HistoryFacts["own"]> = {}, rest: Partial<HistoryFacts> = {}): HistoryFacts => ({ history_facts_format: "x", turn: 4, field_last_move: null, own: side(own), opponent: side(opponent), ...rest });
const fighter = (own: boolean, letter: string | null, name: string): Fighter => ({ paste: "p", hp: null, status: null, boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, mega: false, at: { own, letter, name } });

const worldFacts = (own: Record<string, { last_move: string | null; volatiles?: Record<string, { moved_before: boolean }> } | null>, opponent: typeof own = {}): WorldFacts => ({ own: { active: own }, opponent: { active: opponent } }) as unknown as WorldFacts;

describe("historyDeclarer", () => {
  it("declares each Pokemon's last move and when a volatile began from the world facts, and asks the last move at both ends without them", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "a", "F");
    const base = facts({ active: { a: stint({ volatile_started: { taunt: { started_turn: 3, after_residual: false }, bide: { started_turn: 2, after_residual: true } } }) } }, { active: { a: stint() } });
    const world = worldFacts({ a: { last_move: "tackle", volatiles: { taunt: { moved_before: false } } } }, { a: null });
    const { combatants } = historyDeclarer(base, record, exists, world)([[own], [foe]]);
    const mine = combatants[0]?.[0] as { last_move?: unknown; volatiles?: { value: { id: string; moved_before?: boolean }[] } };
    expect(mine.last_move).toEqual({ kind: "known", value: "tackle" });
    expect(mine.volatiles?.value.map((v) => [v.id, v.moved_before])).toEqual([["bide", undefined], ["taunt", false]]);
    expect((combatants[1]?.[0] as { last_move?: unknown }).last_move).toEqual({ kind: "known", value: null }); // no move yet
    const refused = historyDeclarer(base, record, exists, null)([[own], [foe]]).combatants[0]?.[0] as { last_move?: unknown };
    expect(refused.last_move).toEqual({ kind: "unknown", ends: [null, "tackle"] });
    expect((historyDeclarer(facts({ active: { a: { name: "A", unknown: true } } }), record, exists, world)([[own], [foe]]).combatants[0]?.[0] as { last_move?: unknown }).last_move).toEqual({ kind: "unknown", ends: [null, "tackle"] });
  });

  it("declares a known stint's facts, own moves_used only, and the roster's berry and last item", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "a", "F");
    const declare = historyDeclarer(
      facts(
        { active: { a: stint() }, roster: { A: { ate_berry: false, last_item: null } } },
        { active: { a: stint({ moves_used: null, stall_counter: 3, last_move_failed: null }) }, roster: { F: { ate_berry: true, last_item: "oranberry" } } },
      ),
      record,
      exists,
    );
    const { combatants, battle } = declare([[own], [foe]]);
    expect(combatants[0]?.[0]).toEqual({
      move_actions: { kind: "known", value: 2 },
      stall_counter: { kind: "known", value: 0 },
      times_attacked: { kind: "known", value: 1 },
      hurt_this_turn: { kind: "known", value: false },
      last_move_failed: { kind: "known", value: false },
      stats_raised_this_turn: { kind: "known", value: false },
      stats_lowered_this_turn: { kind: "known", value: true },
      moves_used: { kind: "known", value: ["tackle"] },
      last_move: { kind: "unknown", ends: [null, "tackle"] }, // no world facts were given
      ate_berry: { kind: "known", value: false },
      last_item: { kind: "known", value: null },
    });
    const foeFacts = combatants[1]?.[0] as Record<string, unknown>;
    expect(foeFacts.moves_used).toBeUndefined(); // opponent PP is never declared (and with no revealed move there is nothing to ask)
    const attacker = { ...foe, revealed: ["tackle"] };
    const revealed = historyDeclarer(facts({}, { active: { a: stint({ moves_used: null }) } }), record, exists)([[own], [attacker]], [attacker]).combatants[1]?.[0] as Record<string, unknown>;
    expect(revealed.moves_used).toEqual({ kind: "unknown", ends: [[], ["tackle"]] }); // the declared revealed moves used once against unused, for the Pokemon whose move is asked
    // a foe whose action is only queued (not the asked one) is not asked: its queued move's success is not read
    expect((historyDeclarer(facts({}, { active: { a: stint({ moves_used: null }) } }), record, exists)([[own], [attacker]]).combatants[1]?.[0] as Record<string, unknown>).moves_used).toBeUndefined();
    expect(foeFacts.stall_counter).toEqual({ kind: "known", value: 3 });
    expect(foeFacts.last_move_failed).toEqual({ kind: "unknown", ends: [false, true] });
    expect(foeFacts.ate_berry).toEqual({ kind: "known", value: true });
    expect(foeFacts.last_item).toEqual({ kind: "known", value: "oranberry" });
    expect(battle?.sides).toEqual([{ total_fainted: { kind: "known", value: 0 } }, { total_fainted: { kind: "known", value: 0 } }]);
    expect(battle?.turn).toBeUndefined(); // nothing is dated
    expect(battle?.field_last_move).toEqual({ kind: "known", value: null });
  });

  it("asks an unsettled Pokemon, a refused channel and an unsettled roster entry at both ends", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "a", "F");
    const settled = historyDeclarer(facts({ active: { a: { name: "A", unknown: true } }, roster: { A: { ate_berry: null, last_item: UNSETTLED } } }), record, exists)([[own], [foe]]).combatants[0]?.[0] as Record<string, unknown>;
    expect(settled.move_actions).toEqual({ kind: "unknown", ends: [0, 20] });
    expect(settled.hurt_this_turn).toEqual({ kind: "unknown", ends: [false, true] });
    expect(settled.ate_berry).toEqual({ kind: "unknown", ends: [false, true] });
    expect(settled.last_item).toEqual({ kind: "unknown", ends: [null, "sitrusberry"] });
    // an unsettled stint and a refused channel: volatiles are unknown too (none against the record's ids started), and a foe's used moves are unknown
    const revealedFoe: Fighter = { ...foe, revealed: ["tackle", "protect"] };
    expect(settled.volatiles).toEqual({ kind: "unknown", ends: [[], [{ id: "stall", started_turn: 4, after_residual: false }, { id: "stockpile", started_turn: 4, after_residual: false }]] });
    const unsettledFoe = historyDeclarer(facts({}, { active: { a: { name: "F", unknown: true } } }), record, exists)([[own], [revealedFoe]], [revealedFoe]).combatants[1]?.[0] as Record<string, unknown>;
    expect(unsettledFoe.moves_used).toEqual({ kind: "unknown", ends: [[], ["tackle", "protect"]] });
    const refused = historyDeclarer(null, record, exists)([[own], [revealedFoe]], [revealedFoe]);
    expect(refused.combatants[1]?.[0]).toMatchObject({ volatiles: { kind: "unknown" }, moves_used: { kind: "unknown", ends: [[], ["tackle", "protect"]] } });
    expect(refused.battle?.turn).toBe(1); // dated: the turn is 1 without facts
    expect(refused.combatants[1]?.[0]).toMatchObject({ move_actions: { kind: "unknown", ends: [0, 20] }, last_item: { kind: "unknown", ends: [null, "sitrusberry"] } });
    expect(refused.battle?.field_last_move).toEqual({ kind: "unknown", ends: [null, "tackle"] });
    expect(refused.battle?.sides?.[0]).toEqual({ total_fainted: { kind: "unknown", ends: [0, 20] } });
  });

  it("declares attackers among the query's Pokemon at both ends of the damage, and nothing for attackers it does not hold", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "a", "F");
    const other = fighter(false, "b", "G");
    const hit = historyDeclarer(facts({ active: { a: stint({ attackers_this_turn: ["opponent:a", "opponent:c"] }) } }), record, exists)([[own], [foe, other]]).combatants[0]?.[0] as { attacked_by?: unknown };
    // one declared and one not ("opponent:c"): the fact is unknown, the second end naming every declared other Pokemon
    expect(hit.attacked_by).toEqual({ kind: "unknown", ends: [[], [{ side: 1, position: 0, damage: 20 }, { side: 1, position: 1, damage: 20 }]] });
    // a hit whose source is unknown, or by a Pokemon the query does not declare: unknown, nobody at the first end and every declared other at the second
    const unplaced = historyDeclarer(facts({ active: { a: stint({ attackers_this_turn: null }) } }), record, exists)([[own], [foe, other]]).combatants[0]?.[0] as { attacked_by?: { ends: unknown[][] } };
    expect(unplaced.attacked_by?.ends[0]).toEqual([]);
    expect(unplaced.attacked_by?.ends[1]).toHaveLength(2);
    const absent = historyDeclarer(facts({ active: { a: stint({ attackers_this_turn: ["opponent:c"] }) } }), record, exists)([[own], [foe]]).combatants[0]?.[0] as { attacked_by?: { ends: unknown[][] } };
    expect(absent.attacked_by).toEqual({ kind: "unknown", ends: [[], [{ side: 1, position: 0, damage: 20 }]] });
    const fromOwn = historyDeclarer(facts({}, { active: { a: stint({ attackers_this_turn: ["own:a"] }) } }), record, exists)([[own], [foe]]).combatants[1]?.[0] as { attacked_by?: unknown };
    expect(fromOwn.attacked_by).toEqual({ kind: "unknown", ends: [[{ side: 0, position: 0, damage: 0 }], [{ side: 0, position: 0, damage: 20 }]] });
    const exact = historyDeclarer(facts({ active: { a: stint({ attackers_this_turn: ["opponent:a"] }) } }), record, exists)([[own], [foe, other]]).combatants[0]?.[0] as { attacked_by?: unknown };
    expect(exact.attacked_by).toEqual({ kind: "unknown", ends: [[{ side: 1, position: 0, damage: 0 }], [{ side: 1, position: 0, damage: 20 }]] });
    const none = historyDeclarer(facts({ active: { a: stint() } }), record, exists)([[own], [foe]]).combatants[0]?.[0] as { attacked_by?: unknown };
    expect(none.attacked_by).toBeUndefined();
  });

  it("declares every volatile and pending slot condition with its turn: derived names as their condition, an id the engine lacks as it is", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "b", "F");
    const declare = historyDeclarer(
      facts(
        { active: { a: stint({ volatile_started: { taunt: { started_turn: 3, after_residual: false }, bide: { started_turn: 2, after_residual: true }, stockpile2: { started_turn: 2, after_residual: false }, protosynthesisatk: { started_turn: 1, after_residual: false }, perish3: { started_turn: 1, after_residual: false } } }) }, pending_slot: { a: { wish: { started_turn: 3, after_residual: false, remaining: 1 }, nothing: { started_turn: 1, after_residual: false, remaining: 1 } }, c: { wish: { started_turn: 3, after_residual: false, remaining: 1 } } } },
        {},
        { field_last_move: "nomove" },
      ),
      record,
      exists,
    );
    const { combatants, battle } = declare([[own], [foe]]);
    expect((combatants[0]?.[0] as { volatiles?: unknown }).volatiles).toEqual({ kind: "known", value: [{ id: "bide", started_turn: 2, after_residual: true }, { id: "perish3", started_turn: 1, after_residual: false }, { id: "protosynthesis", started_turn: 1, after_residual: false }, { id: "stockpile", started_turn: 2, after_residual: false, starts: 2 }, { id: "taunt", started_turn: 3, after_residual: false }] });
    expect(battle?.sides?.[0]?.pending_slots).toEqual({ kind: "known", value: [{ position: 0, id: "nothing", started_turn: 1, after_residual: false }, { position: 0, id: "wish", started_turn: 3, after_residual: false }] });
    expect(battle?.turn).toBe(4);
    expect(battle?.field_last_move).toEqual({ kind: "known", value: "nomove" }); // declared as it is: the engine refuses an id it lacks and the answer is history_unknown
  });

  it("leaves the volatiles out when the record says omitted, and still dates the pending slots", () => {
    const own = fighter(true, "a", "A");
    const foe = fighter(false, "a", "F");
    const declare = historyDeclarer(facts({ active: { a: stint({ volatile_started: { taunt: { started_turn: 3, after_residual: false } } }) }, pending_slot: { a: { wish: { started_turn: 3, after_residual: false, remaining: 1 } } } }), { ...record, volatiles: "omitted" }, exists);
    const { combatants, battle } = declare([[own], [foe]]);
    expect((combatants[0]?.[0] as { volatiles?: unknown }).volatiles).toBeUndefined();
    expect(battle?.sides?.[0]?.pending_slots).toBeDefined();
    expect(battle?.turn).toBe(4);
  });

  it("leaves a placeholder combatant and a benched Pokemon without stint facts", () => {
    const placeholder: Fighter = { ...fighter(true, "a", "A") };
    delete placeholder.at;
    const bench = fighter(true, null, "B");
    const declare = historyDeclarer(facts({ active: { a: stint() }, roster: { B: { ate_berry: true, last_item: "noitem" } } }), record, exists);
    const { combatants, battle } = declare([[bench, placeholder], [placeholder]]);
    expect(combatants[0]?.[0]).toEqual({ ate_berry: { kind: "known", value: true }, last_item: { kind: "known", value: "noitem" } }); // a benched Pokemon: the roster's facts only
    expect(combatants[0]?.[1]).toBeUndefined();
    expect(battle?.sides?.[1]).toEqual({}); // a side of placeholders declares nothing
    // a position that holds no Pokemon now, and a name the roster does not know
    const vacant = historyDeclarer(facts({ active: { a: null } }), record, exists)([[fighter(true, "a", "Z")], [fighter(false, "a", "Y")]]);
    expect(vacant.combatants[0]?.[0]).toBeUndefined();
  });
});
