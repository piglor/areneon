// The history facts of the player's own channel as the `history` blocks of probe queries (ADR-022 erratum E-22.5.2, E-22.5.3,
// slice H3). `historyFacts` (packages/public-state) derives the facts from the own channel; this module turns them into the
// query's declarations for the Pokemon a query holds, and makes the two-ends rule's question when a fact is `unknown`.
//
// Only public facts and own facts enter: a foe's `moves_used` is never declared (opponent PP stays full), nothing of the real
// opponent team is read, and an own Pokemon's facts are exact. A fact the channel cannot settle (an Illusion, a delayed attack
// whose user left play, a last item another Pokemon's Pickup may have taken, a line the module refused) is asked at its two ends
// (`unknown_count_end`, `unknown_item`, `unknown_move` of the reviewed record); the probe keeps equal answers and makes different
// ones `unsupported` (`history_unknown`). Nothing is guessed.
//
// Known limits (the declaration is per query, over the Pokemon the query holds): an attacker of a Pokemon that the query does
// not declare is left out of `attacked_by`; the damage an attacker dealt is not a fact of the channel's facts, so a non-empty
// `attacked_by` is asked at damage 0 and at the largest count (a move that reads the amount is then `history_unknown`);
// a volatile whose id is not an engine condition (a derived name such as `stockpile2`) is not declared, and none is when the
// reviewed record says `volatiles: omitted` (the committed value: the engine refuses many volatiles' start handlers in a scenario).
import type { HistoryFacts, PokemonHistory, WorldFacts } from "@areneon/public-state";
import { UNSETTLED } from "@areneon/public-state";
import type { BattleHistory, CombatantHistory } from "@areneon/schema/types/v1/mechanics-query.ts";
import type { Fighter } from "./damage-aware-view.ts";
import type { OutcomeConfig } from "./outcome-config.ts";

/** What a query's Pokemon declare of the battle's history: one block per fighter, in the lists' order, and the battle's. */
export interface DeclaredHistory {
  combatants: (CombatantHistory | undefined)[][];
  battle: BattleHistory | undefined;
}

/** The history a query declares for `sides` (the query's own order: each side's declared Pokemon). */
export type HistoryDeclarer = (sides: readonly (readonly Fighter[])[]) => DeclaredHistory;

/** The engine-data predicates the volatile ids are resolved with (a printed name such as `stockpile2` is its condition and a layer count). */
export interface HistoryExists {
  condition(id: string): boolean;
  /** The condition restarts (a stacking volatile such as Stockpile: its start handler runs once per layer). */
  stacking(id: string): boolean;
}

/** The numbers and stat ids the engine appends to a volatile's printed name (the reader's matching rule). */
const NUMBER = /\d+$/;
const STAT = /(atk|def|spa|spd|spe)$/;


type CountFact = NonNullable<CombatantHistory["move_actions"]>;
type BoolFact = NonNullable<CombatantHistory["hurt_this_turn"]>;
type AttackersFact = NonNullable<CombatantHistory["attacked_by"]>;
type Attacker = { side: number; position: number; damage: number };

const known = <T>(value: T): { kind: "known"; value: T } => ({ kind: "known", value });
/** Order of two distinct ids (a Pokemon holds a volatile once). */
const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : 1);

/**
 * The declarer of one decision. `facts` null means the module refused a line of the channel: every fact it would have
 * settled is then unknown, for every Pokemon.
 */
export function historyDeclarer(facts: HistoryFacts | null, config: OutcomeConfig, exists: HistoryExists, world: WorldFacts | null = null): HistoryDeclarer {
  const end = config.unknown_count_end;
  const unknownCount = (): CountFact => ({ kind: "unknown", ends: [0, end] });
  const unknownBool = (): BoolFact => ({ kind: "unknown", ends: [false, true] });
  const unknownMove = (): NonNullable<CombatantHistory["last_move"]> => ({ kind: "unknown", ends: [null, config.unknown_move] });

  /** The facts of an unsettled Pokemon: every one the channel cannot give. */
  const unsettled = (): CombatantHistory => ({
    move_actions: unknownCount(),
    stall_counter: unknownCount(),
    times_attacked: unknownCount(),
    hurt_this_turn: unknownBool(),
    last_move_failed: unknownBool(),
    stats_raised_this_turn: unknownBool(),
    stats_lowered_this_turn: unknownBool(),
    last_move: unknownMove(),
  });

  /**
   * Attackers of `fighter` among the query's Pokemon, by position strings of the facts (`own:a`, `opponent:b`). A hit by a Pokemon the query
   * does not declare (or by one the channel cannot place) makes the fact unknown: its first end names nobody, its second every declared other
   * Pokemon at the largest damage, so an answer that reads the attackers or the damage differs between the ends (`history_unknown`).
   */
  const attackersOf = (fighter: Fighter, positions: readonly string[] | null, sides: readonly (readonly Fighter[])[]): AttackersFact | undefined => {
    const others: Attacker[] = [];
    sides.forEach((list, side) =>
      list.forEach((other, position) => {
        if (other !== fighter) others.push({ side, position, damage: 0 });
      }),
    );
    const named =
      positions === null
        ? []
        : others.filter((slot) => {
            const there = ((sides[slot.side] as readonly Fighter[])[slot.position] as Fighter).at;
            return there !== undefined && positions.includes(`${there.own ? "own" : "opponent"}:${there.letter}`);
          });
    if (positions !== null && named.length === positions.length) return { kind: "unknown", ends: [named.map((slot) => ({ ...slot, damage: 0 })), named.map((slot) => ({ ...slot, damage: end }))] as [Attacker[], Attacker[]] };
    return { kind: "unknown", ends: [[], others.map((slot) => ({ ...slot, damage: end }))] as [Attacker[], Attacker[]] };
  };

  /**
   * The condition a printed volatile id names, declared through the engine's own start handler: the id itself, a stacking condition with
   * its layer count (`stockpile2` is Stockpile started twice), or the condition without a stat suffix (`protosynthesisatk`). An id that
   * resolves to none is declared as it is: the engine refuses it and the answer is `history_unknown`, never a dropped fact.
   */
  const resolved = (printed: string): { id: string; starts?: number } => {
    if (exists.condition(printed)) return { id: printed };
    const layers = NUMBER.exec(printed);
    const base = printed.replace(NUMBER, "");
    if (layers !== null && exists.condition(base) && exists.stacking(base)) return { id: base, starts: Number(layers[0]) };
    const stat = printed.replace(STAT, "");
    return stat !== printed && exists.condition(stat) ? { id: stat } : { id: printed };
  };

  const one = (fighter: Fighter, sides: readonly (readonly Fighter[])[]): CombatantHistory | undefined => {
    const at = fighter.at;
    if (at === undefined) return undefined; // a placeholder combatant: nothing is declared for it
    const out: CombatantHistory = {};
    if (facts === null) {
      Object.assign(out, unsettled());
      out.ate_berry = unknownBool();
      out.last_item = { kind: "unknown", ends: [null, config.unknown_item] };
      return out;
    }
    const side = at.own ? facts.own : facts.opponent;
    const stint: PokemonHistory | null | undefined = at.letter === null ? undefined : side.active[at.letter];
    if (stint !== undefined && stint !== null) {
      if (stint.unknown) Object.assign(out, unsettled());
      else {
        out.move_actions = known(stint.move_actions);
        out.stall_counter = known(stint.stall_counter ?? 0);
        out.times_attacked = known(stint.times_attacked);
        out.hurt_this_turn = known(stint.hurt_this_turn);
        out.last_move_failed = stint.last_move_failed === null ? unknownBool() : known(stint.last_move_failed);
        out.stats_raised_this_turn = known(stint.stats_raised_this_turn);
        out.stats_lowered_this_turn = known(stint.stats_lowered_this_turn);
        // Opponent PP is never declared: `moves_used` is an own fact (`moves_used` is null for an opponent).
        if (at.own && stint.moves_used !== null) out.moves_used = known([...stint.moves_used]);
        // The Pokemon's own last move (Encore and Disable read it when they start), from the world facts: unknown when those refused a line.
        const mover = world === null ? undefined : (at.own ? world.own : world.opponent).active[at.letter as string];
        out.last_move = world === null ? unknownMove() : known(mover?.last_move ?? null);
        const volatiles =
          config.volatiles === "omitted"
            ? []
            : Object.entries(stint.volatile_started)
                .map(([printed, v]) => ({ ...resolved(printed), started_turn: v.started_turn, after_residual: v.after_residual, ...(mover?.volatiles[printed] === undefined ? {} : { moved_before: mover.volatiles[printed]?.moved_before }) }))
                .sort(byId);
        if (volatiles.length > 0) out.volatiles = known(volatiles);
        const attackers = stint.attackers_this_turn === null || stint.attackers_this_turn.length > 0 ? attackersOf(fighter, stint.attackers_this_turn, sides) : undefined;
        if (attackers !== undefined) out.attacked_by = attackers;
      }
    }
    const roster = side.roster[at.name];
    if (roster !== undefined) {
      if (roster.ate_berry === null) out.ate_berry = unknownBool();
      else out.ate_berry = known(roster.ate_berry);
      if (roster.last_item === UNSETTLED) out.last_item = { kind: "unknown", ends: [null, config.unknown_item] };
      else if (roster.last_item === null || roster.last_item === "") out.last_item = known(null);
      else out.last_item = known(roster.last_item);
    }
    return Object.keys(out).length === 0 ? undefined : out;
  };

  return (sides) => {
    const combatants = sides.map((list) => list.map((fighter) => one(fighter, sides)));
    let dated = false;
    const battleSides = sides.map((list) => {
      const own = (list[0] as Fighter).at?.own;
      if (own === undefined) return {} as NonNullable<BattleHistory["sides"]>[number];
      if (facts === null) return { total_fainted: unknownCount() } as NonNullable<BattleHistory["sides"]>[number];
      const side = own ? facts.own : facts.opponent;
      const entry: NonNullable<BattleHistory["sides"]>[number] = { total_fainted: known(side.total_fainted) };
      const slots: { position: number; id: string; started_turn: number; after_residual: boolean }[] = [];
      for (const [letter, byCondition] of Object.entries(side.pending_slot).sort()) {
        const position = list.findIndex((fighter) => fighter.at?.letter === letter);
        if (position < 0) continue;
        for (const [id, slot] of Object.entries(byCondition).sort()) slots.push({ position, id, started_turn: slot.started_turn, after_residual: slot.after_residual });
      }
      if (slots.length > 0) {
        dated = true;
        entry.pending_slots = known(slots);
      }
      return entry;
    });
    const battle: BattleHistory = { sides: battleSides as NonNullable<BattleHistory["sides"]> };
    if (facts === null) battle.field_last_move = { kind: "unknown", ends: [null, config.unknown_move] };
    else battle.field_last_move = known(facts.field_last_move);
    if (facts !== null && combatants.some((list) => list.some((h) => h?.volatiles !== undefined))) dated = true;
    if (dated && facts !== null) battle.turn = facts.turn;
    return { combatants, battle };
  };
}
