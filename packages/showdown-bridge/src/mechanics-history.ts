// The declared battle history of a probe query (ADR-022 erratum E-22.5.3, slice H2). The history facts of E-22.5.2
// (`activeMoveActions`, the `stall` counter, `timesAttacked`, `hurtThisTurn`, `attackedBy`, `moveLastTurnResult`,
// `statsRaisedThisTurn` / `statsLoweredThisTurn`, `ateBerry`, `lastItem`, `moveSlots[].used`, `side.totalFainted`,
// `battle.lastMove`, volatile ages and pending slot conditions) are declared in the scenario battle through the
// engine's own fields, methods and conditions, as the ADR-026 world builder writes `activeMoveActions`,
// `totalFainted`, `lastMove` and the stall counter (this module shares its writers). The probe holds no rule about
// which move reads which fact: it declares and the engine answers.
//
// Two ends rule: a fact the channel cannot settle is `unknown` in the query with its two ends. A query that holds
// one is asked once with every unknown fact at its first end, and once more per unknown fact with that fact alone at its
// second end (a corner of every combination would be 2^n queries; the facts of a Pokemon are read by different moves
// and the interaction of two unknown facts is not modelled: stated as a decision, ADR-022 E-22.5.3 leaves the
// combination open). `probeMechanics` keeps equal answers and makes different ones `unsupported` (`history_unknown`).

import type { MechanicsQuery } from "./mechanics-query.ts";
import { need } from "./mechanics-guard.ts";
import type { EffectState, WBattle, WMon, WSide } from "./world-engine.ts";
import { addVolatileAt, remainingTurns, writeStallCounter } from "./world-state.ts";

type AnyQuery = MechanicsQuery["query"];

export interface Attacker {
  side: number;
  position: number;
  damage: number;
}
export interface VolatileEntry {
  id: string;
  started_turn: number;
  after_residual: boolean;
  starts?: number;
  moved_before?: boolean;
}
export interface SlotEntry {
  position: number;
  id: string;
  started_turn: number;
  after_residual: boolean;
}

/** The facts of one Pokemon with every unknown fact resolved to one of its ends. */
export interface CombatantFacts {
  move_actions?: number;
  stall_counter?: number;
  times_attacked?: number;
  hurt_this_turn?: boolean;
  attacked_by?: Attacker[];
  last_move_failed?: boolean;
  stats_raised_this_turn?: boolean;
  stats_lowered_this_turn?: boolean;
  ate_berry?: boolean;
  last_item?: string | null;
  moves_used?: string[];
  volatiles?: VolatileEntry[];
}

export interface ResolvedHistory {
  turn?: number;
  field_last_move?: string | null;
  sides: { total_fainted?: number; pending_slots?: SlotEntry[] }[];
  /** The Pokemon with declared facts: `place` counts the declared Pokemon side by side (the position a draw reports). */
  combatants: { side: number; position: number; place: number; facts: CombatantFacts }[];
}

/** A resolved history and the engine condition the stall counter is written to (reviewed data: the world writers record). */
export interface HistoryDeclaration {
  resolved: ResolvedHistory;
  stall: string;
}

type Fact<T> = { kind: "known"; value: T } | { kind: "unknown"; ends: readonly T[] };

const COMBATANT_FACTS = ["move_actions", "stall_counter", "times_attacked", "hurt_this_turn", "attacked_by", "last_move_failed", "stats_raised_this_turn", "stats_lowered_this_turn", "ate_berry", "last_item", "moves_used", "volatiles"] as const;

/** The declared Pokemon of a query in the order a result's positions count them, with the history each carries. */
function combatantsOf(query: AnyQuery): { side: number; position: number; history: unknown }[] {
  if (query.kind === "damage") {
    return [
      { side: 0, position: 0, history: query.attacker.history },
      { side: 1, position: 0, history: query.defender.history },
    ];
  }
  if (query.kind === "speed_order" || query.kind === "move_outcome") return query.sides.flatMap((side, s) => side.map((combatant, p) => ({ side: s, position: p, history: combatant.history })));
  return [];
}

/** The battle-level history block of a query, if the kind has one. */
const battleHistoryOf = (query: AnyQuery): unknown => (query.kind === "stats" ? undefined : query.history);

/** The values a fact can take: its value, or both ends. */
const valuesOf = (fact: Fact<unknown>): readonly unknown[] => (fact.kind === "known" ? [fact.value] : fact.ends);

interface Resolution {
  history: ResolvedHistory | null;
  /** Per unknown fact, in the order they are met: whether its two ends differ. */
  unknowns: boolean[];
}

/** Resolve every fact of a query: known ones to their value, unknown ones to their second end when `ones` holds their index and to their first end otherwise. */
export function resolveHistory(query: AnyQuery, ones: ReadonlySet<number>): Resolution {
  const unknowns: boolean[] = [];
  let any = false;
  const pick = <T>(fact: Fact<T>): T => {
    any = true;
    if (fact.kind === "known") return fact.value;
    const index = unknowns.length;
    unknowns.push(JSON.stringify(fact.ends[0]) !== JSON.stringify(fact.ends[1]));
    return fact.ends[ones.has(index) ? 1 : 0] as T;
  };
  const battle = battleHistoryOf(query) as { turn?: number; field_last_move?: Fact<string | null>; sides?: { total_fainted?: Fact<number>; pending_slots?: Fact<SlotEntry[]> }[] } | undefined;
  const resolved: ResolvedHistory = { sides: [{}, {}], combatants: [] };
  if (battle !== undefined) {
    any = true;
    if (battle.turn !== undefined) resolved.turn = battle.turn;
    if (battle.field_last_move !== undefined) resolved.field_last_move = pick(battle.field_last_move);
    for (const [index, side] of (battle.sides ?? []).entries()) {
      if (side.total_fainted !== undefined) (resolved.sides[index] as ResolvedHistory["sides"][number]).total_fainted = pick(side.total_fainted);
      if (side.pending_slots !== undefined) (resolved.sides[index] as ResolvedHistory["sides"][number]).pending_slots = pick(side.pending_slots);
    }
  }
  for (const [place, { side, position, history }] of combatantsOf(query).entries()) {
    if (history === undefined) continue;
    const declared = history as Record<string, Fact<unknown> | undefined>;
    const facts: Record<string, unknown> = {};
    for (const key of COMBATANT_FACTS) {
      const fact = declared[key];
      if (fact !== undefined) facts[key] = pick(fact);
    }
    resolved.combatants.push({ side, position, place, facts: facts as CombatantFacts });
  }
  return { history: any ? resolved : null, unknowns };
}

/**
 * The histories a query is asked with: the first end of every unknown fact, then one more per unknown fact whose two ends differ, with that
 * fact alone at its second end. A query with no history is `[null]` (the probe's old path, byte for byte).
 */
export function historyVariants(query: AnyQuery): (ResolvedHistory | null)[] {
  const base = resolveHistory(query, new Set());
  const out: (ResolvedHistory | null)[] = [base.history];
  for (const [index, differs] of base.unknowns.entries()) if (differs) out.push(resolveHistory(query, new Set([index])).history);
  return out;
}

/** What the profile cannot say of the history of a query: names resolve to declared Pokemon and the dating facts have a turn to count from. */
export function historyProblems(query: AnyQuery): string[] {
  const problems: string[] = [];
  const declared = combatantsOf(query);
  const sizes = query.kind === "damage" ? [1, 1] : query.kind === "stats" ? [] : query.sides.map((side) => side.length);
  const battle = battleHistoryOf(query) as { turn?: number; sides?: { pending_slots?: Fact<SlotEntry[]> }[] } | undefined;
  let dated = false;
  for (const { side, position, history } of declared) {
    if (history === undefined) continue;
    const facts = history as Record<string, Fact<unknown> | undefined>;
    for (const [key, fact] of Object.entries(facts)) {
      if (fact === undefined) continue;
      for (const value of valuesOf(fact)) {
        if (key === "volatiles" && (value as VolatileEntry[]).length > 0) dated = true;
        if (key === "attacked_by") {
          for (const attacker of value as Attacker[]) {
            if ((sizes[attacker.side] ?? 0) <= attacker.position) problems.push(`/history of side ${side} position ${position} names an attacker the sides do not declare`);
          }
        }
      }
    }
  }
  for (const [side, entry] of (battle?.sides ?? []).entries()) {
    if (entry.pending_slots === undefined) continue;
    for (const value of valuesOf(entry.pending_slots) as SlotEntry[][]) {
      if (value.length > 0) dated = true;
      for (const slot of value) if ((sizes[side] ?? 0) <= slot.position) problems.push(`/history/sides/${side} names a slot position the side does not declare`);
    }
  }
  if (dated && battle?.turn === undefined) problems.push("/history needs a turn to date its volatiles and slot conditions");
  return problems;
}

// ---- the engine's objects, narrowed here so the shared engine types stay as they are ----

export interface HMon extends WMon {
  timesAttacked: number;
  hurtThisTurn: number | null;
  attackedBy: unknown[];
  moveLastTurnResult: boolean | null | undefined;
  statsRaisedThisTurn: boolean;
  statsLoweredThisTurn: boolean;
  ateBerry: boolean;
  lastItem: string;
  moveSlots: { id: string; pp: number; maxpp: number; used?: boolean }[];
  side: HSide;
  gotAttacked(move: string, damage: number, source: WMon): void;
}

export interface HSide extends WSide {
  pokemon: HMon[];
  slotConditions: Record<string, EffectState | undefined>[];
  addSlotCondition(target: number, id: string, source: WMon): unknown;
}

export interface HBattle extends WBattle {
  sides: HSide[];
  lastMove: { id: string } | null;
  dex: WBattle["dex"] & { moves: { get(id: string): { exists: boolean } } };
}

/** Where a declaration reports the Pokemon it is working on, so a random draw names it. */
export interface HistoryContext {
  stall: string;
  at(place: number): void;
}

/** Declare the history on the scenario battle. Every value is absolute; nothing declared is left to the scenario's fresh value. */
export function declareHistory(battle: HBattle, history: ResolvedHistory, context: HistoryContext): void {
  if (history.turn !== undefined) battle.turn = history.turn;
  for (const [index, entry] of history.sides.entries()) {
    const side = battle.sides[index];
    need(side !== undefined, "the history names a side the scenario does not have");
    if (entry.total_fainted !== undefined) (side as HSide).totalFainted = entry.total_fainted;
    for (const slot of entry.pending_slots ?? []) declareSlot(battle, side as HSide, slot);
  }
  if (history.field_last_move !== undefined) {
    if (history.field_last_move === null) battle.lastMove = null;
    else {
      need(battle.dex.moves.get(history.field_last_move).exists, "the declared last move is not an engine move");
      battle.lastMove = battle.dex.getActiveMove(history.field_last_move);
    }
  }
  for (const { side, position, place, facts } of history.combatants) {
    context.at(place);
    const mon = battle.sides[side]?.pokemon[position];
    need(mon !== undefined, "the history names a Pokemon the scenario does not have");
    declareFacts(battle, mon as HMon, facts, context.stall, history.turn);
  }
}

/** A timed slot condition added through the engine as if it began at `started_turn`: its own start handler reads the turn, so the engine's own end turn follows. */
function declareSlot(battle: HBattle, side: HSide, slot: SlotEntry): void {
  const found = battle.dex.conditions.get(slot.id);
  need(found.exists, "the declared slot condition is not an engine condition");
  const holder = side.pokemon[slot.position];
  need(holder !== undefined, "the declared slot condition names a position the side does not have");
  const now = battle.turn;
  battle.turn = slot.started_turn;
  try {
    side.addSlotCondition(slot.position, slot.id, holder as HMon);
  } finally {
    battle.turn = now;
  }
  need(side.slotConditions[slot.position]?.[slot.id] !== undefined, "the engine refused the declared slot condition");
}

function declareFacts(battle: HBattle, mon: HMon, facts: CombatantFacts, stall: string, turn: number | undefined): void {
  if (facts.move_actions !== undefined) mon.activeMoveActions = facts.move_actions;
  if (facts.stall_counter !== undefined) {
    writeStallCounter(mon, stall, facts.stall_counter);
    need(facts.stall_counter === 0 || mon.volatiles[stall] !== undefined, "the engine refused the declared stall counter");
  }
  if (facts.times_attacked !== undefined) mon.timesAttacked = facts.times_attacked;
  if (facts.hurt_this_turn !== undefined) mon.hurtThisTurn = facts.hurt_this_turn ? 1 : null;
  if (facts.attacked_by !== undefined) {
    mon.attackedBy = [];
    for (const attacker of facts.attacked_by) {
      const source = battle.sides[attacker.side]?.pokemon[attacker.position];
      need(source !== undefined && source !== mon, "the declared attacker is not another declared Pokemon");
      mon.gotAttacked("", attacker.damage, source as HMon);
    }
  }
  // The engine's `moveLastTurnResult` is false for a failed action and anything else otherwise.
  if (facts.last_move_failed !== undefined) mon.moveLastTurnResult = !facts.last_move_failed;
  if (facts.stats_raised_this_turn !== undefined) mon.statsRaisedThisTurn = facts.stats_raised_this_turn;
  if (facts.stats_lowered_this_turn !== undefined) mon.statsLoweredThisTurn = facts.stats_lowered_this_turn;
  if (facts.ate_berry !== undefined) mon.ateBerry = facts.ate_berry;
  if (facts.last_item !== undefined) {
    need(facts.last_item === null || battle.dex.items.get(facts.last_item).exists, "the declared last item is not an engine item");
    mon.lastItem = facts.last_item ?? "";
  }
  if (facts.moves_used !== undefined) {
    const used = new Set(facts.moves_used);
    for (const id of used) need(mon.moveSlots.some((slot) => slot.id === id), "a declared used move is not a move of the Pokemon");
    for (const slot of mon.moveSlots) slot.used = used.has(slot.id);
  }
  for (const volatile of facts.volatiles ?? []) {
    need(battle.dex.conditions.get(volatile.id).exists, "the declared volatile is not an engine condition");
    // A stacking volatile (Stockpile) restarts once per start; each start is the engine's own handler.
    for (let start = 0; start < (volatile.starts ?? 1); start += 1) addVolatileAt(battle, mon, volatile.id, volatile.moved_before ?? true);
    need(mon.volatiles[volatile.id] !== undefined, "the engine refused the declared volatile");
    if (turn !== undefined) {
      const remaining = remainingTurns(mon.volatiles[volatile.id], volatile, turn);
      need(remaining === null || remaining >= 1, "the declared volatile has run out");
    }
  }
}
