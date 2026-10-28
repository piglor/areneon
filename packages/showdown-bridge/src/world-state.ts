// The state classes of the scenario world (ADR-026 Evidence 1, B2 and N5, slice W part 2). Every state item
// a player's view can show is classified by the reviewed record (`WorldWriters`): E set by an engine method
// (part 1), W set by a reviewed writer from the world facts, X resampled by the engine's own draw and counted,
// T reset by the engine's own turn rollover, R refused. This module classifies one input (`classify`: it reads
// the player's request, the shared reader's state and the world facts, never a hidden value) and writes the W
// items into the engine's objects (`writeMon`, `writeSide`, `writeConditionDuration`, `rollTurn`). The ids a
// writer or a refusal names come from the record; nothing here lists a move, ability, item or condition.

import type { ActiveFacts, PublicMon, WorldFacts } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { PokemonSet } from "./engine.ts";
import { type WBattle, type WMon, type WSide, toId } from "./world-engine.ts";
import { Refuse, type WorldRefusalCode } from "./world-refusal.ts";
import type { StateClass, StateItemId, WorldWriters } from "./world-writers.ts";

/** The record, indexed. */
export interface StatePlan {
  classOf(item: StateItemId): StateClass;
  /** The engine condition a writer adds (protect_counter, choice_lock, confusion_duration). */
  conditionOf(item: StateItemId): string;
  /** The refusal row that names an id (a volatile, a move, an ability), if any. */
  rowOf(id: string): StateItemId | undefined;
  /** The refusal row that names a protocol line kind. */
  lineRow(kind: string): boolean;
  admittedVolatile(id: string): StateClass | undefined;
  admittedTrigger(id: string): boolean;
}

export function statePlan(writers: WorldWriters): StatePlan {
  const classes = new Map(writers.items.map((item) => [item.id, item.class]));
  const conditions = new Map(writers.items.flatMap((item) => (item.condition === undefined ? [] : [[item.id, item.condition] as const])));
  const rows = new Map<string, StateItemId>();
  const lines = new Set<string>();
  for (const item of writers.items) {
    for (const id of item.ids ?? []) {
      if (item.id === "unavailable_mechanics") lines.add(id);
      else rows.set(id, item.id);
    }
  }
  const volatiles = new Map(writers.admitted.filter((entry) => entry.kind === "volatile").map((entry) => [entry.id, entry.class]));
  const triggers = new Set(writers.admitted.filter((entry) => entry.kind === "start_trigger").map((entry) => entry.id));
  return {
    classOf: (item) => classes.get(item) as StateClass,
    conditionOf: (item) => conditions.get(item) as string,
    rowOf: (id) => rows.get(id),
    lineRow: (kind) => lines.has(kind),
    admittedVolatile: (id) => volatiles.get(id),
    admittedTrigger: (id) => triggers.has(id),
  };
}

/** A refusal of a state item: its record class goes with it. */
export const refuseItem = (plan: StatePlan, item: StateItemId, detail: string, redraw = false, code: WorldRefusalCode = "state_unwritten"): never => {
  throw new Refuse(code, detail, { item, state_class: plan.classOf(item), redraw });
};

/** A W or X item whose class is R refuses a world that holds it. */
export const holds = (plan: StatePlan, item: StateItemId, detail: string, redraw = false): void => {
  if (plan.classOf(item) === "R") refuseItem(plan, item, detail, redraw);
};

/** One own Pokemon of the world: the request's entry and the brought set it is. */
export interface OwnEntry {
  name: string;
  set: PokemonSet;
  hp: number;
  maxhp: number;
  status: string | null;
  shown: PublicMon | undefined;
}

export interface Foe {
  index: number;
  set: PokemonSet;
  mon: PublicMon;
}

/** What one Pokemon of the world needs written beyond part 1's E items. */
export interface MonPlan {
  /** Position in `side.pokemon` order. */
  index: number;
  /** The facts of its position (null on the bench). */
  facts: ActiveFacts | null;
  /** Volatiles to add, in the reader's order. */
  volatiles: { id: string; class: StateClass; began: { started_turn: number; after_residual: boolean; moved_before: boolean } | null }[];
  remove_item: boolean;
  /** The Toxic stage to write, null when it is not badly poisoned and active. */
  toxic_stage: number | null;
  /** The shown major status of a Pokemon on the bench (written by the bench writer). */
  bench_status: boolean;
  /** Whether it is a draw of the engine's: a sleep counter. */
  sleeping: boolean;
}

export type Resample = { side: "own" | "opponent"; position: number; item: "sleep_counter" | "confusion_duration" | "opponent_pp" | "bench_order" };

export interface Classified {
  own: MonPlan[];
  foes: MonPlan[];
  resampled: Resample[];
}

const LETTERS = ["a", "b"] as const;

function volatilePlans(plan: StatePlan, mon: PublicMon | undefined, facts: ActiveFacts | null, index: number, side: "own" | "opponent", resampled: Resample[]): MonPlan["volatiles"] {
  const out: MonPlan["volatiles"] = [];
  for (const id of mon?.volatiles ?? []) {
    const row = plan.rowOf(id);
    if (row !== undefined) refuseItem(plan, row, `${side} Pokemon ${index} has a volatile of a refused kind`);
    if (index >= 2) refuseItem(plan, "unadmitted_id", `${side} Pokemon ${index} is on the bench with a volatile`);
    if (id === plan.conditionOf("confusion_duration")) {
      holds(plan, "confusion_duration", `${side} Pokemon ${index} is confused`);
      resampled.push({ side, position: index, item: "confusion_duration" });
      out.push({ id, class: "X", began: null });
      continue;
    }
    const admitted = plan.admittedVolatile(id);
    if (admitted === undefined) refuseItem(plan, "unadmitted_id", `${side} Pokemon ${index} has a volatile that is not admitted`);
    if (admitted === "W") {
      holds(plan, "volatile_duration", `${side} Pokemon ${index} has a volatile with a remaining duration`);
      const began = facts?.volatiles[id];
      if (began === undefined) refuseItem(plan, "volatile_duration", `${side} Pokemon ${index}: the log does not date a volatile`);
      out.push({ id, class: "W", began: began as { started_turn: number; after_residual: boolean; moved_before: boolean } });
    } else out.push({ id, class: admitted as StateClass, began: null });
  }
  return out;
}

/**
 * Classify the input before any battle is started: every state item present is E, a W item the record has a
 * writer for, an X item (counted), or it refuses the world with its item and class. Reads only the request,
 * the reader's state, the flags and the world facts.
 */
export function classify(input: {
  request: EngineRequest;
  state: { turn: number };
  own: readonly OwnEntry[];
  foes: readonly Foe[];
  facts: WorldFacts;
  plan: StatePlan;
}): Classified {
  const { request, own, foes, facts, plan } = input;
  const resampled: Resample[] = [];
  const mons = (side: "own" | "opponent", shown: readonly (PublicMon | undefined)[], statuses: readonly (string | null)[]): MonPlan[] =>
    shown.map((mon, index) => {
      const letter = LETTERS[index];
      const factsOf = side === "own" ? facts.own : facts.opponent;
      const active = letter === undefined ? null : (factsOf.active[letter] ?? null);
      const living = mon?.fainted !== true && !(side === "own" && request.side.pokemon[index]?.condition === "0 fnt");
      if (index < 2 && living) {
        if (active === null || (mon?.name != null && active.name !== mon.name)) throw new Refuse("facts_mismatch", `${side} Pokemon ${index}: the world facts name another Pokemon in its position`);
      }
      const status = statuses[index] ?? null;
      const volatiles = living ? volatilePlans(plan, mon, active, index, side, resampled) : [];
      let toxic: number | null = null;
      if (living && status === "tox" && index < 2) {
        holds(plan, "toxic_stage", `${side} Pokemon ${index} is badly poisoned`);
        if (active?.toxic_ticks == null) refuseItem(plan, "toxic_stage", `${side} Pokemon ${index}: the log shows no Toxic stage`);
        toxic = active?.toxic_ticks as number;
      }
      if (living && status !== null && index >= 2) holds(plan, "bench_status", `${side} Pokemon ${index} is on the bench with a status`);
      if (living && status === "slp") {
        holds(plan, "sleep_counter", `${side} Pokemon ${index} is asleep`);
        resampled.push({ side, position: index, item: "sleep_counter" });
      }
      return { index, facts: active, volatiles, remove_item: false, toxic_stage: toxic, bench_status: living && status !== null && index >= 2, sleeping: living && status === "slp" };
    });

  // A Pokemon that can hide as another (the record's ability ids) makes the names the channel shows unreliable: Illusion pairs are refused.
  // The same for a turn-parity ability (Truant, Slow Start) in play: its counter is not in the log.
  const byAbility = (ability: string, whose: string, index: number, redraw: boolean): void => {
    const row = plan.rowOf(toId(ability));
    if (row === "illusion_transform") refuseItem(plan, row, `${whose} Pokemon ${index} can hide as another Pokemon`, redraw);
    if (row === "truant_slow_start") refuseItem(plan, row, `${whose} Pokemon ${index} has a turn-parity ability`, redraw);
  };
  own.forEach((entry, index) => {
    if (index < 2 && request.side.pokemon[index]?.condition !== "0 fnt") byAbility(request.side.pokemon[index]?.baseAbility ?? entry.set.ability, "own", index, false);
  });
  // The opponent's real Pokemon in a disguise cannot be told from the bench, so every living member counts for a hiding ability; a turn-parity ability only matters in play.
  foes.forEach((foe, index) => {
    if (foe.mon.fainted) return;
    if (index < 2 || plan.rowOf(toId(foe.set.ability)) === "illusion_transform") byAbility(foe.set.ability, "opponent", index, true);
  });

  const ownPlans = mons("own", own.map((entry) => entry.shown), own.map((entry) => entry.status));
  const foePlans = mons("opponent", foes.map((foe) => foe.mon), foes.map((foe) => foe.mon.status));

  // Items and abilities.
  own.forEach((entry, index) => {
    const requested = request.side.pokemon[index];
    const shown = entry.shown;
    if (requested === undefined) return;
    const brought = toId(entry.set.item);
    const now = toId(requested.item);
    if (now !== brought) {
      if (now === "") {
        holds(plan, "item_removed", `own Pokemon ${index} no longer holds its item`);
        (ownPlans[index] as MonPlan).remove_item = true;
      } else refuseItem(plan, "item_replaced", `own Pokemon ${index} holds another item than it was brought with`);
    }
    // A Mega Evolution replaces the permanent ability too, so a Mega Evolved Pokemon's request shows the Mega forme's.
    if (shown?.mega !== true && toId(requested.baseAbility) !== toId(entry.set.ability)) throw new Refuse("own_team_mismatch", `own Pokemon ${index} has another ability than its set`);
    if (shown?.mega !== true && toId(requested.ability ?? requested.baseAbility) !== toId(requested.baseAbility)) refuseItem(plan, "ability_changed", `own Pokemon ${index} has another ability than it was brought with`);
    if (shown !== undefined && (shown.ability_suppressed || shown.ability_received)) refuseItem(plan, "ability_changed", `own Pokemon ${index} has a changed ability`);
    const slot = request.active?.[index];
    // A request that lists fewer moves than the set holds is a locked, charging or recharging Pokemon.
    if (slot !== undefined && slot.moves.length < requested.moves.length) refuseItem(plan, "locked_moves", `own Pokemon ${index} is locked into a move`);
  });
  foes.forEach((foe, index) => {
    const mon = foe.mon;
    if (mon.item_received) refuseItem(plan, "item_replaced", `opponent Pokemon ${index} has received an item`);
    if (mon.item_lost) {
      holds(plan, "item_removed", `opponent Pokemon ${index} has lost its item`);
      (foePlans[index] as MonPlan).remove_item = true;
    }
    if (mon.ability_received || mon.ability_suppressed) refuseItem(plan, "ability_changed", `opponent Pokemon ${index} has a changed ability`);
  });

  // Slot conditions and locked moves the log shows as the last move.
  for (const [side, plans] of [["own", ownPlans], ["opponent", foePlans]] as const) {
    for (const monPlan of plans) {
      const last = monPlan.facts?.last_move;
      if (last == null) continue;
      const row = plan.rowOf(last);
      if (row === "slot_conditions" || row === "locked_moves") refuseItem(plan, row, `${side} Pokemon ${monPlan.index} last used a move whose effect outlasts the turn`);
    }
  }

  // Hidden state declared (X): the opponent's PP and the order of its unseen bench.
  foes.forEach((foe, index) => {
    const uses = foe.mon.name == null ? undefined : facts.opponent.move_uses[foe.mon.name];
    if (uses !== undefined && Object.values(uses).some((n) => n > 4)) {
      holds(plan, "opponent_pp", `opponent Pokemon ${index} has used a move more than 4 times`);
      resampled.push({ side: "opponent", position: index, item: "opponent_pp" });
    }
  });
  const unseen = foes.map((foe, index) => ({ foe, index })).filter(({ foe, index }) => index >= 2 && !foe.mon.fainted && foe.mon.brought !== true);
  if (unseen.length >= 2) {
    holds(plan, "bench_order", "the opponent has two unseen Pokemon on its bench", true);
    resampled.push({ side: "opponent", position: (unseen[0] as { index: number }).index, item: "bench_order" });
  }
  return { own: ownPlans, foes: foePlans, resampled };
}

/** Whether the line kinds of the flags are all mechanics this format does not have. */
export function flagItem(plan: StatePlan, flags: readonly { line: string }[]): boolean {
  return flags.length > 0 && flags.every((flag) => plan.lineRow(flag.line.slice(1).split("|")[0] as string));
}

// ---- the writers: each takes engine objects and writes one W item through the engine's own state ----

/** A bench status: the engine's `setStatus` refuses a Pokemon out of play, so the writer sets the status and runs the engine's own Start handler (a sleep counter is then the engine's draw). */
export function writeBenchStatus(battle: WBattle, mon: WMon, status: string): void {
  mon.status = status;
  mon.statusState = battle.initEffectState({ id: status, target: mon });
  battle.singleEvent("Start", battle.dex.conditions.get(status), mon.statusState, mon, null, null);
}

/** The Toxic stage reached: the engine's residual increments `statusState.stage` once per tick. */
export function writeToxicStage(mon: WMon, ticks: number): void {
  mon.statusState.stage = ticks;
}

/** The item is gone (consumed or knocked off). */
export function writeItemRemoved(battle: WBattle, mon: WMon): void {
  mon.item = "";
  mon.itemState = battle.initEffectState({ id: "", target: mon });
}

/** An item or ability a scenario-start trigger changed goes back to the set's. */
export function restoreStart(battle: WBattle, mon: WMon, set: { item: string; ability: string }): void {
  if (toId(mon.item) !== toId(set.item)) {
    mon.item = toId(set.item);
    mon.itemState = battle.initEffectState({ id: mon.item, target: mon });
  }
  if (toId(mon.ability) !== toId(set.ability)) {
    mon.ability = toId(set.ability);
    mon.abilityState = battle.initEffectState({ id: mon.ability, target: mon });
  }
}

/** The last move the Pokemon chose (Encore, Disable and Instruct read it). */
export function writeLastMove(battle: WBattle, mon: WMon, move: string): void {
  const active = battle.dex.getActiveMove(move);
  mon.lastMove = active;
  mon.lastMoveUsed = active;
}

/** The choice lock: the engine's own condition added with the move it locks to. Returns whether it was added. */
export function writeChoiceLock(battle: WBattle, mon: WMon, condition: string, move: string): boolean {
  if (battle.dex.items.get(mon.item).isChoice !== true || !mon.moveSlots.some((slot) => slot.id === move)) return false;
  const before = battle.activeMove;
  battle.activeMove = battle.dex.getActiveMove(move);
  try {
    mon.addVolatile(condition);
  } finally {
    battle.activeMove = before;
  }
  return mon.volatiles[condition] !== undefined;
}

/** The Protect counter: the engine's stall condition added once per consecutive success, so its own restart logic sets the counter; the last use was last turn, so one turn of it is left. */
export function writeProtectCounter(mon: WMon, condition: string, streak: number): void {
  for (let use = 0; use < streak; use += 1) mon.addVolatile(condition);
  const held = mon.volatiles[condition];
  if (held !== undefined) held.duration = 1;
}

/** Turns of a side's and the field's counters from the log. */
export function writeSide(side: WSide, facts: { total_fainted: number; fainted_last_turn: boolean }): void {
  side.totalFainted = facts.total_fainted;
  // The engine's turn rollover moves `faintedThisTurn` to `faintedLastTurn`: set it so the rollover leaves the log's value.
  side.faintedThisTurn = facts.fainted_last_turn ? (side.pokemon.find((mon) => mon.fainted) ?? (side.pokemon[0] as WMon)) : null;
}

/**
 * Add a volatile the way the engine does when its target has not moved yet this turn: the engine's start
 * handlers read the queue (Taunt lasts a turn longer on a target that has already moved), so a target that had
 * not moved when the log started the volatile gets a pending move action in the queue for the call only.
 */
export function addVolatileAt(battle: WBattle, mon: WMon, id: string, movedBefore: boolean): void {
  const pending = { choice: "move", pokemon: mon };
  if (!movedBefore) battle.queue.list.push(pending);
  try {
    mon.addVolatile(id);
  } finally {
    if (!movedBefore) battle.queue.list.splice(battle.queue.list.indexOf(pending), 1);
  }
}

/** Remaining turns of an effect that counts down: the engine's own base length less the turns since it began, never a fresh full duration. Returns the remaining turns, null when the effect has no duration. */
export function remainingTurns(state: { duration?: number } | undefined, began: { started_turn: number; after_residual: boolean }, turn: number): number | null {
  if (state === undefined || typeof state.duration !== "number" || state.duration <= 0) return null;
  // The residual of the turn it began in counts it down once, unless it began after that residual.
  state.duration -= turn - began.started_turn - (began.after_residual ? 1 : 0);
  return state.duration;
}

/** Roll the world to its turn through the engine's own turn rollover: it resets the per-turn fields (T), counts `activeTurns`, moves the fainted-this-turn flag and runs the lock, disable and trap events that make the request. */
export function rollTurn(battle: WBattle, turn: number, inPlay: readonly { mon: WMon; turns: number; actions: number }[]): void {
  battle.turn = turn - 1;
  for (const { mon, turns, actions } of inPlay) {
    mon.activeTurns = turns - 1;
    mon.activeMoveActions = actions;
  }
  battle.endTurn();
}
