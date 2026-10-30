// Observed orders of same-priority actions in one player's own channel (ADR-025 section 7, ticket #1133; owner answer Q6).
// The order in which an opponent Pokemon and one of the player's own used a move in the same turn is evidence about the
// opponent's Speed-relevant fields only when the engine's turn-start order is what produced it: the two moves have the
// same priority and nothing that changes either Pokemon's Speed or the order happened before both. This module reads the
// shared reader's frames and finds the pairs, each with the field the order was decided under, and says why a pair is NOT
// isolated. What counts as a speed-changing event the reader's state does not show is reviewed data
// (`BeliefConfig.speed_evidence.non_isolating`), passed in; what it does show (stat stages, status, an item or ability
// revealed or lost, a volatile, weather, terrain, a field or side condition, a Mega Evolution) is compared in the state
// itself, so no list is needed for it. The only protocol facts written here are the line grammar (`|turn|`, `|move|` and
// its `[from]` tag). It computes no Speed or priority and imports no engine.

import type { PublicFrame, PublicMon, PublicSide, PublicState } from "@areneon/public-state";
import { parseLine as parse } from "@areneon/public-state";

const idOf = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const parseLine = (raw: string): ReturnType<typeof parse> => (raw.startsWith("|") ? parse(raw) : { raw, kind: "", args: [], tags: {} });

/** The reviewed rules (`BeliefConfig.speed_evidence.non_isolating`). */
export interface SpeedNonIsolating {
  readonly abilities: readonly string[];
  readonly lasting_moves: readonly string[];
  readonly moves: readonly string[];
}

export interface PairContext {
  readonly rules: SpeedNonIsolating;
  /** The engine's priority of a move id, or null for a move it lacks. */
  priorityOf(move: string): number | null;
  /** The abilities a Pokemon in play may hold: its known ability alone, else every ability its species can have. */
  abilitiesOf(mon: PublicMon): readonly string[];
  /** The Pokemon's species may hide its identity (an identity-hiding ability in the vocabulary). */
  mayHide(mon: PublicMon): boolean;
  /** The probe cannot declare this field (a pseudo-weather but Trick Room, a terrain it does not take). */
  declarable(state: PublicState): boolean;
  /** The player's own Pokemon holds an item or ability the probe does not model (`BeliefConfig.speed_evidence.unmodelled`). */
  unmodelledOwn(mon: PublicMon): boolean;
}

export type NotPaired =
  | "repeated_actor"
  | "priority"
  | "changed_state"
  | "volatile"
  | "changed_mon"
  | "unmodelled_own"
  | "listed_move"
  | "lasting_move"
  | "listed_ability"
  | "possible_disguise"
  | "undeclared_field";

export interface PairSide {
  /** The `|move|` frame of the Pokemon's action. */
  readonly frame: number;
  /** The index into the side's roster. */
  readonly index: number;
  readonly letter: string;
  readonly move: string;
}

export interface SpeedPair {
  readonly turn: number;
  readonly own: PairSide;
  readonly foe: PairSide;
  /** Which of the two acted first. */
  readonly first: "own" | "foe";
  /** The state just before the later action: the field and stages the order was decided under (equal to the turn's start in every field the pair reads). */
  readonly state: PublicState;
  /** The state at the turn's start: priority is fixed there, so the hit points a priority bonus reads (Gale Wings at full health) are this state's. */
  readonly start: PublicState;
  /** Null for an isolated pair, else the first reason it is not. */
  readonly reason: NotPaired | null;
}

interface Action {
  readonly frame: number;
  readonly own: boolean;
  readonly index: number;
  readonly letter: string;
  readonly move: string;
}

function sideOf(state: PublicState, ref: string): { own: boolean; letter: string; index: number } | null {
  const m = /^(p[1-4])([a-z]): /.exec(ref);
  if (m === null) return null;
  const own = m[1] === state.own.player;
  const side = own ? state.own : state.opponent;
  const index = side.slots[m[2] as string] ?? -1;
  return index < 0 ? null : { own, letter: m[2] as string, index };
}

/** What the two acting Pokemon and the field show of Speed: every field of the state the speed order reads. */
function signature(state: PublicState, own: number, foe: number): string {
  const mon = (side: PublicSide, index: number): unknown => {
    const m = side.roster[index] as PublicMon;
    return [m.status, Object.entries(m.boosts).filter(([, v]) => v !== 0).sort(([a], [b]) => (a < b ? -1 : 1)), m.item, m.item_lost, m.ability, m.ability_suppressed, [...m.volatiles].sort(), m.temp_species, m.mega, m.species];
  };
  return JSON.stringify([state.weather, state.terrain, [...state.field_conditions].sort(), state.own.conditions, state.opponent.conditions, mon(state.own, own), mon(state.opponent, foe)]);
}

/** The pairs of a channel's frames, isolated or not (`reason`): one per (own Pokemon, opponent Pokemon) that each used a move in a turn. */
export function extractPairs(frames: readonly PublicFrame[], ctx: PairContext): SpeedPair[] {
  const out: SpeedPair[] = [];
  const lines = frames.map((f) => parseLine(f.line));
  const turns = lines.flatMap((l, i) => (l.kind === "turn" ? [i] : []));
  turns.forEach((start, k) => {
    const end = turns[k + 1] ?? frames.length;
    // The chosen actions of the turn: `|move|` lines that no other move called (the `[from]` tag).
    const actions: Action[] = [];
    for (let i = start + 1; i < end; i += 1) {
      const line = lines[i] as ReturnType<typeof parse>;
      if (line.kind !== "move" || (frames[i] as PublicFrame).flagged !== null || "from" in line.tags) continue;
      const who = sideOf((frames[i] as PublicFrame).state, String(line.args[0]));
      if (who !== null) actions.push({ frame: i, ...who, move: idOf(String(line.args[1])) });
    }
    // The turn starts at its decision request when the channel has one before the first action (the player's own values are
    // updated from it, after the `|turn|` line), else at the `|turn|` line.
    const requestAt = frames.findIndex((f, i) => i > start && i < end && f.kind === "request");
    const started = requestAt >= 0 && requestAt < (actions[0]?.frame ?? end) ? requestAt : start;
    const count = (a: Action): number => actions.filter((b) => b.own === a.own && b.index === a.index).length;
    for (const own of actions.filter((a) => a.own)) {
      for (const foe of actions.filter((a) => !a.own)) {
        const later = Math.max(own.frame, foe.frame);
        const state = (frames[later - 1] as PublicFrame).state;
        const startState = (frames[started] as PublicFrame).state;
        const fail = (): NotPaired | null => {
          if (count(own) !== 1 || count(foe) !== 1) return "repeated_actor";
          const [pa, pb] = [ctx.priorityOf(own.move), ctx.priorityOf(foe.move)];
          if (pa === null || pb === null || pa !== pb) return "priority";
          // At no line between the turn's start and the later action did either Pokemon's Speed or the field differ from the start
          // (a change and its undoing, a Psych Up after a Weak Armor boost, is still a change the order was decided under).
          const before = signature(startState, own.index, foe.index);
          for (let i = started + 1; i < later; i += 1) if (signature((frames[i] as PublicFrame).state, own.index, foe.index) !== before) return "changed_state";
          const [mine, theirs] = [state.own.roster[own.index] as PublicMon, state.opponent.roster[foe.index] as PublicMon];
          for (const m of [mine, theirs]) {
            if (m.volatiles.length > 0 || m.temp_species !== null || m.ability_suppressed || m.phantom === true) return "volatile";
          }
          if ([theirs.item_received, theirs.ability_received, theirs.item_lost].some(Boolean)) return "changed_mon";
          if (ctx.unmodelledOwn(mine)) return "unmodelled_own";
          for (let i = started; i < later; i += 1) {
            const earlier = lines[i] as ReturnType<typeof parse>;
            if (earlier.kind === "move" && ctx.rules.moves.includes(idOf(String(earlier.args[1])))) return "listed_move";
          }
          for (let i = 0; i < later; i += 1) {
            const earlier = lines[i] as ReturnType<typeof parse>;
            if (earlier.kind === "move" && ctx.rules.lasting_moves.includes(idOf(String(earlier.args[1])))) return "lasting_move";
          }
          const involved = new Set([mine, theirs]);
          for (const side of [state.own, state.opponent]) {
            for (const index of Object.values(side.slots)) {
              const m = index === null ? null : (side.roster[index] as PublicMon);
              if (m === null || m.fainted || involved.has(m)) continue;
              if (ctx.abilitiesOf(m).some((a) => ctx.rules.abilities.includes(a))) return "listed_ability";
            }
          }
          // A shown Pokemon may be an unrevealed identity-hiding member of the preview: the order is not about the Pokemon it seems to be.
          if (state.opponent.roster.some((m) => ctx.mayHide(m) && !(m.brought === true || m.fainted || m.active_slot !== null))) return "possible_disguise";
          if (!ctx.declarable(state)) return "undeclared_field";
          return null;
        };
        const side = (a: Action): PairSide => ({ frame: a.frame, index: a.index, letter: a.letter, move: a.move });
        out.push({ turn: state.turn, own: side(own), foe: side(foe), first: own.frame < foe.frame ? "own" : "foe", state, start: startState, reason: fail() });
      }
    }
  });
  return out;
}
