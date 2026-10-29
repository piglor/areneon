// Isolated hits in one player's own channel (ADR-025 section 7, ticket #1132; finding N9). A hit is evidence about a
// hidden set only when the probe can account for everything that shaped it: one attacker, one defender, one hit, and
// no modifier the reader cannot see. This module reads the shared reader's frames (the state after every line) and
// finds the hits, each with the shown hit points before and after and the crit line, and says why a hit is NOT
// isolated. Which modifiers count is reviewed data (`BeliefConfig.damage_evidence.non_isolating`), passed in; the only
// protocol facts written here are the line grammar (`|move|`, `|-damage|`, `|-crit|`) and the lines a single hit may
// carry. It computes no damage and imports no engine.

import type { PublicFrame, PublicMon, PublicState } from "@areneon/public-state";
import { parseLine as parse } from "@areneon/public-state";

const idOf = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** A line the reader can parse, else an empty-kind line (a blank line separates actions). */
const parseLine = (raw: string): ReturnType<typeof parse> => (raw.startsWith("|") ? parse(raw) : { raw, kind: "", args: [], tags: {} });

/** The reviewed rules (`BeliefConfig.damage_evidence.non_isolating`). */
export interface NonIsolating {
  readonly abilities: readonly string[];
  readonly lasting_moves: readonly string[];
  readonly moves: readonly string[];
  readonly targets: readonly string[];
}

export interface HitContext {
  readonly rules: NonIsolating;
  /** The engine's target type of a move id, or null for a move it lacks. */
  targetOf(move: string): string | null;
  /** The abilities a Pokemon in play may hold: its known ability alone, else every ability its species can have. */
  abilitiesOf(mon: PublicMon): readonly string[];
  /** The Pokemon's species may hide its identity (an identity-hiding ability in the vocabulary, `BeliefConfig.identity_hiding_abilities`). */
  mayHide(mon: PublicMon): boolean;
  /** The probe cannot declare this field (a pseudo-weather other than Trick Room, a terrain it does not take): the hit is not isolated. */
  declarable(state: PublicState): boolean;
}

export type NotIsolated =
  | "move_tags"
  | "no_target"
  | "same_side"
  | "extra_line"
  | "damage_count"
  | "damage_tag"
  | "unseen_mon"
  | "volatile"
  | "changed_mon"
  | "listed_move"
  | "lasting_move"
  | "listed_target"
  | "listed_ability"
  | "possible_disguise"
  | "undeclared_field";

export interface HitSide {
  readonly own: boolean;
  readonly letter: string;
  /** The index into the side's roster in the state at the move line. */
  readonly index: number;
}

export interface Hit {
  /** The index of the `|move|` frame. */
  readonly frame: number;
  readonly turn: number;
  readonly move: string;
  readonly attacker: HitSide;
  readonly defender: HitSide;
  readonly crit: boolean;
  /** The defender's shown hit points at the move line (before the damage) and at the damage line. */
  readonly before: { readonly num: number; readonly den: number };
  readonly after: { readonly num: number; readonly den: number };
  /** The state at the move line: the field and stages the damage was dealt under. */
  readonly state: PublicState;
  /** Null for an isolated hit, else the first reason it is not. */
  readonly reason: NotIsolated | null;
}

/** The kinds that end the lines of one action. */
const ENDS = new Set(["move", "switch", "drag", "turn", "upkeep", "cant", "request", "win", "tie", "replace", "detailschange", "teampreview", "start", "-mega"]);
/** The kinds a single hit may carry between its move line and the next action. */
const ALLOWED = new Set(["-damage", "-crit", "-supereffective", "-resisted", "faint"]);

function sideOf(state: PublicState, ref: string): { own: boolean; letter: string; mon: PublicMon | null; index: number } | null {
  const m = /^(p[1-4])([a-z]): /.exec(ref);
  if (m === null) return null;
  const own = m[1] === state.own.player;
  const side = own ? state.own : state.opponent;
  if (!own && m[1] !== state.opponent.player) return null;
  const index = side.slots[m[2] as string];
  const mon = index === null || index === undefined ? null : (side.roster[index] ?? null);
  return { own, letter: m[2] as string, mon, index: index ?? -1 };
}

/** The hits of a channel's frames, isolated or not (`reason`). A move whose lines hold no damage line is not a hit. */
export function extractHits(frames: readonly PublicFrame[], ctx: HitContext): Hit[] {
  const out: Hit[] = [];
  let turnStart = 0;
  frames.forEach((frame, i) => {
    const line = parseLine(frame.line);
    if (line.kind === "turn") turnStart = i;
    if (line.kind !== "move" || frame.flagged !== null) return;
    // The lines of this action.
    const body: { kind: string; args: string[]; tags: Record<string, string>; frame: number }[] = [];
    for (let j = i + 1; j < frames.length; j += 1) {
      const next = parseLine((frames[j] as PublicFrame).line);
      if (ENDS.has(next.kind) || next.kind === "") break;
      body.push({ kind: next.kind, args: next.args, tags: next.tags, frame: j });
    }
    const damages = body.filter((l) => l.kind === "-damage");
    if (damages.length === 0) return;
    const state = frame.state;
    const attacker = sideOf(state, line.args[0] ?? "");
    const target = sideOf(state, damages[0]?.args[0] ?? "");
    if (attacker === null || target === null) return;
    const move = idOf(line.args[1] ?? "");
    const damage = damages[0] as (typeof body)[number];
    const afterFrame = frames[damage.frame] as PublicFrame;
    const after = (afterFrame.state[target.own ? "own" : "opponent"].roster[target.index] as PublicMon | undefined) ?? null;
    const before = target.mon;
    if (before === null || after === null || before.hp_num === null || before.hp_den === null || after.hp_num === null || after.hp_den === null) return;

    const fail = (): NotIsolated | null => {
      if (Object.keys(line.tags).length > 0) return "move_tags";
      if ((line.args[2] ?? "") === "") return "no_target";
      if (attacker.own === target.own) return "same_side";
      if (damages.length !== 1) return "damage_count";
      if (Object.keys(damage.tags).length > 0) return "damage_tag";
      if (body.some((l) => !ALLOWED.has(l.kind))) return "extra_line";
      if (body.filter((l) => l.kind === "-crit").length > 1) return "extra_line";
      const mons = [attacker.mon, target.mon];
      if (mons.some((m) => m === null)) return "unseen_mon";
      for (const m of mons as PublicMon[]) {
        if (m.volatiles.length > 0 || m.temp_species !== null || m.ability_suppressed || m.phantom === true) return "volatile";
        if (!m.own && (m.item_received || m.ability_received || m.item_lost)) return "changed_mon";
      }
      if (ctx.rules.moves.includes(move)) return "listed_move";
      for (let k = turnStart; k < i; k += 1) {
        const earlier = parseLine((frames[k] as PublicFrame).line);
        if (earlier.kind === "move" && ctx.rules.moves.includes(idOf(earlier.args[1] ?? ""))) return "listed_move";
      }
      for (let k = 0; k < i; k += 1) {
        const earlier = parseLine((frames[k] as PublicFrame).line);
        if (earlier.kind === "move" && ctx.rules.lasting_moves.includes(idOf(earlier.args[1] ?? ""))) return "lasting_move";
      }
      const kind = ctx.targetOf(move);
      if (kind !== null && ctx.rules.targets.includes(kind)) return "listed_target";
      const involved = new Set([attacker.mon, target.mon]);
      for (const side of [state.own, state.opponent]) {
        for (const index of Object.values(side.slots)) {
          const mon = index === null ? undefined : side.roster[index];
          if (mon === undefined || mon.fainted || involved.has(mon)) continue;
          if (ctx.abilitiesOf(mon).some((a) => ctx.rules.abilities.includes(a))) return "listed_ability";
        }
      }
      // A shown Pokemon may be an unrevealed identity-hiding member of the preview: the hit is not about the Pokemon it seems to be.
      if (state.opponent.roster.some((m) => ctx.mayHide(m) && !(m.brought === true || m.fainted || m.active_slot !== null))) return "possible_disguise";
      if (!ctx.declarable(state)) return "undeclared_field";
      return null;
    };
    out.push({
      frame: i,
      turn: state.turn,
      move,
      attacker: { own: attacker.own, letter: attacker.letter, index: attacker.index },
      defender: { own: target.own, letter: target.letter, index: target.index },
      crit: body.some((l) => l.kind === "-crit"),
      before: { num: before.hp_num, den: before.hp_den },
      after: { num: after.hp_num, den: after.hp_den },
      state,
      reason: fail(),
    });
  });
  return out;
}
