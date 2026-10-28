// Trapping evidence of the player's own request (ADR-025 section 3, ticket #1127; asked by ADR-026 N-I).
//
// The player's own move request carries `trapped` and `maybeTrapped` for its active Pokemon. Both are the
// player's own information, and `maybeTrapped` is evidence about the opponent's abilities: the engine
// raises it when a foe's species could hold a trapping ability that the foe does not hold. This module turns
// the request flags into hard constraints on the abilities of the opponent actives, through an engine-backed
// port (`TrappingCheck`) that the bridge implements with a scenario battle of the pinned engine. Nothing is
// decided here: no ability, move, item or type list is written in this package, only the logic that joins the
// engine's verdicts with the observed flags.
//
// What the port says. For each informative own active k (one whose request carried no `trapped: true`, so
// the engine's own causes are known) the check gives `observed[k]` (0: no flag, 1: `maybeTrapped`) and, for a
// candidate foe (the species it is shown as and an ability), `shows(candidate)[k]`: what that own active would
// show with that foe in play. The engine raises the flag from each foe independently and the request shows
// the highest, so the flag a pair of foes shows is the larger of their two levels. Two constraints follow, both
// sound (the real world satisfies them):
//
//   Upper bound: every foe shows at most what was observed, so an ability whose level exceeds `observed[k]`
//                is excluded. (With `observed[k] = 0` a foe whose species could trap and does not hold the
//                trapping ability would have shown `maybeTrapped`.)
//   Lower bound: where `observed[k] = 1` some foe shows 1. If exactly one active foe could show it (it has an
//                ability not excluded that shows 1) that foe must; with two or more candidates the
//                disjunction would couple the slots and is left out (the belief stays wider, never wrong),
//                and with none the flag has another cause and adds nothing. The bound is also left out when
//                an active foe is not eligible (its ability is known to have changed or be suppressed, or a
//                disguise mixture is in play): such a foe's contribution is unknown.
//
// The belief applies the result as a zero weight on the excluded abilities of a slot's forme lists.

/** What a own active shows of the trapping flags: 0 nothing (or a hidden trap), 1 `maybeTrapped`. */
export type TrapLevel = 0 | 1;

/** A candidate opponent active as the engine sees it: the species it is shown as, an ability, the item the forme requires (or null). */
export interface TrapCandidate {
  readonly species: string;
  readonly ability: string;
  readonly item: string | null;
}

/**
 * The port the bridge implements over a scenario battle of the pinned engine (as `SetLegality` is). It is
 * built from the player's own request and nothing else of the own side reaches the belief.
 */
export interface TrappingCheck {
  /** Per informative own active, the level the request showed. May be empty (no information). */
  readonly observed: readonly TrapLevel[];
  /** Per informative own active, the level it would show with this candidate as both foes. Same length as `observed`. */
  shows(candidate: TrapCandidate): readonly TrapLevel[];
}

/** One opponent slot as the constraint reads it. */
export interface TrapSlot {
  /** Roster position. */
  readonly index: number;
  /** An active, living Pokemon: it is a foe of the own actives in the engine's trap loop. */
  readonly active: boolean;
  /** The species it is shown as (temporary forme included); null when the preview hides it. */
  readonly species: string | null;
  /** Its ability is its own and unchanged (not received, suppressed or copied) and no disguise mixture is in play. */
  readonly eligible: boolean;
  /** The abilities the vocabulary lists for the species it is shown as; null when the species is not in the roster. */
  readonly abilities: readonly string[] | null;
  /** The item its shown forme requires (a Mega forme's stone), else null. */
  readonly item: string | null;
}

/** Abilities to give weight 0 per roster position, and the fixed-word notes of what was applied. */
export interface TrapExclusions {
  readonly excluded: ReadonlyMap<number, ReadonlySet<string>>;
  readonly notes: ReadonlyMap<number, readonly string[]>;
  readonly constraints: readonly string[];
}

export class TrappingCheckError extends Error {
  override name = "TrappingCheckError";
}

export function trapExclusions(check: TrappingCheck, slots: readonly TrapSlot[]): TrapExclusions {
  const excluded = new Map<number, Set<string>>();
  const notes = new Map<number, string[]>();
  const none: TrapExclusions = { excluded, notes, constraints: [] };
  const K = check.observed.length;
  if (K === 0) return none;
  if (check.observed.some((level) => level !== 0 && level !== 1)) throw new TrappingCheckError("an observed trapping level is neither 0 nor 1");

  const levels = new Map<number, Map<string, readonly TrapLevel[]>>();
  for (const slot of slots) {
    if (!slot.active || !slot.eligible || slot.species === null || slot.abilities === null) continue;
    const per = new Map<string, readonly TrapLevel[]>();
    for (const ability of slot.abilities) {
      const shown = check.shows({ species: slot.species, ability, item: slot.item });
      if (shown.length !== K || shown.some((level) => level !== 0 && level !== 1)) throw new TrappingCheckError("the trapping check answered with another shape than its observation");
      per.set(ability, shown);
    }
    levels.set(slot.index, per);
  }

  const mark = (index: number, ability: string, why: string): void => {
    const set = excluded.get(index) ?? new Set<string>();
    if (!set.has(ability)) notes.set(index, [...(notes.get(index) ?? []), `trapping_${why}:${ability}`]);
    excluded.set(index, set.add(ability));
  };
  for (const [index, per] of levels) {
    for (const [ability, shown] of per) if (shown.some((level, k) => level > (check.observed[k] as TrapLevel))) mark(index, ability, "excluded");
  }

  if (levels.size === 0) return none;

  // The lower bound needs every active foe's contribution to be known.
  const known = slots.every((slot) => !slot.active || (slot.eligible && slot.species !== null && slot.abilities !== null));
  const constraints: string[] = ["trapping:upper"];
  if (known) {
    for (let k = 0; k < K; k += 1) {
      if (check.observed[k] !== 1) continue;
      const capable = [...levels].filter(([index, per]) => [...per].some(([ability, shown]) => shown[k] === 1 && !excluded.get(index)?.has(ability)));
      if (capable.length !== 1) continue;
      const [index, per] = capable[0] as [number, Map<string, readonly TrapLevel[]>];
      for (const [ability, shown] of per) if (shown[k] !== 1) mark(index, ability, "required");
      constraints.push("trapping:lower");
    }
  }
  return { excluded, notes, constraints: [...new Set(constraints)] };
}
