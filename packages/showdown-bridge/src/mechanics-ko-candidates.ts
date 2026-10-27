// Which roster-legal items and abilities MAY change a knock-out, read from the
// pinned engine's data (ADR-016 section 2, #1166). The KO counts of a damage
// answer are exact only while nothing acts when damage is dealt, so the
// mechanics config lists the items and abilities that stop them
// (`ko_blockers`) and the ones it reviewed and found harmless
// (`ko_non_blockers`, each with a reason). This module keeps that list tied to
// the roster: it finds the candidates and `checkKoBlockers` fails when one is
// in neither list, so a pin or regulation bump that adds an item or ability
// with such a handler cannot silently give KO counts for it.
//
// No game rule is written here. The only knowledge in code is the NAMES of the
// engine's event handlers that run when the holder is hit (below); which items
// and abilities have them is read from the engine's data for the format's mod,
// and every decision about a candidate is reviewed data in the config. A
// handler's presence is a prompt to review, not a verdict: most candidates are
// non-blockers (an immunity, a boost on the holder, damage to the attacker).
// Limit: an effect the engine implements in its simulation code instead of an
// item or ability handler has no hook to find (Tera Shell's effectiveness is in
// sim/pokemon.ts); it is not roster-legal at this pin, and a pin bump that
// makes it so needs a reviewer, because no derivation can see it.

import type { Engine } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/**
 * Engine handlers (data/items.ts and data/abilities.ts event names) that act on
 * the holder of an item or ability when it takes a hit, so that a count taken
 * from the roll vectors alone may differ from what the engine deals:
 *
 * - `onDamage`: runs as damage is about to be dealt (Focus Sash, Sturdy, Disguise).
 * - `onDamagingHit`: after the holder was damaged by a move (Stamina, Weak Armor,
 *   the contact effects).
 * - `onSourceModifyDamage`: a modifier the holder applies to the damage it takes
 *   (Multiscale at full HP, the type-resist berries, constant reductions).
 * - `onAnyModifyDamage`, `onAnyDamage`: the same for damage to other Pokemon.
 * - `onTryHit`: before the move hits (an immunity, or a full-HP condition).
 * - `onEffectiveness`: the type effectiveness the holder takes (Disguise, Ice Face).
 * - `onEat`, `onTryEatItem`: a berry eaten on a hit or at an HP threshold.
 * - `onEmergencyExit`, `onAfterMoveSecondary`: the holder leaves, or an item
 *   moves, after the hit (Emergency Exit, Eject Button, Red Card, Pickpocket).
 */
export const KO_HOOKS: readonly string[] = [
  "onAfterMoveSecondary",
  "onAnyDamage",
  "onAnyModifyDamage",
  "onDamage",
  "onDamagingHit",
  "onEat",
  "onEffectiveness",
  "onEmergencyExit",
  "onSourceModifyDamage",
  "onTryEatItem",
  "onTryHit",
];

/** Candidate ids with the KO hooks their engine data defines, each list in `KO_HOOKS` order. */
export interface KoCandidates {
  abilities: Record<string, string[]>;
  items: Record<string, string[]>;
}

const hooksOf = (entry: unknown): string[] => KO_HOOKS.filter((hook) => typeof (entry as Record<string, unknown>)[hook] === "function");

/**
 * The roster-legal items (the vocabulary's item list) and abilities (every
 * ability some roster species may hold) whose engine data, in the format's mod,
 * defines a KO hook.
 */
export function koCandidates(engine: Engine, formatId: string, vocabulary: TeamVocabulary): KoCandidates {
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(formatId).mod);
  const abilityIds = [...new Set(vocabulary.roster.flatMap((species) => species.abilities))].sort();
  const collect = (ids: readonly string[], get: (id: string) => unknown): Record<string, string[]> => {
    const found: Record<string, string[]> = {};
    for (const id of ids) {
      const hooks = hooksOf(get(id));
      if (hooks.length > 0) found[id] = hooks;
    }
    return found;
  };
  return {
    abilities: collect(abilityIds, (id) => dex.abilities.get(id)),
    items: collect([...vocabulary.items].sort(), (id) => dex.items.get(id)),
  };
}

/**
 * Check the config's KO lists against the roster: every candidate is a blocker
 * or a reviewed non-blocker (never both), and a non-blocker entry names a
 * candidate (so a stale entry does not outlive a pin bump). Returns the
 * problems; none means ok.
 */
export function checkKoBlockers(engine: Engine, config: MechanicsConfig, vocabulary: TeamVocabulary): string[] {
  if (vocabulary.showdown_commit !== engine.commit) return ["ko_blockers: the team vocabulary is for another engine commit"];
  if (vocabulary.format_id !== config.format_id) return ["ko_blockers: the team vocabulary is for another format"];
  const problems: string[] = [];
  const candidates = koCandidates(engine, config.format_id, vocabulary);
  for (const kind of ["abilities", "items"] as const) {
    const noun = kind === "abilities" ? "ability" : "item";
    const blockers = new Set(config.ko_blockers[kind]);
    const reviewed = config.ko_non_blockers?.[kind] ?? [];
    const nonBlockers = new Set(reviewed.map((entry) => entry.id));
    for (const [id, hooks] of Object.entries(candidates[kind])) {
      if (blockers.has(id) && nonBlockers.has(id)) problems.push(`ko_non_blockers: ${noun} ${id} is also in ko_blockers`);
      else if (!blockers.has(id) && !nonBlockers.has(id)) problems.push(`ko_blockers: ${noun} ${id} has ${hooks.join(", ")} and is in neither ko_blockers nor ko_non_blockers`);
    }
    for (const id of nonBlockers) {
      if (!(id in candidates[kind])) problems.push(`ko_non_blockers: ${noun} ${id} is not a roster-legal ${noun} with a KO handler`);
    }
  }
  return problems;
}
