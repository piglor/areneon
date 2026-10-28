// The evidence allowlist of slice 2 (ADR-025 section 3): which parts of the shared reader's state
// the belief treats as hard constraints. The belief reads the reader's state at the request and
// never folds lines itself, so this module is a projection of one opponent `PublicMon`. The
// allowlist is protocol semantics that the reader (and the Python reducer) already encode, not a
// game rule, and a test pins it to the reader's line grammar:
//
//   species, forme, level, gender   `|poke|` and the switch-in details
//   brought                         the Pokemon has been in play, so it is in the brought set
//   moves_revealed                  a `|move|` the Pokemon chose (no `[from]` call by another effect)
//   ability, item                   an ability or item revealed on the Pokemon itself (not one it received or copied)
//   mega, mega_stone, species       a Mega Evolution (`detailschange`, `-mega`): forme and stone
//
// Everything else the reader tracks (hit points, status, boosts, volatiles, conditions) adds no
// constraint in this slice: the belief stays wider, never wrong. Damage taken and move order are
// slice 4. A Pokemon under Transform shows the copied Pokemon's moves, so its moves add none.

import type { PublicMon } from "@areneon/public-state";

/** The `PublicMon` fields the belief reads. */
export const EVIDENCE_FIELDS = [
  "base_species",
  "species",
  "name",
  "level",
  "gender",
  "brought",
  "fainted",
  "active_slot",
  "item",
  "item_known",
  "item_received",
  "ability",
  "ability_received",
  "ability_suppressed",
  "temp_species",
  "moves_revealed",
  "volatiles",
  "mega",
  "mega_stone",
  "phantom",
] as const satisfies readonly (keyof PublicMon)[];

/** The `PublicMon` fields the belief deliberately does not read. */
export const IGNORED_FIELDS = [
  "own",
  "shiny",
  "hp_num",
  "hp_den",
  "status",
  "boosts",
  "item_lost",
  "stats",
  "moves_own",
  "pp",
  "can_mega_evo",
] as const satisfies readonly (keyof PublicMon)[];

/** The reader's id for a Pokemon that has copied another (its moves are the copied Pokemon's). */
const TRANSFORM = "transform";

export interface SlotEvidence {
  readonly name: string | null;
  /** The forme the Pokemon is shown as, null while the preview hides it. */
  readonly shown: string | null;
  readonly level: number | null;
  readonly gender: string | null;
  /** Known to be in the brought set: it has been in play. */
  readonly settled: boolean;
  /** The item revealed on the Pokemon, if any. */
  readonly item: string | null;
  readonly ability: string | null;
  /** Moves the Pokemon chose, empty while it is transformed. */
  readonly moves: readonly string[];
  readonly transformed: boolean;
  readonly mega: boolean;
  readonly megaStone: string | null;
}

export function slotEvidence(mon: PublicMon): SlotEvidence {
  const transformed = mon.volatiles.includes(TRANSFORM);
  return {
    name: mon.name,
    shown: mon.species,
    level: mon.level,
    gender: mon.gender,
    settled: mon.brought === true || mon.fainted || mon.active_slot !== null,
    // An item or ability the Pokemon received or copied (Trick, Trace, Mummy, ...) is not its own: no evidence about its set.
    item: mon.item_known && mon.item !== null && !mon.item_received ? mon.item : null,
    ability: mon.ability_received ? null : mon.ability,
    moves: transformed ? [] : [...mon.moves_revealed],
    transformed,
    mega: mon.mega,
    megaStone: mon.mega ? mon.mega_stone : null,
  };
}

/**
 * What the trapping constraint reads of one opponent Pokemon: whether it is a foe of the own actives (in play
 * and living), the species it is shown as (a temporary forme counts: the engine reads the current species'
 * ability list), whether its ability is still its own (received, suppressed and Transformed abilities are not
 * the species' list), and the vocabulary's ability list and stone for that species.
 */
export function trapEvidence(mon: PublicMon, roster: ReadonlyMap<string, { readonly abilities: readonly string[]; readonly required_item?: string }>, undisguised: boolean): { active: boolean; species: string | null; eligible: boolean; abilities: readonly string[] | null; item: string | null } {
  const species = mon.temp_species ?? mon.species;
  const entry = species === null ? undefined : roster.get(species);
  return {
    active: mon.active_slot !== null && !mon.fainted,
    species,
    eligible: undisguised && !mon.ability_received && !mon.ability_suppressed && !mon.volatiles.includes(TRANSFORM),
    abilities: entry?.abilities ?? null,
    item: entry?.required_item ?? null,
  };
}
