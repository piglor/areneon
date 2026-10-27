// The evidence allowlist of slice 2 (ADR-025 section 3): which parts of the shared reader's state
// the belief treats as hard constraints. The belief reads the reader's state at the request and
// never folds lines itself, so this module is a projection of one opponent `PublicMon`. The
// allowlist is protocol semantics that the reader (and the Python reducer) already encode, not a
// game rule, and a test pins it to the reader's line grammar:
//
//   species, forme, level, gender   `|poke|` and the switch-in details
//   brought                         the Pokemon has been in play, so it is in the brought set
//   moves_revealed                  a `|move|` the Pokemon chose (no `[from]` call by another effect)
//   ability, item                   an ability or item revealed on the Pokemon itself
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
  "ability",
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
  "temp_species",
  "hp_num",
  "hp_den",
  "status",
  "boosts",
  "item_lost",
  "ability_suppressed",
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
    item: mon.item_known && mon.item !== null ? mon.item : null,
    ability: mon.ability,
    moves: transformed ? [] : [...mon.moves_revealed],
    transformed,
    mega: mon.mega,
    megaStone: mon.mega ? mon.mega_stone : null,
  };
}
