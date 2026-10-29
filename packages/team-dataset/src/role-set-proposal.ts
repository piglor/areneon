// Component R of the mixture proposal (ADR-020 section 2): teams drawn from the
// engine's role sets. The module chooses WHAT a team is made of; the proposal
// turns the choice into sets with the same item, nature and Stat Point draws as
// the uniform component, so only the species, role, moves and ability differ.
//
// What one R draw does, in order:
//   1. choose `team_size` table entries uniformly and without repeats;
//   2. for each, a role (one of the entry's sets) uniformly;
//   3. the ability (from the role's list), the moves (up to `moves_per_set`,
//      without replacement from the role's pool), then item, nature and Stat
//      Points exactly as the uniform path draws them.
// A table entry that is a Mega forme is played as the forme it shows as before
// the Evolution, holding its stone (the table names that forme's abilities).

import {
  type Engine,
  type EngineData,
  megaPreviewSpecies,
  readRoleSetTable,
  type RoleSetTable,
  roleSetTableHash,
  type TeamVocabulary,
} from "@areneon/showdown-bridge";
import type { Rng } from "./stream.ts";

export class RoleSetProposalError extends Error {
  override name = "RoleSetProposalError";
}

/** The table and what the engine says about its Mega formes: the input of component R. */
export interface RoleSetsSource {
  table: RoleSetTable;
  /** `megaPreviewSpecies`: Mega roster entry to the species it shows as. */
  shows_as: ReadonlyMap<string, string>;
  /** The table's hash, which the config's component must name. */
  hash: string;
}

/** Read a table against the vocabulary (the reader's checks) and derive the Mega map from the engine. */
export function roleSetsSource(engine: Engine, vocabulary: TeamVocabulary, engineData: EngineData, table: unknown): RoleSetsSource {
  const read = readRoleSetTable(table, vocabulary);
  return { table: read, shows_as: megaPreviewSpecies(engine, vocabulary, engineData), hash: roleSetTableHash(read) };
}

/** One member of an R team, before item, nature and Stat Points. */
export interface RolePick {
  /** The roster entry (its required item is the member's item). */
  entry: string;
  /** The species the set is written for: the entry, or the forme a Mega forme shows as. */
  species: string;
  moves: readonly string[];
  abilities: readonly string[];
}

/** Draw the members of one R team. */
export function drawRolePicks(rng: Rng, source: RoleSetsSource, size: number): RolePick[] {
  const entries = source.table.species;
  if (entries.length < size) throw new RoleSetProposalError("the role-set table lists fewer species than a team holds");
  const chosen: RoleSetsSource["table"]["species"] = [];
  const used = new Set<number>();
  while (chosen.length < size) {
    const at = rng.below(entries.length);
    if (used.has(at)) continue;
    used.add(at);
    chosen.push(entries[at] as RoleSetsSource["table"]["species"][number]);
  }
  return chosen.map((entry) => {
    const role = entry.sets[rng.below(entry.sets.length)] as RoleSetsSource["table"]["species"][number]["sets"][number];
    return { entry: entry.id, species: source.shows_as.get(entry.id) ?? entry.id, moves: role.moves, abilities: role.abilities };
  });
}
