// The reader rules of `candidate_features` that the contract profile cannot say (ADR-022 section 3,
// ADR-008 section 3): rows sorted and unique by (slot, move, target, mega), and, for a query, a row for
// every damaging move part of every candidate. Pure and engine-free: the coverage rule takes the
// damaging-move test as a function, so the Python worker (`ml/areneon_ml/worker/features.py`) applies the
// same rule with the engine-data table. Each function returns a problem as fixed text, or null.

import { parseJointChoice } from "./damage-aware-choice.ts";

/** The rows of a `candidate_features` object as the wire carries them. */
export interface WireFeatureRows {
  feature_set: string;
  feature_set_hash: string;
  rows: readonly { slot: number; move: number; target: number; mega: boolean }[];
}

/**
 * The request target types that hit every foe or a random one. A damaging move part of such a type that
 * names no target location still needs at least one row. The Python reader repeats the list; a test holds
 * them equal. (The other types name one foe by location, or no foe.)
 */
export const FOE_SPREAD_TARGET_TYPES: readonly string[] = ["allAdjacent", "allAdjacentFoes", "randomNormal"];

const keyOf = (row: WireFeatureRows["rows"][number]): number[] => [row.slot, row.move, row.target, Number(row.mega)];

/** Null when the rows are strictly ascending by (slot, move, target, mega false first): sorted, and none twice. */
export function featureRowsProblem(features: WireFeatureRows): string | null {
  const rows = features.rows;
  for (let index = 1; index < rows.length; index++) {
    const before = keyOf(rows[index - 1] as WireFeatureRows["rows"][number]);
    const now = keyOf(rows[index] as WireFeatureRows["rows"][number]);
    const order = before.findIndex((value, i) => value !== now[i]);
    if (order < 0 || (before[order] as number) > (now[order] as number)) {
      return "the candidate_features rows must be sorted and unique by (slot, move, target, mega)";
    }
  }
  return null;
}

interface RequestMove {
  id?: unknown;
  target?: unknown;
}

/**
 * Null when every damaging move part of every candidate has its row, as far as the query itself says. A part that
 * aims at a foe location, and a part of a spread or random-target move that names none, need at least one row of
 * (slot, move, mega). The row of the exact location is not required: the engine lists a candidate that aims at a
 * fainted foe's location (`move 1 2` with the second foe down), and the builder rightly gives that location no row;
 * telling an absent foe from a missing row would need the battle state, which the worker does not hold. (A battle
 * always has a living foe, and every location of a move is a candidate, so a move with a damaging part has a row.)
 * `observation.log` ends in the request; `isDamaging` tells a damaging move from a status move by its id.
 */
export function featureCoverageProblem(
  query: { observation: { log: readonly string[] }; candidates: readonly { choice: string }[] },
  features: WireFeatureRows,
  isDamaging: (moveId: string) => boolean,
): string | null {
  const last = query.observation.log.at(-1) ?? "";
  if (!last.startsWith("|request|")) return "the observation's log does not end in a request";
  const request = JSON.parse(last.slice("|request|".length)) as { active?: { moves?: RequestMove[] }[] };
  if (request.active === undefined) return null; // Team Preview, a forced switch or a wait: no move to score
  const some = new Set(features.rows.map((row) => `${row.slot}/${row.move}/${row.mega}`));
  for (const { choice } of query.candidates) {
    const parts = parseJointChoice(choice);
    for (const [slot, part] of parts.entries()) {
      if (part.kind !== "move") continue;
      const move = request.active[slot]?.moves?.[part.index - 1];
      if (typeof move?.id !== "string" || !isDamaging(move.id)) continue;
      const aimsAtFoe = part.target !== null ? part.target > 0 : typeof move.target === "string" && FOE_SPREAD_TARGET_TYPES.includes(move.target);
      if (aimsAtFoe && !some.has(`${slot}/${part.index}/${part.mega}`)) return "a damaging move part of a candidate has no candidate_features row";
    }
  }
  return null;
}
