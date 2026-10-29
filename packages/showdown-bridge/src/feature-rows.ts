// The reader rules of `candidate_features` that the contract profile cannot say (ADR-022 section 3,
// ADR-008 section 3): rows sorted and unique by (slot, move, target, mega), and, for a query, a row for
// every damaging move part of every candidate. Pure and engine-free: the coverage rule takes the
// damaging-move test as a function, so the Python worker (`ml/areneon_ml/worker/features.py`) applies the
// same rule with the engine-data table. Each function returns a problem as fixed text, or null.
//
// Under `probe-damage/2` (ADR-022 erratum E-22.3.2) the object also carries `threats`, `speeds`, `own_priorities` and
// `foe_priorities`: present exactly then, each sorted and unique by its key; and, for a query, the own positions, Mega
// flags, own-priority rows and speed rows that the request and the candidates imply (`featureCoverageProblem`).

import { parseJointChoice } from "./damage-aware-choice.ts";

/** The rows of a `candidate_features` object as the wire carries them. */
export interface WireFeatureRows {
  feature_set: string;
  feature_set_hash: string;
  rows: readonly { slot: number; move: number; target: number; mega: boolean }[];
  /** `probe-damage/2` only (all four, or none). */
  threats?: readonly { kind?: string; own: number; mega: boolean; foe: number; move: string }[];
  speeds?: readonly { kind?: string; own: number; mega: boolean; foe: number }[];
  own_priorities?: readonly { kind?: string; slot: number; move: number; mega: boolean }[];
  foe_priorities?: readonly { kind?: string; foe: number; move: string }[];
}

/** The feature set whose object carries the four extra arrays. */
export const FEATURE_SET_WITH_ARRAYS = "probe-damage/2";
const ARRAYS = ["threats", "speeds", "own_priorities", "foe_priorities"] as const;

/**
 * The request target types that hit every foe or a random one. A damaging move part of such a type that
 * names no target location still needs at least one row. The Python reader repeats the list; a test holds
 * them equal. (The other types name one foe by location, or no foe.)
 */
export const FOE_SPREAD_TARGET_TYPES: readonly string[] = ["allAdjacent", "allAdjacentFoes", "randomNormal"];

const keyOf = (row: WireFeatureRows["rows"][number]): number[] => [row.slot, row.move, row.target, Number(row.mega)];

type Key = readonly (number | string)[];

/** True when `keys` are strictly ascending, compared element by element (numbers numerically, ids by code unit). */
function ascending(keys: readonly Key[]): boolean {
  for (let index = 1; index < keys.length; index++) {
    const before = keys[index - 1] as Key;
    const now = keys[index] as Key;
    const order = before.findIndex((value, i) => value !== now[i]);
    if (order < 0 || (before[order] as number | string) > (now[order] as number | string)) return false;
  }
  return true;
}

/**
 * Null when the rows are strictly ascending by (slot, move, target, mega false first): sorted, and none twice. Under
 * `probe-damage/2` the four arrays must be present and strictly ascending by their keys (threats by (own, mega, foe, move
 * id), speeds by (own, mega, foe), own priorities by (slot, move, mega), foe priorities by (foe, move id)); under any other
 * set none may be present.
 */
export function featureRowsProblem(features: WireFeatureRows): string | null {
  if (!ascending(features.rows.map(keyOf))) return "the candidate_features rows must be sorted and unique by (slot, move, target, mega)";
  const present = ARRAYS.filter((name) => features[name] !== undefined);
  if (features.feature_set === FEATURE_SET_WITH_ARRAYS ? present.length !== ARRAYS.length : present.length > 0) {
    return `the candidate_features arrays ${ARRAYS.join(", ")} are present exactly under ${FEATURE_SET_WITH_ARRAYS}`;
  }
  const orders: [string, Key[]][] = [
    ["threats", (features.threats ?? []).map((r) => [r.own, Number(r.mega), r.foe, r.move])],
    ["speeds", (features.speeds ?? []).map((r) => [r.own, Number(r.mega), r.foe])],
    ["own_priorities", (features.own_priorities ?? []).map((r) => [r.slot, r.move, Number(r.mega)])],
    ["foe_priorities", (features.foe_priorities ?? []).map((r) => [r.foe, r.move])],
  ];
  for (const [name, keys] of orders) if (!ascending(keys)) return `the candidate_features ${name} must be sorted and unique by their key`;
  return null;
}

interface RequestMove {
  id?: unknown;
  target?: unknown;
}

interface RequestBody {
  active?: { moves?: RequestMove[] }[];
  forceSwitch?: unknown[];
  side?: { pokemon?: { condition?: unknown }[] };
}

/**
 * Null when every damaging move part of every candidate has its row, as far as the query itself says. A part that
 * aims at a foe location, and a part of a spread or random-target move that names none, need at least one row of
 * (slot, move, mega). The row of the exact location is not required: the engine lists a candidate that aims at a
 * fainted foe's location (`move 1 2` with the second foe down), and the builder rightly gives that location no row;
 * telling an absent foe from a missing row would need the battle state, which the worker does not hold. (A battle
 * always has a living foe, and every location of a move is a candidate, so a move with a damaging part has a row.)
 * `observation.log` ends in the request; `isDamaging` tells a damaging move from a status move by its id. Under
 * `probe-damage/2` the arrays are judged against the request and the candidates too (`arrayCoverageProblem`).
 */
export function featureCoverageProblem(
  query: { observation: { log: readonly string[] }; candidates: readonly { choice: string }[] },
  features: WireFeatureRows,
  isDamaging: (moveId: string) => boolean,
): string | null {
  const last = query.observation.log.at(-1) ?? "";
  if (!last.startsWith("|request|")) return "the observation's log does not end in a request";
  const request = JSON.parse(last.slice("|request|".length)) as RequestBody;
  if (request.active !== undefined) {
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
  } // Team Preview, a forced switch or a wait: no move to score
  return features.feature_set === FEATURE_SET_WITH_ARRAYS ? arrayCoverageProblem(request, query.candidates, features) : null;
}

const fixed = (text: string): string => `the candidate_features ${text}`;
const keyText = (key: Key): string => key.join("/");

/**
 * The `probe-damage/2` rules that need the request: the own Pokemon the arrays are about are the living actives of a move
 * request, every living benched Pokemon a `switch N` candidate names, and the own Mega variant of an active slot where a
 * candidate Mega Evolves; `threats` name only those (position, Mega flag); `speeds` hold exactly one row per (own position,
 * own Mega flag, foe location) for each of them against every foe location the arrays show; `own_priorities` hold exactly
 * one row per own move part of the candidates; a request with no move to score (Team Preview, a wait) has no row in the
 * arrays that need one, and a forced switch has no priority row.
 */
function arrayCoverageProblem(request: RequestBody, candidates: readonly { choice: string }[], features: WireFeatureRows): string | null {
  const threats = features.threats ?? [];
  const speeds = features.speeds ?? [];
  const own = features.own_priorities ?? [];
  const foes = features.foe_priorities ?? [];
  const slots = request.active ?? request.forceSwitch;
  const pokemon = request.side?.pokemon ?? [];
  const alive = (position: number): boolean => {
    const condition = pokemon[position]?.condition;
    return typeof condition === "string" && !/ fnt$/.test(condition);
  };
  const parsed = slots === undefined ? [] : candidates.map(({ choice }) => parseJointChoice(choice));

  const variants = new Set<string>(); // "position/mega"
  const megaSlots = new Set<number>();
  const parts = new Set<string>(); // "slot/move/mega"
  for (const list of parsed) {
    for (const [slot, part] of list.entries()) {
      if (part.kind === "move") {
        parts.add(`${slot}/${part.index}/${part.mega}`);
        if (part.mega) megaSlots.add(slot);
      } else if (part.kind === "switch") {
        if (pokemon[part.to - 1] === undefined) return fixed("arrays name a Pokemon the request does not list");
        if (alive(part.to - 1)) variants.add(`${part.to - 1}/false`);
      }
    }
  }
  if (request.active !== undefined) {
    for (let slot = 0; slot < request.active.length; slot++) {
      if (!alive(slot)) continue;
      variants.add(`${slot}/false`);
      if (megaSlots.has(slot)) variants.add(`${slot}/true`);
    }
  }
  const foeLocations = new Set<number>([...features.rows.map((r) => r.target), ...threats.map((r) => r.foe), ...speeds.map((r) => r.foe), ...foes.map((r) => r.foe)]);

  if (threats.some((r) => !variants.has(`${r.own}/${r.mega}`))) return fixed("threats name an own Pokemon or Mega variant the request and candidates do not imply");
  const expectedSpeeds = [...variants]
    .flatMap((variant) => [...foeLocations].map((foe): Key => [...variant.split("/").map((v, i) => (i === 0 ? Number(v) : Number(v === "true"))), foe]))
    .sort((a, b) => (a[0] as number) - (b[0] as number) || (a[1] as number) - (b[1] as number) || (a[2] as number) - (b[2] as number))
    .map(keyText);
  if (speeds.map((r) => keyText([r.own, Number(r.mega), r.foe])).join(",") !== expectedSpeeds.join(",")) {
    return fixed("speeds must hold exactly one row per own position, own Mega flag and foe location");
  }
  const expectedParts = request.active === undefined ? [] : [...parts].map((part) => part.split("/").map((v, i) => (i === 2 ? Number(v === "true") : Number(v))));
  expectedParts.sort((a, b) => (a[0] as number) - (b[0] as number) || (a[1] as number) - (b[1] as number) || (a[2] as number) - (b[2] as number));
  if (own.map((r) => keyText([r.slot, r.move, Number(r.mega)])).join(",") !== expectedParts.map(keyText).join(",")) {
    return fixed("own_priorities must hold exactly one row per own move part of the candidates");
  }
  if (request.active === undefined && foes.length > 0) return fixed("foe_priorities are empty when the request has no move to score");
  return null;
}
