// The reader rules of `candidate_features` that the contract profile cannot say (ADR-022 section 3,
// ADR-008 section 3): rows sorted and unique by (slot, move, target, mega), and, for a query, a row for
// every damaging move part of every candidate. Pure and engine-free: the coverage rule takes the
// damaging-move test as a function, so the Python worker (`ml/areneon_ml/worker/features.py`) applies the
// same rule with the engine-data table. Each function returns a problem as fixed text, or null.
//
// Under `probe-damage/2` (ADR-022 erratum E-22.3.2) the object also carries `threats`, `speeds`, `own_priorities` and
// `foe_priorities`: present exactly then, each sorted and unique by its key; and, for a query, the own positions, Mega
// flags, own-priority rows and speed rows that the request and the candidates imply (`featureCoverageProblem`).
//
// Under `probe-outcome/1` (ADR-022 erratum E-22.5.3) the object carries those four arrays and `outcomes` (one row per own move part
// of the candidates and opposing-action branch); the reason `history_unknown` of an `unsupported` row is that set's alone.

import { parseJointChoice } from "./damage-aware-choice.ts";

/** The `reason` an `unsupported` row carries. */
interface Reasoned {
  reason?: string;
}

/** One `outcomes` row as the wire carries it (a `computed` row has the four values, an `unsupported` one a reason). */
export type WireOutcomeRow = {
  kind?: string;
  slot: number;
  move: number;
  target: number;
  mega: boolean;
  action: string;
  foe_move: string | null;
  assumed: boolean;
  precondition?: string;
  precondition_ppm?: number;
  hit_ppm?: number;
  success_ppm?: number;
} & Reasoned;

/** The rows of a `candidate_features` object as the wire carries them. */
export interface WireFeatureRows {
  feature_set: string;
  feature_set_hash: string;
  rows: readonly ({ slot: number; move: number; target: number; mega: boolean } & Reasoned)[];
  /** `probe-damage/2` and `probe-outcome/1` only (all four, or none). */
  threats?: readonly ({ kind?: string; own: number; mega: boolean; foe: number; move: string } & Reasoned)[];
  speeds?: readonly ({ kind?: string; own: number; mega: boolean; foe: number } & Reasoned)[];
  own_priorities?: readonly ({ kind?: string; slot: number; move: number; mega: boolean } & Reasoned)[];
  foe_priorities?: readonly ({ kind?: string; foe: number; move: string } & Reasoned)[];
  /** `probe-outcome/1` only (beside the four): per own move part and opposing-action branch, whether the move works now. */
  outcomes?: readonly WireOutcomeRow[];
  /** `probe-damage/3` only (beside the four): the belief's moves, with their inclusion in ppm. */
  belief_threats?: readonly { kind?: string; own: number; mega: boolean; foe: number; move: string; inclusion_ppm: number }[];
  /** `probe-damage/3` only: the model's refusal code, exactly when the belief was refused (then `belief_threats` is empty). */
  belief_threats_refused?: string;
}

/** The feature set whose object carries the four extra arrays. */
export const FEATURE_SET_WITH_ARRAYS = "probe-damage/2";
/** The feature set that carries the four and `belief_threats`. */
export const FEATURE_SET_WITH_BELIEF_THREATS = "probe-damage/3";
/** The feature set that carries the four and `outcomes` (`history_unknown` reasons are its alone). */
export const FEATURE_SET_WITH_OUTCOMES = "probe-outcome/1";
const ARRAYS = ["threats", "speeds", "own_priorities", "foe_priorities"] as const;
const BELIEF_ARRAYS = [...ARRAYS, "belief_threats"] as const;
const OUTCOME_ARRAYS = [...ARRAYS, "outcomes"] as const;
const EVERY_ARRAY = [...ARRAYS, "belief_threats", "outcomes"] as const;
/** The arrays a set carries: none, the four of `/2`, those and `belief_threats` of `/3`, or those and `outcomes` of `probe-outcome/1`. */
const arraysOf = (featureSet: string): readonly (typeof EVERY_ARRAY)[number][] =>
  featureSet === FEATURE_SET_WITH_BELIEF_THREATS ? BELIEF_ARRAYS : featureSet === FEATURE_SET_WITH_ARRAYS ? ARRAYS : featureSet === FEATURE_SET_WITH_OUTCOMES ? OUTCOME_ARRAYS : [];
/** The most branches one own move part has in the `outcomes` (the switch branch and the targeted foe's moves): the reviewed record's own ceiling. */
export const MAX_OUTCOME_BRANCHES = 5;
const MILLION = 1_000_000;

/**
 * The request target types that hit every foe or a random one. A damaging move part of such a type that
 * names no target location still needs at least one row. The Python reader repeats the list; a test holds
 * them equal. (The other types name one foe by location, or no foe.)
 */
export const FOE_SPREAD_TARGET_TYPES: readonly string[] = ["allAdjacent", "allAdjacentFoes", "randomNormal"];

const keyOf = (row: WireFeatureRows["rows"][number]): number[] => [row.slot, row.move, row.target, Number(row.mega)];
/** The part of an outcome row: the own move part, without its branch. */
const partKey = (r: WireOutcomeRow): Key => [r.slot, r.move, r.target, Number(r.mega)];

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
 * set none may be present. Under `probe-outcome/1` the four and `outcomes` (by (slot, move, target, mega, action with switch first,
 * foe move id)), whose branches per part are checked here too (`outcomesProblem`).
 */
export function featureRowsProblem(features: WireFeatureRows): string | null {
  if (!ascending(features.rows.map(keyOf))) return "the candidate_features rows must be sorted and unique by (slot, move, target, mega)";
  const wanted = arraysOf(features.feature_set);
  const present = EVERY_ARRAY.filter((name) => features[name] !== undefined);
  if (present.length !== wanted.length || !wanted.every((name) => present.includes(name))) {
    return `the candidate_features arrays ${ARRAYS.join(", ")} are present exactly under ${FEATURE_SET_WITH_ARRAYS}, and with belief_threats under ${FEATURE_SET_WITH_BELIEF_THREATS}, and with outcomes under ${FEATURE_SET_WITH_OUTCOMES}`;
  }
  const orders: [string, Key[]][] = [
    ["threats", (features.threats ?? []).map((r) => [r.own, Number(r.mega), r.foe, r.move])],
    ["speeds", (features.speeds ?? []).map((r) => [r.own, Number(r.mega), r.foe])],
    ["own_priorities", (features.own_priorities ?? []).map((r) => [r.slot, r.move, Number(r.mega)])],
    ["foe_priorities", (features.foe_priorities ?? []).map((r) => [r.foe, r.move])],
    ["belief_threats", (features.belief_threats ?? []).map((r) => [r.own, Number(r.mega), r.foe, r.move])],
    ["outcomes", (features.outcomes ?? []).map((r) => [...partKey(r), Number(r.action !== "switch"), r.foe_move ?? ""])],
  ];
  for (const [name, keys] of orders) if (!ascending(keys)) return `the candidate_features ${name} must be sorted and unique by their key`;
  if (features.belief_threats_refused !== undefined && (features.feature_set !== FEATURE_SET_WITH_BELIEF_THREATS || (features.belief_threats ?? []).length > 0)) {
    return "a candidate_features belief_threats_refused is a probe-damage/3 field and means the belief_threats are empty";
  }
  // A belief threat is a move the channel has not shown: never one that is already a revealed threat of that foe.
  const revealed = new Set((features.threats ?? []).map((r) => `${r.foe}/${r.move}`));
  if ((features.belief_threats ?? []).some((r) => revealed.has(`${r.foe}/${r.move}`))) return "a candidate_features belief_threats row repeats a move already in threats for that foe";
  if (features.feature_set !== FEATURE_SET_WITH_OUTCOMES && [features.rows, ...ARRAYS.map((name) => features[name] ?? [])].some((list) => list.some((r) => r.reason === HISTORY_UNKNOWN))) {
    return `a candidate_features reason ${HISTORY_UNKNOWN} is a ${FEATURE_SET_WITH_OUTCOMES} value`;
  }
  return outcomesProblem(features.outcomes ?? []);
}

const HISTORY_UNKNOWN = "history_unknown";

/**
 * The rules of the `outcomes` that need no request: a `switch` branch has no foe move and is never assumed, a `move` branch names its
 * foe move; a computed row's precondition fits its ppm (passes 10^6, fails 0, a chance in between) and `success_ppm` is
 * floor(precondition x hit / 10^6); each part (the rows are sorted, so a part's rows are adjacent) has its `switch` branch first, then
 * one to four move branches, and a foe that revealed nothing has exactly one, `assumed`.
 */
function outcomesProblem(rows: readonly WireOutcomeRow[]): string | null {
  for (const r of rows) {
    if (r.action === "switch" ? r.foe_move !== null || r.assumed : r.foe_move === null) return fixed("outcomes: a switch branch has no foe move and is not assumed, a move branch names its foe move");
    if (r.kind !== "computed") continue;
    const pre = r.precondition_ppm as number;
    if (r.precondition === "passes" ? pre !== MILLION : r.precondition === "fails" ? pre !== 0 : pre <= 0 || pre >= MILLION) return fixed("outcomes: the precondition does not fit its ppm");
    if (r.success_ppm !== Math.floor((pre * (r.hit_ppm as number)) / MILLION)) return fixed("outcomes: success_ppm must be floor(precondition_ppm x hit_ppm / 10^6)");
  }
  for (let from = 0; from < rows.length; ) {
    const part = partKey(rows[from] as WireOutcomeRow).join("/");
    let to = from;
    while (to < rows.length && partKey(rows[to] as WireOutcomeRow).join("/") === part) to++;
    const group = rows.slice(from, to);
    const moves = group.slice(1);
    if (group[0]?.action !== "switch") return fixed("outcomes: each own move part starts with its switch branch");
    if (moves.length < 1 || moves.length > MAX_OUTCOME_BRANCHES - 1 || moves.some((r) => r.action !== "move")) return fixed(`outcomes: each own move part has a switch branch and one to ${MAX_OUTCOME_BRANCHES - 1} move branches`);
    if (moves.some((r) => r.assumed) && (moves.length !== 1 || moves[0]?.assumed !== true)) return fixed("outcomes: an assumed branch is the only move branch of its part");
    from = to;
  }
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
  return arraysOf(features.feature_set).length > 0 ? arrayCoverageProblem(request, query.candidates, features) : null;
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
  const located = new Set<string>(); // "slot/move/target/mega": the own move parts the outcomes are about
  for (const list of parsed) {
    for (const [slot, part] of list.entries()) {
      if (part.kind === "move") {
        parts.add(`${slot}/${part.index}/${part.mega}`);
        located.add(`${slot}/${part.index}/${part.target ?? 0}/${part.mega}`);
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
  if ((features.belief_threats ?? []).some((r) => !variants.has(`${r.own}/${r.mega}`))) return fixed("belief_threats name an own Pokemon or Mega variant the request and candidates do not imply");
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
  if (arraysOf(features.feature_set).includes("outcomes")) {
    // One block of rows per own move part of the candidates whose Pokemon lives and whose foe location holds a living foe (the
    // untargeted and ally-targeted parts are about the lowest foe, always present); none in a request with no move to score.
    const expectedOutcomes = request.active === undefined ? [] : [...located].map((part) => part.split("/").map((v, i) => (i === 3 ? Number(v === "true") : Number(v)))).filter(([slot, , target]) => alive(slot as number) && foeLocations.size > 0 && ((target as number) <= 0 || foeLocations.has(target as number)));
    expectedOutcomes.sort((a, b) => (a[0] as number) - (b[0] as number) || (a[1] as number) - (b[1] as number) || (a[2] as number) - (b[2] as number) || (a[3] as number) - (b[3] as number));
    const shown = [...new Set((features.outcomes ?? []).map((r) => keyText(partKey(r))))];
    if (shown.join(",") !== expectedOutcomes.map(keyText).join(",")) return fixed("outcomes must hold the rows of exactly the own move parts of the candidates that have a foe to ask about");
  }
  if (request.active !== undefined) {
    // A move request: the builder asks the priority of every revealed foe move, so each threat's foe move has its row.
    const priced = new Set(foes.map((r) => `${r.foe}/${r.move}`));
    if (threats.some((r) => !priced.has(`${r.foe}/${r.move}`))) return fixed("foe_priorities must hold a row for the foe move of every threat in a move request");
  }
  return null;
}
