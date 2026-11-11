// The candidate feature builder, tier A, fourth feature set `probe-outcome/1` (ADR-022 erratum E-22.5.3, slice H3).
// It holds `probe-damage/2`'s five arrays (`rows`, `threats`, `speeds`, `own_priorities`, `foe_priorities`) with the same keys and row
// shapes, built by the same code with the history facts of the player's own channel DECLARED in every probe query (so they are not
// `/2`'s bytes: a Last Respects row scales with the faint count, a Rage Fist threat with the hits taken), plus one more array:
//
//   outcomes  per own move part (slot, move position, target, Mega flag) and opposing-action branch: whether the move works NOW and with
//             what chance, as the pinned engine answers it (`move_outcome`): `precondition` (passes, fails, or a chance), its ppm, the
//             accuracy chance `hit_ppm` and `success_ppm` = floor(10^6 x precondition x hit / 10^12).
//
// Branches (the opponent's choice is hidden, so each part is asked once per branch, rows never depending on how the engine happened to
// answer): `switch` (every foe switches) and one per revealed move of the targeted foe (an untargeted or ally-targeted move: the foe at
// the lowest location), in move-id order; a foe with no revealed move gets one branch from the reviewed record's assumed rule, flagged
// `assumed`. The other foe acts with its first revealed damaging move, else its assumed move; the ally is declared with no action (R16:
// an ally's action is not modelled). At most `max_branches` per part; every branch is listed even when the answers are equal.
//
// The foe side is ADR-018's assumed set (revealed facts as revealed, the rest from the v1 record); the foe's history facts are the public
// lines as the player saw them; the opponent's PP is never declared; nothing of the opponent's unrevealed sets enters. The own side is
// exact. A part whose target location holds no living foe has no rows (the engine lists such candidates; there is nothing to ask).
// A request with no move to score has no rows (Team Preview, wait); a forced switch has `/2`'s rule and no outcomes. A declared state the
// engine refuses is an `unsupported` row with `query_kind` (the #1246 rule), never a refused request.
//
// Pure and deterministic: the same bytes with or without the memo and in any candidate order. The input is a `PlayerChannel`, the player's
// own team and the candidate strings; no opponent team, hidden verdict, `BattleRecord` or seed is a parameter of anything here.
import { type HistoryFacts, HistoryFactsError, historyFacts, type HistoryFactsData, ownHistoryChannel } from "@areneon/public-state";
import { buildArrays, type ArrayContext, type CostTrace, type FoePriorityRow, type OwnPriorityRow, type SpeedRow, type ThreatRow } from "./candidate-features-2.ts";
import { type FeatureConfigs, type FeatureMemo, type FeatureRow, FEATURE_SET_OUTCOME, createFeatureMemo } from "./candidate-features.ts";
import { type HistoryDeclarer, historyDeclarer } from "./candidate-history.ts";
import type { Candidate } from "./choices.ts";
import { array, DamageAwareError, type DamageView, type Foe, object, ProbeRefusal } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_FORMAT } from "./golden.ts";
import { goldenHistoryFactsData } from "./history-facts-data.ts";
import type { MechanicsQuery } from "./mechanics-query.ts";
import type { PlainReason } from "./mechanics-result.ts";
import type { PlayerChannel } from "./observation.ts";
import type { OutcomeConfig } from "./outcome-config.ts";

export { FEATURE_SET_OUTCOME };

/** Why an `outcomes` row (or, with history declared, any row of the set) is `unsupported`: the probe's reasons, `history_unknown` included. */
export type OutcomeReason = PlainReason | "history_unknown";

/** `T` with its `unsupported` rows' reason widened to `history_unknown`: the rows of this set, whose queries declare history. */
type Widen<T> = T extends { kind: "unsupported"; reason: infer R } ? Omit<T, "reason"> & { reason: R | "history_unknown" } : T;

export type OutcomeValue =
  | { kind: "computed"; precondition: "passes" | "fails" | "chance"; precondition_ppm: number; hit_ppm: number; success_ppm: number }
  | { kind: "unsupported"; reason: OutcomeReason };

/** One own move part against one opposing-action branch. `target` is the part's choice location (0 for none). */
export type OutcomeRow = { slot: number; move: number; target: number; mega: boolean; action: "switch" | "move"; foe_move: string | null; assumed: boolean } & OutcomeValue;

export interface CandidateFeaturesOutcome {
  feature_set: typeof FEATURE_SET_OUTCOME;
  feature_set_hash: string;
  rows: Widen<FeatureRow>[];
  threats: Widen<ThreatRow>[];
  speeds: Widen<SpeedRow>[];
  own_priorities: Widen<OwnPriorityRow>[];
  foe_priorities: Widen<FoePriorityRow>[];
  outcomes: OutcomeRow[];
}

const MILLION = 1_000_000;
/** Order of two distinct ids (a duplicate is never compared). */
const compare = (a: string, b: string): number => (a < b ? -1 : 1);

// ---------------------------------------------------------------- the history data and the declarer

const dataOf = new WeakMap<Engine, HistoryFactsData>();
/** The history facts' engine data of the pinned checkout (read once per engine; only the committed format has a golden table). */
export function historyDataOf(engine: Engine, configs: FeatureConfigs): HistoryFactsData {
  const given = configs.outcome?.history_data;
  if (given !== undefined) return given;
  if (configs.config.format_id !== GOLDEN_FORMAT) throw new DamageAwareError(`probe-outcome/1 has history facts data for ${GOLDEN_FORMAT} only`);
  let data = dataOf.get(engine);
  if (data === undefined) dataOf.set(engine, (data = goldenHistoryFactsData()));
  return data;
}

/** The facts of the channel, or null when the module refuses a line (every fact it would settle is then unknown for every Pokemon). */
export function factsOf(channel: PlayerChannel, data: HistoryFactsData): HistoryFacts | null {
  try {
    return historyFacts({ channel: ownHistoryChannel(channel.player, channel.lines), data });
  } catch (error) {
    if (!(error instanceof HistoryFactsError)) throw error;
    return null;
  }
}

function declarerOf(engine: Engine, configs: FeatureConfigs, setup: NonNullable<FeatureConfigs["outcome"]>, channel: PlayerChannel): HistoryDeclarer {
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(configs.config.format_id).mod) as unknown as {
    conditions: { get(id: string): { exists: boolean } };
    moves: { get(id: string): { exists: boolean } };
    items: { get(id: string): { exists: boolean } };
  };
  return historyDeclarer(factsOf(channel, historyDataOf(engine, configs)), setup.config, {
    condition: (id) => dex.conditions.get(id).exists,
    move: (id) => dex.moves.get(id).exists,
    item: (id) => dex.items.get(id).exists,
  });
}

// ---------------------------------------------------------------- branches

/** One opposing-action branch of a part. */
export interface Branch {
  action: "switch" | "move";
  foe_move: string | null;
  assumed: boolean;
}

/**
 * The branches of a part against a foe: `switch`, then one per revealed move in move-id order, or, with no revealed move, one assumed
 * branch (`assumed`); at most `max` in all (the revealed moves are cut in id order).
 */
export function branchesOf(revealed: readonly string[], assumed: string, max: number): Branch[] {
  const moves = [...new Set(revealed)].sort(compare).slice(0, max - 1);
  return [{ action: "switch", foe_move: null, assumed: false }, ...(moves.length === 0 ? [{ action: "move" as const, foe_move: assumed, assumed: true }] : moves.map((move) => ({ action: "move" as const, foe_move: move, assumed: false })))];
}

/** The move a foe is assumed to use when it has revealed none (the record's rule `assumed-set-filler`): the assumed set's filler, else its first move. */
export function assumedMoveOf(foe: Foe, dex: DamageView["dex"]): string {
  const id = foe.assumed.filler_moves[0] ?? foe.assumed.set.moves[0];
  if (id === undefined) throw new DamageAwareError("an assumed set holds no move");
  return dex.moves.get(id).id;
}

/** The move the other foe acts with: its first revealed damaging move, else its assumed one. */
export function otherFoeMove(foe: Foe, dex: DamageView["dex"]): string {
  return foe.assumed.revealed_moves.find((id) => dex.moves.get(id).exists && dex.moves.get(id).category !== "Status") ?? assumedMoveOf(foe, dex);
}

// ---------------------------------------------------------------- the build

type OutcomeQuery = Extract<MechanicsQuery["query"], { kind: "move_outcome" }>;
type Action = OutcomeQuery["actions"][number];

/** The features of one decision under `probe-outcome/1`. Throws `FeatureBuildError` as `candidateFeatures2` does. */
export function candidateFeaturesOutcome(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
  trace?: CostTrace,
): CandidateFeaturesOutcome {
  const setup = configs.outcome;
  if (setup === undefined) throw new DamageAwareError("probe-outcome/1 needs its reviewed record");
  // The declarer is built only for a request that has rows (the facts replay the channel): the empty ones are cheap.
  const out = buildArrays(
    engine,
    configs,
    channel,
    team,
    candidates,
    memo,
    trace,
    FEATURE_SET_OUTCOME,
    (context) => ({ outcomes: outcomeRows(setup.config, context) }),
    { outcomes: [] as OutcomeRow[] },
    lazyDeclarer(() => declarerOf(engine, configs, setup, channel)),
  );
  return out as unknown as CandidateFeaturesOutcome;
}

/** A declarer that builds its facts on first use (a request with nothing to declare never replays the channel). */
function lazyDeclarer(make: () => HistoryDeclarer): HistoryDeclarer {
  let declarer: HistoryDeclarer | null = null;
  return (sides) => (declarer ??= make())(sides);
}

/** The `outcomes` of one move request. */
function outcomeRows(record: OutcomeConfig, { view, state, entries, foes, parsed, body, measured }: ArrayContext): OutcomeRow[] {
  const active = Array.isArray(body.active) ? (body.active as unknown[]) : null;
  if (active === null) return []; // a forced switch asks no move
  const actives = entries.filter((entry) => entry.slot !== null);
  if (actives.length === 0 || foes.length === 0) return [];

  // Every distinct own move part of the candidates: (slot, move position, target location, Mega flag).
  const parts = new Map<string, { slot: number; move: number; target: number | null; mega: boolean; id: string }>();
  for (const list of parsed) {
    list.forEach((part, slot) => {
      if (part.kind !== "move") return;
      const key = `${slot}/${part.index}/${part.target}/${part.mega}`;
      if (parts.has(key)) return;
      const moves = array(object(active[slot], "an active entry").moves, "moves");
      const id = String(object(moves[part.index - 1], "a move of the request").id);
      parts.set(key, { slot, move: part.index, target: part.target, mega: part.mega, id });
    });
  }
  const ownFighters = actives.map((entry) => entry.fighter);
  const foeFighters = foes.map((foe) => foe.fighter);
  const out: OutcomeRow[] = [];
  const ask = (query: OutcomeQuery): OutcomeValue => {
    let result: ReturnType<DamageView["query"]>;
    try {
      result = view.query(query);
    } catch (error) {
      if (!(error instanceof ProbeRefusal)) throw error;
      view.noteRefused();
      return { kind: "unsupported", reason: "query_kind" };
    }
    if (result?.kind === "unsupported") return { kind: "unsupported", reason: view.reasonOf(result.reason) };
    if (result?.kind !== "computed" || result.answer.kind !== "move_outcome") throw new DamageAwareError("the probe gave no outcome to a move-outcome query");
    const { precondition, precondition_ppm: pre, hit_ppm: hit } = result.answer;
    return { kind: "computed", precondition, precondition_ppm: pre, hit_ppm: hit, success_ppm: Math.floor((pre * hit) / MILLION) };
  };

  for (const part of [...parts.values()]) {
    const actor = actives.findIndex((entry) => entry.slot === part.slot);
    if (actor < 0) continue; // the slot's Pokemon is fainted: its parts are not asked
    // The foe the part is about: the one at its target location, else (an untargeted or ally-targeted move) the lowest location.
    const aimed = part.target !== null && part.target > 0 ? foes.find((foe) => foe.loc === part.target) : [...foes].sort((a, b) => a.loc - b.loc)[0];
    if (aimed === undefined) continue; // a target location with no living foe: nothing to ask
    const targetIndex = foes.indexOf(aimed);
    const known = view.dex.moves.get(part.id).exists;
    const revealed = aimed.assumed.revealed_moves.filter((id) => view.dex.moves.get(id).exists).map((id) => view.dex.moves.get(id).id);
    const branches = branchesOf(revealed, assumedMoveOf(aimed, view.dex), record.max_branches);
    const key = { slot: part.slot, move: part.move, target: part.target ?? 0, mega: part.mega };
    for (const branch of branches) {
      const row = measured("outcomes", () => {
        if (!known) return { kind: "unsupported", reason: "query_kind" } as OutcomeValue;
        const actions: [Action, ...Action[]] = [{ side: 0, position: actor, move: part.id, mega: part.mega }];
        foes.forEach((foe, position) => {
          const move = branch.action === "switch" ? null : foe === aimed ? (branch.foe_move as string) : otherFoeMove(foe, view.dex);
          actions.push({ side: 1, position, move, mega: false });
        });
        const { sides, history } = view.declared([ownFighters, foeFighters]);
        return ask({ kind: "move_outcome", sides: sides as unknown as OutcomeQuery["sides"], actions, asked: 0, target: { side: 1, position: targetIndex }, state: view.field(state, true), history: history as NonNullable<typeof history> });
      });
      out.push({ ...key, action: branch.action, foe_move: branch.foe_move, assumed: branch.assumed, ...row });
    }
  }
  return out.sort((a, b) => a.slot - b.slot || a.move - b.move || a.target - b.target || Number(a.mega) - Number(b.mega) || Number(a.action !== "switch") - Number(b.action !== "switch") || compare(a.foe_move ?? "", b.foe_move ?? ""));
}

