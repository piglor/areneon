// The candidate feature builder, tier A, third feature set `probe-damage/3` (ADR-022, the belief threats of the opponent model).
// It holds `probe-damage/2`'s rows and four arrays byte for byte (the same code builds them, `buildArrays`) and one more array:
//
//   belief_threats  what each opposing active's moves that the BELIEF says it may hold (and the channel has not shown) would do
//                   to each own Pokemon, with the belief's probability that the foe holds the move (`inclusion_ppm`).
//
// `/2`'s `threats` are the revealed moves only; a foe that has shown nothing has none, which is exactly when a player most needs
// the question "what might it hit me with". The opponent model answers it: per roster position, the marginal inclusion of each
// move over the committed usage prior and the evidence of the channel (`BeliefThreatsPort.moveInclusions`; the bridge does not
// import the opponent model). Per foe the builder keeps the first `per_foe` (4) of those moves that are damaging, reach an opposing
// Pokemon and that the foe's assumed set can hold (the damaging filter and the placement are applied BEFORE the cut, so the cut is
// four threats, not four moves of which some were status moves). The damage is ADR-018's assumed foe set plus that move (revealed
// moves, then the belief move; every other field from the v1 record), the own Pokemon at its exact HP: values on an assumed set,
// never facts about the real opponent. A move the belief gives to a foe is listed under `inclusion_ppm`; the values do not
// depend on it. Reader rules (feature-rows.ts): no move that is already one of `/2`'s `threats` for that foe, sorted and unique
// by (own, mega, foe, move id), at most 128 rows.
//
// A refused belief (a code of the model, or an unsettled identity-hiding member) leaves `belief_threats` empty; a defect of the
// run throws. Pure and deterministic: the input is a `PlayerChannel`, the player's own team and the candidate strings; the
// belief takes the reader's opponent half and the reader's flags and nothing else, so no opponent secret can enter.
import type { Candidate } from "./choices.ts";
import { type CandidateFeatures2, buildArrays, type CostTrace, type ArrayContext } from "./candidate-features-2.ts";
import type { BeliefMoveInclusion } from "./belief-features.ts";
import { type BeliefThreatsSetup, createFeatureMemo, type DamageValues, damageValuesOf, FEATURE_SET_3, type FeatureConfigs, type FeatureMemo } from "./candidate-features.ts";
import { assumedWithMove } from "./damage-aware-assumed-set.ts";
import { DamageAwareError, type Foe, foeTargetKind } from "./damage-aware-view.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { PlayerChannel } from "./observation.ts";

export { FEATURE_SET_3 };

/** What a belief move would do to one own Pokemon, and how likely the belief says the foe holds it. */
export type BeliefThreatRow = { own: number; mega: boolean; foe: number; move: string; inclusion_ppm: number } & DamageValues;

export interface CandidateFeatures3 extends Omit<CandidateFeatures2, "feature_set"> {
  feature_set: typeof FEATURE_SET_3;
  belief_threats: BeliefThreatRow[];
}

// A foe's belief moves are distinct ids, so two never compare equal.
const compare = (a: string, b: string): number => (a < b ? -1 : 1);

/** The features of one decision under `probe-damage/3`. Throws `FeatureBuildError` as `candidateFeatures2` does. */
export function candidateFeatures3(
  engine: Engine,
  configs: FeatureConfigs,
  channel: PlayerChannel,
  team: readonly Readonly<PokemonSet>[],
  candidates: readonly Pick<Candidate, "choice" | "pending">[],
  memo: FeatureMemo = createFeatureMemo(),
  trace?: CostTrace,
): CandidateFeatures3 {
  const setup = configs.belief_threats;
  if (setup === undefined) throw new DamageAwareError("probe-damage/3 needs the belief threats port");
  return buildArrays(engine, configs, channel, team, candidates, memo, trace, FEATURE_SET_3, (context) => ({ belief_threats: beliefThreatRows(engine, configs, setup, context) }), { belief_threats: [] as BeliefThreatRow[] });
}

function beliefThreatRows(engine: Engine, configs: FeatureConfigs, setup: BeliefThreatsSetup, { view, state, entries, variants, asMega, foes, threats, flags }: ArrayContext): BeliefThreatRow[] {
  const inclusions = setup.port.moveInclusions({ opponent: state.opponent, flags });
  if (inclusions === null) return [];
  const out: BeliefThreatRow[] = [];
  // Each foe's moves, in the belief's order (inclusion descending, then id): damaging ones the assumed set can hold, cut to `per_foe`.
  type Kept = { move: string; inclusion_ppm: number; paste: string };
  const kept = new Map<Foe, Kept[]>();
  for (const foe of foes) {
    const position = state.opponent.roster.indexOf(foe.mon);
    const revealed = new Set(threats.filter((t) => t.foe === foe.loc).map((t) => t.move));
    const list: Kept[] = [];
    for (const { id, inclusion_ppm } of inclusions[position] as readonly BeliefMoveInclusion[]) {
      if (list.length >= setup.per_foe) break;
      if (revealed.has(id) || foe.assumed.revealed_moves.includes(id)) continue;
      const data = view.dex.moves.get(id);
      if (!data.exists || data.category === "Status" || foeTargetKind(data.target) === "none") continue;
      const placed = assumedWithMove({ engine, format_id: configs.config.format_id }, foe.assumed, id);
      if (placed !== null) list.push({ move: id, inclusion_ppm, paste: placed.paste });
    }
    kept.set(foe, list);
  }
  for (const entry of entries) {
    for (const mega of variants(entry)) {
      for (const foe of foes) {
        for (const { move, inclusion_ppm, paste } of kept.get(foe) as Kept[]) {
          out.push({ own: entry.position, mega, foe: foe.loc, move, inclusion_ppm, ...damageValuesOf(view, () => view.damageResult({ ...foe.fighter, paste }, asMega(entry, mega), move, state, false)) });
        }
      }
    }
  }
  return out.sort((a, b) => a.own - b.own || Number(a.mega) - Number(b.mega) || a.foe - b.foe || compare(a.move, b.move));
}
