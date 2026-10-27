// The generator's draw of a nature and a Stat Point spread, as the opponent model's `SpreadSampler`
// port (ADR-025 section 4, N-G). A `generator-matched` usage prior names no spread of its own: its
// "other spread" entry stands for what the generator draws, and the belief sampler asks this object
// for it on the world's own stream. The nature is uniform over the vocabulary's natures, the Stat
// Points come from `drawStatPoints`, the routine the proposal itself calls (ADR-013 section 2), with
// the rule table's total and the config's per-stat maximum. No likelihood belongs to it: no evidence
// touches spreads before slice 4.
//
// The port is declared structurally here: `packages/opponent-model` never imports this package, and
// only the composition root of `damage-aware@v2` (`damage-aware-worlds.ts`) imports both: it hands this
// object to `createSampler`.

import type { TeamGeneratorConfig, TeamVocabulary } from "@areneon/showdown-bridge";
import { drawStatPoints, ProposalError } from "./proposal.ts";

export type StatPointSpread = Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;

/** The shape of the opponent model's `SpreadSampler` port (`below(T)` is uniform in [0, T)). */
export interface GeneratorSpreadSampler {
  other(species: string, forme: string, below: (total: bigint) => bigint): { readonly nature: string; readonly stat_points: StatPointSpread };
}

/** The sampler for a generator config on a vocabulary; refuses what the proposal refuses. */
export function generatorSpreadSampler(vocabulary: Pick<TeamVocabulary, "natures" | "rules">, config: Pick<TeamGeneratorConfig, "stat_point_max_per_stat">): GeneratorSpreadSampler {
  if (vocabulary.natures.length === 0) throw new ProposalError("the vocabulary lists no nature");
  const total = vocabulary.rules.ev_limit;
  const cap = config.stat_point_max_per_stat;
  if (cap * 6 < total) throw new ProposalError(`6 stats of at most ${cap} cannot hold ${total} Stat Points`);
  return {
    other: (_species, _forme, below) => {
      const pick = (n: number): number => Number(below(BigInt(n)));
      const nature = vocabulary.natures[pick(vocabulary.natures.length)] as TeamVocabulary["natures"][number];
      return { nature: nature.id, stat_points: drawStatPoints(pick, total, cap) as StatPointSpread };
    },
  };
}
