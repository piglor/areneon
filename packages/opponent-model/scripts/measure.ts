// Not shipped. The cost and rank figures of ADR-025 slice 2 part 4, printed as JSON for the ticket journal:
//   node packages/opponent-model/scripts/measure.ts [--random N]
// Beliefs and worlds are timed one at a time, one thread, over the committed channels with the real pinned validator as the
// legality port; the uniform-prior rank table (N6) is computed over the same requests. Needs the built engine.

import { createSetLegality } from "../../showdown-bridge/src/set-legality.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import { BeliefError, createSampler, type SpreadSampler } from "../src/index.ts";
import { beliefAt, games, idsOf, points, rankTable, ranks, skewedPrior } from "../test/corpus.ts";
import { measure } from "../test/measure.ts";
import { realVocabulary, uniformPrior } from "../test/real.ts";

const flag = process.argv.indexOf("--random");
const list = games(flag >= 0 ? { random: Number(process.argv[flag + 1]) } : {});
const engine = loadEngine();
const spreads: SpreadSampler = {
  other: (_s, _f, below) => ({
    nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id,
    stat_points: { hp: Number(below(33n)), atk: Number(below(33n)), def: 0, spa: 0, spd: 0, spe: Number(below(2n)) },
  }),
};
const sampler = createSampler({ legality: createSetLegality({ engine, vocabulary: realVocabulary }), spreads });

const cost = { uniform: measure(list, uniformPrior, sampler), skewed: measure(list, skewedPrior, sampler) };

const all = [];
let requests = 0;
for (const game of list) {
  for (const point of points(game)) {
    requests += 1;
    if (point.truth === null) continue;
    try {
      const b = beliefAt(point, uniformPrior(idsOf(point)));
      for (const rank of ranks(b, { ...point, truth: point.truth })) all.push({ phase: point.request === 0 ? ("preview" as const) : ("battle" as const), rank });
    } catch (e) {
      if (!(e instanceof BeliefError)) throw e;
    }
  }
}
console.log(JSON.stringify({ games: list.length, requests, cost, uniform_prior_rank_table: rankTable(all) }, null, 1));
