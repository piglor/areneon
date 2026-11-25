// A corpus of traces through a battle backend: every compared step is one item of the level-1 tally (entities present and
// exercised, from the trace), tagged with its interaction classes; the first difference of each battle is kept with its fixed
// words. Battles can be added in chunks, so a corpus never has to be held in memory whole.
import { type BattleClassesCount, classCounts } from "./battle-report.ts";
import type { BattleBackend } from "./battle-backend.ts";
import { type BattleDifference, compareBattles } from "./battle-compare.ts";
import type { ProjectionOracle } from "./battle-projection.ts";
import { classifyStep, type InteractionClasses } from "./battle-classes.ts";
import { stepFacts } from "./battle-level.ts";
import type { BattleTrace } from "./battle-trace.ts";
import type { DifferenceReason } from "./compare.ts";
import { sha256Jcs } from "./hash.ts";
import { addItem, type CorpusSpec, type CorpusTally, newTally } from "./tally.ts";

/** One battle's first difference, with the battle's name. */
export interface BattleCorpusDifference extends BattleDifference {
  battle_id: string;
}

export interface BattleCorpus {
  tally: CorpusTally;
  classes: Map<string, BattleClassesCount>;
  battles: number;
  steps: number;
  /** Steps after a battle's first difference, never compared. */
  unreached: number;
  differences: BattleCorpusDifference[];
}

export function newBattleCorpus(spec: CorpusSpec, classes: InteractionClasses): BattleCorpus {
  return { tally: newTally(spec), classes: classCounts(classes), battles: 0, steps: 0, unreached: 0, differences: [] };
}

const REASON: Record<BattleDifference["kind"], DifferenceReason> = { channel: "value", request: "value", mask: "value", draws: "value", state: "value", outcome: "value", refusal: "settled", count: "count", projection: "value" };

/** Compare `traces` with `backend` (and its projection, given an oracle) and add every compared step to the corpus, battle by battle in trace order. */
export function addBattles(corpus: BattleCorpus, classes: InteractionClasses, backend: BattleBackend, traces: readonly BattleTrace[], batchSize = 64, projection?: ProjectionOracle): void {
  const seen: (BattleDifference | null)[][] = traces.map(() => []);
  compareBattles(
    backend,
    traces,
    batchSize,
    (n, _step, difference) => {
      (seen[n] as (BattleDifference | null)[]).push(difference);
    },
    projection,
  );
  traces.forEach((trace, n) => {
    const facts = stepFacts(trace);
    const compared = seen[n] as (BattleDifference | null)[];
    corpus.battles += 1;
    corpus.steps += trace.steps.length;
    corpus.unreached += trace.steps.length - compared.length;
    compared.forEach((difference, step) => {
      const { entities, exercised } = facts[step] as { entities: string[]; exercised: string[] };
      const position = corpus.tally.items;
      const key = sha256Jcs({ seed: trace.seed, step, team_hashes: trace.team_hashes });
      addItem(corpus.tally, { position, key, entities, exercised, difference: difference === null ? null : { position, reason: REASON[difference.kind], field: `${difference.kind}:${difference.seat ?? "-"}`, entities } });
      for (const id of classifyStep(classes, trace.steps[step] as BattleTrace["steps"][number])) {
        const row = corpus.classes.get(id) as BattleClassesCount;
        row.tagged += 1;
        if (difference === null) row.equal += 1;
      }
      if (difference !== null) corpus.differences.push({ battle_id: trace.battle_id, ...difference });
    });
  });
}
