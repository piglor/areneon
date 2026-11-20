// The summary of a battle corpus run: integers and fixed words only (counts per difference kind, per entity and per interaction
// class), so the M-D spike only fills counts. No line of a battle appears in it.
import type { BattleCorpus } from "./battle-corpus.ts";
import type { InteractionClasses } from "./battle-classes.ts";
import type { Json } from "./canonical.ts";
import { tallyHash } from "./tally.ts";

export interface BattleClassesCount {
  tagged: number;
  equal: number;
}

/** A zero count for every class of the file, in file order. */
export function classCounts(classes: InteractionClasses): Map<string, BattleClassesCount> {
  return new Map(classes.classes.map((c) => [c.id, { tagged: 0, equal: 0 }]));
}

/** The corpus's counts: battles, steps, differences by kind, per entity kind and per entity, per class. */
export function battleCorpusSummary(corpus: BattleCorpus): Json {
  const byKind: Record<string, number> = {};
  for (const d of corpus.differences) byKind[d.kind] = (byKind[d.kind] ?? 0) + 1;
  const kinds: Record<string, { entities: number; exercised: number }> = {};
  const entities = [...corpus.tally.entities].sort(([a], [b]) => Number(a > b) - Number(a < b));
  for (const [id, row] of entities) {
    const kind = id.slice(0, id.indexOf(":"));
    const k = (kinds[kind] ??= { entities: 0, exercised: 0 });
    k.entities += 1;
    k.exercised += Number(row.exercised > 0);
  }
  return {
    battles: corpus.battles,
    steps: corpus.steps,
    compared_steps: corpus.tally.items,
    unreached_steps: corpus.unreached,
    differences: corpus.differences.length,
    differences_by_kind: byKind,
    first_differences: corpus.differences.slice(0, 20).map((d) => ({ battle_id: d.battle_id, kind: d.kind, seat: d.seat, step: d.step, where: d.where })),
    entity_kinds: kinds,
    entities: entities.map(([id, r]) => ({ id, present: r.present, exercised: r.exercised, equal: r.equal, differing: r.differing })),
    classes: [...corpus.classes].map(([id, c]) => ({ id, tagged: c.tagged, equal: c.equal })),
    tally_hash: tallyHash(corpus.tally),
  };
}
