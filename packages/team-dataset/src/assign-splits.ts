// The split assignment of ADR-013 section 2 (and ADR-010 E1): a public pool of
// teams, each already in its stratum's bin, is dealt to dev, validation and
// training. Within a stratum the teams are ordered by
//   HMAC-SHA256(assignment_key, canonical_team_hash)
// and the first `quota[dev]` go to dev, the next `quota[validation]` to
// validation, the rest to training. The order depends on the key and on the
// team alone, never on the stream position or on the order the pool is given
// in, so the same pool, plan and key give the same split in every run.
//
// The function is pure: it decides nothing about legality (the pool's teams
// are the engine's) and it refuses a pool that does not fit the plan, so a
// stratum is never short, over full, or holding one team twice (L1).

import { createHmac } from "node:crypto";
import { type QuotaPlan, type Split } from "./plan.ts";

export class AssignError extends Error {
  override name = "AssignError";
}

const fail = (message: string): never => {
  throw new AssignError(message);
};

/** What the assignment needs of a pooled team: its identity and its stratum. */
export interface PoolEntry {
  canonical_team_hash: string;
  stratum: number;
}

const HEX_KEY = /^[0-9a-f]{64}$/;

/**
 * Deal the pool to the plan's splits. The result lists, for each split of the
 * plan, its teams stratum by stratum (ascending) and within a stratum in HMAC
 * order. Refused: a key that is not 64 lowercase hex digits; a team in a
 * stratum the plan does not have; a stratum that does not hold exactly its
 * bin size; a canonical hash twice; a quota below `nStratumMin` (E1, when
 * given).
 */
export function assignSplits<T extends PoolEntry>(pool: Iterable<T>, plan: QuotaPlan, assignmentKey: string, nStratumMin?: number): Record<Split, T[]> {
  if (!HEX_KEY.test(assignmentKey)) fail("the assignment key must be 64 lowercase hex digits");
  const key = Buffer.from(assignmentKey, "hex");
  const byStratum = new Map<number, { entry: T; order: string }[]>(plan.strata.map((s) => [s.stratum, []]));
  const seen = new Set<string>();
  for (const entry of pool) {
    const bin = byStratum.get(entry.stratum) ?? fail(`team ${entry.canonical_team_hash} is in stratum ${entry.stratum}, which the plan does not have`);
    if (seen.has(entry.canonical_team_hash)) fail(`team ${entry.canonical_team_hash} is in the pool twice`);
    seen.add(entry.canonical_team_hash);
    bin.push({ entry, order: createHmac("sha256", key).update(entry.canonical_team_hash, "utf8").digest("hex") });
  }
  const out = { dev: [], validation: [], training: [] } as Record<Split, T[]>;
  for (const stratum of plan.strata) {
    const bin = byStratum.get(stratum.stratum) as { entry: T; order: string }[];
    if (bin.length !== stratum.total) fail(`stratum ${stratum.stratum} holds ${bin.length} teams, its bin is ${stratum.total}`);
    let sum = 0;
    for (const split of plan.splits) {
      const quota = stratum.quotas[split] as number;
      if (nStratumMin !== undefined && quota < nStratumMin) fail(`stratum ${stratum.stratum}: the ${split} quota ${quota} is below n_stratum_min ${nStratumMin}`);
      sum += quota;
    }
    if (sum !== stratum.total) fail(`stratum ${stratum.stratum}: the quotas sum to ${sum}, its bin is ${stratum.total}`);
    bin.sort((a, b) => (a.order < b.order ? -1 : a.order > b.order ? 1 : 0));
    let at = 0;
    for (const split of plan.splits) {
      const quota = stratum.quotas[split] as number;
      for (const { entry } of bin.slice(at, at + quota)) out[split].push(entry);
      at += quota;
    }
  }
  return out;
}
