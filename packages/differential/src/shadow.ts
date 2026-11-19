// The shadow check (ADR-030 rule 4, ADR-028 section 5): every generation a deterministic sample of a backend's answers is
// recomputed by the oracle and compared; any difference stops the generation and names the entities to revoke. The sample is
// #1195's bottom-k (`inference-divergence.ts`): the items with the smallest keys, a key being the first 8 bytes of SHA-256 over a
// stable key of the item, so the same items are sampled whatever order they were answered in. One function; the size rule is a
// parameter (level 1: max(1% of backend answers, 256); level 2: max(5% of games, 64)). Wiring it into the loop is not here.
import type { Json } from "./canonical.ts";
import { compareItem, type Difference } from "./compare.ts";
import { key64 } from "./hash.ts";
import type { Backend, Level, Settled } from "./backend.ts";

export interface SizeRule {
  /** The share of the answers to sample, in parts per million. */
  share_ppm: number;
  /** The least sample (all of them when fewer were answered). */
  min: number;
}

/** Level 1, mechanics queries: max(1% of the backend's answers, 256). */
export const LEVEL_1_RULE: SizeRule = { share_ppm: 10_000, min: 256 };
/** Level 2, battles: max(5% of the games, 64). */
export const LEVEL_2_RULE: SizeRule = { share_ppm: 50_000, min: 64 };

export const SHADOW_PURPOSE = "differential-shadow/1";

/** How many of `n` answers the sample holds: the share rounded up, at least `min`, at most `n` (integers only). */
export function shadowSize(n: number, rule: SizeRule): number {
  const share = Number((BigInt(n) * BigInt(rule.share_ppm) + 999_999n) / 1_000_000n);
  return Math.min(n, Math.max(rule.min, share));
}

/** The sampling key of an item: the first 8 bytes of SHA-256 over the JCS of its stable key and the purpose. */
export const shadowKey = (stable: Json): bigint => key64({ key: stable, purpose: SHADOW_PURPOSE });

/** The positions of the sampled items: the `shadowSize` smallest keys (ties by position), in ascending position order. */
export function shadowSample(stableKeys: readonly Json[], rule: SizeRule): number[] {
  const keyed = stableKeys.map((stable, position) => ({ key: shadowKey(stable), position }));
  keyed.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : a.position - b.position));
  return keyed
    .slice(0, shadowSize(stableKeys.length, rule))
    .map((k) => k.position)
    .sort((a, b) => a - b);
}

/** A shadow difference: stop the generation (nothing committed) and revoke `entities` (the whole backend when `whole_backend`). */
export class ShadowDivergence extends Error {
  override name = "ShadowDivergence";
  readonly differences: Difference[];
  /** Every entity of every differing item (of every sampled item after a count mismatch), sorted. */
  readonly entities: string[];
  /** The recorded answers were not one per input: nothing of the backend's generation can be attributed, so revoke it whole. */
  readonly whole_backend: boolean;
  constructor(differences: Difference[], entities: string[], whole_backend: boolean) {
    super(
      whole_backend
        ? "shadow check: the recorded answers are not one per query; stop the generation and revoke the backend"
        : `shadow check: ${differences.length} of the sampled answers differ from the oracle; stop the generation and revoke their entities`,
    );
    this.differences = differences;
    this.entities = [...new Set(entities)].sort();
    this.whole_backend = whole_backend;
  }
}

/**
 * Recompute the sampled inputs with `oracle` and compare them with the answers the generation actually used (`recorded`, one per
 * input, in input order); the backend is never asked again. Returns the sampled positions; throws `ShadowDivergence` on any
 * difference, and with `whole_backend` when `recorded` does not hold exactly one answer per input.
 */
export function shadowCheck<In, Out>(level: Level<In, Out>, oracle: Backend<In, Out>, inputs: readonly In[], recorded: readonly Settled<Out>[], stableKey: (input: In) => Json, rule: SizeRule): number[] {
  const sample = shadowSample(inputs.map(stableKey), rule);
  const chosen = sample.map((p) => inputs[p] as In);
  if (recorded.length !== inputs.length) throw new ShadowDivergence([], chosen.flatMap((input) => level.entities(input)), true);
  const expected = oracle.run(chosen);
  const differences = chosen.flatMap((input, i) => {
    const item = compareItem(level, input, sample[i] as number, expected[i] as Settled<Out>, recorded[sample[i] as number]);
    return item.difference === null ? [] : [item.difference];
  });
  if (differences.length > 0) throw new ShadowDivergence(differences, differences.flatMap((d) => d.entities), false);
  return sample;
}
