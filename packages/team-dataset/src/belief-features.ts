// The composition root of the opponent sheet block (ADR-025 erratum E4 points 1, 4 and 6; ADR-027 section 2). It joins the
// opponent model (`packages/opponent-model`, engine-free: `belief`, `opponentSheet`) with the committed prior and belief
// config, behind the bridge's structural `BeliefFeaturesPort`. The bridge imports none of them. The arm, the prior files and
// their checks are `search-worlds.ts`'s `armParts` (one place reads and checks the committed records: a prior of another
// format, one not derived from the committed generator config, or a role-sets prior over another fill or table is refused).
//
// The block is a pure function of the reader's opponent half, its flags and the trapping check, so it is memoised by the
// canonical evidence projection of the opponent half (what `slotEvidence` reads of each Pokemon, the phantom flag, and
// whether any line was flagged). The memo is bypassed whenever the trapping check observes something: it then depends on the
// player's own side, which the key does not hold. Memoised and recomputed blocks are equal (a test).

import { belief, opponentSheet, refusedSheet, SHEET_K, sheetFeatureSetHash, sheetRefusalOf, slotEvidence } from "@areneon/opponent-model";
import type { BeliefFeaturesPort, SheetKValues } from "@areneon/showdown-bridge";
import { type SearchWorldsContext, type SearchWorldsFiles, armParts, searchWorldsFiles } from "./search-worlds.ts";

export type BeliefFeaturesContext = Omit<SearchWorldsContext, "arm"> & {
  /** The prior arm. Today's synthetic opponents are the generator's, so `generator-matched` is the primary arm (an upper bound for this evaluation). */
  readonly arm?: SearchWorldsContext["arm"];
  readonly k?: SheetKValues;
};

/** At most this many memoised blocks (oldest first out). */
const MEMO_LIMIT = 256;

// The reader's types come through the port (this package does not depend on the reader's package).
type SheetArgs = Parameters<BeliefFeaturesPort["sheet"]>[0];

/** The canonical evidence projection of an opponent half: the key the memo uses. */
export function evidenceKey(opponent: SheetArgs["opponent"], flags: SheetArgs["flags"]): string {
  return JSON.stringify([flags.length > 0, opponent.roster.map((mon) => [slotEvidence(mon), mon.phantom === true, mon.base_species])]);
}

/** The port: belief over the committed prior, then the closed-regime block. */
export function createBeliefFeatures(context: BeliefFeaturesContext, files: SearchWorldsFiles = searchWorldsFiles(context.vocabulary.format_id)): BeliefFeaturesPort {
  const k = context.k ?? SHEET_K;
  const { prior, view, beliefConfig, prior_hash, belief_config_hash, identity_hash } = armParts({ ...context, arm: context.arm ?? "generator-matched" }, files);
  const features_hash = sheetFeatureSetHash(identity_hash, k);
  const memo = new Map<string, ReturnType<BeliefFeaturesPort["sheet"]>>();
  return {
    identity_hash,
    features_hash,
    prior_hash,
    belief_config_hash,
    prior_format_id: prior.format_id,
    k,
    sheet({ opponent, flags, trapping }) {
      // The belief ignores a trapping check that observes nothing (`observed` empty), so such a call is a pure function of the
      // evidence projection; with observed levels the exclusions depend on the player's own side, which the key does not hold.
      const informative = trapping !== null && trapping !== undefined && trapping.observed.length > 0;
      const key = informative ? null : evidenceKey(opponent, flags);
      const hit = key === null ? undefined : memo.get(key);
      if (hit !== undefined) return structuredClone(hit);
      let block;
      try {
        block = opponentSheet(belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags, trapping }), opponent, features_hash, k);
      } catch (error) {
        const code = sheetRefusalOf(error);
        if (code === null) throw error;
        block = refusedSheet(opponent, code, features_hash);
      }
      if (key !== null) {
        if (memo.size >= MEMO_LIMIT) memo.delete(memo.keys().next().value as string);
        memo.set(key, structuredClone(block));
      }
      return block;
    },
  };
}
