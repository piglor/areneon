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

import { belief, EVIDENCE_NONE, moveInclusions, opponentSheet, refusedSheet, SHEET_K, sheetFeatureSetHash, sheetRefusalOf, slotEvidence, uninformativeIdentityHash, uninformativeOpponent } from "@areneon/opponent-model";
import type { BeliefFeaturesPort, BeliefThreatsPort, SheetKValues } from "@areneon/showdown-bridge";
import { type SearchWorldsContext, type SearchWorldsFiles, armParts, searchWorldsFiles } from "./search-worlds.ts";

export type BeliefFeaturesContext = Omit<SearchWorldsContext, "arm"> & {
  /** The prior arm. Today's synthetic opponents are the generator's, so `generator-matched` is the primary arm (an upper bound for this evaluation). */
  readonly arm?: SearchWorldsContext["arm"];
  readonly k?: SheetKValues;
  /**
   * `"none"` selects the uninformative mode (#1309): every block and every belief threat is built from the prior alone, from the
   * opponent as the Team Preview gives it, whatever the channel reveals. It has its own identity (so its own source and feature set
   * hashes). Absent: the informative belief, with every byte and hash it had.
   */
  readonly evidence?: typeof EVIDENCE_NONE;
};

/** At most this many memoised blocks (oldest first out). */
const MEMO_LIMIT = 256;

// The reader's types come through the port (this package does not depend on the reader's package).
type SheetArgs = Parameters<BeliefFeaturesPort["sheet"]>[0];

/** The canonical evidence projection of an opponent half: the key the memo uses. */
export function evidenceKey(opponent: SheetArgs["opponent"], flags: SheetArgs["flags"]): string {
  return JSON.stringify([flags.length > 0, opponent.roster.map((mon) => [slotEvidence(mon), mon.phantom === true, mon.base_species])]);
}

/**
 * Where the informative belief refuses for a reader reason: a reader flag or a phantom (Illusion twin) roster entry (`reader_flagged`).
 * The uninformative mode refuses there too and reads no evidence otherwise, so the two arms' refused shares (E4's 1% validity rule) are comparable.
 */
const readerRefuses = (input: { readonly opponent: SheetArgs["opponent"]; readonly flags: SheetArgs["flags"] }): boolean => input.flags.length > 0 || input.opponent.roster.some((mon) => mon.phantom === true);

/** Run `run`; a refusal of the model (a code a consumer falls back on) becomes `otherwise(code)`, a defect of the run throws. */
function refusing<T>(run: () => T, otherwise: (code: NonNullable<ReturnType<typeof sheetRefusalOf>>) => T): T {
  try {
    return run();
  } catch (error) {
    const code = sheetRefusalOf(error);
    if (code === null) throw error;
    return otherwise(code);
  }
}

/** Store `value` under `key`, the oldest entry out beyond `MEMO_LIMIT`. */
function remember<V>(memo: Map<string, V>, key: string, value: V): V {
  if (memo.size >= MEMO_LIMIT) memo.delete(memo.keys().next().value as string);
  memo.set(key, value);
  return value;
}

/** The port: belief over the committed prior, then the closed-regime block. */
export function createBeliefFeatures(context: BeliefFeaturesContext, files: SearchWorldsFiles = searchWorldsFiles(context.vocabulary.format_id)): BeliefFeaturesPort & BeliefThreatsPort {
  const k = context.k ?? SHEET_K;
  const parts = armParts({ ...context, arm: context.arm ?? "generator-matched" }, files);
  const { prior, view, beliefConfig, prior_hash, belief_config_hash } = parts;
  const uninformative = context.evidence === EVIDENCE_NONE;
  const identity_hash = uninformative ? uninformativeIdentityHash(parts.identity_hash) : parts.identity_hash;
  const features_hash = sheetFeatureSetHash(identity_hash, k);
  const memo = new Map<string, ReturnType<BeliefFeaturesPort["sheet"]>>();
  const inclusionMemo = new Map<string, ReturnType<BeliefThreatsPort["moveInclusions"]>>();
  return {
    identity_hash,
    features_hash,
    prior_hash,
    belief_config_hash,
    prior_format_id: prior.format_id,
    k,
    // The belief's move inclusions for `probe-damage/3` (no trapping check: the threats read the evidence projection only, so the
    // memo key is complete). A refusal of the model keeps its code (`{ refused }`); a defect throws.
    moveInclusions(input) {
      // The uninformative mode reads the preview opponent, no flags and no trapping: the answer is one value per roster. It refuses
      // exactly where the informative belief refuses for a reader reason (a flag or a phantom entry), so the refused shares of the two arms match.
      if (uninformative && readerRefuses(input)) return { refused: "reader_flagged" };
      const opponent = uninformative ? uninformativeOpponent(input.opponent) : input.opponent;
      const flags = uninformative ? [] : input.flags;
      const key = evidenceKey(opponent, flags);
      const hit = inclusionMemo.get(key);
      if (hit !== undefined) return hit;
      const found = refusing(
        () => moveInclusions(belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags }), opponent),
        (refused) => ({ refused }),
      );
      return remember(inclusionMemo, key, found);
    },
    sheet(input) {
      if (uninformative && readerRefuses(input)) return refusedSheet(input.opponent, "reader_flagged", features_hash);
      const opponent = uninformative ? uninformativeOpponent(input.opponent) : input.opponent;
      const flags = uninformative ? [] : input.flags;
      const trapping = uninformative ? null : input.trapping;
      // The belief ignores a trapping check that observes nothing (`observed` empty), so such a call is a pure function of the
      // evidence projection; with observed levels the exclusions depend on the player's own side, which the key does not hold.
      const informative = trapping !== null && trapping !== undefined && trapping.observed.length > 0;
      const key = informative ? null : evidenceKey(opponent, flags);
      const hit = key === null ? undefined : memo.get(key);
      if (hit !== undefined) return structuredClone(hit);
      const block = refusing(
        () => opponentSheet(belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags, trapping }), opponent, features_hash, k),
        (code) => refusedSheet(opponent, code, features_hash),
      );
      if (key !== null) remember(memo, key, structuredClone(block));
      return block;
    },
  };
}
