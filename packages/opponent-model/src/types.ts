// The belief filter's input and output types (ADR-025 sections 1, 3 and 8).
//
// The INPUT is the whole privacy statement of the filter (a type test holds it): a usage prior, the
// vocabulary's rule values and per-species lists, the belief config, and the opponent half of the
// shared reader's state at the request, built from ONE player's own channel, with the lines that
// reader flagged. The opponent's real team, the omniscient log, hidden verdicts, a battle record
// and the seed have no field to travel in.

import type { PublicFlag, PublicSide } from "@areneon/public-state";
import type { BeliefConfig } from "@areneon/schema/types/v1/belief-config.ts";
import type { TeamVocabulary } from "@areneon/schema/types/v1/team-vocabulary.ts";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import type { Ratio } from "./ratio.ts";
import type { Dist } from "./fields.ts";
import type { MoveModel } from "./moves.ts";

/** Bumped when the filter changes how it computes a value (as `adapter_format` is for the probe). */
export const BELIEF_FORMAT = "belief/1";

/**
 * The part of the pinned engine's `TeamVocabulary` the filter reads. A `TeamVocabulary` is one;
 * `rules.item_clause` may be absent (no constraint), and any value but 1 is refused.
 */
export interface BeliefVocabulary {
  readonly format_id: TeamVocabulary["format_id"];
  readonly engine_data_hash: TeamVocabulary["engine_data_hash"];
  readonly rules: Pick<TeamVocabulary["rules"], "min_team_size" | "max_team_size" | "picked_team_size"> & { readonly item_clause?: number };
  readonly items: TeamVocabulary["items"];
  readonly roster: readonly TeamVocabulary["roster"][number][];
}

export interface BeliefInput {
  readonly prior: UsagePrior;
  readonly vocabulary: BeliefVocabulary;
  readonly config: BeliefConfig;
  /** The opponent half of the shared reader's state at the request (`PublicState.opponent`). */
  readonly opponent: PublicSide;
  /**
   * The lines the reader flagged in this channel. Any flagged line makes the belief refuse
   * (`reader_flagged`): the consumer decides which flags concern the request it asks about.
   */
  readonly flags: readonly PublicFlag[];
}

export interface Marginal<T> {
  readonly id: T;
  readonly p: Ratio;
}

/** One forme a slot may hold, with the model its fields are drawn from. */
export interface FormeBelief {
  readonly id: string;
  /** A Mega forme of the preview species (its stone is its item). */
  readonly mega: boolean;
  /** P(the slot holds this forme), after the evidence and the Item Clause. */
  readonly p: Ratio;
  /** The ability list in force: the pre-Evolution list for a Mega forme that has not evolved. */
  readonly abilities: Dist<string>;
  readonly spreads: Dist<string>;
  readonly moves: MoveModel;
}

/** One (forme, item) cell of a slot with its integer weight on the scale of the belief's normaliser. */
export interface Cell {
  readonly forme: string;
  /** An item id, or null for no item. */
  readonly item: string | null;
  readonly weight: bigint;
}

export interface SlotBelief {
  /** The 0-based position in the opponent's preview roster. */
  readonly index: number;
  /** The preview species id (the prior's species entry). */
  readonly species: string;
  readonly shown: { readonly name: string | null; readonly species: string | null; readonly level: number | null; readonly gender: string | null };
  /** In the brought set for sure (it has been in play). */
  readonly settled: boolean;
  readonly p_brought: Ratio;
  readonly formes: readonly FormeBelief[];
  readonly abilities: readonly Marginal<string>[];
  /** `null` is "holds no item". */
  readonly items: readonly Marginal<string | null>[];
  readonly moves: readonly Marginal<string>[];
  readonly spreads: readonly Marginal<string>[];
  /** The most likely forme, ability and item (diagnostic). */
  readonly mode: { readonly forme: string; readonly ability: string; readonly item: string | null };
  readonly cells: readonly Cell[];
  /** The evidence applied to this slot and the evidence that was left out, as fixed words. */
  readonly constraints: readonly string[];
}

export interface OpponentBelief {
  readonly format: typeof BELIEF_FORMAT;
  readonly slots: readonly SlotBelief[];
  readonly itemClause: {
    /** False when the rule table has no Item Clause. */
    readonly enforced: boolean;
    /** The sum of the joint (forme, item) weights, the divisor of every `Cell` product. */
    readonly normaliser: bigint;
  };
  readonly brought: {
    readonly picked: number;
    /** Roster positions settled in the brought set. */
    readonly settled: readonly number[];
    /** The number of subsets of the roster of the picked size that hold every settled Pokemon. */
    readonly subsets: bigint;
  };
  readonly constraints: readonly string[];
}
