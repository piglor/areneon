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
export const BELIEF_FORMAT = "belief/3";

/**
 * The part of the pinned engine's `TeamVocabulary` the filter reads. A `TeamVocabulary` is one;
 * `rules.item_clause` may be absent (no constraint), and any value but 1 is refused.
 */
export interface BeliefVocabulary {
  readonly format_id: TeamVocabulary["format_id"];
  readonly engine_data_hash: TeamVocabulary["engine_data_hash"];
  readonly rules: Pick<TeamVocabulary["rules"], "adjust_level" | "min_team_size" | "max_team_size" | "picked_team_size"> & { readonly item_clause?: number };
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

/** One slot's belief inside ONE disguise assignment: the full per-forme models and the (forme, item) cells. */
export interface SlotBelief {
  /** The 0-based position in the opponent's preview roster. */
  readonly index: number;
  /** The preview species id (the prior's species entry). */
  readonly species: string;
  readonly shown: { readonly name: string | null; readonly species: string | null; readonly level: number | null; readonly gender: string | null };
  /** In the brought set for sure under this assignment. */
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

/** One slot's belief mixed over the disguise assignments (with a single assignment, that assignment's own numbers). */
export type SlotMarginal = Omit<SlotBelief, "formes" | "cells"> & {
  readonly formes: readonly { readonly id: string; readonly mega: boolean; readonly p: Ratio }[];
};

/**
 * One disguise assignment (ADR-025 section 3, N-B): where each shown position's evidence goes. A
 * belief with no unsettled identity-hiding species has exactly one, the identity.
 */
export interface DisguiseAssignment {
  /** The positions whose evidence is attributed to another slot: `position` is shown as itself, the evidence is `slot`'s. Empty for the identity. */
  readonly moved: readonly { readonly position: number; readonly slot: number }[];
  /** The assignment's integer weight: its Item Clause normaliser on the common scale of `DisguiseMixture.total`. */
  readonly weight: bigint;
  /** weight / total. */
  readonly p: Ratio;
  /** The Item Clause DP total of this assignment, the divisor of its cells' products. */
  readonly normaliser: bigint;
  /** Roster slots in the brought set under this assignment: the slots the shown positions are assigned to. */
  readonly required: readonly number[];
  readonly slots: readonly SlotBelief[];
}

export interface DisguiseMixture {
  /** Preview slots whose species can hide its identity (from the config's abilities against the vocabulary). */
  readonly hiding: readonly number[];
  /** The hiding slots not yet settled: the members a shown position may really be. */
  readonly unsettled: readonly number[];
  /** The shown positions whose evidence is assigned (empty when nothing is shown or no hiding slot is unsettled). */
  readonly positions: readonly number[];
  /** The sum of the assignments' weights, the divisor of every `weight`. */
  readonly total: bigint;
}

export interface OpponentBelief {
  readonly format: typeof BELIEF_FORMAT;
  /** The marginal beliefs, mixed over `assignments`. */
  readonly slots: readonly SlotMarginal[];
  /** At least one; the identity first when it is possible. */
  readonly assignments: readonly DisguiseAssignment[];
  readonly disguise: DisguiseMixture;
  readonly itemClause: {
    /** False when the rule table has no Item Clause. */
    readonly enforced: boolean;
  };
  readonly brought: {
    readonly picked: number;
    /** Roster positions in the brought set under every assignment. */
    readonly settled: readonly number[];
    /** Per assignment: the number of subsets of the roster of the picked size that hold its required slots. */
    readonly subsets: bigint;
  };
  readonly constraints: readonly string[];
  /** What the sampler reads besides the slots: the format's level and the config's legality redraw bound. */
  readonly sampling: {
    /** The level of a world's set when the channel shows none (the rule table's adjustLevel). */
    readonly level: number;
    /** `BeliefConfig.max_legality_redraws`. */
    readonly max_legality_redraws: number;
  };
}
