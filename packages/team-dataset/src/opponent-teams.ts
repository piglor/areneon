// The composition root of the full-team opponent draw (ADR-029 section 3, ADR-025 erratum E-25.1): the Team Preview
// solver's hypothetical opponent teams, a set for ALL six preview slots, from ADR-025's belief. It is the same arm
// wiring as `createSearchWorlds` (`armParts`: the committed prior, belief config, pinned-engine set legality and the
// generator's spread routine) with the model's `sampleTeam` behind it instead of `sample`; there is no second sampler.
//
// A previewed species the prior has no entry for is a refusal the belief raises (`species_not_in_prior`): it comes
// back as a coded `refused` result and is counted, never guessed. A team whose every redraw held an illegal set is
// refused the same way (`legality_redraws_exhausted`). Defects of the run (the model's `BELIEF_DEFECTS`) are thrown by
// the callers that treat them so; here every coded refusal is counted and returned, anything else propagates.

import { belief, type OpponentTeam, type SetLegality, teamStream } from "@areneon/opponent-model";
import { refusalCode, type WorldSamplePort } from "@areneon/showdown-bridge";
import { armParts, type SearchWorldsContext, type SearchWorldsFiles, searchWorldsFiles } from "./search-worlds.ts";

type OpenInput = Parameters<WorldSamplePort["open"]>[0];

export type TeamDraw = { readonly kind: "team"; readonly team: OpponentTeam; readonly redraws: number } | { readonly kind: "refused"; readonly code: string };

export type TeamSourceResult = { readonly kind: "refused"; readonly code: string } | { readonly kind: "source"; readonly source: { draw(stream: Uint8Array, index: number): TeamDraw } };

/** What the port has done so far: integers only, refusals by the model's code. */
export interface OpponentTeamCounts {
  readonly opened: number;
  readonly belief_refusals: Readonly<Record<string, number>>;
  readonly drawn: number;
  readonly draw_refusals: Readonly<Record<string, number>>;
  /** The legality redraws summed over the drawn teams. */
  readonly legality_redraws: number;
}

export interface OpponentTeamPort {
  readonly identity_hash: string;
  readonly prior_hash: string;
  readonly belief_config_hash: string;
  readonly prior_format_id: string;
  /** The belief of the reader's opponent half, then teams by index (team `i` is `sampleTeam` over `teamStream(stream, i)`). */
  open(input: Pick<OpenInput, "opponent" | "flags">): TeamSourceResult;
  counts(): OpponentTeamCounts;
}

/** `legality` replaces the pinned engine's set legality: a test seam (the default is the real validator). */
export function createOpponentTeams(context: SearchWorldsContext, files: SearchWorldsFiles = searchWorldsFiles(context.vocabulary.format_id), legality?: SetLegality): OpponentTeamPort {
  const { prior, view, beliefConfig, sampler, prior_hash, belief_config_hash, identity_hash } = armParts(context, files, legality);
  let opened = 0;
  let drawn = 0;
  let legality_redraws = 0;
  const belief_refusals: Record<string, number> = {};
  const draw_refusals: Record<string, number> = {};
  const bump = (table: Record<string, number>, code: string): void => {
    table[code] = (table[code] ?? 0) + 1;
  };
  return {
    identity_hash,
    prior_hash,
    belief_config_hash,
    prior_format_id: prior.format_id,
    open({ opponent, flags }) {
      opened += 1;
      let computed;
      try {
        computed = belief({ prior, vocabulary: view, config: beliefConfig, opponent, flags });
      } catch (error) {
        const code = refusalCode(error);
        if (code === null) throw error;
        bump(belief_refusals, code);
        return { kind: "refused", code };
      }
      return {
        kind: "source",
        source: {
          draw(stream, index) {
            try {
              const { team, redraws } = sampler.sampleTeam(computed, teamStream(stream, index));
              drawn += 1;
              legality_redraws += redraws;
              return { kind: "team", team, redraws };
            } catch (error) {
              const code = refusalCode(error);
              if (code === null) throw error;
              bump(draw_refusals, code);
              return { kind: "refused", code };
            }
          },
        },
      };
    },
    counts: () => ({ opened, belief_refusals: { ...belief_refusals }, drawn, draw_refusals: { ...draw_refusals }, legality_redraws }),
  };
}
