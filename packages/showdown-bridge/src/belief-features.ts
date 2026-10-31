// The opponent sheet block from the opponent belief (ADR-025 erratum E4 points 1, 3 to 6; ADR-027 section 2): the
// closed-regime `OpponentSheet` a learned policy may read beside its observation. The opponent model is a package of
// its own (`packages/opponent-model`, engine-free) that the bridge does not import; this module is the structural
// shape the bridge needs of it (`BeliefFeaturesPort`) and the engine-backed part around it: the trapping evidence of the
// player's own request (ticket #1127) and the counts of refusals per game.
//
// The block is a pure function of the player's own view: the reader's opponent half and flags at the request, and the
// trapping check of the player's own move request. It takes no sampled world, seed, opponent team, `BattleRecord`,
// `OmniscientLog` or hidden verdict (the type of `SheetInput` has no field for them; `test/belief-features.test.ts` holds
// a type test and a source scan).
//
// A refused belief is a block with `source.refusal` (a fixed code) and every field `unknown`, counted per code: no
// silent zero and no rule fallback. A defect of the run (`BELIEF_DEFECTS`) is thrown by the port and propagates.

import { interpretChannel, type PublicFlag, type PublicSide, type PublicState } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import type { VersionManifest } from "./formats.ts";
import { createTrappingCheck, type TrappingCheckPort } from "./trapping-check.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The reviewed `k` values the block was cut with (E4 point 1). */
export interface SheetKValues {
  readonly item: number;
  readonly ability: number;
  readonly nature: number;
  readonly moves: number;
}

/**
 * What the sheet builder needs of the opponent model: the block of the reader's opponent half at a request, or a thrown
 * `BeliefError` for a defect of the run. A refusal is not thrown: it comes back as a block whose `source.refusal` is a code.
 */
export interface BeliefFeaturesPort {
  /** `belief_identity_hash` (ADR-025 section 5). */
  readonly identity_hash: string;
  /** The feature set hash: the block's `source.hash` (names the belief identity, the k values, the rounding and the builder format). */
  readonly features_hash: string;
  readonly prior_hash: string;
  readonly belief_config_hash: string;
  /** The usage prior's `format_id`. */
  readonly prior_format_id: string;
  readonly k: SheetKValues;
  sheet(input: { readonly opponent: PublicSide; readonly flags: readonly PublicFlag[]; readonly trapping?: TrappingCheckPort | null }): OpponentSheet;
}

export interface BeliefFeaturesContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  readonly port: BeliefFeaturesPort;
  /** The scenario format id the trapping check starts its battles in (`checkScenario`). */
  readonly scenario_id: string;
}

/** Everything the block is a function of: the player's own request, own team, the reader's state and its flags. */
export interface SheetInput {
  /** The player's own request, as the engine sent it. */
  readonly request: EngineRequest;
  /** The player's own team as brought. */
  readonly team: readonly Readonly<PokemonSet>[];
  /** The shared reader's state at the request. */
  readonly state: PublicState;
  readonly flags: readonly PublicFlag[];
}

export interface SheetCounts {
  /** Blocks built. */
  readonly requests: number;
  /** Blocks that were refused. */
  readonly refused: number;
  /** Refused blocks by their fixed code, sorted by code. */
  readonly refusals: readonly { readonly code: string; readonly count: number }[];
}

export class BeliefFeaturesError extends Error {
  override name = "BeliefFeaturesError";
}

export interface SheetBuilder {
  /** The block at a request. */
  build(input: SheetInput): OpponentSheet;
  /** The counts so far (per game: make one builder per game). */
  counts(): SheetCounts;
}

/** A builder for one game. The port's prior must be of the vocabulary's format (a prior of another format is never used silently). */
export function createSheetBuilder(context: BeliefFeaturesContext): SheetBuilder {
  const { engine, vocabulary, port, scenario_id } = context;
  if (port.prior_format_id !== vocabulary.format_id) throw new BeliefFeaturesError("the usage prior is for another format than the vocabulary's: it is never used silently");
  let requests = 0;
  let refused = 0;
  const byCode = new Map<string, number>();
  return {
    build({ request, team, state, flags }) {
      const trapping = createTrappingCheck({ engine, vocabulary, scenario_id }, { request, team, state });
      const block = port.sheet({ opponent: state.opponent, flags, trapping });
      requests += 1;
      const code = block.source.refusal;
      if (code !== undefined) {
        refused += 1;
        byCode.set(code, (byCode.get(code) ?? 0) + 1);
      }
      return block;
    },
    counts: () => ({ requests, refused, refusals: [...byCode].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([code, count]) => ({ code, count })) }),
  };
}

/** The share of refused blocks in parts per million (an experiment with more than 10,000 ppm refused is invalid, E4 point 5). */
export const refusedSharePpm = (counts: Pick<SheetCounts, "requests" | "refused">): number => (counts.requests === 0 ? 0 : Math.floor((counts.refused * 1_000_000) / counts.requests));

/** What a seat or the training loader needs to build the block of a request (ADR-027 slice 4). */
export interface SheetSetup {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  readonly port: BeliefFeaturesPort;
  /** The scenario format id the trapping check starts its battles in. */
  readonly scenario_id: string;
}

/** What the worker is expected to state at the handshake: the block's source. */
export const sheetStatement = (setup: Pick<SheetSetup, "port">): { feature_set: string; feature_set_hash: string } => ({ feature_set: SHEET_FEATURE_SET, feature_set_hash: setup.port.features_hash });

/** The feature set id of the first sheet (the opponent model names it too; a test holds them equal). */
export const SHEET_FEATURE_SET = "opponent-sheet/1";

/** Builds the port of the composition root for a vocabulary, reader and pin. */
export type BeliefFeaturesFactory = (context: { readonly engine: Engine; readonly vocabulary: TeamVocabulary; readonly reader_sha256: string; readonly manifest: VersionManifest }) => BeliefFeaturesPort;

/** A channel that does not end in a request cannot be given a block. */
export class SheetInputError extends Error {
  override name = "SheetInputError";
}

/**
 * The block at the request a player's own channel ends in. The one function the play-time seat and the training loader
 * both call, so a block recomputed from an Episode's channel prefix and own team equals the one that was sent: it reads the
 * lines with the shared reader, takes the request it ends in and the reader's opponent half and flags, and builds.
 */
export function sheetForChannel(builder: SheetBuilder, input: { readonly player: string; readonly lines: readonly string[]; readonly team: readonly Readonly<PokemonSet>[] }): OpponentSheet {
  const run = interpretChannel(input.player, input.lines);
  const frame = run.frames[run.frames.length - 1];
  if (frame === undefined || frame.kind !== "request") throw new SheetInputError("the channel does not end in a request");
  const request = run.requests[frame.request - 1] as EngineRequest;
  return builder.build({ request, team: input.team, state: frame.state, flags: run.flags });
}
