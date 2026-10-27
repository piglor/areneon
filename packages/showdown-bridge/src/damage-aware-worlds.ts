// The belief port of `damage-aware@v2` (ADR-025 sections 4 to 6). The opponent model is a package of its
// own (`packages/opponent-model`, engine-free) that the bridge does not import (adding it would change the
// workspace dependencies the importer image copies); the composition root joins the two. This module is the
// structural shape the baseline needs of it: sample N worlds of the opponent from the reader's opponent
// half, on a stream the baseline derives from the player's own request, or refuse with a coded error.
//
// Nothing here knows a game rule or a species: a world is the opponent model's `OpponentWorld`, a list of
// sets, and the refusal codes are strings the model defines.

import type { PublicFlag, PublicSide } from "@areneon/public-state";
import { jcs } from "./canonical-json.ts";
import type { DamageAwareConfigV2 } from "./damage-aware-config.ts";
import type { Engine } from "./engine.ts";
import type { VersionManifest } from "./formats.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";
import type { WorldSetLike } from "./damage-aware-assumed-set.ts";
import { sha256Hex } from "./rules.ts";

/** The purpose string of the baseline's world stream (ADR-025 section 4). */
export const WORLDS_PURPOSE = "damage-aware-worlds/1";

/** One preview slot of a sampled world: brought or not, and the set of a brought one. */
export interface WorldSlotLike {
  readonly index: number;
  readonly brought: boolean;
  readonly set: WorldSetLike | null;
}

export interface WorldLike {
  readonly slots: readonly WorldSlotLike[];
}

export interface SampledWorldsLike {
  readonly worlds: readonly { readonly world: WorldLike }[];
  /** Redraws summed over the worlds (a world with an illegal set is drawn again). */
  readonly redraws: number;
}

/**
 * What `damage-aware@v2` needs of the opponent model. `sample` is `belief` then `sampleWorlds`: a pure
 * function of the reader's opponent half, its flags and the 32-byte stream. A refusal is an `Error` with
 * `name` "BeliefError" and a string `code` (the model's `BeliefError`).
 */
export interface BeliefWorldsPort {
  /** `belief_identity_hash` (ADR-025 section 5): the prior, the belief config, `belief_format`, the vocabulary, the reader and the pin. */
  readonly identity_hash: string;
  /** SHA-256 of the JCS bytes of the usage prior the belief is computed from. */
  readonly prior_hash: string;
  /** SHA-256 of the JCS bytes of the belief config. */
  readonly belief_config_hash: string;
  /** The usage prior's `format_id`. */
  readonly prior_format_id: string;
  sample(input: { readonly opponent: PublicSide; readonly flags: readonly PublicFlag[]; readonly stream: Uint8Array; readonly n: number }): SampledWorldsLike;
}

/** The 32-byte stream of one request: SHA-256 of the JCS of `{request_hash, request_seq, player, purpose}`, all of the player's own request. */
export function worldsStream(request: { readonly request_hash: string; readonly request_seq: number; readonly player: string }): Uint8Array {
  const hex = sha256Hex(jcs({ request_hash: request.request_hash, request_seq: request.request_seq, player: request.player, purpose: WORLDS_PURPOSE }));
  return Uint8Array.from(hex.match(/../g) as string[], (byte) => parseInt(byte, 16));
}

/** The belief refusal code of a thrown value, or null when it is not one. */
export function refusalCode(error: unknown): string | null {
  if (!(error instanceof Error) || error.name !== "BeliefError") return null;
  const code = (error as Error & { code?: unknown }).code;
  return typeof code === "string" ? code : null;
}

/**
 * Refusals that say the run is set up wrongly, not that this request cannot be modelled: a prior or vocabulary
 * of another format (after a pin bump the M-C prior must never be used silently), an Item Clause value the
 * model does not read, a prior that needs a spread sampler nobody supplied. They propagate; every other code
 * is counted and the request is answered by the `actor-rule` candidate.
 */
export const BELIEF_DEFECTS: ReadonlySet<string> = new Set(["format_mismatch", "item_clause_unsupported", "spread_sampler_required"]);

/** What the composition root needs to build the port of a v2 record. */
export interface BeliefWorldsContext {
  readonly engine: Engine;
  readonly vocabulary: TeamVocabulary;
  readonly config: DamageAwareConfigV2;
  /** SHA-256 of the public-state reader's file (a part of the belief identity). */
  readonly reader_sha256: string;
  readonly manifest: VersionManifest;
}

/** Builds the belief port for a v2 record. It loads the usage prior and the belief config the record names by hash and refuses any other. */
export type BeliefWorldsFactory = (context: BeliefWorldsContext) => BeliefWorldsPort;
