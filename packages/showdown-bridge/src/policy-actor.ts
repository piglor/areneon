// The A1 policy actor (ADR-010 §2): the online observation adapter and the
// hash-based sampler. For each request it builds the player's observation
// from their own stream so far, with the candidates `enumerateCandidates`
// returns for it, and asks the policy worker. The policy sees only that and
// `u`, never the battle seed, an opponent's hidden information or a
// CloneJudge verdict.

import { createHash } from "node:crypto";
import type { PolicyQuery } from "@areneon/schema/types/v1/policy-query.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { NoCandidateError } from "./actor-rule.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { buildFeatures, isFeatureSetId } from "./feature-build.ts";
import { createFeatureMemo, type FeatureBuildCode, FeatureBuildError, type FeatureConfigs, type FeatureMemo } from "./candidate-features.ts";
import { enumerateCandidates } from "./choices.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { guardActor, type LatencyBudget, type PlayedGame, playGame, UnsupportedCandidateError } from "./evaluation-harness.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { policyStreamLabel } from "./league-seed.ts";
import { assertOwnTeamStats, pinnedEngine } from "./own-team-stats.ts";
import { observe, playerChannel } from "./player-channel.ts";
import { type PolicyClient, PolicyReplyError } from "./policy-worker.ts";
import { PolicyProtocolError, type PlaySettings } from "./policy-protocol.ts";
import { type Actor, ForfeitSignal, type Player } from "./runner.ts";
import { battleClass, type DataClass, stampTeam, type SyntheticTeamHashes, teamHash } from "./team-class.ts";

/** Names the uniformly random Team Preview of the Preview ablation in its stream label and policy id (#940). */
export const RANDOM_PREVIEW_POLICY = "random-preview@v1";

/**
 * The Mulberry32 seed of a random Preview: the first 8 hex digits of SHA-256 over the JCS of the policy stream label of
 * this seat (protocol D9: seed, game, seat, policy id), read as a uint32 by `parseInt`. The label carries the seat, so
 * the two seats of a game, and the two games of a pair (candidate on p1, then on p2), draw from different streams.
 */
export function randomPreviewSeed(seed: string, game: number, player: Player): number {
  const digest = createHash("sha256").update(jcs(policyStreamLabel(seed, game, player, RANDOM_PREVIEW_POLICY) as unknown as Json)).digest("hex");
  return Number.parseInt(digest.slice(0, 8), 16);
}

/** What the sampler hashes (ADR-010 §2). */
export interface SamplerInput {
  /** The battle seed, as `BattleOptions.seed` holds it. */
  seed: string;
  /** The game's index within its protocol block or league match. */
  game: number;
  player: Player;
  policy_id: string;
  request_seq: number;
}

/**
 * The uniform draw of one decision, as an integer in `[0, 2^53 - 1]`: the
 * first 53 bits of SHA-256 over the JCS of the input. The worker turns it
 * into `u / 2^53`. The game index gives games that share a battle seed (games
 * 1 and 3 of a block) independent streams; the policy id and seat keep two
 * policies, or two seats, from sharing one.
 */
export function sampleU(input: SamplerInput): number {
  const digest = createHash("sha256").update(jcs({ ...policyStreamLabel(input.seed, input.game, input.player, input.policy_id), request_seq: input.request_seq } as unknown as Json)).digest();
  return Number(digest.readBigUInt64BE(0) >> 11n);
}

/** What a seat needs to build candidate features (ADR-022 section 3). */
export interface ActorFeatures {
  configs: FeatureConfigs;
  /** This game's memo of probe answers (default: a fresh one); never changes a byte. */
  memo?: FeatureMemo;
}

export interface PolicyActorOptions {
  client: PolicyClient;
  /** Names this game to the client, which allows one outstanding query per battle. */
  battle: string;
  player: Player;
  /** This player's own team, as the observation records it. */
  team: PrivateTeamSpec;
  manifest: VersionManifest;
  /** The class of the player's channel: the battle's, from both teams (`battleClass`). */
  channel_class: DataClass;
  /**
   * The battle seed `runBattle` is given. The sampler hashes it, so it must be
   * the same value; `policyGame` takes it once and passes it to both. Prefer it.
   */
  seed: string;
  game: number;
  policy_id: string;
  sampling: "argmax" | "sample";
  /**
   * Temperature and truncation of a `sample` seat (ADR-023 §1), put in each query. The caller computes the matching
   * `policy_id` with `policyId(pair, "sample", play)`; argmax takes none.
   */
  play?: PlaySettings;
  /**
   * The reviewed records the candidate features (ADR-022 section 3) are built from. Rows are attached to a query only
   * when the worker stated a feature set (`client.featureSet`), and then they must be that set. A request the builder
   * cannot feature (`FeatureBuildError`) forfeits the game as `crash` with the build code in the detail, counted like
   * any policy failure: the seat's decision stack could not answer, and no choice is invented for it (no rule
   * fallback, which would make the played policy partly scripted, and no incident, which would let a reader limit
   * stop a generation). Any other failure of the builder propagates to `guardActor`, which treats an unexpected
   * error of a policy seat as a `crash` too (a defect shows in the detail, not in the cause).
   */
  features?: ActorFeatures;
  /** The engine that checks the team's stats against the Team Preview request (default: the pinned checkout). */
  engine?: Engine;
  /**
   * Answers the Team Preview request instead of the worker (the Preview ablation's random Preview). It sees only this
   * player's own request, whose legal choice set the engine judged; the worker is never asked for the Preview.
   */
  preview_actor?: Actor;
}

/**
 * An A1 `Actor` that asks the policy worker. Run it under `guardActor` with
 * role `policy`: a dead worker or an invalid reply forfeits the game; a
 * candidate the worker cannot parse, a candidate list above `N_max` or a
 * request the adapter cannot observe is a harness incident.
 */
export function policyActor(options: PolicyActorOptions): Actor {
  const { client, battle, player, team, manifest, channel_class, seed, game, policy_id, sampling, play, engine, preview_actor, features } = options;
  const memo = features?.memo ?? createFeatureMemo();
  if (sampling === "argmax" && play !== undefined && (play.temperature_ppm !== undefined || play.truncation !== undefined)) {
    throw new PolicyProtocolError("a temperature or a truncation applies to sample mode only");
  }
  return async (request, context) => {
    if (context === undefined || context.player !== player) {
      throw new PolicyProtocolError(`the policy actor of ${player} needs its own player's channel`);
    }
    if (preview_actor !== undefined && request.phase === "preview") {
      // The worker is not asked, but the team is still checked against the stats the engine sent this player.
      try {
        assertOwnTeamStats(engine ?? pinnedEngine(), manifest.format_id, team, playerChannel({ schema_version: 1, data_class: channel_class, player, lines: context.channel }));
      } catch (error) {
        throw new PolicyProtocolError(`cannot observe the request: ${(error as Error).message}`);
      }
      return preview_actor(request, context);
    }
    let candidates;
    try {
      candidates = enumerateCandidates(request).candidates;
    } catch (error) {
      throw new UnsupportedCandidateError((error as Error).message);
    }
    if (candidates.length === 0) throw new NoCandidateError("the request has no candidate");
    let observation;
    let channel;
    try {
      channel = playerChannel({ schema_version: 1, data_class: channel_class, player, lines: context.channel });
      observation = await observe({ channel, team, manifest });
      // The team the observation records must give the stats the engine sent
      // this player at Team Preview (once per game: the first request).
      if (request.request_seq === 1) assertOwnTeamStats(engine ?? pinnedEngine(), manifest.format_id, team, channel);
    } catch (error) {
      throw new PolicyProtocolError(`cannot observe the request: ${(error as Error).message}`);
    }
    const { identity } = observation;
    if (identity.request_seq !== request.request_seq || identity.request_hash !== request.request_hash) {
      throw new PolicyProtocolError("the observation is of another request than the one asked");
    }
    const stated = client.featureSet ?? null;
    let candidate_features;
    if (features !== undefined && stated !== null) {
      try {
        if (!isFeatureSetId(stated.feature_set)) throw new PolicyProtocolError("the worker stated a candidate feature set this side cannot build");
        candidate_features = buildFeatures(engine ?? pinnedEngine(), features.configs, channel, team.team, candidates, memo, stated.feature_set);
      } catch (error) {
        if (error instanceof FeatureBuildError) throw new ForfeitSignal("crash", `no candidate features: ${error.code satisfies FeatureBuildCode}`);
        throw error;
      }
      if (candidate_features.feature_set_hash !== stated.feature_set_hash || candidate_features.feature_set !== stated.feature_set) {
        throw new PolicyProtocolError("the candidate features this side builds are of another feature set than the worker stated");
      }
    }
    const query = {
      schema_version: 1,
      identity,
      observation,
      candidates: candidates.map(({ choice, pending }) => ({ choice, pending })),
      ...(candidate_features === undefined ? {} : { candidate_features }),
      sampling:
        sampling === "argmax"
          ? { kind: "argmax" }
          : {
              kind: "sample",
              u: sampleU({ seed, game, player, policy_id, request_seq: identity.request_seq }),
              ...(play?.temperature_ppm === undefined ? {} : { temperature_ppm: play.temperature_ppm }),
              ...(play?.truncation === undefined ? {} : { truncation: play.truncation }),
            },
      policy_id,
    } as Omit<PolicyQuery, "query_id">;
    try {
      return (await client.ask(battle, query)).choice;
    } catch (error) {
      if (error instanceof PolicyReplyError) throw new ForfeitSignal("invalid_reply", error.message);
      throw error;
    }
  };
}

/** One seat of a `policyGame`: a learned policy behind a worker. */
export interface PolicySeat {
  client: PolicyClient;
  policy_id: string;
  sampling: "argmax" | "sample";
  /** Temperature and truncation of a `sample` seat (ADR-023 §1); see `PolicyActorOptions.play`. */
  play?: PlaySettings;
  /** The candidate-feature records of this seat (see `PolicyActorOptions.features`); rows go only to a worker that stated a set. */
  features?: ActorFeatures;
  /** Play the Team Preview uniformly at random among the legal Preview choices (the Preview ablation), seeded per seat. */
  random_preview?: true;
  /**
   * Build the seat's actor from the options `policyGame` would give `policyActor` (the same battle name, team, class, seed and game),
   * for a policy that wraps the pair: the decision-time search and its no-search control (ADR-026). The result is guarded as role
   * `policy` like any seat; the seat's `client`, `policy_id` and `sampling` stay the pair's own.
   */
  make?: (options: PolicyActorOptions) => Actor;
}

export interface PolicyGameOptions {
  engine: Engine;
  formatId: string;
  /** The battle seed: the one value the engine runs on and the sampler hashes. */
  seed: string;
  /** The game's index within its protocol block or league match. */
  game: number;
  teams: Record<Player, PokemonSet[]>;
  known_synthetic: SyntheticTeamHashes;
  /** A policy seat, or an actor already guarded by the caller (a scripted baseline). */
  seats: Record<Player, PolicySeat | Actor>;
  latency_budget: LatencyBudget;
  maxTurns?: number;
}

/**
 * Play one A1 game with policy seats. The seed is taken once and used for
 * both the battle and the sampler, so the two cannot differ; each policy seat
 * is guarded as role `policy`. Call `client.beginGame()` before it.
 */
export async function policyGame(options: PolicyGameOptions): Promise<PlayedGame> {
  const { engine, formatId, seed, game, teams, known_synthetic, seats, latency_budget, maxTurns } = options;
  const manifest = resolveFormat(engine, formatId).manifest;
  const channel_class = battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, known_synthetic);
  const actorOf = (player: Player): Actor => {
    const seat = seats[player];
    if (typeof seat === "function") return seat;
    const { random_preview, make, ...asked } = seat;
    const build = make ?? policyActor;
    const actor = build({
      ...asked,
      ...(random_preview === undefined ? {} : { preview_actor: randomLegalActor(randomPreviewSeed(seed, game, player)) }),
      battle: `game-${game}-${player}`,
      player,
      team: stampTeam(teams[player] as PrivateTeamSpec["team"], known_synthetic),
      manifest,
      channel_class,
      seed,
      game,
      engine,
    });
    return guardActor(actor, { role: "policy", latency_budget });
  };
  return playGame({
    engine,
    formatId,
    seed,
    teams,
    actors: { p1: actorOf("p1"), p2: actorOf("p2") },
    ...(maxTurns === undefined ? {} : { maxTurns }),
  });
}
