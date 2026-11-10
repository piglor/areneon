// The S0 game source (ADR-026 section 8 S0 data; #1147, #1308): a seeded A1 game of the frozen pair against `damage-aware@v1`,
// as an `OracleSpec` the fork oracle can replay. TEST TOOLING (it feeds the oracle). The pair seat is a `policyActor` (with the
// run's candidate features when the pair reads them) and the opponent seat a `damageAwareActor`, each built fresh from a factory,
// so a rollout can ask a fresh copy of exactly the policy that played the game (`OracleSpec.make_actor`).

import {
  type DamageAwareSetup,
  type Engine,
  type EvaluationClient,
  type FeatureConfigs,
  battleClass,
  damageAwareActor,
  policyActor,
  policyId,
  type PokemonSet,
  type PolicyClient,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  type VersionManifest,
} from "@areneon/showdown-bridge";
import type { OracleSpec } from "../../../showdown-bridge/test/fidelity/oracle.ts";

type Player = "p1" | "p2";
export type KnownSynthetic = ReturnType<typeof syntheticTeamHashes>;

export interface PairGameInput {
  engine: Engine;
  manifest: VersionManifest;
  /** The worker of the frozen pair and its checkpoint ids; `features` are the configs of a pair that reads candidate features (its worker states the set). */
  client: PolicyClient & EvaluationClient;
  pair: { preview: string; battle: string };
  sampling: "argmax" | "sample";
  features?: FeatureConfigs;
  /** The opponent: `damage-aware@v1` from the committed records. */
  damage: DamageAwareSetup;
  id: string;
  seed: string;
  teams: Record<Player, PokemonSet[]>;
  /** The seat the pair plays. */
  seat: Player;
  game: number;
  known: KnownSynthetic;
}

/** The pair's policy id for this game (the feature set enters it, as in training). */
export const gamePolicyId = (input: Pick<PairGameInput, "client" | "pair" | "sampling">): string => policyId(input.pair, input.sampling, {}, input.client.featureSet?.feature_set_hash);

/** The pair actor of one game, fresh on every call. `suffix` names the worker's battle so concurrent uses never share one. */
export function pairActor(input: PairGameInput, suffix: string, seed: string, game: number) {
  const { teams, seat, known, manifest } = input;
  return policyActor({
    client: input.client,
    battle: `s0-${input.id}-${suffix}`,
    player: seat,
    team: stampTeam(teams[seat] as never, known),
    manifest,
    channel_class: battleClass({ p1: teamHash(teams.p1), p2: teamHash(teams.p2) }, known),
    seed,
    game,
    policy_id: gamePolicyId(input),
    sampling: input.sampling,
    engine: input.engine,
    ...(input.features === undefined ? {} : { features: { configs: input.features } }),
  });
}

/** A fresh `damage-aware@v1` actor for the opponent seat (one seat of one game: never reuse it across games or rollouts). */
export const damageAwareFoe = (input: Pick<PairGameInput, "engine" | "damage">) => damageAwareActor({ ...input.damage, engine: input.engine, data_class: "synthetic" });

/** The oracle spec of one game of the pair against damage-aware@v1 on the given teams. */
export function pairGameSpec(input: PairGameInput): OracleSpec {
  const foe: Player = input.seat === "p1" ? "p2" : "p1";
  return {
    id: input.id,
    seed: input.seed,
    actor_seeds: { p1: 0xabc0 + input.game, p2: 0xdef0 + input.game },
    teams: input.teams,
    make_actor: {
      [input.seat]: () => pairActor(input, "play", input.seed, input.game),
      [foe]: () => damageAwareFoe(input),
    },
  };
}
