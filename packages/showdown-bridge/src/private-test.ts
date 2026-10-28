// The private test set's play (ADR-015 erratum E1, item 2): one frozen pair
// plays the scripted baselines (and, optionally, another frozen pair) on teams
// the caller hands in, in sample and argmax mode, on A1 with one query per
// forward pass. Teams are ordered and cut into disjoint pairs by HMAC under the
// per-test salt (`private-test-pairing/1`), every pair is played from both
// seats with one battle seed (`private-test-seed/1`), and the result is a
// PrivateTestRecord of COUNTS only. It reuses the progress evaluation's seat
// builders, outcome reading and tally; it writes no game record, omniscient log,
// Episode or page: a played game's record is dropped when its outcome is read.
// A harness incident fails the test and returns nothing; a policy forfeit is a
// scored loss for the failing side with its cause.

import type { Engine, PokemonSet } from "./engine.ts";
import { guardActor, type LatencyBudget } from "./evaluation-harness.ts";
import { resolveFormat } from "./formats.ts";
import { firstLegal } from "./golden.ts";
import type { DamageAwareSetup } from "./league-generation.ts";
import { type PolicyClient } from "./policy-worker.ts";
import type { FeatureConfigs } from "./candidate-features.ts";
import { policyGame, type PolicySeat } from "./policy-actor.ts";
import { DAMAGE_AWARE_POLICY_ID, damageAwareBaseline, PROGRESS_BASELINES, PROGRESS_MODES, type ProgressMode, ProgressError, rowOf, type BaselineReport } from "./progress-evaluation.ts";
import { tallyRows } from "./progress-records.ts";
import { PRIVATE_TEST_PAIRING, PRIVATE_TEST_SEED_DERIVATION, privateTestCommitment, privateTestPairs, privateTestStreams } from "./private-test-seed.ts";
import { type PrivateTestRecord, type PrivateTestMatch, readPrivateTestRecord } from "./private-test-record.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, Player } from "./runner.ts";
import { type SyntheticTeamHashes, syntheticTeamHashes } from "./team-class.ts";
import { runConfigHash, type TrainingRunConfig } from "./training-run-config.ts";

/** A team of the test set: its canonical hash (the pairing identity) and the sets a battle starts with. */
export interface PrivateTestTeam {
  canonical_team_hash: string;
  sets: PokemonSet[];
}

type PairClient = PolicyClient & { beginGame?: () => Promise<void> };

export interface PrivateTestOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** SHA-256 of `generations/g<N>/manifest.json`. */
  generation_manifest_sha256: string;
  checkpoints: { preview: string; battle: string };
  policy_ids: Record<ProgressMode, string>;
  /** The data class of the pair's checkpoint records. */
  candidate_data_class: "synthetic" | "private";
  /** A started worker client with the pair loaded; one query at a time. */
  client: PairClient;
  /** The candidate features the pair reads (`featureSetup(...).configs`; absent: none); its policy ids then carry the `features` member. */
  features?: FeatureConfigs;
  /** Another frozen pair as an opponent, behind a worker of its own: sample against sample, argmax against argmax. */
  opponent_pair?: { checkpoints: { preview: string; battle: string }; policy_ids: Record<ProgressMode, string>; client: PairClient; /** The other pair's own features, from its own run config. */ features?: FeatureConfigs };
  /** The damage-aware baseline as a scripted opponent (opt-in; a match over its wall-clock budget fails the test). */
  damage_aware?: { setup: DamageAwareSetup; match_budget_ms: number; clock?: () => number };
  /** The teams of the set, each once. */
  teams: readonly PrivateTestTeam[];
  source: { kind: "staging" } | { kind: "snapshot"; snapshot_hash: string } | { kind: "dataset"; dataset_hash: string };
  subset: "all" | "holdout";
  /** The per-test 256-bit salt (64 hex digits). */
  salt: string;
  /** Play only the first this-many pairs (default: every disjoint pair). */
  max_pairs?: number;
  /** Teams proven synthetic (a public dataset's); default none, so the games are classed private. */
  known_synthetic?: SyntheticTeamHashes;
}

export interface PrivateTestResult {
  record: PrivateTestRecord;
  /** Wall-clock milliseconds per match: a measurement for the operator, never stored. */
  wall_ms: Record<string, number>;
}

const seatOf = (index: number): Player => (index % 2 === 0 ? "p1" : "p2");
const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");
/** The candidate's half-games in one game: win 2, tie 1, loss 0. */
const HALVES = { candidate: 2, tie: 1, opponent: 0 } as const;

type Opponent = {
  id: string;
  budget_ms?: number;
  policy_id: (mode: ProgressMode) => string;
  checkpoints?: { preview: string; battle: string };
  seat: (mode: ProgressMode, pair: number) => { seat: PolicySeat | Actor; report?: () => BaselineReport };
  begin: () => Promise<void>;
};

/** How many disjoint pairs a set of teams gives. */
export const pairsAvailable = (teams: number): number => Math.floor(teams / 2);

/**
 * Play the test and return its record. Refused before any game (`ProgressError`): fewer than two teams, a team
 * twice, more pairs than the set gives, a damage-aware setup for another format. A harness incident throws.
 */
export async function playPrivateTest(options: PrivateTestOptions): Promise<PrivateTestResult> {
  const { engine, config, client, teams } = options;
  const byHash = new Map(teams.map((team) => [team.canonical_team_hash, team]));
  if (byHash.size !== teams.length) throw new ProgressError("a team appears twice in the test set");
  const available = pairsAvailable(teams.length);
  if (available < 1) throw new ProgressError("the test set needs at least two teams");
  const wanted = options.max_pairs ?? available;
  if (!Number.isSafeInteger(wanted) || wanted < 1 || wanted > available) throw new ProgressError(`the test set gives ${available} disjoint pairs`);
  const pairs = privateTestPairs(options.salt, [...byHash.keys()]).slice(0, wanted);
  const budget: LatencyBudget = { preview_ms: config.generation.latency_budget_ms.preview_ms, battle_ms: config.generation.latency_budget_ms.battle_ms };
  const manifest = resolveFormat(engine, config.format_id).manifest;
  const known = options.known_synthetic ?? syntheticTeamHashes([]);
  const clock = options.damage_aware?.clock ?? (() => performance.now());

  const opponents: Opponent[] = PROGRESS_BASELINES.map((baseline) => ({
    id: baseline.actor,
    policy_id: () => baseline.policy_id,
    seat: (_mode, pair) => ({ seat: guardActor(baseline.actor === "actor-rule" ? firstLegal : randomLegalActor(privateTestStreams(options.salt, pair).baseline_seed), { role: "scripted", latency_budget: budget }) }),
    begin: () => Promise.resolve(),
  }));
  const damage = options.damage_aware;
  if (damage !== undefined) {
    if (!Number.isSafeInteger(damage.match_budget_ms) || damage.match_budget_ms < 1) throw new ProgressError("the damage-aware match budget must be a positive whole number of milliseconds");
    if (damage.setup.config.epsilon_ppm !== 0) throw new ProgressError("the damage-aware baseline plays an evaluation with epsilon_ppm 0: it draws nothing");
    if (damage.setup.config.format_id !== config.format_id) throw new ProgressError("the damage-aware baseline's records are for another format");
    const baseline = damageAwareBaseline({ engine, setup: damage.setup, manifest, data_class: "private", budget });
    opponents.push({
      id: "damage-aware",
      budget_ms: damage.match_budget_ms,
      policy_id: () => DAMAGE_AWARE_POLICY_ID,
      seat: (_mode, pair) => baseline.seat(privateTestStreams(options.salt, pair).baseline_seed),
      begin: () => Promise.resolve(),
    });
  }
  const other = options.opponent_pair;
  if (other !== undefined) {
    opponents.push({
      id: "pair",
      policy_id: (mode) => other.policy_ids[mode],
      checkpoints: other.checkpoints,
      seat: (mode) => ({ seat: { client: other.client, policy_id: other.policy_ids[mode], sampling: mode, ...(other.features === undefined ? {} : { features: { configs: other.features } }) } }),
      begin: async () => {
        await other.client.beginGame?.();
      },
    });
  }

  const matches: PrivateTestMatch[] = [];
  const wall_ms: Record<string, number> = {};
  for (const opponent of opponents) {
    for (const mode of PROGRESS_MODES) {
      const match_id = `${opponent.id}-${mode}`;
      const policy_id = options.policy_ids[mode];
      const rows: { outcome: ReturnType<typeof rowOf>["outcome"] }[] = [];
      const pairScores: number[] = [];
      const started = clock();
      for (let index = 0; index < pairs.length; index++) {
        const [first, second] = pairs[index] as [string, string];
        const sets: Record<Player, PokemonSet[]> = { p1: (byHash.get(first) as PrivateTestTeam).sets, p2: (byHash.get(second) as PrivateTestTeam).sets };
        const streams = privateTestStreams(options.salt, index);
        let halves = 0;
        // Both games of a pair: the teams stay on their seats, the policies swap, one battle seed.
        for (let leg = 0; leg < 2; leg++) {
          const candidate = seatOf(leg);
          const seat: PolicySeat = { client, policy_id, sampling: mode, ...(options.features === undefined ? {} : { features: { configs: options.features } }) };
          await client.beginGame?.();
          await opponent.begin();
          const opposing = opponent.seat(mode, index);
          const played = await policyGame({
            engine,
            formatId: config.format_id,
            seed: streams.seed,
            game: index,
            teams: sets,
            known_synthetic: known,
            seats: { [candidate]: seat, [rival(candidate)]: opposing.seat } as Record<Player, PolicySeat | Actor>,
            latency_budget: budget,
            maxTurns: config.generation.max_turns,
          });
          opposing.report?.(); // checks the baseline's identity; its counts are not stored
          const { outcome } = rowOf(match_id, leg, index, candidate, played);
          rows.push({ outcome });
          halves += HALVES[outcome.kind === "forfeit" ? (outcome.loser === "candidate" ? "opponent" : "candidate") : outcome.winner];
          if (opponent.budget_ms !== undefined && clock() - started > opponent.budget_ms) throw new ProgressError(`${match_id} passed its wall-clock budget: nothing is written`);
        }
        pairScores.push(halves);
      }
      wall_ms[match_id] = Math.round(clock() - started);
      const tally = tallyRows(rows, config.generation.max_turns);
      const pair_score_counts: [number, number, number, number, number] = [0, 0, 0, 0, 0];
      for (const score of pairScores) pair_score_counts[score as 0 | 1 | 2 | 3 | 4]++;
      matches.push({
        match_id,
        policy_id,
        mode,
        opponent_policy_id: opponent.policy_id(mode),
        ...(opponent.checkpoints === undefined ? {} : { opponent_checkpoints: opponent.checkpoints }),
        games: rows.length,
        wins: tally.wins,
        ties: tally.ties,
        losses: tally.losses,
        turn_cap_ties: tally.turn_cap_ties,
        forfeits: tally.forfeits.map(({ side, cause, count }) => ({ side, cause, count })),
        pairs: pairs.length,
        pair_score_counts,
      });
    }
  }
  matches.sort((a, b) => (a.match_id < b.match_id ? -1 : 1));
  const record = readPrivateTestRecord({
    schema_version: 1,
    data_class: "private",
    run_id: config.seeds.run_key,
    run_config_hash: await runConfigHash(config),
    generation: options.generation,
    generation_manifest_sha256: options.generation_manifest_sha256,
    checkpoints: options.checkpoints,
    candidate_data_class: options.candidate_data_class,
    version_manifest: manifest,
    test_set: {
      commitment: privateTestCommitment(options.salt, [...byHash.keys()]),
      team_count: teams.length,
      pairs: pairs.length,
      source_kind: options.source.kind,
      ...(options.source.kind === "snapshot" ? { snapshot_hash: options.source.snapshot_hash } : {}),
      ...(options.source.kind === "dataset" ? { dataset_hash: options.source.dataset_hash } : {}),
      subset: options.subset,
    },
    plan: {
      kind: "disjoint-pairs",
      seed_derivation: PRIVATE_TEST_SEED_DERIVATION,
      pairing: PRIVATE_TEST_PAIRING,
      games_per_match: pairs.length * 2,
      max_turns: config.generation.max_turns,
      latency_budget_ms: budget,
    },
    matches,
  });
  return { record, wall_ms };
}
