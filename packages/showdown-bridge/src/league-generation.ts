// One generation of the training loop's A1 play (ADR-010 §1): the frozen pair
// plays its match plan, every seat's Episode is written (a league collection
// record, the seed the battle ran with), and only the Episodes of the pair's
// own seats go through the training loader, which refuses any other policy.
// The plan is a pure function of the run config, the generation and the team
// pool, so a generation that was killed is replayed from the same plan.
//
// Teams are drawn from the committed synthetic fixtures (the golden team, the
// legal paste fixtures and the random-play corpus): the ADR-013 team
// generator is not built yet, and this pool is the interim. Every team drawn
// is reported by its hash, which becomes the team-set manifest of the run.

import { closeSync, fsyncSync, mkdirSync, openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import type { Engine, PokemonSet } from "./engine.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { buildLeagueRunRecord, type LeagueRunRecord } from "./league-run-record.ts";
import { leagueGameStreams } from "./league-seed.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { episodeHash } from "./episode.ts";
import { guardActor, type LatencyBudget, type PlayedGame } from "./evaluation-harness.ts";
import { firstLegal, syntheticTeam, TEAMS_DIR, teamFixtureManifest } from "./golden.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { nodeSha256 } from "./player-channel.ts";
import { type PolicyClient } from "./policy-worker.ts";
import { type PolicyGameOptions, policyGame, type PolicySeat } from "./policy-actor.ts";
import { loadRandomCorpus } from "./random-corpus.ts";
import { randomLegalActor } from "./random-legal.ts";
import type { Actor, ForfeitCause, Player } from "./runner.ts";
import { importPaste } from "./teams.ts";
import { stampTeam, type SyntheticTeamHashes, teamHash } from "./team-class.ts";
import { loadTrainingSet, type TrainingSet } from "./training-loader.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import type { Episode } from "./episode.ts";

/** The teams a run may draw: synthetic by source, in a fixed order. */
export interface TrainingPool {
  /** Sorted by team hash, so an index names the same team on every machine. */
  teams: readonly PokemonSet[][];
  hashes: readonly string[];
  /** Every team of the pool, proven synthetic by where it was read from. */
  known_synthetic: SyntheticTeamHashes;
}

/**
 * The interim team pool (the team generator is ADR-013, not built): the golden
 * team, every legal paste fixture and every team of the random-play corpus.
 * All three are committed under `fixtures/` as CC0 synthetic data. A team with
 * a set whose ability is in `excludedAbilities` (the run config's list) is left out.
 */
export function trainingTeamPool(engine: Engine, excludedAbilities: readonly string[] = []): TrainingPool {
  const byHash = new Map<string, PokemonSet[]>();
  const add = (team: PokemonSet[]): void => void byHash.set(teamHash(team), team);
  add(syntheticTeam());
  const manifest = teamFixtureManifest();
  for (const [file, fixture] of Object.entries(manifest.fixtures)) {
    if (fixture.expect === "legal") add(importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8")));
  }
  for (const battle of loadRandomCorpus()) {
    add(battle.teams.p1);
    add(battle.teams.p2);
  }
  const excluded = new Set(excludedAbilities);
  const hashes = [...byHash.keys()].filter((hash) => !(byHash.get(hash) as PokemonSet[]).some((set) => excluded.has(set.ability))).sort();
  if (hashes.length === 0) throw new GenerationError("the team pool is empty after the excluded abilities");
  return { teams: hashes.map((hash) => byHash.get(hash) as PokemonSet[]), hashes, known_synthetic: new Set(hashes) };
}

/** The scripted opponent of a baseline game; the policy takes the other seat. */
export interface BaselineOpponent {
  actor: "actor-rule" | "random-legal";
  policy_id: string;
}

export interface GamePlan {
  /** The game's index within the generation, the sampler's `game`. */
  game: number;
  seed: string;
  /** Indices into the pool. */
  teams: Record<Player, number>;
  /** `self`: the pair plays both seats. Otherwise the pair takes `policy_seat` and a baseline the other. */
  opponent: "self" | BaselineOpponent;
  policy_seat: Player | "both";
  /** Seeds the random-legal baseline's stream (an unsigned 32-bit integer). */
  baseline_seed: number;
}

/**
 * The generation's match plan: `generation.games` self-play games, then each
 * listed baseline's games. Every number comes from `leagueGameStreams` (league-seed/1: SHA-256
 * of the JCS of the run key, the generation and the game index), so the plan
 * is reproducible from the config alone. A baseline game alternates the policy's seat.
 */
export function planGeneration(config: TrainingRunConfig, generation: number, poolSize: number): GamePlan[] {
  if (!Number.isInteger(generation) || generation < 1) throw new RangeError("a generation is numbered from 1");
  if (!Number.isInteger(poolSize) || poolSize < 1) throw new RangeError("the team pool is empty");
  const { games, baseline_games } = config.generation;
  const opponents: GamePlan["opponent"][] = [];
  for (let i = 0; i < games; i++) opponents.push("self");
  for (const baseline of baseline_games) {
    for (let i = 0; i < baseline.games; i++) opponents.push({ actor: baseline.actor, policy_id: baseline.policy_id });
  }
  return opponents.map((opponent, game) => {
    const streams = leagueGameStreams(config.seeds.run_key, generation, game);
    return {
      game,
      seed: streams.seed,
      teams: { p1: streams.team_draws.p1 % poolSize, p2: streams.team_draws.p2 % poolSize },
      opponent,
      policy_seat: opponent === "self" ? "both" : game % 2 === 0 ? "p1" : "p2",
      baseline_seed: streams.baseline_seed,
    };
  });
}

export interface GenerationOptions {
  engine: Engine;
  config: TrainingRunConfig;
  generation: number;
  /** The frozen pair's `behaviour_policy_id` (`policy:` + SHA-256). */
  policy_id: string;
  /** A started worker client; `beginGame` (when it has one) runs before each wave, as the client's overrun rule asks. */
  client: PolicyClient & { beginGame?: () => Promise<void> };
  pool: TrainingPool;
}

export interface GameRow {
  game: number;
  seed: string;
  opponent: string;
  outcome: "played" | "forfeit";
  winner: Player | null;
  turns: number;
  /** Only on a forfeited game: the seat that lost, why, at which request of its channel, and the harness's detail (#912). */
  forfeit?: { loser: Player; cause: ForfeitCause; request_seq: number; detail: string };
}

/** The forfeits of a generation by cause, sorted by cause; empty when none. */
export function forfeitCauses(games: readonly GameRow[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const game of games) if (game.forfeit !== undefined) counts[game.forfeit.cause] = (counts[game.forfeit.cause] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => (a < b ? -1 : 1)));
}

export interface GenerationResult {
  generation: number;
  plan: GamePlan[];
  games: GameRow[];
  /** The Episodes of every seat that was played (the league record), as stored. */
  episodes: Episode[];
  /** Episode hashes, in `episodes` order. */
  episode_hashes: string[];
  stream: string;
  set: TrainingSet;
  /** What the generation played, tied to the run config and the seed derivation (`league-run.json`). */
  run_record: LeagueRunRecord;
  /** SHA-256 of the TrainingSet's JCS bytes: what `CheckpointRecord.training_set_hashes` holds. */
  training_set_hash: string;
  manifest: VersionManifest;
  /** Hashes of the teams the plan drew, sorted and unique. */
  team_hashes: string[];
}

/** A game that could not be played: a harness incident, which fails the generation. */
export class GenerationError extends Error {
  override name = "GenerationError";
}

type StampSets = Parameters<typeof stampTeam>[0];

const rival = (player: Player): Player => (player === "p1" ? "p2" : "p1");

function baselineActor(opponent: BaselineOpponent, plan: GamePlan, budget: LatencyBudget): Actor {
  const actor = opponent.actor === "actor-rule" ? firstLegal : randomLegalActor(plan.baseline_seed);
  return guardActor(actor, { role: "scripted", latency_budget: budget });
}

async function playOne(options: GenerationOptions, plan: GamePlan): Promise<PlayedGame> {
  const { engine, config, policy_id, client, pool } = options;
  const { generation: settings } = config;
  const budget = { preview_ms: settings.latency_budget_ms.preview_ms, battle_ms: settings.latency_budget_ms.battle_ms };
  const seat: PolicySeat = { client, policy_id, sampling: "sample" };
  const seats: PolicyGameOptions["seats"] =
    plan.opponent === "self"
      ? { p1: seat, p2: seat }
      : plan.policy_seat === "p1"
        ? { p1: seat, p2: baselineActor(plan.opponent, plan, budget) }
        : { p1: baselineActor(plan.opponent, plan, budget), p2: seat };
  return policyGame({
    engine,
    formatId: config.format_id,
    seed: plan.seed,
    game: plan.game,
    teams: { p1: pool.teams[plan.teams.p1] as PokemonSet[], p2: pool.teams[plan.teams.p2] as PokemonSet[] },
    known_synthetic: pool.known_synthetic,
    seats,
    latency_budget: budget,
    maxTurns: settings.max_turns,
  });
}

const idOf = (opponent: GamePlan["opponent"], policy_id: string): string => (opponent === "self" ? policy_id : opponent.policy_id);

/**
 * Play the generation's plan against `client` (started, with the frozen pair
 * loaded) in waves of `n_parallel` games, write an Episode for every played
 * seat, and load the pair's own seats through the training loader.
 *
 * Refused: a harness incident in any game (the generation fails, nothing is
 * retried); no Episode of the pair at all (every game forfeited). A game that
 * a policy forfeited is not an Episode (it is not training data) and is
 * reported in `games`.
 */
export async function playGeneration(options: GenerationOptions): Promise<GenerationResult> {
  const { engine, config, generation, policy_id, client, pool } = options;
  const plan = planGeneration(config, generation, pool.teams.length);
  const rows: GameRow[] = [];
  const episodes: Episode[] = [];
  for (let start = 0; start < plan.length; start += config.generation.n_parallel) {
    await client.beginGame?.();
    const wave = plan.slice(start, start + config.generation.n_parallel);
    const played = await Promise.all(wave.map((entry) => playOne(options, entry)));
    for (const [index, game] of played.entries()) {
      const entry = wave[index] as GamePlan;
      const { outcome, record } = game;
      if (outcome.kind === "incident") throw new GenerationError(`game ${entry.game} (${entry.seed}): ${outcome.incident}: ${outcome.message}`);
      const row = { game: entry.game, seed: entry.seed, opponent: idOf(entry.opponent, policy_id) };
      if (outcome.kind === "forfeit" || record === null) {
        if (outcome.kind !== "forfeit") {
          rows.push({ ...row, outcome: "forfeit", winner: null, turns: 0 });
        } else {
          const { loser, cause, request_seq, detail } = outcome;
          rows.push({ ...row, outcome: "forfeit", winner: outcome.winner, turns: outcome.turns, forfeit: { loser, cause, request_seq, detail } });
        }
        continue;
      }
      rows.push({ ...row, outcome: "played", winner: record.winner, turns: record.turns });
      const sides: Record<Player, string> = {
        p1: entry.opponent !== "self" && entry.policy_seat !== "p1" ? entry.opponent.policy_id : policy_id,
        p2: entry.opponent !== "self" && entry.policy_seat !== "p2" ? entry.opponent.policy_id : policy_id,
      };
      for (const player of ["p1", "p2"] as const) {
        episodes.push(
          await episodeFromRecord(record, player, {
            team: stampTeam(pool.teams[entry.teams[player]] as StampSets, pool.known_synthetic),
            opponent_team: stampTeam(pool.teams[entry.teams[rival(player)]] as StampSets, pool.known_synthetic),
            known_synthetic: pool.known_synthetic,
            behaviour_policy_id: sides[player],
            collection: { kind: "league", run_id: config.seeds.run_key, seed: record.seed, opponent_policy_id: sides[rival(player)] },
          }),
        );
      }
    }
  }
  const own = episodes.filter((episode) => episode.provenance.behaviour_policy_id === policy_id);
  if (own.length === 0) throw new GenerationError("no Episode of the frozen pair: every game was forfeited");
  const { stream, set } = await loadTrainingSet({
    episodes: own,
    evaluation_manifests: [],
    known_synthetic: pool.known_synthetic,
    behaviour_policy_id: policy_id,
  });
  const drawn = new Set<string>();
  for (const entry of plan) for (const player of ["p1", "p2"] as const) drawn.add(pool.hashes[entry.teams[player]] as string);
  return {
    generation,
    plan,
    games: rows,
    episodes,
    episode_hashes: await Promise.all(episodes.map((episode) => episodeHash(episode, nodeSha256))),
    stream,
    set,
    run_record: await buildLeagueRunRecord({ config, generation, pair_policy_id: policy_id, plan, games: rows }),
    training_set_hash: await nodeSha256(new TextEncoder().encode(jcs(set as unknown as Json))),
    manifest: resolveFormat(engine, config.format_id).manifest,
    team_hashes: [...drawn].sort(),
  };
}

/**
 * Write a generation's files under `dir` (created if needed), all canonical JSON
 * except the stream, which is the loader's own bytes:
 * `episodes/<episode hash>.json` (every played seat), `stream.jsonl`,
 * `training-set.json`, `plan.json`, `league-run.json` (the LeagueRunRecord) and `generation.json` (the per-game rows and
 * the hashes the Python driver records). Nothing in them depends on the clock.
 */
export function writeGeneration(dir: string, result: GenerationResult): void {
  const episodes = join(dir, "episodes");
  mkdirSync(episodes, { recursive: true });
  const put = (name: string, value: unknown): void => writeDurable(join(dir, name), `${jcs(value as Json)}\n`);
  result.episodes.forEach((episode, index) => put(`episodes/${result.episode_hashes[index]}.json`, episode));
  writeDurable(join(dir, "stream.jsonl"), result.stream);
  put("training-set.json", result.set);
  put("plan.json", result.plan);
  put("league-run.json", result.run_record);
  // generation.json last: it is the file that says the rest is whole.
  put("generation.json", {
    schema_version: 1,
    generation: result.generation,
    games: result.games,
    forfeits: result.games.filter((game) => game.outcome === "forfeit").length,
    // Present only when a game was forfeited, so a run without forfeits writes the bytes it always did.
    ...(result.games.some((game) => game.outcome === "forfeit") ? { forfeit_causes: forfeitCauses(result.games) } : {}),
    episode_hashes: [...result.episode_hashes].sort(),
    training_set_hash: result.training_set_hash,
    examples: result.set.examples,
    manifest: result.manifest,
    team_hashes: result.team_hashes,
  });
  syncDir(episodes);
  syncDir(dir);
}

/** Write a file and flush it to disk before returning. */
function writeDurable(path: string, data: string): void {
  const descriptor = openSync(path, "w");
  try {
    writeSync(descriptor, data);
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

/** Flush a directory's entries to disk. */
function syncDir(path: string): void {
  const descriptor = openSync(path, "r");
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}
