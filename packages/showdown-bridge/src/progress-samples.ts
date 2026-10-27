// The progress contracts' share of the sample corpus (ADR-008 section 2):
// valid, invalid (schema) and reader-invalid samples of the
// ProgressEvaluationRecord and the ProgressGameRecord. The game is a real
// golden-team battle played under the seed `progress-seed/1` gives it; the
// evaluation record is a small, consistent one around it (the run, dataset and
// generation hashes are placeholders that pin a shape).

// The mutators below edit nested sample fields by path, one fault per sample.
/* eslint-disable @typescript-eslint/no-explicit-any */

import type { Engine } from "./engine.ts";
import { playGame } from "./evaluation-harness.ts";
import { firstLegal, GOLDEN_FORMAT, syntheticTeam } from "./golden.ts";
import { buildProgressGameRecord } from "./progress-evaluation.ts";
import { omniscientLogHash, progressRecordHash } from "./progress-records.ts";
import { progressGameStreams } from "./progress-seed.ts";

const hexOf = (c: string): string => c.repeat(64);

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, any>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

export interface ProgressSamples {
  valid: Record<string, unknown>;
  invalid: Record<string, unknown>;
  readerInvalid: Record<string, unknown>;
}

export async function progressSamples(engine: Engine): Promise<ProgressSamples> {
  const run_id = "tiny-run";
  const team = syntheticTeam();
  const played = await playGame({
    engine,
    formatId: GOLDEN_FORMAT,
    seed: progressGameStreams(run_id, 0).seed,
    teams: { p1: team, p2: team },
    actors: { p1: firstLegal, p2: firstLegal },
    maxTurns: 3, // a short game keeps the corpus small: the turn cap ends it as a tie
  });
  const pair = `policy:${hexOf("a")}`;
  const game = buildProgressGameRecord({ run_id, generation: 1, match_id: "actor-rule-argmax", game: 0, teams: { p1: team, p2: team }, policies: { p1: pair, p2: "actor-rule@v1" }, played });
  // The sample game is played to an end; the candidate sits on p1, so a p1 win is the candidate's.
  const seen = { p1: "candidate", p2: "opponent", tie: "tie" } as const;
  const { winner, turns } = game.outcome as { winner: keyof typeof seen; turns: number };
  const candidateWins = seen[winner];
  const gameHash = progressRecordHash(game);
  const checkpoints = { preview: hexOf("1"), battle: hexOf("2") };
  const row = (match_id: string, index: number, outcome: unknown) => ({ match_id, game: index, pair: index, candidate_seat: index % 2 === 0 ? "p1" : "p2", outcome });
  const evaluation = {
    schema_version: 1,
    data_class: "synthetic",
    run_id,
    run_config_hash: hexOf("3"),
    generation: 1,
    generation_manifest_sha256: hexOf("4"),
    checkpoints,
    version_manifest: game.version_manifest,
    teams: { dataset_hash: hexOf("5"), split: "validation", pairs_used: 2 },
    plan: {
      seed_derivation: "progress-seed/1",
      pairing: "progress-pairing/1",
      games_per_match: 2,
      recorded_games_per_match: 1,
      max_turns: 100,
      latency_budget_ms: { preview_ms: 30000, battle_ms: 30000 },
    },
    matches: [
      {
        match_id: "actor-rule-argmax",
        policy_id: pair,
        opponent: { kind: "baseline", policy_id: "actor-rule@v1" },
        mode: "argmax",
        games: 2,
        // Game 1 is a forfeit by the candidate: one more loss.
        ...{ candidate: { wins: 1, ties: 0, losses: 1 }, tie: { wins: 0, ties: 1, losses: 1 }, opponent: { wins: 0, ties: 0, losses: 2 } }[candidateWins],
        turn_cap_ties: 0,
        turns_total: turns + 7,
        forfeits: [{ side: "candidate", cause: "latency_overrun", count: 1 }],
        game_record_hashes: [gameHash],
      },
    ],
    games: [
      row("actor-rule-argmax", 0, { kind: "played", winner: candidateWins, turns }),
      row("actor-rule-argmax", 1, { kind: "forfeit", loser: "candidate", cause: "latency_overrun", request_seq: 3, turns: 7 }),
    ],
  };
  const withPrevious = mutated(evaluation, (e) => {
    const previous = { kind: "previous", policy_id: `policy:${hexOf("b")}`, checkpoints: { preview: hexOf("6"), battle: hexOf("7") } };
    const match = { ...e.matches[0], match_id: "previous-sample", mode: "sample", opponent: previous, forfeits: [{ side: "opponent", cause: "crash", count: 1 }], wins: 2, losses: 0, ties: 0, game_record_hashes: [gameHash] };
    e.matches = [match];
    e.games = [row("previous-sample", 0, { kind: "played", winner: "candidate", turns }), row("previous-sample", 1, { kind: "forfeit", loser: "opponent", cause: "crash", request_seq: 2, turns: 7 })];
  });
  // Schema version 2 (ADR-017 erratum E2): every pair is played from both seats, so 2n games for n pairs. Two pairs here.
  const v2 = mutated(evaluation, (e) => {
    e.schema_version = 2;
    e.plan.games_per_match = 4;
    e.teams.pairs_used = 2;
    e.matches = [{ ...e.matches[0], games: 4, wins: 2, ties: 1, losses: 1, turn_cap_ties: 0, turns_total: 4 * turns, forfeits: [], game_record_hashes: [gameHash] }];
    e.games = (["candidate", "opponent", "candidate", "tie"] as const).map((winner, index) => ({ match_id: "actor-rule-argmax", game: index, pair: index >> 1, candidate_seat: index % 2 === 0 ? "p1" : "p2", outcome: { kind: "played", winner, turns } }));
  });
  const withV2 = (change?: (e: Record<string, any>) => void) => mutated(v2, (e) => change?.(e));
  const renameMatch = (e: Record<string, any>, id: string) => {
    e.matches[0].match_id = id;
    e.games.forEach((g: any) => (g.match_id = id));
  };
  const v2Previous = withV2((e) => {
    renameMatch(e, "previous-sample");
    Object.assign(e.matches[0], { mode: "sample", opponent: { kind: "previous", policy_id: `policy:${hexOf("b")}`, checkpoints: { preview: hexOf("6"), battle: hexOf("7") } } });
  });
  const v2Anchor = (change?: (e: Record<string, any>) => void) =>
    withV2((e) => {
      renameMatch(e, "anchor-g0-sample");
      Object.assign(e.matches[0], { mode: "sample", opponent: { kind: "anchor", generation: 0, policy_id: `policy:${hexOf("b")}`, checkpoints: { preview: hexOf("6"), battle: hexOf("7") } } });
      e.plan.anchor_games_per_match = 4;
      change?.(e);
    });
  // A cross-run match: another run's frozen pair (a different run key), run-<mode>.
  const v2Run = (change?: (e: Record<string, any>) => void) =>
    withV2((e) => {
      renameMatch(e, "run-sample");
      Object.assign(e.matches[0], { mode: "sample", opponent: { kind: "run", run_id: "other-run", run_config_hash: hexOf("c"), generation: 3, policy_id: `policy:${hexOf("b")}`, checkpoints: { preview: hexOf("6"), battle: hexOf("7") } } });
      e.plan.run_games_per_match = 4;
      change?.(e);
    });
  // The Preview ablation (#940): one Battle network with three Previews against one baseline, argmax.
  const v2Ablation = (change?: (e: Record<string, any>) => void) =>
    withV2((e) => {
      const variant = (id: string, candidate_preview: unknown, policy_id: string) => ({ ...e.matches[0], match_id: id, policy_id, candidate_preview });
      const learned = variant("preview-learned-actor-rule-argmax", { kind: "learned" }, `policy:${hexOf("a")}`);
      const random = variant("preview-random-actor-rule-argmax", { kind: "random" }, `policy:${hexOf("c")}`);
      const earlier = variant("preview-g0-actor-rule-argmax", { kind: "generation", generation: 0, checkpoint: hexOf("8") }, `policy:${hexOf("d")}`);
      e.matches = [earlier, learned, random];
      e.games = e.matches.flatMap((m: any) => (["candidate", "opponent", "candidate", "tie"] as const).map((winner, index) => ({ match_id: m.match_id, game: index, pair: index >> 1, candidate_seat: index % 2 === 0 ? "p1" : "p2", outcome: { kind: "played", winner, turns } })));
      change?.(e);
    });
  // The damage-aware baseline (ADR-018 slice 4): the seat that is p2 here plays it, with its integer counts.
  const answeredByP2 = game.decisions.filter((d) => d.player === "p2").length;
  const gameCounts = { decisions: answeredByP2, preview_decisions: 1, scored_decisions: answeredByP2 - 1, queries: 12, probe_runs: 9, unsupported_queries: 2, reader_fallbacks: 0, assumed_set_fallbacks: 0, epsilon_plays: 0 };
  const identity = hexOf("9");
  const damageGame = mutated(game, (g) => {
    g.match_id = "damage-aware-argmax";
    g.policies.p2 = "damage-aware@v1";
    g.baseline = { seat: "p2", identity_hash: identity, counts: gameCounts };
  }) as unknown as typeof game;
  const damageHash = progressRecordHash(damageGame);
  // A faulty damage-aware game keeps a few lines, like the other faulty games; its counts are those of the short game.
  const shortDamage = (change: (g: Record<string, any>) => void) =>
    mutated(damageGame, (g) => {
      g.omniscient_log = g.omniscient_log.slice(0, 3);
      g.omniscient_log_sha256 = omniscientLogHash(g.omniscient_log);
      g.decisions = g.decisions.slice(0, 2);
      g.baseline.counts.decisions = g.decisions.filter((d: any) => d.player === "p2").length;
      g.baseline.counts.scored_decisions = 0;
      change(g);
    });
  const v2Damage = (change?: (e: Record<string, any>) => void) =>
    withV2((e) => {
      renameMatch(e, "damage-aware-argmax");
      Object.assign(e.matches[0], { opponent: { kind: "baseline", policy_id: "damage-aware@v1" }, game_record_hashes: [damageHash], baseline: { identity_hash: identity, counts: Object.fromEntries(Object.entries(gameCounts).map(([k, v]) => [k, v * 4])) } });
      e.games.forEach((g: any) => (g.baseline_counts = { ...gameCounts }));
      change?.(e);
    });
  // Both modes of the baseline in one record: the matches sort as damage-aware-argmax, damage-aware-sample.
  const v2DamageBoth = (change?: (e: Record<string, any>) => void) =>
    v2Damage((e) => {
      const sample = structuredClone({ ...e.matches[0], match_id: "damage-aware-sample", mode: "sample" });
      e.matches.push(sample);
      e.games.push(...e.games.map((g: any) => ({ ...structuredClone(g), match_id: "damage-aware-sample" })));
      change?.(e);
    });
  const withEval = (change: (e: Record<string, any>) => void) => mutated(evaluation, change);
  const withGame = (change: (g: Record<string, any>) => void) => mutated(game, change);
  // A faulty sample whose fault does not depend on the log's length carries a few lines, so the corpus stays small.
  const withShortGame = (change: (g: Record<string, any>) => void) =>
    withGame((g) => {
      g.omniscient_log = g.omniscient_log.slice(0, 3);
      g.omniscient_log_sha256 = omniscientLogHash(g.omniscient_log);
      g.decisions = g.decisions.slice(0, 2);
      change(g);
    });
  // A game of a both-seats evaluation: game 1 is pair 0 played again with the seats' policies swapped, on pair 0's seed.
  const pairGame = withGame((g) => ((g.game = 1), (g.pair = 0)));
  const pairKeyGame = withShortGame((g) => ((g.game = 1), (g.pair = 0), (g.evaluation_key = "paired-key"), (g.seed = progressGameStreams("paired-key", 0).seed)));
  const forfeitGame = withGame((g) => (g.outcome = { kind: "forfeit", loser: "p2", cause: "invalid_reply", request_seq: 4, turns: 3 }));
  return {
    valid: {
      "progress-evaluation-record/baseline-match": evaluation,
      "progress-evaluation-record/previous-match": withPrevious,
      "progress-evaluation-record-2/baseline-match": v2,
      "progress-evaluation-record-2/evaluation-dataset": withV2((e) => (e.teams.evaluation_dataset_hash = hexOf("6"))),
      "progress-evaluation-record-2/plan-label": withV2((e) => (e.plan.evaluation_key = "paired-key")),
      "progress-evaluation-record-2/previous-match": v2Previous,
      "progress-evaluation-record-2/anchor-match": v2Anchor(),
      "progress-evaluation-record-2/run-match": v2Run(),
      "progress-evaluation-record-2/ablation-match": v2Ablation(),
      "progress-evaluation-record-2/damage-aware-match": v2Damage(),
      "progress-game-record/damage-aware": damageGame,
      "progress-game-record/played": game,
      "progress-game-record/played-pair": pairGame,
      // Game 1 of pair 0 on a declared evaluation key: the seed is progress-seed/1's for the key, not for the run id.
      "progress-game-record/plan-label": pairKeyGame,
      "progress-game-record/forfeit": forfeitGame,
    },
    invalid: {
      // The guard refuses a committed "private" label anywhere, so the sample says "public"; a test flips it to private.
      "progress-evaluation-record/public-class": withEval((e) => (e.data_class = "public")),
      "progress-evaluation-record/extra-field": withEval((e) => (e.notes = "x")),
      "progress-evaluation-record/fractional-wins": withEval((e) => (e.matches[0].wins = 0.5)),
      "progress-evaluation-record/next-seed-derivation": withEval((e) => (e.plan.seed_derivation = "progress-seed/2")),
      "progress-evaluation-record/test-split": withEval((e) => (e.teams.split = "dev")),
      "progress-evaluation-record/unknown-outcome": withEval((e) => (e.games[0].outcome = { kind: "abandoned", turns: 1 })),
      "progress-evaluation-record/no-manifest-hash": withEval((e) => delete e.generation_manifest_sha256),
      "progress-evaluation-record-2/public-class": withV2((e) => (e.data_class = "public")),
      "progress-evaluation-record-2/schema-version-1": withV2((e) => (e.schema_version = 1)),
      "progress-evaluation-record-2/evaluation-dataset-not-hash": withV2((e) => (e.teams.evaluation_dataset_hash = "not-a-hash")),
      "progress-evaluation-record-2/anchor-no-generation": v2Anchor((e) => delete e.matches[0].opponent.generation),
      "progress-evaluation-record-2/run-no-run-id": v2Run((e) => delete e.matches[0].opponent.run_id),
      "progress-evaluation-record-2/ablation-no-checkpoint": v2Ablation((e) => delete e.matches[0].candidate_preview.checkpoint),
      "progress-evaluation-record-2/plan-label-malformed": withV2((e) => (e.plan.evaluation_key = "not a token")),
      "progress-evaluation-record/plan-label": withEval((e) => (e.plan.evaluation_key = "paired-key")),
      "progress-game-record/plan-label-malformed": withShortGame((g) => ((g.game = 1), (g.pair = 0), (g.evaluation_key = ""))),
      "progress-evaluation-record-2/fractional-wins": withV2((e) => (e.matches[0].wins = 0.5)),
      "progress-evaluation-record-2/baseline-fractional-count": v2Damage((e) => (e.games[0].baseline_counts.queries = 0.5)),
      "progress-game-record/baseline-no-identity": shortDamage((g) => delete g.baseline.identity_hash),
      "progress-game-record/public-class": withShortGame((g) => (g.data_class = "public")),
      "progress-game-record/extra-field": withShortGame((g) => (g.notes = "x")),
      "progress-game-record/bad-seed-form": withShortGame((g) => (g.seed = "1,2,3,4")),
      "progress-game-record/unknown-outcome": withShortGame((g) => (g.outcome = { kind: "abandoned", turns: 1 })),
      "progress-game-record/no-log-hash": withShortGame((g) => delete g.omniscient_log_sha256),
    },
    readerInvalid: {
      "progress-evaluation-record/counts-disagree": withEval((e) => (e.matches[0].wins += 1)),
      "progress-evaluation-record/rows-incomplete": withEval((e) => e.games.pop()),
      "progress-evaluation-record/unsorted-matches": withEval((e) => (e.matches = [{ ...e.matches[0], match_id: "b-match" }, { ...e.matches[0], match_id: "a-match" }])),
      "progress-evaluation-record/turn-cap-ties-wrong": withEval((e) => (e.matches[0].turn_cap_ties = 1)),
      "progress-evaluation-record/forfeits-disagree": withEval((e) => (e.matches[0].forfeits = [{ side: "opponent", cause: "latency_overrun", count: 1 }])),
      "progress-evaluation-record/hashes-unsorted": withEval((e) => ((e.plan.recorded_games_per_match = 2), (e.matches[0].game_record_hashes = [hexOf("f"), hexOf("1")]))),
      "progress-evaluation-record/hash-count-wrong": withEval((e) => (e.matches[0].game_record_hashes = [])),
      "progress-evaluation-record/seat-wrong": withEval((e) => (e.games[1].candidate_seat = "p1")),
      "progress-evaluation-record/pairs-used-wrong": withEval((e) => (e.teams.pairs_used = 3)),
      "progress-evaluation-record-2/plan-label-is-run-id": withV2((e) => (e.plan.evaluation_key = e.run_id)),
      "progress-game-record/plan-label-is-run-id": withShortGame((g) => ((g.game = 1), (g.pair = 0), (g.evaluation_key = g.run_id))),
      "progress-game-record/plan-label-without-pair": withShortGame((g) => ((g.evaluation_key = "paired-key"), (g.seed = progressGameStreams("paired-key", 0).seed))),
      "progress-game-record/plan-label-seed-from-run-id": withShortGame((g) => ((g.game = 1), (g.pair = 0), (g.evaluation_key = "paired-key"), (g.seed = progressGameStreams(g.run_id, 0).seed))),
      "progress-evaluation-record-2/evaluation-dataset-is-own": withV2((e) => (e.teams.evaluation_dataset_hash = e.teams.dataset_hash)),
      "progress-evaluation-record-2/ablation-vs-anchor": v2Ablation((e) => ((e.plan.anchor_games_per_match = 4), (e.matches[0].opponent = { kind: "anchor", generation: 0, policy_id: `policy:${hexOf("b")}`, checkpoints: { preview: hexOf("6"), battle: hexOf("7") } }))),
      "progress-evaluation-record-2/ablation-misnamed": v2Ablation((e) => ((e.matches[0].match_id = "preview-g0-random-legal-argmax"), e.games.slice(0, 4).forEach((g: any) => (g.match_id = "preview-g0-random-legal-argmax")))),
      "progress-evaluation-record-2/ablation-not-earlier": v2Ablation((e) => (e.matches[0].candidate_preview.generation = 1)),
      "progress-evaluation-record-2/counts-disagree": withV2((e) => (e.matches[0].wins += 1)),
      "progress-evaluation-record-2/pair-is-game": withV2((e) => (e.games[2].pair = 2)),
      "progress-evaluation-record-2/seat-wrong": withV2((e) => (e.games[1].candidate_seat = "p1")),
      "progress-evaluation-record-2/odd-games": withV2((e) => ((e.plan.games_per_match = 3), (e.matches[0].games = 3), (e.teams.pairs_used = 2), e.games.pop())),
      "progress-evaluation-record-2/pairs-used-wrong": withV2((e) => (e.teams.pairs_used = 3)),
      "progress-evaluation-record-2/rows-incomplete": withV2((e) => e.games.pop()),
      "progress-evaluation-record-2/anchor-games-missing": v2Anchor((e) => delete e.plan.anchor_games_per_match),
      "progress-evaluation-record-2/anchor-games-odd": v2Anchor((e) => (e.plan.anchor_games_per_match = 3)),
      "progress-evaluation-record-2/anchor-not-earlier": v2Anchor((e) => ((e.matches[0].opponent.generation = 1), renameMatch(e, "anchor-g1-sample"))),
      "progress-evaluation-record-2/anchor-misnamed": v2Anchor((e) => renameMatch(e, "anchor-g5-sample")),
      "progress-evaluation-record-2/run-games-missing": v2Run((e) => delete e.plan.run_games_per_match),
      "progress-evaluation-record-2/run-games-odd": v2Run((e) => (e.plan.run_games_per_match = 3)),
      "progress-evaluation-record-2/run-misnamed": v2Run((e) => renameMatch(e, "run-argmax")),
      "progress-evaluation-record-2/run-same-run-id": v2Run((e) => (e.matches[0].opponent.run_id = e.run_id)),
      "progress-evaluation-record-2/run-plan-without-match": withV2((e) => (e.plan.run_games_per_match = 4)),
      "progress-evaluation-record-2/anchor-plan-without-match": withV2((e) => (e.plan.anchor_games_per_match = 4)),
      "progress-evaluation-record-2/baseline-counts-disagree": v2Damage((e) => (e.matches[0].baseline.counts.queries += 1)),
      "progress-evaluation-record-2/baseline-row-missing": v2Damage((e) => delete e.games[0].baseline_counts),
      "progress-evaluation-record-2/baseline-on-other-match": withV2((e) => (e.matches[0].baseline = { identity_hash: identity, counts: gameCounts })),
      "progress-evaluation-record-2/baseline-probe-runs-exceed-queries": v2Damage((e) => ((e.games[0].baseline_counts.probe_runs = 13), (e.matches[0].baseline.counts.probe_runs += 4))),
      "progress-evaluation-record-2/baseline-identities-differ": v2DamageBoth((e) => (e.matches[1].baseline.identity_hash = hexOf("8"))),
      "progress-game-record/baseline-seat-wrong": shortDamage((g) => (g.baseline.seat = "p1")),
      "progress-game-record/baseline-missing": shortDamage((g) => delete g.baseline),
      "progress-game-record/baseline-decisions-differ": shortDamage((g) => (g.baseline.counts.decisions += 5)),
      "progress-game-record/pair-wrong": withShortGame((g) => ((g.game = 3), (g.pair = 0))),
      "progress-game-record/pair-seed-wrong": withShortGame((g) => ((g.game = 1), (g.pair = 0), (g.seed = progressGameStreams(g.run_id, 1).seed))),
      "progress-game-record/chosen-not-candidate": withShortGame((g) => (g.decisions[0].chosen = "not a choice")),
      "progress-game-record/team-hash-wrong": withShortGame((g) => (g.teams.p1.team_hash = hexOf("0"))),
      "progress-game-record/log-hash-wrong": withShortGame((g) => (g.omniscient_log_sha256 = hexOf("0"))),
      "progress-game-record/seed-wrong": withShortGame((g) => (g.seed = `sodium,${"0".repeat(32)}`)),
    },
  };
}
