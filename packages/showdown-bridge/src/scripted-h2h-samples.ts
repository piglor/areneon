// The ScriptedHeadToHeadRecord's share of the sample corpus (ADR-008 section 2, ADR-025 section 13).
// Placeholders that pin a shape: counts of nothing real, hashes that are single repeated digits.

import { sumBaselineCounts } from "./progress-records.ts";
import { sumBeliefCounts, tallyScripted } from "./scripted-h2h-record.ts";

type Cases = Record<string, unknown>;

/* eslint-disable @typescript-eslint/no-explicit-any */
const hexOf = (c: string): string => c.repeat(64);
const mutated = (value: unknown, change: (copy: Record<string, any>) => void): Record<string, unknown> => {
  const copy = structuredClone(value) as Record<string, any>;
  change(copy);
  return copy;
};

const counts = (decisions: number, extra: Record<string, number> = {}): Record<string, number> => ({
  decisions,
  preview_decisions: 1,
  scored_decisions: decisions - 1,
  queries: decisions * 4,
  probe_runs: decisions * 3,
  unsupported_queries: 1,
  reader_fallbacks: 0,
  assumed_set_fallbacks: 0,
  epsilon_plays: 0,
  ...extra,
});
const belief = (decisions: number, refusals: { code: string; count: number }[] = []): Record<string, unknown> => ({
  decisions,
  worlds_sampled: decisions * 8,
  redraws: 2,
  contradicted_worlds: 1,
  no_consistent_world: 0,
  refusals,
});

const manifest = { showdown_commit: "a".repeat(40), format_id: "gen9championsvgc2026regmc", regulation: "M-C", ruleset_hash: hexOf("5"), format_rules_hash: hexOf("6") };

type Row = Record<string, any>;
/** Two pairs (four games) of damage-aware@v1 (a) against damage-aware@v2 (b); a forfeit and a tie in the mix. */
function damageRows(): Row[] {
  const row = (game: number, outcome: Record<string, unknown>, refusals: { code: string; count: number }[] = []): Row => ({
    game,
    pair: game >> 1,
    a_seat: game % 2 === 0 ? "p1" : "p2",
    outcome,
    a_counts: counts(10 + game),
    b_counts: counts(12 + game, game === 3 ? { assumed_set_fallbacks: 1 } : {}),
    b_belief: belief(10 + game, refusals),
  });
  return [
    row(0, { kind: "played", winner: "b", turns: 9 }),
    row(1, { kind: "played", winner: "b", turns: 11 }),
    row(2, { kind: "played", winner: "a", turns: 14 }, [{ code: "reader_flagged", count: 1 }]),
    row(3, { kind: "forfeit", loser: "a", cause: "latency_overrun", request_seq: 3, turns: 2 }, [{ code: "legality_redraws_exhausted", count: 1 }, { code: "reader_flagged", count: 2 }]),
  ];
}

function record(rows: Row[], arms: Record<string, unknown>, extraPlan: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    data_class: "synthetic",
    run_id: "tiny-run",
    version_manifest: manifest,
    teams: { dataset_hash: hexOf("9"), split: "validation", pairs_used: rows.length / 2 },
    plan: { seed_derivation: "progress-seed/1", pairing: "progress-pairing/1", games: rows.length, max_turns: 100, latency_budget_ms: { preview_ms: 1000, battle_ms: 1000 }, ...extraPlan },
    arms,
    result: tallyScripted(rows as never, 100),
    games: rows,
  };
}

export function scriptedH2HSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const rows = damageRows();
  const armA = { policy_id: "damage-aware@v1", identity_hash: hexOf("1"), counts: sumBaselineCounts(rows.map((r) => r.a_counts)) };
  const armB = {
    policy_id: "damage-aware@v2",
    identity_hash: hexOf("2"),
    counts: sumBaselineCounts(rows.map((r) => r.b_counts)),
    belief: { identity_hash: hexOf("3"), prior_hash: hexOf("4"), belief_config_hash: hexOf("7"), worlds: 8, totals: sumBeliefCounts(rows.map((r) => r.b_belief) as never) },
  };
  const damage = record(rows, { a: armA, b: armB }, { wall_time_cap_ms: 28_800_000 });
  const simpleRows: Row[] = [0, 1, 2, 3].map((game) => ({ game, pair: game >> 1, a_seat: game % 2 === 0 ? "p1" : "p2", outcome: { kind: "played", winner: game === 1 ? "tie" : "a", turns: game === 1 ? 101 : 20 + game } }));
  const simple = record(simpleRows, { a: { policy_id: "actor-rule@v1" }, b: { policy_id: "random-legal@v1" } });
  const of = (change: (m: Record<string, any>) => void): Record<string, unknown> => mutated(damage, change);

  return {
    valid: {
      "scripted-head-to-head-record/v1-against-v2": damage,
      "scripted-head-to-head-record/actor-rule-against-random-legal": simple,
    },
    invalid: {
      "scripted-head-to-head-record/public-class": of((m) => (m.data_class = "public")),
      "scripted-head-to-head-record/extra-field": of((m) => (m.notes = "x")),
      "scripted-head-to-head-record/next-version": of((m) => (m.schema_version = 2)),
      "scripted-head-to-head-record/test-split": of((m) => (m.teams.split = "dev")),
      "scripted-head-to-head-record/fractional-turns": of((m) => (m.games[0].outcome.turns = 1.5)),
      "scripted-head-to-head-record/free-text-cause": of((m) => (m.games[3].outcome.cause = "Worker said x")),
      "scripted-head-to-head-record/bare-policy-id": of((m) => (m.arms.a.policy_id = "damage-aware")),
      "scripted-head-to-head-record/unknown-seat": of((m) => (m.games[0].a_seat = "p3")),
      "scripted-head-to-head-record/unknown-winner": of((m) => (m.games[0].outcome.winner = "candidate")),
      "scripted-head-to-head-record/no-games": of((m) => delete m.games),
      "scripted-head-to-head-record/no-arms": of((m) => delete m.arms),
      "scripted-head-to-head-record/short-identity-hash": of((m) => (m.arms.a.identity_hash = "abc")),
    },
    readerInvalid: {
      "scripted-head-to-head-record/same-arm-twice": of((m) => (m.arms.b = { ...m.arms.a })),
      "scripted-head-to-head-record/games-odd": of((m) => {
        m.games.pop();
        m.plan.games = 3;
        m.teams.pairs_used = 2;
        m.result = tallyScripted(m.games, 100);
      }),
      "scripted-head-to-head-record/pairs-not-half": of((m) => (m.teams.pairs_used = 3)),
      "scripted-head-to-head-record/game-rows-short": of((m) => {
        m.plan.games = 6;
        m.teams.pairs_used = 3;
      }),
      "scripted-head-to-head-record/pair-not-half-game": of((m) => (m.games[2].pair = 0)),
      "scripted-head-to-head-record/a-seat-not-parity": of((m) => (m.games[0].a_seat = "p2")),
      "scripted-head-to-head-record/result-not-rows": of((m) => (m.result.turns_total += 1)),
      "scripted-head-to-head-record/wins-not-games": of((m) => (m.result.ties += 1)),
      "scripted-head-to-head-record/forfeits-unsorted": of((m) => {
        m.games[1].outcome = { kind: "forfeit", loser: "b", cause: "crash", request_seq: 1, turns: 1 };
        m.result = tallyScripted(m.games, 100);
        m.result.forfeits.reverse();
      }),
      "scripted-head-to-head-record/counts-on-rule-arm": of((m) => (m.arms.a = { policy_id: "actor-rule@v1", identity_hash: hexOf("1"), counts: m.arms.a.counts })),
      "scripted-head-to-head-record/counts-missing-on-damage-arm": of((m) => delete m.arms.a.counts),
      "scripted-head-to-head-record/identity-missing-on-damage-arm": of((m) => delete m.arms.a.identity_hash),
      "scripted-head-to-head-record/arm-counts-not-sum": of((m) => (m.arms.a.counts.decisions += 1)),
      "scripted-head-to-head-record/row-counts-missing": of((m) => delete m.games[0].a_counts),
      "scripted-head-to-head-record/probe-runs-over-queries": of((m) => {
        m.games[0].a_counts.probe_runs = m.games[0].a_counts.queries + 1;
        m.arms.a.counts = sumBaselineCounts(m.games.map((g: Row) => g.a_counts));
      }),
      "scripted-head-to-head-record/previews-over-decisions": of((m) => {
        m.games[0].a_counts.preview_decisions = m.games[0].a_counts.decisions;
        m.arms.a.counts = sumBaselineCounts(m.games.map((g: Row) => g.a_counts));
      }),
      "scripted-head-to-head-record/belief-missing-on-v2": of((m) => delete m.arms.b.belief),
      "scripted-head-to-head-record/belief-on-v1": of((m) => (m.arms.a.belief = m.arms.b.belief)),
      "scripted-head-to-head-record/belief-row-missing": of((m) => delete m.games[0].b_belief),
      "scripted-head-to-head-record/belief-totals-not-sum": of((m) => (m.arms.b.belief.totals.redraws += 1)),
      "scripted-head-to-head-record/refusals-unsorted": of((m) => {
        m.games[3].b_belief.refusals.reverse();
        m.arms.b.belief.totals = sumBeliefCounts(m.games.map((g: Row) => g.b_belief));
      }),
      "scripted-head-to-head-record/refusals-over-decisions": of((m) => {
        m.games[0].b_belief.refusals = [{ code: "reader_flagged", count: 99 }];
        m.arms.b.belief.totals = sumBeliefCounts(m.games.map((g: Row) => g.b_belief));
      }),
    },
  };
}
