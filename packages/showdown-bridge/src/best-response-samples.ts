// The BestResponseRecord's share of the sample corpus (ADR-008 section 2, ADR-029 section 2).
// Placeholders that pin a shape: counts of nothing real, hashes that are single repeated digits.

import { tallyArm } from "./best-response-record.ts";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Cases = Record<string, unknown>;
type Row = Record<string, any>;

const hexOf = (c: string): string => c.repeat(64);
const mutated = (value: unknown, change: (copy: Record<string, any>) => void): Record<string, unknown> => {
  const copy = structuredClone(value) as Record<string, any>;
  change(copy);
  return copy;
};

const manifest = { showdown_commit: "a".repeat(40), format_id: "gen9championsvgc2026regmc", regulation: "M-C", ruleset_hash: hexOf("5"), format_rules_hash: hexOf("6") };

const counts = (worldsAsked: number, extra: Record<string, number> = {}): Record<string, number> => ({
  worlds_asked: worldsAsked,
  worlds_used: worldsAsked,
  preview_queries: worldsAsked + 1,
  entry_attempts: worldsAsked * 12,
  entries: worldsAsked * 12,
  value_queries: worldsAsked * 12,
  rollout_choices: 0,
  back_draws: 0,
  clamped: 1,
  clamped_low: 1,
  clamped_high: 0,
  game_end_entries: 0,
  engine_refusals: 0,
  not_move_request: 0,
  worlds_dropped: 0,
  rollout_capped: 0,
  missing_entries: 0,
  ...extra,
});

const table = (choices: Row[], extra: Row = {}, worlds = 4): Row => ({ kind: "table", counts: counts(worlds), worlds_unstarted: 0, worlds_refused: 0, own: 4, columns: 3, complete: true, choices, ...extra });
const choice = (arm: string, rank: number, key: string, expected: number): Row => ({ arm, rank, key, expected_ppm: expected, columns_used: 3 });
const bTable = (): Row => table([choice("responder", 1, "13|24", 640000), choice("blind", 0, "12|34", 510000)]);
const aTable = (): Row => table([choice("room", 2, "21|34", 530000)]);

const played = (winner: "focal" | "other" | "tie", turns: number): Row => ({ kind: "played", winner, turns });
const forfeit = (loser: "focal" | "other", cause: string, turns: number): Row => ({ kind: "forfeit", loser, cause, request_seq: 3, turns });

/** Two pairs (four games): the responder gains on both, a forfeit and a tie in the mix. */
function rows(): Row[] {
  const row = (game: number, mirror: Row, responder: Row, blind: Row, room: Row, b: Row = bTable(), a: Row = aTable()): Row => ({
    game,
    pair: game >> 1,
    a_seat: game % 2 === 0 ? "p1" : "p2",
    mirror,
    responder,
    blind,
    room,
    b_decision: b,
    a_decision: a,
  });
  return [
    row(0, played("other", 9), played("focal", 11), played("other", 9), played("focal", 12)),
    row(1, played("other", 10), played("focal", 14), played("tie", 100), played("tie", 100)),
    row(2, played("focal", 8), played("focal", 8), played("focal", 8), played("other", 13), { kind: "fallback", code: "no_world" }, aTable()),
    row(3, played("other", 12), forfeit("other", "latency_overrun", 2), played("other", 12), played("focal", 9), bTable(), { kind: "fallback", code: "species_not_in_prior" }),
  ];
}

function record(games: Row[], extraPlan: Record<string, unknown> = {}, extraTeams: Record<string, unknown> = {}, leaf: Record<string, unknown> = { kind: "first-request" }): Record<string, unknown> {
  const maxTurns = 100;
  return {
    schema_version: 1,
    data_class: "synthetic",
    run_id: "tiny-run",
    version_manifest: manifest,
    evaluated: { generation: 5, generation_manifest_sha256: hexOf("7"), checkpoints: { preview: hexOf("1"), battle: hexOf("2") }, mode: "argmax", policy_id: "policy:argmax-example" },
    teams: { dataset_hash: hexOf("9"), split: "validation", pairs_used: games.length / 2, ...extraTeams },
    plan: {
      seed_derivation: "progress-seed/1",
      pairing: "progress-pairing/1",
      games: games.length,
      max_turns: maxTurns,
      latency_budget_ms: { preview_ms: 1000, battle_ms: 1000 },
      responder_preview_budget_ms: 3600000,
      responder: {
        format: 1,
        payoff_leaf_format: 1,
        hypothetical_battle_format: 1,
        worlds: 4,
        k_own: 4,
        own_rule: "prior-top-k",
        leaf,
        columns: "all-lead-pairs",
        belief: { identity_hash: hexOf("3"), prior_hash: hexOf("4"), belief_config_hash: hexOf("8") },
      },
      ...extraPlan,
    },
    result: Object.fromEntries((["mirror", "responder", "blind", "room"] as const).map((arm) => [arm, tallyArm(games.map((g) => g[arm]), maxTurns)])),
    games,
  };
}

export function bestResponseSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base = record(rows());
  const second = record(rows().slice(0, 2), { evaluation_key: "e0p-key" }, { evaluation_dataset_hash: hexOf("c") }, { kind: "rollout", turns: 2 });
  const valid: Cases = {
    "best-response-record/two-pairs": base,
    "best-response-record/evaluation-dataset-rollout": mutated(second, (c) => {
      c.evaluated.mode = "sample";
      c.plan.responder.own_rule = "distinct-leads";
    }),
  };
  const bad = (change: (copy: Record<string, any>) => void): Record<string, unknown> => mutated(base, change);
  const invalid: Cases = {
    "best-response-record/extra-field": bad((c) => {
      c.extra = 1;
    }),
    "best-response-record/public-class": bad((c) => {
      c.data_class = "public";
    }),
    "best-response-record/next-version": bad((c) => {
      c.schema_version = 2;
    }),
    "best-response-record/unknown-mode": bad((c) => {
      c.evaluated.mode = "greedy";
    }),
    "best-response-record/short-identity-hash": bad((c) => {
      c.plan.responder.belief.identity_hash = "abc";
    }),
    "best-response-record/fractional-turns": bad((c) => {
      c.games[0].mirror.turns = 1.5;
    }),
    "best-response-record/free-text-cause": bad((c) => {
      c.games[3].responder.cause = "the worker said no";
    }),
    "best-response-record/no-room-arm": bad((c) => {
      delete c.games[0].room;
    }),
    "best-response-record/unknown-decision-kind": bad((c) => {
      c.games[0].b_decision.kind = "guess";
    }),
    "best-response-record/strategy-label-malformed": bad((c) => {
      c.games[0].b_decision.choices[0].key = "1234";
    }),
    "best-response-record/expected-over-one": bad((c) => {
      c.games[0].b_decision.choices[0].expected_ppm = 1000001;
    }),
    "best-response-record/columns-not-all": bad((c) => {
      c.plan.responder.columns = "top-8";
    }),
    "best-response-record/test-split": bad((c) => {
      c.teams.split = "test";
    }),
  };
  const readerInvalid: Cases = {
    "best-response-record/games-odd": record(rows().slice(0, 3), {}, { pairs_used: 1 }),
    "best-response-record/pairs-not-half": bad((c) => {
      c.teams.pairs_used = 1;
    }),
    "best-response-record/game-rows-short": bad((c) => {
      c.games.pop();
    }),
    "best-response-record/a-seat-not-parity": bad((c) => {
      c.games[1].a_seat = "p1";
    }),
    "best-response-record/tally-not-rows": bad((c) => {
      c.result.blind.wins += 1;
      c.result.blind.losses -= 1;
    }),
    "best-response-record/forfeits-unsorted": bad((c) => {
      c.result.mirror.forfeits = [
        { side: "other", cause: "crash", count: 1 },
        { side: "focal", cause: "crash", count: 1 },
      ];
    }),
    "best-response-record/evaluation-dataset-is-own": bad((c) => {
      c.teams.evaluation_dataset_hash = c.teams.dataset_hash;
    }),
    "best-response-record/worlds-not-plan": bad((c) => {
      c.games[0].b_decision.worlds_refused = 1;
    }),
    "best-response-record/choices-wrong-arms": bad((c) => {
      c.games[0].b_decision.choices.reverse();
    }),
    "best-response-record/rank-over-own": bad((c) => {
      c.games[0].a_decision.choices[0].rank = 4;
    }),
    "best-response-record/complete-with-missing": bad((c) => {
      c.games[0].b_decision.counts.missing_entries = 1;
    }),
    "best-response-record/clamped-not-sum": bad((c) => {
      c.games[0].b_decision.counts.clamped = 3;
    }),
    "best-response-record/columns-used-over-columns": bad((c) => {
      c.games[0].b_decision.choices[0].columns_used = 4;
    }),
  };
  return { valid, invalid, readerInvalid };
}
