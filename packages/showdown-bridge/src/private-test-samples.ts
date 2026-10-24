// The PrivateTestRecord's share of the sample corpus (ADR-008 section 2,
// ADR-015 erratum E1). Placeholders that pin a shape: counts of nothing real,
// hashes that are single repeated digits. A `valid` and a `reader-invalid`
// sample carry the private label, so each file is listed in
// `PRIVATE_TEST_RECORD_MARKERS` (scripts/check_data_class.py), which accepts it
// only in the strict record shape (hashes, id tokens, fixed words, bounded
// integers; no paste, no free text). An `invalid` sample says
// `data_class: synthetic` as well as its own fault (the guard holds no private
// file with a broken field); a test flips the label back and requires it to
// stay invalid.

type Cases = Record<string, unknown>;

/* eslint-disable @typescript-eslint/no-explicit-any */
const hexOf = (c: string): string => c.repeat(64);
const mutated = (value: unknown, change: (copy: Record<string, any>) => void): Record<string, unknown> => {
  const copy = structuredClone(value) as Record<string, any>;
  change(copy);
  return copy;
};

const pairPolicy = (mode: string): string => `policy:${hexOf(mode === "sample" ? "a" : "b")}`;
const otherPolicy = (mode: string): string => `policy:${hexOf(mode === "sample" ? "c" : "d")}`;

/** One match around three pairs (six games): wins, ties, losses and the pair-score histogram agree. */
function match(opponent: string, mode: "sample" | "argmax", extra: Record<string, unknown> = {}): Record<string, unknown> {
  const pair = opponent === "pair";
  return {
    match_id: `${opponent}-${mode}`,
    policy_id: pairPolicy(mode),
    mode,
    opponent_policy_id: pair ? otherPolicy(mode) : `${opponent}@v1`,
    ...(pair ? { opponent_checkpoints: { preview: hexOf("e"), battle: hexOf("f") } } : {}),
    games: 6,
    wins: 4,
    ties: 0,
    losses: 2,
    turn_cap_ties: 0,
    forfeits: [],
    pairs: 3,
    pair_score_counts: [0, 0, 2, 0, 1],
    ...extra,
  };
}

const base = (matches: Record<string, unknown>[], testSet: Record<string, unknown>, candidate: "synthetic" | "private"): Record<string, unknown> => ({
  schema_version: 1,
  data_class: "private",
  run_id: "tiny-run",
  run_config_hash: hexOf("1"),
  generation: 2,
  generation_manifest_sha256: hexOf("2"),
  checkpoints: { preview: hexOf("3"), battle: hexOf("4") },
  candidate_data_class: candidate,
  version_manifest: { showdown_commit: "a".repeat(40), format_id: "gen9championsvgc2026regmc", regulation: "M-C", ruleset_hash: hexOf("5"), format_rules_hash: hexOf("6") },
  test_set: { commitment: hexOf("7"), team_count: 7, pairs: 3, ...testSet },
  plan: { kind: "disjoint-pairs", seed_derivation: "private-test-seed/1", pairing: "private-test-pairing/1", games_per_match: 6, max_turns: 100, latency_budget_ms: { preview_ms: 1000, battle_ms: 1000 } },
  matches,
});

const baselines = (): Record<string, unknown>[] => [match("actor-rule", "argmax"), match("actor-rule", "sample"), match("random-legal", "argmax"), match("random-legal", "sample")];

export function privateTestSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const staging = base(baselines(), { source_kind: "staging", subset: "all" }, "synthetic");
  const snapshot = base([...baselines(), match("pair", "argmax"), match("pair", "sample")].sort((a, b) => ((a.match_id as string) < (b.match_id as string) ? -1 : 1)), { source_kind: "snapshot", snapshot_hash: hexOf("8"), subset: "holdout" }, "private");
  const dataset = base(
    [
      match("damage-aware", "argmax", { forfeits: [{ side: "candidate", cause: "latency_overrun", count: 1 }, { side: "opponent", cause: "crash", count: 2 }], ties: 0 }),
      match("damage-aware", "sample", { wins: 3, ties: 2, losses: 1, turn_cap_ties: 1 }),
    ],
    { source_kind: "dataset", dataset_hash: hexOf("9"), subset: "all" },
    "private",
  );
  const synthetic = (change: (m: Record<string, any>) => void): Record<string, unknown> =>
    mutated(staging, (m) => {
      m.data_class = "synthetic";
      change(m);
    });
  const first = (m: Record<string, any>): Record<string, any> => m.matches[0];
  const of = (from: Record<string, unknown>, change: (m: Record<string, any>) => void): Record<string, unknown> => mutated(from, change);

  return {
    valid: {
      "private-test-record/staging": staging,
      "private-test-record/snapshot-holdout": snapshot,
      "private-test-record/dataset": dataset,
    },
    invalid: {
      "private-test-record/synthetic-class": synthetic(() => undefined),
      "private-test-record/extra-field": synthetic((m) => (m.notes = "x")),
      "private-test-record/team-list": synthetic((m) => (m.test_set.teams = [hexOf("1")])),
      "private-test-record/next-version": synthetic((m) => (m.schema_version = 2)),
      "private-test-record/short-commitment": synthetic((m) => (m.test_set.commitment = "abc")),
      "private-test-record/unknown-source-kind": synthetic((m) => (m.test_set.source_kind = "bucket")),
      "private-test-record/unknown-subset": synthetic((m) => (m.test_set.subset = "some")),
      "private-test-record/all-pairs-plan": synthetic((m) => (m.plan.kind = "all-pairs")),
      "private-test-record/unknown-candidate-class": synthetic((m) => (m.candidate_data_class = "public")),
      "private-test-record/fractional-wins": synthetic((m) => (first(m).wins = 1.5)),
      "private-test-record/negative-ties": synthetic((m) => (first(m).ties = -1)),
      "private-test-record/four-score-counts": synthetic((m) => (first(m).pair_score_counts = [0, 0, 2, 1])),
      "private-test-record/free-text-cause": synthetic((m) => (first(m).forfeits = [{ side: "candidate", cause: "Worker said x", count: 1 }])),
      "private-test-record/bare-pair-id": synthetic((m) => (first(m).policy_id = hexOf("a"))),
      "private-test-record/no-matches": synthetic((m) => delete m.matches),
      "private-test-record/empty-matches": synthetic((m) => (m.matches = [])),
      "private-test-record/no-plan": synthetic((m) => delete m.plan),
    },
    readerInvalid: {
      "private-test-record/snapshot-hash-on-staging": of(staging, (m) => (m.test_set.snapshot_hash = hexOf("8"))),
      "private-test-record/snapshot-without-hash": of(snapshot, (m) => delete m.test_set.snapshot_hash),
      "private-test-record/dataset-hash-on-staging": of(staging, (m) => (m.test_set.dataset_hash = hexOf("9"))),
      "private-test-record/holdout-on-staging": of(staging, (m) => (m.test_set.subset = "holdout")),
      "private-test-record/private-pair-on-staging": of(staging, (m) => (m.candidate_data_class = "private")),
      "private-test-record/private-pair-on-whole-snapshot": of(snapshot, (m) => (m.test_set.subset = "all")),
      "private-test-record/pairs-over-half": of(staging, (m) => (m.test_set.pairs = 4)),
      "private-test-record/games-not-twice-pairs": of(staging, (m) => (m.plan.games_per_match = 8)),
      "private-test-record/match-games-not-plan": of(staging, (m) => (first(m).games = 4)),
      "private-test-record/counts-not-games": of(staging, (m) => (first(m).wins = 5)),
      "private-test-record/histogram-not-pairs": of(staging, (m) => (first(m).pair_score_counts = [0, 0, 2, 0, 2])),
      "private-test-record/histogram-not-score": of(staging, (m) => (first(m).pair_score_counts = [0, 1, 1, 1, 0])),
      "private-test-record/matches-unsorted": of(staging, (m) => (m.matches = [...m.matches].reverse())),
      "private-test-record/match-twice": of(staging, (m) => (m.matches = [m.matches[0], m.matches[0], ...m.matches.slice(1)])),
      "private-test-record/match-id-not-opponent-mode": of(staging, (m) => (first(m).match_id = "actor-rule-sample")),
      "private-test-record/unknown-baseline": of(staging, (m) => Object.assign(first(m), { match_id: "aardvark-argmax", opponent_policy_id: "aardvark@v1" })),
      "private-test-record/pair-without-checkpoints": of(snapshot, (m) => delete m.matches.find((x: Record<string, any>) => x.match_id === "pair-argmax").opponent_checkpoints),
      "private-test-record/baseline-with-checkpoints": of(staging, (m) => (first(m).opponent_checkpoints = { preview: hexOf("e"), battle: hexOf("f") })),
      "private-test-record/forfeits-unsorted": of(dataset, (m) => (first(m).forfeits = [...first(m).forfeits].reverse())),
      "private-test-record/forfeits-over-losses": of(staging, (m) => (first(m).forfeits = [{ side: "candidate", cause: "crash", count: 3 }])),
      "private-test-record/turn-cap-over-ties": of(staging, (m) => (first(m).turn_cap_ties = 1)),
    },
  };
}
