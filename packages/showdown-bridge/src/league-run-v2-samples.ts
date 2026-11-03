// The LeagueRunRecord version 2 share of the sample corpus (ADR-021 section 8, ADR-008 section 2):
// a population generation with two learning members and an exploiter seat, and one fault per sample.
// Counting rule: a member counts every game in which it holds a seat; a game between two members is in
// both entries; episodes are the member's own seats; the root counts distinct games.

/* eslint-disable @typescript-eslint/no-explicit-any */

const hexOf = (c: string): string => c.repeat(64);

function mutated<T>(value: T, change: (copy: Record<string, any>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

export interface LeagueRunV2Samples {
  valid: Record<string, unknown>;
  invalid: Record<string, unknown>;
  readerInvalid: Record<string, unknown>;
}

/** The hashes are those of the population sample configs (2 members and 1 exploiter; 1 member and none). */
export function leagueRunV2Samples(configHashes: { population: string; single: string }, baselineCounts: Record<string, number>): LeagueRunV2Samples {
  const pair = (c: string) => `policy:${hexOf(c)}`;
  const damageId = "damage-aware@v1";
  const checkpoints = (a: string, b: string) => ({ preview: hexOf(a), battle: hexOf(b) });
  const op = (policy: string, games: number, played: number, member?: string) => ({ opponent_policy_id: policy, ...(member === undefined ? {} : { member }), games, played });
  // m0: self-play 2 (both played), m1 3 (all played), one forfeited game against the baseline.
  const m0 = {
    member: "m0",
    pair_policy_id: pair("a"),
    checkpoints: checkpoints("1", "2"),
    games: 6,
    played: 5,
    forfeits: 1,
    forfeit_causes: [{ cause: "crash", count: 1 }],
    episodes: 7,
    opponents: [op(damageId, 1, 0), op(pair("a"), 2, 2, "m0"), op(pair("b"), 3, 3, "m1")],
    baselines: [{ policy_id: damageId, identity_hash: hexOf("c"), games: 1, counts: baselineCounts }],
  };
  // m1: 3 games with m0, 1 self-play, 2 with the exploiter (one forfeited), 1 against a past snapshot.
  const m1 = {
    member: "m1",
    pair_policy_id: pair("b"),
    checkpoints: checkpoints("3", "4"),
    games: 7,
    played: 6,
    forfeits: 1,
    forfeit_causes: [{ cause: "latency_overrun", count: 1 }],
    episodes: 7,
    opponents: [op(pair("a"), 3, 3, "m0"), op(pair("b"), 1, 1, "m1"), op(pair("c"), 2, 1, "x0"), op(pair("d"), 1, 1)],
  };
  const x0 = { member: "x0", pair_policy_id: pair("c"), checkpoints: checkpoints("5", "6"), games: 2, played: 1, forfeits: 1, forfeit_causes: [{ cause: "latency_overrun", count: 1 }], episodes: 1, opponents: [op(pair("b"), 2, 1, "m1")] };
  // Distinct games: (6 + 7 + 2) - 3 (m0 vs m1) - 2 (m1 vs x0) = 10; played (5 + 6 + 1) - 3 - 1 = 8.
  const population = {
    schema_version: 2,
    run_id: "tiny-run",
    run_config_hash: configHashes.population,
    generation: 3,
    seed_derivation: "league-seed/1",
    plan_hash: hexOf("b"),
    games: 10,
    played: 8,
    forfeits: 2,
    forfeit_causes: [{ cause: "crash", count: 1 }, { cause: "latency_overrun", count: 1 }],
    episodes: 16,
    copy_decision_hash: hexOf("e"),
    members: [m0, m1, x0],
  };
  const withPop = (change: (r: Record<string, any>) => void) => mutated(population, change);
  const withMember = (index: number, change: (m: Record<string, any>) => void) => withPop((r) => change(r.members[index]));
  // One member: two self-play games and one against a past snapshot.
  const single = {
    schema_version: 2,
    run_id: "tiny-run",
    run_config_hash: configHashes.single,
    generation: 1,
    seed_derivation: "league-seed/1",
    plan_hash: hexOf("b"),
    games: 3,
    played: 3,
    forfeits: 0,
    episodes: 6,
    copy_decision_hash: null,
    members: [{ member: "m0", pair_policy_id: pair("a"), checkpoints: checkpoints("1", "2"), games: 3, played: 3, forfeits: 0, episodes: 5, opponents: [op(pair("a"), 2, 2, "m0"), op(pair("d"), 1, 1)] }],
  };
  return {
    valid: { "league-run-record-2/population": population, "league-run-record-2/single-member": single },
    invalid: {
      "league-run-record-2/extra-field": withPop((r) => (r.notes = "x")),
      "league-run-record-2/version-1": withPop((r) => (r.schema_version = 1)),
      "league-run-record-2/pair-policy-id-at-root": withPop((r) => (r.pair_policy_id = pair("a"))),
      "league-run-record-2/no-copy-decision-hash": withPop((r) => delete r.copy_decision_hash),
      "league-run-record-2/short-copy-decision-hash": withPop((r) => (r.copy_decision_hash = "abc")),
      "league-run-record-2/no-members": withPop((r) => (r.members = [])),
      "league-run-record-2/member-leading-zero": withMember(0, (m) => (m.member = "m01")),
      "league-run-record-2/member-unknown-kind": withMember(0, (m) => (m.member = "y0")),
      "league-run-record-2/member-bare-pair-id": withMember(0, (m) => (m.pair_policy_id = hexOf("a"))),
      "league-run-record-2/member-no-checkpoints": withMember(0, (m) => delete m.checkpoints),
      "league-run-record-2/member-short-checkpoint": withMember(0, (m) => (m.checkpoints.battle = "abc")),
      "league-run-record-2/member-extra-field": withMember(1, (m) => (m.notes = "x")),
      "league-run-record-2/member-fractional-games": withMember(1, (m) => (m.games = 4.5)),
      "league-run-record-2/member-empty-forfeit-causes": withMember(1, (m) => (m.forfeit_causes = [])),
      "league-run-record-2/member-unknown-forfeit-cause": withMember(0, (m) => (m.forfeit_causes = [{ cause: "timeout", count: 1 }])),
      "league-run-record-2/opponent-no-played": withMember(1, (m) => delete m.opponents[0].played),
      "league-run-record-2/opponent-bad-member": withMember(1, (m) => (m.opponents[0].member = "z1")),
      "league-run-record-2/opponent-zero-games": withMember(1, (m) => (m.opponents[3].games = 0)),
    },
    readerInvalid: {
      "league-run-record-2/members-unsorted": withPop((r) => (r.members = [m0, x0, m1])),
      "league-run-record-2/members-unsorted-numeric": withPop((r) => {
        r.members = [{ ...m0, member: "m10" }, { ...m1, member: "m2" }, x0];
      }),
      "league-run-record-2/duplicate-member": withPop((r) => (r.members = [m0, m0, x0])),
      "league-run-record-2/member-counts-disagree": withMember(0, (m) => (m.forfeits = 2)),
      "league-run-record-2/member-episodes-disagree": withMember(1, (m) => (m.episodes = 6)),
      "league-run-record-2/member-episodes-ignore-self-play": withMember(0, (m) => (m.episodes = 5)),
      "league-run-record-2/member-opponents-unsorted": withMember(0, (m) => (m.opponents = [...m.opponents].reverse())),
      "league-run-record-2/member-duplicate-opponent": withMember(1, (m) => (m.opponents = [op(pair("a"), 3, 3, "m0"), op(pair("a"), 3, 3, "m0"), op(pair("c"), 2, 1, "x0"), op(pair("d"), 1, 1)])),
      "league-run-record-2/member-opponents-games-do-not-sum": withMember(1, (m) => (m.opponents[3].games = 2)),
      "league-run-record-2/member-opponents-played-do-not-sum": withMember(1, (m) => ((m.opponents[3].played = 0), (m.opponents[3].games = 1))),
      "league-run-record-2/opponent-played-exceeds-games": withMember(0, (m) => ((m.opponents[0].played = 2), (m.opponents[0].games = 1))),
      "league-run-record-2/opponent-member-unknown": withMember(1, (m) => (m.opponents[2].member = "x1")),
      "league-run-record-2/opponent-policy-is-not-the-members": withMember(1, (m) => (m.opponents[0].opponent_policy_id = pair("9"))),
      "league-run-record-2/self-entry-wrong-policy": withMember(0, (m) => (m.opponents[1].opponent_policy_id = pair("b"))),
      "league-run-record-2/symmetry-games-differ": withMember(0, (m) => ((m.opponents[2].games = 4), (m.games = 7), (m.forfeits = 2), (m.forfeit_causes = [{ cause: "crash", count: 2 }]))),
      "league-run-record-2/symmetry-played-differ": withMember(2, (m) => ((m.opponents[0].played = 2), (m.played = 2), (m.forfeits = 0), delete m.forfeit_causes, (m.episodes = 2))),
      "league-run-record-2/symmetry-missing": withMember(2, (m) => ((m.opponents = [op(pair("d"), 2, 1)]), (m.episodes = 1))),
      "league-run-record-2/member-causes-missing": withMember(0, (m) => delete m.forfeit_causes),
      "league-run-record-2/member-causes-without-forfeits": withMember(1, (m) => ((m.forfeits = 0), (m.played = 7), (m.opponents[3].played = 1), (m.opponents[2].played = 2), (m.episodes = 8))),
      "league-run-record-2/member-causes-do-not-sum": withMember(0, (m) => (m.forfeit_causes = [{ cause: "crash", count: 2 }])),
      "league-run-record-2/member-causes-unsorted": withMember(1, (m) => ((m.forfeits = 2), (m.played = 5), (m.opponents[3].played = 0), (m.forfeit_causes = [{ cause: "latency_overrun", count: 1 }, { cause: "crash", count: 1 }]), (m.episodes = 6))),
      "league-run-record-2/member-baselines-unsorted": withMember(0, (m) => {
        m.baselines = [{ ...m.baselines[0], policy_id: "random-legal@tiny", games: 1 }, m.baselines[0]];
        m.opponents = [...m.opponents, op("random-legal@tiny", 1, 1)];
        m.games += 1;
        m.played += 1;
        m.episodes += 1;
      }),
      "league-run-record-2/member-baseline-games-disagree": withMember(0, (m) => (m.baselines[0].games = 2)),
      "league-run-record-2/member-baseline-not-an-opponent": withMember(0, (m) => (m.baselines[0].policy_id = "damage-aware@v2")),
      "league-run-record-2/member-baseline-probe-runs-exceed-queries": withMember(0, (m) => (m.baselines[0].counts = { ...baselineCounts, probe_runs: 411 })),
      "league-run-record-2/member-baseline-decisions-exceed": withMember(0, (m) => (m.baselines[0].counts = { ...baselineCounts, preview_decisions: 10 })),
      "league-run-record-2/total-games-is-the-members-sum": withPop((r) => ((r.games = 15), (r.played = 12), (r.forfeits = 3), (r.episodes = 24))),
      "league-run-record-2/total-games-differs": withPop((r) => ((r.games = 11), (r.forfeits = 3))),
      "league-run-record-2/total-played-differs": withPop((r) => ((r.played = 7), (r.forfeits = 3), (r.episodes = 14))),
      "league-run-record-2/total-counts-disagree": withPop((r) => (r.forfeits = 3)),
      "league-run-record-2/total-episodes-differs": withPop((r) => (r.episodes = 15)),
      "league-run-record-2/total-causes-missing": withPop((r) => delete r.forfeit_causes),
      "league-run-record-2/total-causes-without-forfeits": withPop((r) => {
        r.members = [{ ...single.members[0] }];
        r.games = 3;
        r.played = 3;
        r.forfeits = 0;
        r.episodes = 6;
      }),
      "league-run-record-2/total-causes-do-not-sum": withPop((r) => (r.forfeit_causes = [{ cause: "crash", count: 1 }])),
      "league-run-record-2/total-cause-not-a-members": withPop((r) => (r.forfeit_causes = [{ cause: "crash", count: 1 }, { cause: "out_of_memory", count: 1 }])),
      "league-run-record-2/total-cause-above-members-sum": withPop((r) => ((r.forfeit_causes = [{ cause: "crash", count: 2 }]), (r.forfeits = 2))),
    },
  };
}
