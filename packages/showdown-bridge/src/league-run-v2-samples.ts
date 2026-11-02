// The LeagueRunRecord version 2 share of the sample corpus (ADR-021 section 8, ADR-008 section 2):
// a population generation with two learning members and an exploiter seat, and one fault per sample.

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

/** `run_config_hash` is the tiny config's; `baselineCounts` the damage-aware counts of the version 1 samples. */
export function leagueRunV2Samples(run_config_hash: string, baselineCounts: Record<string, number>): LeagueRunV2Samples {
  const pair = (c: string) => `policy:${hexOf(c)}`;
  const damageId = "damage-aware@v1";
  const member = (id: string, c: string, rest: Record<string, unknown>) => ({ member: id, pair_policy_id: pair(c), checkpoints: { preview: hexOf(c === "a" ? "1" : c === "b" ? "3" : "5"), battle: hexOf(c === "a" ? "2" : c === "b" ? "4" : "6") }, ...rest });
  const m0 = member("m0", "a", {
    games: 5,
    played: 4,
    forfeits: 1,
    forfeit_causes: [{ cause: "crash", count: 1 }],
    episodes: 8,
    opponents: [{ opponent_policy_id: damageId, games: 1 }, { opponent_policy_id: pair("a"), games: 2 }, { opponent_policy_id: pair("b"), games: 2 }],
    baselines: [{ policy_id: damageId, identity_hash: hexOf("c"), games: 1, counts: baselineCounts }],
  });
  const m1 = member("m1", "b", { games: 4, played: 4, forfeits: 0, episodes: 8, opponents: [{ opponent_policy_id: pair("b"), games: 3 }, { opponent_policy_id: pair("d"), games: 1 }] });
  const x0 = member("x0", "c", { games: 3, played: 3, forfeits: 0, episodes: 6, opponents: [{ opponent_policy_id: pair("b"), games: 3 }] });
  const population = {
    schema_version: 2,
    run_id: "tiny-run",
    run_config_hash,
    generation: 3,
    seed_derivation: "league-seed/1",
    plan_hash: hexOf("b"),
    games: 12,
    played: 11,
    forfeits: 1,
    forfeit_causes: [{ cause: "crash", count: 1 }],
    episodes: 22,
    copy_decision_hash: hexOf("e"),
    members: [m0, m1, x0],
  };
  const withPop = (change: (r: Record<string, any>) => void) => mutated(population, change);
  const withMember = (index: number, change: (m: Record<string, any>) => void) => withPop((r) => change(r.members[index]));
  const single = mutated(population, (r) => {
    r.games = 4;
    r.played = 4;
    r.forfeits = 0;
    delete r.forfeit_causes;
    r.episodes = 8;
    r.copy_decision_hash = null;
    r.members = [{ ...m1, member: "m0" }];
  });
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
    },
    readerInvalid: {
      "league-run-record-2/members-unsorted": withPop((r) => (r.members = [m0, x0, m1])),
      "league-run-record-2/members-unsorted-numeric": withPop((r) => {
        r.members = [{ ...m0, member: "m10" }, { ...m1, member: "m2" }, x0];
      }),
      "league-run-record-2/duplicate-member": withPop((r) => (r.members = [m0, m0, x0])),
      "league-run-record-2/member-counts-disagree": withMember(0, (m) => (m.forfeits = 2)),
      "league-run-record-2/member-episodes-not-two-per-game": withMember(1, (m) => (m.episodes = 9)),
      "league-run-record-2/member-opponents-unsorted": withMember(0, (m) => (m.opponents = [...m.opponents].reverse())),
      "league-run-record-2/member-duplicate-opponent": withMember(1, (m) => (m.opponents = [{ opponent_policy_id: pair("b"), games: 2 }, { opponent_policy_id: pair("b"), games: 2 }])),
      "league-run-record-2/member-opponents-do-not-sum": withMember(1, (m) => (m.opponents = [{ opponent_policy_id: pair("b"), games: 3 }, { opponent_policy_id: pair("d"), games: 2 }])),
      "league-run-record-2/member-causes-missing": withMember(0, (m) => delete m.forfeit_causes),
      "league-run-record-2/member-causes-without-forfeits": withMember(1, (m) => (m.forfeit_causes = [{ cause: "crash", count: 1 }])),
      "league-run-record-2/member-causes-do-not-sum": withMember(0, (m) => (m.forfeit_causes = [{ cause: "crash", count: 2 }])),
      "league-run-record-2/member-causes-unsorted": withPop((r) => {
        r.forfeits = 3;
        r.forfeit_causes = [{ cause: "crash", count: 1 }, { cause: "latency_overrun", count: 2 }];
        Object.assign(r.members[0], { games: 7, played: 4, forfeits: 3, forfeit_causes: [{ cause: "latency_overrun", count: 2 }, { cause: "crash", count: 1 }], opponents: [{ opponent_policy_id: damageId, games: 1 }, { opponent_policy_id: pair("a"), games: 4 }, { opponent_policy_id: pair("b"), games: 2 }] });
        r.games = 14;
      }),
      "league-run-record-2/total-games-differs": withPop((r) => (r.games = 13)),
      "league-run-record-2/total-played-differs": withPop((r) => ((r.played = 10), (r.forfeits = 2))),
      "league-run-record-2/total-episodes-differs": withPop((r) => (r.episodes = 24)),
      "league-run-record-2/total-causes-missing": withPop((r) => delete r.forfeit_causes),
      "league-run-record-2/total-causes-without-forfeits": withPop((r) => {
        r.members = [m1];
        r.games = 4;
        r.played = 4;
        r.forfeits = 0;
        r.episodes = 8;
      }),
      "league-run-record-2/total-causes-differ-from-members": withPop((r) => (r.forfeit_causes = [{ cause: "latency_overrun", count: 1 }])),
      "league-run-record-2/member-baseline-games-disagree": withMember(0, (m) => (m.baselines[0].games = 2)),
      "league-run-record-2/member-baseline-not-an-opponent": withMember(0, (m) => (m.baselines[0].policy_id = "damage-aware@v2")),
      "league-run-record-2/member-baselines-unsorted": withMember(0, (m) => {
        m.baselines = [{ ...m.baselines[0], policy_id: "random-legal@tiny", games: 1 }, m.baselines[0]];
        m.opponents = [...m.opponents, { opponent_policy_id: "random-legal@tiny", games: 1 }];
        m.games += 1;
        m.played += 1;
        m.episodes += 2;
      }),
      "league-run-record-2/member-baseline-probe-runs-exceed-queries": withMember(0, (m) => (m.baselines[0].counts = { ...baselineCounts, probe_runs: 411 })),
      "league-run-record-2/member-baseline-decisions-exceed": withMember(0, (m) => (m.baselines[0].counts = { ...baselineCounts, preview_decisions: 10 })),
    },
  };
}
