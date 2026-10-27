// Measure the cost of damage-aware@v2 (ADR-025 section 13, the cost gate N-H: p99 of a decision at most
// 5 s on one thread): the wall-clock time of every decision of the baseline seat and of every game, on the
// random-play corpus teams against `random-legal` (the baseline seat alternates between p1 and p2), with the
// real mechanics probe, the committed prior and the pinned validator. The same games with --version v1 give
// the ratio to the field-rule baseline.
//
//   node packages/team-dataset/scripts/damage-aware-v2-budget.ts [--version v2|v1] [--games 30] [--start 0] [--max-turns 100] [--out FILE]
//
// It prints one JSON object: per-decision worst, median and 95th percentile (all decisions, and
// the scored ones alone), per-game median, mean, worst and 95th percentile, the counts, and the
// extrapolation to a 200-game match against the run's battle latency budget. The figures are
// measurements of one machine at one time (a shared one: other work running slows them), not a
// claim that they repeat. SHOWDOWN_DIR points at another built checkout.
import { writeFileSync } from "node:fs";
import { readerSha256 } from "../../public-state/src/index.ts";
import { type Actor, damageAwareActor, loadDamageAwareConfig, loadDamageAwareConfigV2, loadEngine, loadMechanicsConfig, loadRulesSnapshot, type Player, randomLegalActor, resolveFormat, runBattle } from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { loadRandomCorpus } from "../../showdown-bridge/src/random-corpus.ts";
import { createBeliefWorlds } from "../src/damage-aware-worlds.ts";

const flag = (name: string, fallback: number, least = 1): number => {
  const at = process.argv.indexOf(`--${name}`);
  if (at < 0) return fallback;
  const value = Number(process.argv[at + 1]);
  if (!Number.isInteger(value) || value < least) throw new Error(`--${name} needs an integer of at least ${least}`);
  return value;
};
const versionAt = process.argv.indexOf("--version");
const version = versionAt < 0 ? "v2" : process.argv[versionAt + 1];
if (version !== "v1" && version !== "v2") throw new Error("--version is v1 or v2");
const out = process.argv.indexOf("--out") >= 0 ? process.argv[process.argv.indexOf("--out") + 1] : undefined;
const games = flag("games", 30);
const start = flag("start", 0, 0);
const maxTurns = flag("max-turns", 100);

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const mechanics = loadMechanicsConfig(GOLDEN_FORMAT);
const snapshot = loadRulesSnapshot(engine.commit);
const manifest = resolveFormat(engine, GOLDEN_FORMAT, snapshot).manifest;
const configV2 = loadDamageAwareConfigV2(vocabulary);
const config = version === "v1" ? loadDamageAwareConfig("v1", vocabulary) : configV2;
const makeOptions = () => ({
  engine,
  vocabulary,
  config,
  mechanics,
  snapshot,
  data_class: "synthetic" as const,
  // the port is built once per game, as the training and progress drivers build one per baseline game
  ...(version === "v2" ? { worlds: createBeliefWorlds({ engine, vocabulary, config: configV2, reader_sha256: readerSha256(), manifest }) } : {}),
});
const corpus = loadRandomCorpus();

const percentile = (sorted: number[], share: number): number => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * share) - 1)] as number;
const summary = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    n: sorted.length,
    median_ms: Math.round(percentile(sorted, 0.5)),
    mean_ms: Math.round(sorted.reduce((a, b) => a + b, 0) / sorted.length),
    p95_ms: Math.round(percentile(sorted, 0.95)),
    p99_ms: Math.round(percentile(sorted, 0.99)),
    worst_ms: Math.round(sorted[sorted.length - 1] as number),
  };
};

const decisionMs: number[] = [];
const scoredMs: number[] = [];
const gameMs: number[] = [];
const totals: Record<string, number> = {};
const decisionsPerGame: number[] = [];
const beliefTotals = { worlds_sampled: 0, redraws: 0, contradicted_worlds: 0, no_consistent_world: 0, refusals: {} as Record<string, number> };
let identityHash = "";
for (let game = start; game < start + games; game++) {
  const battle = corpus[game % corpus.length] as (typeof corpus)[number];
  const seat: Player = game % 2 === 0 ? "p1" : "p2";
  const other: Player = seat === "p1" ? "p2" : "p1";
  const actor = damageAwareActor(makeOptions());
  let scoredBefore = 0;
  const timed: Actor = async (request, context) => {
    const started = performance.now();
    const choice = await actor(request, context);
    const elapsed = performance.now() - started;
    if (request.phase !== "preview") {
      decisionMs.push(elapsed);
      const scored = actor.report().counts.scored_decisions;
      if (scored > scoredBefore) scoredMs.push(elapsed);
      scoredBefore = scored;
    }
    return choice;
  };
  const started = performance.now();
  await runBattle({
    engine,
    formatId: GOLDEN_FORMAT,
    seed: battle.seed,
    teams: battle.teams,
    actors: { [seat]: timed, [other]: randomLegalActor(battle.actor_seeds[other]) } as Record<Player, Actor>,
    maxTurns,
    snapshot,
  });
  gameMs.push(performance.now() - started);
  const report = actor.report();
  identityHash = report.identity_hash;
  if (report.belief !== undefined) {
    beliefTotals.worlds_sampled += report.belief.worlds_sampled;
    beliefTotals.redraws += report.belief.redraws;
    beliefTotals.contradicted_worlds += report.belief.contradicted_worlds;
    beliefTotals.no_consistent_world += report.belief.no_consistent_world;
    for (const [code, n] of Object.entries(report.belief.refusals)) beliefTotals.refusals[code] = (beliefTotals.refusals[code] ?? 0) + n;
  }
  decisionsPerGame.push(report.counts.decisions);
  for (const [key, value] of Object.entries(report.counts)) totals[key] = (totals[key] ?? 0) + value;
}

// A game's time includes the opponent's (random-legal) decisions and the engine; the baseline's own time is the sum of its decisions.
const perGameDecisionMs = decisionMs.reduce((a, b) => a + b, 0) / games;
const result = {
  machine: { node: process.version, cpus: (await import("node:os")).cpus().length, model: (await import("node:os")).cpus()[0]?.model },
  config: { baseline: `damage-aware@${version}`, games, start, max_turns: maxTurns, opponent: "random-legal", teams: "random-play corpus", baseline_identity_hash: identityHash },
  per_decision: { all: summary(decisionMs), scored: summary(scoredMs) },
  per_game: { wall: summary(gameMs), decisions_mean: Math.round(decisionsPerGame.reduce((a, b) => a + b, 0) / games), baseline_decision_time_mean_ms: Math.round(perGameDecisionMs) },
  counts_total: totals,
  belief_total: version === "v2" ? beliefTotals : undefined,
  // ADR-025 section 13 (N-H): p99 per decision at most 5 s on one thread; if it fails N is lowered, which is a new v2 identity
  cost_gate: version === "v2" ? { p99_limit_ms: 5000, p99_ms: summary(decisionMs).p99_ms, scored_p99_ms: summary(scoredMs).p99_ms, passes: summary(scoredMs).p99_ms <= 5000 && summary(decisionMs).p99_ms <= 5000 } : undefined,
  match_200_games_minutes: {
    // the games here end early (random-legal plays badly); ADR-018 assumes about 25 baseline decisions per game against a learned policy
    at_25_decisions_per_game_from_mean_decision: Number(((200 * 25 * (decisionMs.reduce((a, b) => a + b, 0) / decisionMs.length)) / 60000).toFixed(1)),
    at_25_decisions_per_game_from_p95_decision: Number(((200 * 25 * summary(decisionMs).p95_ms) / 60000).toFixed(1)),
    from_mean_game: Number(((200 * (gameMs.reduce((a, b) => a + b, 0) / games)) / 60000).toFixed(1)),
    from_p95_game: Number(((200 * (summary(gameMs).p95_ms)) / 60000).toFixed(1)),
  },
  budgets: { battle_latency_ms: 30000, match_minutes: 30 },
};
const text = JSON.stringify(result, null, 2);
if (out !== undefined) writeFileSync(out, `${text}\n`);
console.log(text);
