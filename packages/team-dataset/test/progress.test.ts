// The progress evaluation's play, with an in-process stand-in for the policy worker
// (the real worker is driven by progress.python.test.ts).
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  actorRule,
  buildProgressGameRecord,
  committedPlanSeeds,
  guardActor,
  playGame,
  type PlayedGame,
  jcs,
  type Actor,
  type PolicyClient,
  PolicyProtocolError,
  PROGRESS_BASELINES,
  PROGRESS_MODES,
  progressGameStreams,
  progressPlan,
  playProgress,
  ProgressError,
  type ProgressGameRecord,
  type ProgressOptions,
  progressRecordHash,
  readProgressEvaluationRecord,
  readProgressGameRecord,
  readTrainingRunConfig,
  replayActor,
  runBattle,
  omniscientLogHash,
  teamHash,
  writeProgress,
  type Json,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { datasetValidationPool } from "../src/progress-pool.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = new URL("../../..", import.meta.url).pathname;
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const tiny = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
// A turn cap that lets games be decided (the tiny sample's cap of 6 turns ends every game as a tie).
const config = readTrainingRunConfig({ ...tiny, generation: { ...(tiny.generation as object), max_turns: 100 }, team_dataset: { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) } });
const pool = datasetValidationPool(engine, config);
const scratch = mkdtempSync(join(tmpdir(), "progress-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const firstLegal: Actor = (request) => actorRule(request.legal as never);

type Client = PolicyClient & { beginGame: () => Promise<void>; asked: number; modes: Set<string> };
/** An in-process worker: the shared actor rule over the query's own candidates. */
function actorClient(fail?: (n: number) => Error | undefined): Client {
  const client: Client = {
    asked: 0,
    modes: new Set(),
    beginGame: () => Promise.resolve(),
    ask(_battle, query) {
      client.asked++;
      client.modes.add(query.sampling.kind);
      const error = fail?.(client.asked);
      if (error !== undefined) return Promise.reject(error);
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({
        schema_version: 1 as const,
        identity: query.identity,
        choice: actorRule({ choices: open, pending }),
        source_policy_id: query.policy_id,
        elapsed_ms: 0,
        deadline_outcome: "in_budget" as const,
      });
    },
  };
  return client;
}

const options = (change: Partial<ProgressOptions> = {}): ProgressOptions => ({
  engine,
  config,
  generation: 1,
  generation_manifest_sha256: "a".repeat(64),
  checkpoints: { preview: "1".repeat(64), battle: "2".repeat(64) },
  policy_ids: { sample: `policy:${"c".repeat(64)}`, argmax: `policy:${"d".repeat(64)}` },
  client: actorClient(),
  pool,
  games: 2,
  record_games: 1,
  training_teams: new Set(),
  league_seeds: new Set(),
  ...change,
});

describe("progressPlan", () => {
  it("pairs the validation teams disjointly, as many pairs as asked", () => {
    const plan = progressPlan(options({ games: 5 }));
    expect(plan.pairs).toHaveLength(5);
    const used = plan.pairs.flatMap(([a, b]) => [a.canonical_team_hash, b.canonical_team_hash]);
    expect(new Set(used).size).toBe(10);
    expect(plan.seeds).toEqual([0, 1, 2, 3, 4].map((game) => progressGameStreams(config.seeds.run_key, game).seed));
  });

  it("refuses more games than disjoint pairs, never wrapping round", () => {
    const most = Math.floor(pool.teams.length / 2);
    expect(progressPlan(options({ games: most })).pairs).toHaveLength(most);
    expect(() => progressPlan(options({ games: most + 1 }))).toThrow(/disjoint team pairs/);
    expect(() => progressPlan(options({ games: 0 }))).toThrow(/positive integer/);
  });

  it("refuses a pool team the run trained on, and a team not proven synthetic", () => {
    const first = pool.teams[0]?.sets as never;
    expect(() => progressPlan(options({ training_teams: new Set([teamHash(first)]) }))).toThrow(/team-set\.json/);
    const unproven = { ...pool, known_synthetic: new Set<string>() };
    expect(() => progressPlan(options({ pool: unproven }))).toThrow(/not proven synthetic/);
    const twice = { ...pool, teams: [...pool.teams, pool.teams[0] as never] };
    expect(() => progressPlan(options({ pool: twice }))).toThrow(/twice/);
  });

  it("refuses a progress seed that equals a seed of a committed plan", () => {
    const seed = progressGameStreams(config.seeds.run_key, 1).seed;
    expect(() => progressPlan(options({ league_seeds: new Set([seed]) }))).toThrow(/equals a seed of a committed plan/);
    // A seed beyond the games asked for is not in play, so it is not refused.
    expect(progressPlan(options({ games: 1, league_seeds: new Set([seed]) })).seeds).toHaveLength(1);
  });
});

describe("playProgress", () => {
  it("plays four matches on one plan, both modes against both baselines, and records the first games", async () => {
    const client = actorClient();
    const result = await playProgress(options({ client }));
    const { record } = result;
    expect(record.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "random-legal-argmax", "random-legal-sample"]);
    expect(PROGRESS_BASELINES.length * PROGRESS_MODES.length).toBe(4);
    expect(record.games).toHaveLength(8);
    expect(record.plan).toMatchObject({ games_per_match: 2, recorded_games_per_match: 1, max_turns: config.generation.max_turns });
    expect(record.teams).toEqual({ dataset_hash: config.team_dataset?.dataset_hash, split: "validation", pairs_used: 2 });
    // Every match uses the same two pairs and seeds; the candidate sits on p1 for an even game and p2 for an odd one.
    for (const match of record.matches) {
      const rows = record.games.filter((row) => row.match_id === match.match_id);
      expect(rows.map((row) => [row.pair, row.candidate_seat])).toEqual([[0, "p1"], [1, "p2"]]);
      expect(match.game_record_hashes).toHaveLength(1);
    }
    expect([...result.game_records.keys()].sort()).toEqual(record.matches.map((m) => `${m.match_id}-0.json`));
    expect(client.modes).toEqual(new Set(["sample", "argmax"]));
    expect(Object.keys(result.wall_ms)).toHaveLength(4);
    expect(() => readProgressEvaluationRecord(JSON.parse(jcs(record as unknown as Json)))).not.toThrow();
    for (const stored of result.game_records.values()) {
      expect(stored.data_class).toBe("synthetic");
      expect(JSON.parse(jcs(readProgressGameRecord(stored) as unknown as Json))).toEqual(stored);
    }
  }, 120_000);

  describe("against the previous generation's frozen pair", () => {
    const previous = (client = actorClient()): NonNullable<ProgressOptions["previous"]> => ({
      checkpoints: { preview: "3".repeat(64), battle: "4".repeat(64) },
      policy_ids: { sample: `policy:${"e".repeat(64)}`, argmax: `policy:${"f".repeat(64)}` },
      client,
    });

    it("adds previous-sample and previous-argmax, each pair in its own mode, on the same plan and seeds", async () => {
      const client = actorClient();
      const other = actorClient();
      const result = await playProgress(options({ client, previous: previous(other), record_games: 2 }));
      const { record } = result;
      expect(record.matches.map((m) => m.match_id)).toEqual(["actor-rule-argmax", "actor-rule-sample", "previous-argmax", "previous-sample", "random-legal-argmax", "random-legal-sample"]);
      expect(record.games).toHaveLength(12);
      for (const mode of PROGRESS_MODES) {
        const match = record.matches.find((m) => m.match_id === `previous-${mode}`) as (typeof record.matches)[number];
        expect(match.mode).toBe(mode);
        expect(match.policy_id).toBe(options().policy_ids[mode]);
        expect(match.opponent).toEqual({ kind: "previous", policy_id: previous().policy_ids[mode], checkpoints: previous().checkpoints });
      }
      // Both pairs were queried, each in the mode of its match; the candidate's worker also served the baselines.
      expect(other.asked).toBeGreaterThan(0);
      expect(other.modes).toEqual(new Set(["sample", "argmax"]));
      expect(client.asked).toBeGreaterThan(other.asked);
      // Same plan: the previous games use the seeds and pairs of the baseline games, and name both policy ids.
      const baseline = result.game_records.get("actor-rule-sample-1.json") as ProgressGameRecord;
      const versus = result.game_records.get("previous-sample-1.json") as ProgressGameRecord;
      expect(versus.seed).toBe(baseline.seed);
      expect(versus.teams).toEqual(baseline.teams);
      expect(versus.seed).toBe(progressGameStreams(config.seeds.run_key, 1).seed);
      const seats = (g: ProgressGameRecord): string[] => [g.policies.p1, g.policies.p2];
      expect(seats(versus).sort()).toEqual([options().policy_ids.sample, previous().policy_ids.sample].sort());
      // The candidate sits on p2 in an odd game.
      expect(versus.policies.p2).toBe(options().policy_ids.sample);
      expect(Object.keys(result.wall_ms)).toHaveLength(6);
    }, 240_000);

    it("is byte-identical when run twice", async () => {
      const run = () => playProgress(options({ previous: previous() }));
      const [a, b] = [await run(), await run()];
      expect(jcs(a.record as unknown as Json)).toBe(jcs(b.record as unknown as Json));
      for (const [name, stored] of a.game_records) expect(jcs(stored as unknown as Json)).toBe(jcs(b.game_records.get(name) as unknown as Json));
    }, 240_000);

    it("still refuses a progress seed equal to a committed plan's before any game is played", async () => {
      const seed = progressGameStreams(config.seeds.run_key, 0).seed;
      const other = actorClient();
      await expect(playProgress(options({ previous: previous(other), league_seeds: new Set([seed]) }))).rejects.toThrow(/equals a seed of a committed plan/);
      expect(other.asked).toBe(0);
    });
  });

  it("is byte-identical when run twice", async () => {
    const a = await playProgress(options());
    const b = await playProgress(options());
    expect(jcs(a.record as unknown as Json)).toBe(jcs(b.record as unknown as Json));
    expect(progressRecordHash(a.record)).toBe(progressRecordHash(b.record));
    for (const [name, stored] of a.game_records) expect(jcs(stored as unknown as Json)).toBe(jcs(b.game_records.get(name) as unknown as Json));
  }, 120_000);

  it("never plays a team of the pairing twice within a match", async () => {
    const result = await playProgress(options({ games: 3, record_games: 3 }));
    for (const match of result.record.matches) {
      const hashes = [...result.game_records.entries()].filter(([name]) => name.startsWith(`${match.match_id}-`)).flatMap(([, g]) => [g.teams.p1.team_hash, g.teams.p2.team_hash]);
      expect(new Set(hashes).size).toBe(6);
    }
    const trained = new Set(pool.teams.map((t) => teamHash(t.sets)));
    for (const g of result.game_records.values()) for (const seat of ["p1", "p2"] as const) expect(trained.has(g.teams[seat].team_hash)).toBe(true);
  }, 120_000);

  it("replays a stored game on the engine to its omniscient_log_sha256", async () => {
    const { game_records } = await playProgress(options({ games: 2, record_games: 2 }));
    expect(game_records.size).toBe(8);
    for (const stored of game_records.values()) {
      const choices = stored.decisions.map((d) => ({ player: d.player, request_seq: d.request_seq, choice: d.chosen }));
      const replay = await runBattle({
        engine,
        formatId: config.format_id,
        seed: stored.seed,
        teams: { p1: stored.teams.p1.team as never, p2: stored.teams.p2.team as never },
        actors: { p1: replayActor(choices, "p1"), p2: replayActor(choices, "p2") },
        maxTurns: config.generation.max_turns,
      });
      if (stored.outcome.kind === "played") {
        expect(omniscientLogHash(replay.omniscient_log)).toBe(stored.omniscient_log_sha256);
        expect(replay.turns).toBe(stored.outcome.turns);
      }
    }
  }, 120_000);

  it("scores a policy forfeit as a loss for the failing side and keeps its cause", async () => {
    // The stand-in worker crashes on its third query of the evaluation: a candidate forfeit.
    const client = actorClient((n) => (n === 3 ? new Error("worker died") : undefined));
    const { record } = await playProgress(options({ client, games: 1, record_games: 1 }));
    const first = record.matches[0] as (typeof record.matches)[number];
    expect(first.forfeits).toEqual([{ side: "candidate", cause: "crash", count: 1 }]);
    expect(first.losses).toBe(1);
    expect(record.games[0]?.outcome).toMatchObject({ kind: "forfeit", loser: "candidate", cause: "crash" });
  }, 120_000);

  it("fails the whole evaluation on a harness incident and returns nothing", async () => {
    const client = actorClient((n) => (n === 2 ? new PolicyProtocolError("bad query") : undefined));
    await expect(playProgress(options({ client }))).rejects.toThrow(/policy_protocol/);
  });

  it("refuses record_games above games", async () => {
    await expect(playProgress(options({ record_games: 3 }))).rejects.toThrow(ProgressError);
  });

  it("writes progress.json and the games, durable and canonical", async () => {
    const result = await playProgress(options());
    const dir = join(scratch, "out");
    writeProgress(dir, result);
    expect(readdirSync(dir).sort()).toEqual(["games", "progress.json"]);
    expect(readFileSync(join(dir, "progress.json"), "utf8")).toBe(`${jcs(result.record as unknown as Json)}\n`);
    expect(readdirSync(join(dir, "games"))).toHaveLength(4);
  }, 120_000);
});

describe("buildProgressGameRecord", () => {
  const played = async (): Promise<PlayedGame> => {
    const team = pool.teams[0]?.sets as never;
    return playGame({ engine, formatId: FORMAT, seed: progressGameStreams("r", 0).seed, teams: { p1: team, p2: team }, actors: { p1: firstLegal, p2: firstLegal }, maxTurns: 2 });
  };
  const facts = (game: PlayedGame) => ({ run_id: "r", generation: 1, match_id: "m", game: 0, teams: { p1: pool.teams[0]?.sets as never, p2: pool.teams[0]?.sets as never }, policies: { p1: "a@1", p2: "b@1" }, played: game });

  it("records a refused pending choice beside the chosen one", async () => {
    const game = await played();
    ((game.record as { choices: { refused?: string }[] }).choices[0] as { refused?: string }).refused = "[Unavailable choice] a hidden reason";
    const stored = buildProgressGameRecord(facts(game));
    expect(stored.decisions[0]?.refused).toBe("[Unavailable choice] a hidden reason");
    expect(stored.decisions[1]).not.toHaveProperty("refused");
  });

  it("stores decisions by request then seat, whichever seat's worker answered first", async () => {
    const game = await played();
    const expected = buildProgressGameRecord(facts(game));
    const choices = (game.record as { choices: unknown[] }).choices;
    choices.reverse();
    const reversed = buildProgressGameRecord(facts(game));
    expect(jcs(reversed as unknown as Json)).toBe(jcs(expected as unknown as Json));
    expect(expected.decisions.map((d) => [d.request_seq, d.player])).toEqual([...expected.decisions].map((d) => [d.request_seq, d.player]).sort((a, b) => (a[0] as number) - (b[0] as number) || String(a[1]).localeCompare(String(b[1]))));
  });

  it("records a forfeit by seat, with its cause and request", async () => {
    const team = pool.teams[0]?.sets as never;
    const dies = guardActor(() => Promise.reject(new Error("worker died")), { role: "policy", latency_budget: { preview_ms: 1000, battle_ms: 1000 } });
    const game = await playGame({ engine, formatId: FORMAT, seed: progressGameStreams("r", 0).seed, teams: { p1: team, p2: team }, actors: { p1: firstLegal, p2: dies }, maxTurns: 2 });
    const stored = buildProgressGameRecord(facts(game));
    expect(stored.outcome).toMatchObject({ kind: "forfeit", loser: "p2", cause: "crash", request_seq: 1 });
  });

  it("refuses a choice that names a request the record does not hold", async () => {
    const game = await played();
    (game.record as { requests: { p1: unknown[] } }).requests.p1 = [];
    expect(() => buildProgressGameRecord(facts(game))).toThrow(/names a request the record does not hold/);
  });

  it("refuses an incident game", () => {
    const played = { outcome: { kind: "incident" as const, incident: "unclassified" as const, message: "x" }, record: null };
    expect(() => buildProgressGameRecord({ run_id: "r", generation: 1, match_id: "m", game: 0, teams: { p1: [], p2: [] }, policies: { p1: "a@1", p2: "b@1" }, played })).toThrow(/incident/);
  });
});

describe("committedPlanSeeds", () => {
  it("collects the battle seeds of every committed generation's plan, skipping generation 0 and anything else", () => {
    const run = join(scratch, "seeds-run");
    mkdirSync(join(run, "generations", "g000000"), { recursive: true }); // plays nothing: no plan.json
    mkdirSync(join(run, "generations", "g000001"));
    mkdirSync(join(run, "generations", "g000002"));
    mkdirSync(join(run, "generations", ".g000003.partial"));
    writeFileSync(join(run, "generations", "g000001", "plan.json"), JSON.stringify([{ seed: "sodium,a" }, { seed: "sodium,b" }]));
    writeFileSync(join(run, "generations", "g000002", "plan.json"), JSON.stringify([{ seed: "sodium,c" }]));
    writeFileSync(join(run, "generations", ".g000003.partial", "plan.json"), JSON.stringify([{ seed: "sodium,partial" }]));
    expect([...committedPlanSeeds(run)].sort()).toEqual(["sodium,a", "sodium,b", "sodium,c"]);
  });

  it("does not hide a plan it cannot read", () => {
    const run = join(scratch, "seeds-broken");
    mkdirSync(join(run, "generations", "g000001", "plan.json"), { recursive: true }); // a directory where the file belongs
    expect(() => committedPlanSeeds(run)).toThrow(/EISDIR/);
  });
});
