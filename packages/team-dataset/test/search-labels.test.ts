// The off-path search labellers and the label stream (ADR-026 erratum E2 decisions 1 and 3 to 6; SD-d, #1290), with the real engine,
// opponent model and probe and a stand-in worker: the labels of an Episode (determinism, invariance to what the opponent holds
// unrevealed, the negative control, the seed check), and a generation that labels (the PPO stream is byte-identical to an unlabelled
// one, the label stream and its manifest are valid, equal on a rerun and for any split of the games over processes, a missing file
// is recomputed and too many fail the generation).
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type EpisodeLabels,
  episodeFromRecord,
  episodeHash,
  GenerationError,
  jcs,
  type Json,
  labelCounts,
  labelEpisode,
  readLabelError,
  labelPolicy,
  labelSetup,
  labelSelected,
  createReplyModel,
  featureSetup,
  ReplyModelError,
  type ReplyModel,
  writeEpisodeLabels,
  writeLabelFiles,
  LabelError,
  type LabelSetup,
  leagueBattleSeed,
  loadSearchConfig,
  type PolicyClient,
  playGeneration,
  readEpisodeLabels,
  readTrainingExample,
  readTrainingRunConfig,
  
  readTrainingSet,
  runBattle,
  searchConfigHash,
  stampTeam,
  syntheticTeamHashes,
  teamHash,
  type WorldSamplePort,
  writeLabelStream,
} from "@areneon/showdown-bridge";
import { nodeSha256 } from "../../showdown-bridge/src/player-channel.ts";
import { describe, expect, it } from "vitest";
import { firstLegal } from "../../showdown-bridge/src/golden.ts";
import { type GenerationOptions, playEntries, trainingTeamPool, writeGeneration } from "../../showdown-bridge/src/league-generation.ts";
import { searchTargetProblem } from "../../showdown-bridge/src/training-loader.ts";
import { loadRandomCorpus, type RandomBattleFile } from "../../showdown-bridge/src/random-corpus.ts";
import { SAMPLES_DIR } from "../../showdown-bridge/src/samples.ts";
import { comparable, mutated, withGenders } from "../../showdown-bridge/test/damage-aware-invariance-support.ts";
import { progressSearchSetup } from "../src/progress-search-setup.ts";
import { manifest, portFor, setupFor, ValueStandInWorker, vocabulary } from "./search-actor-support.ts";
import { engine, FORMAT } from "./support.ts";

const label = loadSearchConfig(FORMAT, "rerank-1-label.v1");
const pair = { preview: "a".repeat(64), battle: "b".repeat(64) };
const POLICY = `policy:${"a".repeat(64)}`;
const RUN_KEY = "labels-test";
const GENERATION = 3;
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** The stand-in worker as a game client: it plays the first open candidate and answers `evaluate` and `value` as the search tests' does. */
class Worker extends ValueStandInWorker {
  async beginGame(): Promise<void> {}
  override async ask(_battle: string, query: Parameters<PolicyClient["ask"]>[1]): ReturnType<PolicyClient["ask"]> {
    const open = query.candidates.filter((candidate) => !candidate.pending);
    const choice = (open[0] ?? (query.candidates[0] as { choice: string })).choice;
    return { schema_version: 1, identity: query.identity, choice, source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" } as Awaited<ReturnType<PolicyClient["ask"]>>;
  }
}

const labelsFor = (share_ppm: number, change: Partial<LabelSetup> = {}): LabelSetup => ({
  setup: setupFor(portFor(), { config: label }),
  search_config_hash: searchConfigHash(label),
  share_ppm,
  start_generation: GENERATION,
  min_spread_ppm: 0,
  pair,
  manifest,
  ...change,
});

// ---- the labels of one Episode ----

const battles = loadRandomCorpus().slice(0, 2);

async function episodeOf(battle: RandomBattleFile, p2?: RandomBattleFile["teams"]["p2"], game = 0) {
  const teams = { p1: withGenders(battle.teams.p1), p2: withGenders(p2 ?? battle.teams.p2) };
  const seed = leagueBattleSeed(RUN_KEY, GENERATION, game);
  const record = await runBattle({ engine, formatId: FORMAT, seed, teams, actors: { p1: firstLegal, p2: firstLegal }, maxTurns: 3 });
  const known = syntheticTeamHashes([teamHash(teams.p1), teamHash(teams.p2)]);
  const episode = await episodeFromRecord(record, "p1", {
    team: stampTeam(teams.p1 as never, known),
    opponent_team: stampTeam(teams.p2 as never, known),
    known_synthetic: known,
    behaviour_policy_id: POLICY,
    collection: { kind: "league", run_id: RUN_KEY, seed: record.seed, opponent_policy_id: "other@1" },
  });
  return { record, episode, hash: await episodeHash(episode, nodeSha256) };
}

const contextFor = (labels: LabelSetup, client = new Worker()) => ({ engine, labels, client, run_key: RUN_KEY, generation: GENERATION });

describe("the labels of one Episode", () => {
  it("labels the selected Battle move decisions: entries in range over non-pending candidates, deterministic, every refusal a counted reason", async () => {
    const { episode, hash } = await episodeOf(battles[0] as RandomBattleFile);
    const a = await labelEpisode(contextFor(labelsFor(1_000_000)), episode, hash, 0);
    const b = await labelEpisode(contextFor(labelsFor(1_000_000)), episode, hash, 0);
    expect(jcs(b as unknown as Json)).toBe(jcs(a as unknown as Json));
    expect(a.decisions.length).toBeGreaterThan(0);
    expect(a.search_identity_hash).toBe(labelPolicy(labelsFor(1), new Worker()).identity_hash);
    const labelled = a.decisions.filter((d) => "label" in d);
    expect(labelled.length).toBeGreaterThan(0);
    for (const d of a.decisions) {
      if (!("label" in d)) continue;
      expect(d.label.entries.length).toBeGreaterThanOrEqual(2);
      expect(d.label.entries.length).toBeLessThanOrEqual(label.k_max as number);
      for (const e of d.label.entries) {
        expect(e.q_ppm).toBeGreaterThanOrEqual(0);
        expect(e.q_ppm).toBeLessThanOrEqual(1_000_000);
        expect(e.candidate).toBeLessThan(d.label.prior_ppm.length);
      }
      expect(d.label.entries.map((e) => e.candidate)).toEqual([...d.label.entries.map((e) => e.candidate)].sort((x, y) => x - y));
    }
    // selection by hash: a smaller share keeps a subset of the decisions
    const some = await labelEpisode(contextFor(labelsFor(400_000)), episode, hash, 0);
    const all = new Set(a.decisions.map((d) => d.request_seq));
    for (const d of some.decisions) {
      expect(all.has(d.request_seq)).toBe(true);
      expect(labelSelected({ run_key: RUN_KEY, generation: GENERATION, game: 0, player: "p1", request_seq: d.request_seq }, 400_000)).toBe(true);
    }
  }, 600_000);

  it("refuses an Episode whose seed is not the one league-seed/1 gives for the game, and one of another run", async () => {
    const { episode, hash } = await episodeOf(battles[0] as RandomBattleFile);
    await expect(labelEpisode(contextFor(labelsFor(1_000_000)), episode, hash, 1)).rejects.toThrow(LabelError);
    await expect(labelEpisode({ ...contextFor(labelsFor(1_000_000)), run_key: "another" }, episode, hash, 0)).rejects.toThrow("not a league Episode of this run");
  }, 300_000);

  it("counts a decision under the minimum spread as a refusal, not a label", async () => {
    const { episode, hash } = await episodeOf(battles[0] as RandomBattleFile);
    const loose = await labelEpisode(contextFor(labelsFor(1_000_000)), episode, hash, 0);
    const strict = await labelEpisode(contextFor(labelsFor(1_000_000, { min_spread_ppm: 1_000_001 })), episode, hash, 0);
    expect(strict.decisions.filter((d) => "label" in d)).toEqual([]);
    expect(strict.decisions.filter((d) => "refused" in d && d.refused === "spread_below_min").length).toBeGreaterThanOrEqual(loose.decisions.filter((d) => "label" in d).length);
  }, 600_000);

  it("labels from the player's view alone: a hidden-information change in the opponent's team changes no label at an equal player channel", async () => {
    let compared = 0;
    for (const battle of battles) {
      const a = await episodeOf(battle);
      const b = await episodeOf(battle, mutated(battle.teams.p2));
      const la = await labelEpisode(contextFor(labelsFor(1_000_000)), a.episode, a.hash, 0);
      const lb = await labelEpisode(contextFor(labelsFor(1_000_000)), b.episode, b.hash, 0);
      for (const { seq } of comparable(a.record, b.record)) {
        const left = la.decisions.find((d) => d.request_seq === seq);
        const right = lb.decisions.find((d) => d.request_seq === seq);
        expect(right, `${battle.id} request ${seq}`).toEqual(left);
        if (left !== undefined) compared += 1;
      }
    }
    expect(compared).toBeGreaterThanOrEqual(2);
  }, 900_000);

  it("has teeth: a belief handed the opponent's real team labels differently", async () => {
    const cheat = (real: RandomBattleFile["teams"]["p2"]): WorldSamplePort => {
      const inner = portFor();
      return {
        ...inner,
        open: ({ opponent }) => {
          const brought = real.slice(0, 4).map((set) => toId(set.species));
          const world = {
            slots: opponent.roster.map((mon, index) => {
              const set = real.slice(0, 4).find((s) => [mon.base_species, mon.species].includes(toId(s.species)));
              if (set === undefined || !brought.includes(toId(set.species))) return { index, brought: false, set: null };
              return {
                index,
                brought: true,
                set: { species: mon.base_species, forme: mon.base_species, item: set.item === "" ? null : toId(set.item), ability: toId(set.ability), nature: toId(set.nature), stat_points: set.evs, moves: set.moves.map(toId).sort(), level: 50, gender: null },
              };
            }),
          };
          return { kind: "source", source: { draw: () => ({ world: world as never, redraws: 0 }) } };
        },
      };
    };
    const skew = (team: RandomBattleFile["teams"]["p2"], bulky: boolean) => team.map((set) => ({ ...structuredClone(set), evs: bulky ? { hp: 32, atk: 0, def: 32, spa: 0, spd: 2, spe: 0 } : { hp: 0, atk: 32, def: 0, spa: 32, spd: 0, spe: 2 } }));
    const battle = battles[1] as RandomBattleFile;
    const teams = [skew(battle.teams.p2, true), skew(battle.teams.p2, false)];
    const found: EpisodeLabels[] = [];
    for (const team of teams) {
      const { episode, hash } = await episodeOf(battle, team);
      found.push(await labelEpisode(contextFor(labelsFor(1_000_000, { setup: setupFor(cheat(team), { config: label }) })), episode, hash, 0));
    }
    const a = found[0] as EpisodeLabels;
    const b = found[1] as EpisodeLabels;
    // the first request is the same player view whatever the opponent holds, so an honest belief would label alike
    const first = (labels: EpisodeLabels) => labels.decisions.find((d) => "label" in d);
    expect(first(a)).toBeDefined();
    expect(JSON.stringify(first(a))).not.toBe(JSON.stringify(first(b)));
  }, 600_000);
});

// ---- a generation that labels ----

const raw = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/search-distill.json"), "utf8")) as { generation: Record<string, unknown>; seeds: Record<string, unknown> };
raw.seeds.run_key = RUN_KEY;
raw.generation.games = 2;
raw.generation.baseline_games = [{ actor: "random-legal", policy_id: "random-legal@tiny", games: 1 }];
raw.generation.max_turns = 4;
(raw.generation.search_labels as Record<string, unknown>).search_config_hash = searchConfigHash(label);
(raw.generation.search_labels as Record<string, unknown>).label_share_ppm = 400_000;
(raw.generation.search_labels as Record<string, unknown>).start_generation = GENERATION;
const config = readTrainingRunConfig(raw);
const pool = trainingTeamPool(engine, config.generation.excluded_abilities);

const labelled = (labels: LabelSetup | undefined, dirs: { episodes: string; stream: string; labels?: string; labelsPath: string }, change: Partial<GenerationOptions> = {}): GenerationOptions => ({
  engine,
  config,
  generation: GENERATION,
  policy_id: POLICY,
  client: new Worker() as unknown as GenerationOptions["client"],
  pool,
  episodes_dir: dirs.episodes,
  stream_path: dirs.stream,
  ...(labels === undefined ? {} : { labels, labels_dir: dirs.labels, labels_path: dirs.labelsPath }),
  ...change,
});

const tmp = (): string => mkdtempSync(join(tmpdir(), "labels-test-"));
const dirsFor = (root: string, name: string) => {
  const base = join(root, name);
  mkdirSync(join(base, "episodes"), { recursive: true });
  return { episodes: join(base, "episodes"), stream: join(base, "stream.jsonl"), labels: join(base, "scratch"), labelsPath: join(base, "labels.jsonl"), base };
};
const text = (path: string): string => readFileSync(path, "utf8");

describe("a generation that labels", () => {
  it("leaves the PPO stream as it was, writes a valid label stream and manifest, and is the same for a rerun and for two game processes", async () => {
    const root = tmp();
    try {
      const labels = labelsFor(400_000);
      const plain = dirsFor(root, "plain");
      await playGeneration({ ...labelled(undefined, plain) });
      const single = dirsFor(root, "single");
      const result = await playGeneration(labelled(labels, single));
      // the PPO stream of a labelled generation is byte-identical to an unlabelled one
      expect(text(single.stream)).toBe(text(plain.stream));
      expect(result.labels?.search_identity_hash).toBe(labelPolicy(labels, new Worker()).identity_hash);
      const set = result.labels?.set;
      if (set === null || set === undefined) throw new Error("the generation labelled nothing");
      const lines = text(single.labelsPath).split("\n").filter((line) => line !== "");
      expect(lines.length).toBe(set.examples);
      expect(set.search_labels?.labelled).toBe(set.examples);
      for (const line of lines) {
        const example = readTrainingExample(JSON.parse(line));
        expect(example.search_target).toBeDefined();
        expect(searchTargetProblem(example)).toBeNull();
        expect(example.observation.kind).toBe("battle");
      }
      readTrainingSet(JSON.parse(jcs(set as unknown as Json)));
      const totals = result.labels?.totals;
      expect(totals?.selected).toBe((totals?.labelled ?? 0) + Object.values(totals?.refused ?? {}).reduce((a, b) => a + b, 0));
      expect(totals?.recomputed).toBe(0);
      // every file of the generation: the label manifest and the summary's block
      writeGeneration(single.base, result);
      expect(JSON.parse(text(join(single.base, "label-set.json")))).toEqual(set);
      const summary = JSON.parse(text(join(single.base, "generation.json"))) as { search_labels: Record<string, unknown> };
      // the counts are in the summary: how many were selected, labelled and refused (by reason), none of the diagnostic classes
      const counts = labelCounts(result.labels?.totals as NonNullable<typeof result.labels>["totals"]);
      expect(summary.search_labels).toEqual({ search_config_hash: searchConfigHash(label), search_identity_hash: result.labels?.search_identity_hash, ...counts });
      expect(counts.labelled).toBe(set.examples);
      expect(counts.labelled + counts.refused.reduce((a, r) => a + r.count, 0)).toBe(counts.selected);
      expect(counts.refused.map((r) => r.reason)).toEqual([...counts.refused.map((r) => r.reason)].sort());
      expect(JSON.stringify(summary)).not.toContain("unseen_brought");
      const noLabels = dirsFor(root, "plain-written");
      writeGeneration(noLabels.base, await playGeneration(labelled(undefined, noLabels)));
      const { search_labels: _block, ...rest } = summary as Record<string, unknown>;
      expect(rest).toEqual(JSON.parse(text(join(noLabels.base, "generation.json"))));
      expect(readdirSync(noLabels.base).sort()).not.toContain("label-set.json");

      // a rerun, labelling every Episode again from the stored files, gives the same bytes
      const again = dirsFor(root, "again");
      const refs = readdirSync(single.episodes).map((name) => ({ hash: name.replace(".json", ""), path: join(single.episodes, name) }));
      const known = pool.known_synthetic;
      const withGames = [];
      for (const ref of refs) {
        const episode = JSON.parse(text(ref.path)) as { provenance: { collection: { seed: string }; behaviour_policy_id: string } };
        if (episode.provenance.behaviour_policy_id !== POLICY) continue;
        const game = result.plan.find((entry) => entry.seed === episode.provenance.collection.seed)?.game as number;
        withGames.push({ ...ref, game });
      }
      const rerun = await writeLabelStream({
        context: { ...contextFor(labels), known_synthetic: known, behaviour_policy_id: POLICY },
        episodes: withGames,
        dir: again.labels,
        path: again.labelsPath,
        max_missing_percent: 100,
      });
      expect(text(again.labelsPath)).toBe(text(single.labelsPath));
      expect(jcs(rerun.set as unknown as Json)).toBe(jcs(set as unknown as Json));
      expect(rerun.totals.recomputed).toBe(withGames.length);
      expect(readEpisodeLabels(again.labels, withGames[0]?.hash as string, result.labels?.search_identity_hash as string)).toBeDefined();
      // more than 5% of the files missing fails the generation as a harness incident
      await expect(
        writeLabelStream({ context: { ...contextFor(labels), known_synthetic: known, behaviour_policy_id: POLICY }, episodes: withGames, dir: join(root, "empty"), path: join(root, "none.jsonl") }),
      ).rejects.toThrow("harness incident");
      // nothing selected: no label file, a null set
      const none = await writeLabelStream({
        context: { ...contextFor(labelsFor(1)), known_synthetic: known, behaviour_policy_id: POLICY },
        episodes: withGames,
        dir: join(root, "none"),
        path: join(root, "no-labels.jsonl"),
        max_missing_percent: 100,
      });
      expect(none.set).toBeNull();
      expect(() => readFileSync(join(root, "no-labels.jsonl"))).toThrow();

      // two game processes (each labels its own Episodes into the shared scratch directory) give the same files
      const split = dirsFor(root, "split");
      const others = playEntries(labelled(labels, split), { index: 1, of: 2 });
      const parent = await playGeneration(labelled(labels, split, { shard: { index: 0, of: 2 }, others }));
      expect(text(split.stream)).toBe(text(single.stream));
      expect(text(split.labelsPath)).toBe(text(single.labelsPath));
      expect(jcs(parent.labels?.set as unknown as Json)).toBe(jcs(set as unknown as Json));
      expect(parent.labels?.totals.recomputed).toBe(0);

      // every worker of a process labels (Episodes spread over them): the same files
      const peered = dirsFor(root, "peered");
      const withPeers = await playGeneration(labelled(labels, peered, { peers: [new Worker() as never, new Worker() as never] }));
      expect(text(peered.stream)).toBe(text(single.stream));
      expect(text(peered.labelsPath)).toBe(text(single.labelsPath));
      expect(jcs(withPeers.labels?.set as unknown as Json)).toBe(jcs(set as unknown as Json));
      // and the writer's own recomputation, over several workers, is the same too
      const recomputed = dirsFor(root, "recomputed");
      const again2 = await writeLabelStream({ context: { ...contextFor(labels), known_synthetic: known, behaviour_policy_id: POLICY }, clients: [new Worker(), new Worker()], episodes: withGames, dir: recomputed.labels, path: recomputed.labelsPath, max_missing_percent: 100 });
      expect(text(recomputed.labelsPath)).toBe(text(single.labelsPath));
      expect(again2.totals.recomputed).toBe(withGames.length);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }, 1_800_000);

  it("requires labels_dir up front, with a clear error, not a fallback that fails later", async () => {
    const root = tmp();
    try {
      const dirs = dirsFor(root, "no-dir");
      await expect(playGeneration({ ...labelled(labelsFor(400_000), dirs), labels_dir: undefined })).rejects.toThrow("labels need labels_dir");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }, 300_000);

  it("is not labelled before its start generation, and a generation without a spill directory is refused", async () => {
    const root = tmp();
    try {
      const dirs = dirsFor(root, "early");
      const early = await playGeneration(labelled(labelsFor(400_000), dirs, { generation: GENERATION - 1 }));
      expect(early.labels).toBeUndefined();
      const { episodes_dir: _episodes, stream_path: _stream, ...memory } = labelled(labelsFor(400_000), dirsFor(root, "mem"));
      await expect(playGeneration(memory)).rejects.toThrow(GenerationError);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }, 900_000);
});

describe("every reason a selected decision has no label", () => {
  const first = async () => {
    const made = await episodeOf(battles[0] as RandomBattleFile);
    return made;
  };
  const reasons = (found: EpisodeLabels): string[] => found.decisions.map((d) => ("refused" in d ? d.refused : "label"));

  it("a refused belief is belief_refused", async () => {
    const { episode, hash } = await first();
    const refusing: WorldSamplePort = { ...portFor(), open: () => ({ kind: "refused", code: "evidence_has_no_support" }) };
    const found = await labelEpisode(contextFor(labelsFor(1_000_000, { setup: setupFor(refusing, { config: label }) })), episode, hash, 0);
    expect(found.decisions.length).toBeGreaterThan(0);
    expect(new Set(reasons(found))).toEqual(new Set(["belief_refused"]));
  }, 300_000);

  it("a world dropped by the reply model is world_dropped, not a label over fewer worlds", async () => {
    const { episode, hash } = await first();
    const setup = labelsFor(1_000_000);
    const real = createReplyModel({ engine, vocabulary, config: setup.setup.damage_config, mechanics: setup.setup.mechanics, data_class: "synthetic" });
    let calls = 0;
    // the first world of every fourth reply fails: three of the four worlds stay, which is above the minimum
    const failing: ReplyModel = {
      reply: (world: never, memo?: never) => {
        if (calls++ % 4 === 0) throw new ReplyModelError("no reply");
        return real.reply(world, memo);
      },
      counts: () => real.counts(),
    } as unknown as ReplyModel;
    const found = await labelEpisode({ ...contextFor(labelsFor(1_000_000)), reply_model: failing }, episode, hash, 0);
    expect(reasons(found)).toContain("world_dropped");
    expect(reasons(found)).not.toContain("label");
  }, 300_000);

  it("a searched set of one candidate is too_few_searched, and one with no prior mass is no_prior_mass", async () => {
    const { episode, hash } = await first();
    const one = { ...label, top_k_rule: "prior-top-k" as const, top_k: 1 };
    const few = await labelEpisode(contextFor(labelsFor(1_000_000, { setup: setupFor(portFor(), { config: one }) })), episode, hash, 0);
    expect(reasons(few)).toContain("too_few_searched");
    class Massless extends Worker {
      override async evaluate(battle: string, query: Parameters<Worker["evaluate"]>[1]): ReturnType<Worker["evaluate"]> {
        const found = await super.evaluate(battle, query);
        return { ...found, prior_ppm: found.prior_ppm.map(() => 0) } as Awaited<ReturnType<Worker["evaluate"]>>;
      }
    }
    const none = await labelEpisode(contextFor(labelsFor(1_000_000), new Massless()), episode, hash, 0);
    expect(reasons(none)).toContain("no_prior_mass");
    expect(none.decisions.every((d) => "refused" in d)).toBe(true);
  }, 300_000);

  it("scratch files: an unreadable Episode leaves none, a stale one is not used, a stopped process writes none, a present one is kept", async () => {
    const root = tmp();
    try {
      const { episode, hash } = await episodeOf(battles[0] as RandomBattleFile);
      const episodes = join(root, "episodes");
      mkdirSync(episodes);
      const path = join(episodes, `${hash}.json`);
      (await import("node:fs")).writeFileSync(path, `${jcs(episode as unknown as Json)}\n`);
      const labels = labelsFor(1_000_000);
      const context = { ...contextFor(labels), known_synthetic: syntheticTeamHashes([teamHash(episode.team.team as never), teamHash(battles[0]?.teams.p2 as never)]), behaviour_policy_id: POLICY };
      const dir = join(root, "scratch");
      const identity = labelPolicy(labels, context.client).identity_hash;
      expect(await writeLabelFiles(context, [{ hash: "0".repeat(64), path: join(root, "missing.json"), game: 0 }], dir)).toBe(0);
      expect(readEpisodeLabels(dir, "0".repeat(64), identity)).toBeUndefined();
      // the error class is left for the writer, which names it when too many files are missing
      expect(readLabelError(dir, "0".repeat(64))).toBe("Error");
      expect(readLabelError(dir, "1".repeat(64))).toBeUndefined();
      await expect(writeLabelStream({ context, episodes: [{ hash: "0".repeat(64), path: join(root, "missing.json"), game: 0 }], dir, path: join(root, "e.jsonl") })).rejects.toThrow("first labeller error: Error");
      expect(await writeLabelFiles(context, [{ hash, path, game: 0 }], dir, () => true)).toBe(0);
      expect(readEpisodeLabels(dir, hash, identity)).toBeUndefined();
      expect(await writeLabelFiles(context, [{ hash, path, game: 0 }], dir)).toBe(1);
      expect(readEpisodeLabels(dir, hash, identity)?.decisions.length).toBeGreaterThan(0);
      expect(readEpisodeLabels(dir, hash, "f".repeat(64))).toBeUndefined();
      expect(await writeLabelFiles(context, [{ hash, path, game: 0 }], dir)).toBe(0); // kept, not made again
      // a label for a decision the Episode does not have is refused by the stream writer
      writeEpisodeLabels(dir, { format: 1, episode_hash: hash, search_identity_hash: identity, decisions: [{ request_seq: 9999, class: "rest", label: { prior_ppm: [1, 1], entries: [{ candidate: 0, q_ppm: 1 }, { candidate: 1, q_ppm: 2 }] } }] });
      await expect(writeLabelStream({ context, episodes: [{ hash, path, game: 0 }], dir, path: join(root, "bad.jsonl") })).rejects.toThrow("names a decision the Episode does not have");
      // an Episode stored under a hash that is not its own
      const other = join(episodes, `${"1".repeat(64)}.json`);
      (await import("node:fs")).copyFileSync(path, other);
      await expect(writeLabelStream({ context, episodes: [{ hash: "1".repeat(64), path: other, game: 0 }], dir: join(root, "s2"), path: join(root, "bad2.jsonl"), max_missing_percent: 100 })).rejects.toThrow("not its own");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }, 600_000);
});

describe("a pair that reads candidate features", () => {
  it("labels with the rows of the root query, under a search policy that names the worker's feature set", async () => {
    const tinyConfig = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { model: { encoder: { options?: unknown } } };
    const withSet = (hash: string) => {
      const copy = structuredClone(tinyConfig);
      copy.model.encoder.options = { candidate_features: { feature_set: "probe-damage/1", feature_set_hash: hash } };
      return readTrainingRunConfig(copy);
    };
    let hash = "";
    try {
      featureSetup(engine, withSet("0".repeat(64)));
    } catch (error) {
      hash = /computes as ([0-9a-f]{64})/.exec((error as Error).message)?.[1] ?? "";
    }
    const features = featureSetup(engine, withSet(hash));
    expect(features).toBeDefined();
    class FeatureWorker extends Worker {
      override readonly featureSet = { feature_set: "probe-damage/1", feature_set_hash: hash };
    }
    const worker = new FeatureWorker();
    const labels = labelsFor(1_000_000, { features: features?.configs as never });
    const plainIdentity = labelPolicy(labelsFor(1), new Worker()).policy_id;
    expect(labelPolicy(labels, worker).policy_id).not.toBe(plainIdentity);
    const { episode, hash: episodeId } = await episodeOf(battles[0] as RandomBattleFile);
    const found = await labelEpisode(contextFor(labels, worker), episode, episodeId, 0);
    expect(found.decisions.length).toBeGreaterThan(0);
    expect(worker.queries.filter((q) => q.kind === "evaluate" && q.rows).length).toBeGreaterThan(0);
    expect(worker.queries.filter((q) => q.kind === "evaluate").every((q) => q.rows)).toBe(true);
  }, 600_000);
});

describe("labelSetup", () => {
  const pairIds = { preview: pair.preview, battle: pair.battle };
  it("sets the labelling up from the run config through the composition root, and refuses what does not agree", () => {
    const setup = labelSetup(engine, config, pairIds, progressSearchSetup);
    expect(setup?.share_ppm).toBe(400_000);
    expect(setup?.start_generation).toBe(GENERATION);
    expect(setup?.search_config_hash).toBe(searchConfigHash(label));
    expect(setup?.min_spread_ppm).toBe(config.ppo.search_distill?.min_spread_ppm);
    expect(() => labelSetup(engine, config, pairIds)).toThrow("no search setup factory");
    const wrong = structuredClone(raw) as unknown as { generation: { search_labels: { search_config_hash: string } } };
    wrong.generation.search_labels.search_config_hash = "0".repeat(64);
    expect(() => labelSetup(engine, readTrainingRunConfig(wrong), pairIds, progressSearchSetup)).toThrow("which this checkout computes as");
    const plain = structuredClone(raw) as unknown as { generation: Record<string, unknown>; ppo: Record<string, unknown> };
    delete plain.generation.search_labels;
    delete plain.ppo.search_distill;
    expect(labelSetup(engine, readTrainingRunConfig(plain), pairIds, progressSearchSetup)).toBeUndefined();
  });
});

