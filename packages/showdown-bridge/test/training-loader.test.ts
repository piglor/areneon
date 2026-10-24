import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import {
  type BattleRecord,
  type CollectionInput,
  type Episode,
  episodeFromRecord,
  episodeHash,
  evaluationRunManifest,
  decodeRequest,
  enumerateCandidates,
  loadEngine,
  loadTrainingSet,
  nodeSha256,
  readEvaluationRunManifest,
  readTrainingSet,
  stampTeam,
  teamHash,
  trainingExample,
  type TrainingExample,
  writeStoredTrainingStream,
  writeTrainingStream,
  TrainingLoaderError,
} from "../src/index.ts";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";

const KNOWN = syntheticFixtureHashes(loadEngine());
const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], KNOWN);
const league = (record: BattleRecord) => ({ kind: "league", run_id: "league-1", seed: record.seed, opponent_policy_id: "first-legal" }) as const;
const manifest = (episode_hashes: string[], run_id = "eval-1", sealed = true) => ({ schema_version: 1, run_id, sealed, episode_hashes });
const mutate = (episode: Episode, change: (copy: Episode) => void): Episode => {
  const copy = structuredClone(episode);
  change(copy);
  return copy;
};
// A stored Episode as a hostile writer could have changed it: outside the types.
const forged = (value: unknown): never => value as never;

const POLICY = "first-legal";
let record: BattleRecord;
let p1: Episode;
let p2: Episode;
const write = (player: "p1" | "p2", collection: CollectionInput) =>
  episodeFromRecord(record, player, { team, opponent_team: team, known_synthetic: KNOWN, behaviour_policy_id: "first-legal", collection });
beforeAll(async () => {
  record = await goldenBattle(loadEngine());
  p1 = await write("p1", league(record));
  p2 = await write("p2", league(record));
});

const load = (episodes: unknown[], evaluation_manifests: unknown[] = []) =>
  loadTrainingSet({ episodes, evaluation_manifests, known_synthetic: KNOWN, behaviour_policy_id: POLICY });
const refusal = (promise: Promise<unknown>, message: string | RegExp) => expect(promise).rejects.toThrow(message);

describe("loadTrainingSet", () => {
  it("writes one TrainingExample per decision, with no provenance, and a TrainingSet that matches the stream", async () => {
    const { examples, stream, set } = await load([p1]);
    expect(examples.map((e) => e.choice)).toEqual(p1.decisions.map((d) => d.choice));
    expect(examples.map((e) => e.observation.identity.request_seq)).toEqual(p1.decisions.map((d) => d.identity.request_seq));
    expect(examples[0]?.observation.kind).toBe("preview");
    const hash = await episodeHash(p1, nodeSha256);
    for (const example of examples) {
      expect(Object.keys(example).sort()).toEqual(["candidates", "choice", "episode_hash", "observation", "outcome", "schema_version"]);
      expect(example.episode_hash).toBe(hash);
      expect(JSON.stringify(example)).not.toContain(record.seed as string);
    }
    expect(stream.split("\n").slice(0, -1).map((l) => JSON.parse(l))).toEqual(examples);
    expect(stream.endsWith("\n")).toBe(true);
    expect(set).toEqual({
      schema_version: 1,
      data_class: "synthetic",
      episode_hashes: [hash],
      evaluation_manifest_hashes: [],
      learner_ready: true,
      behaviour_policy_id: POLICY,
      examples: examples.length,
      stream_sha256: await nodeSha256(new TextEncoder().encode(stream)),
    });
    expect(readTrainingSet(set)).toEqual(set);
  });

  it("carries the request's candidates in canonical order, and the outcome from the player's view", async () => {
    for (const [player, episode] of [["p1", p1], ["p2", p2]] as const) {
      const { examples, set } = await load([episode]);
      expect(set.learner_ready).toBe(true);
      const outcome = record.winner === null ? "tie" : record.winner === player ? "win" : "loss";
      for (const example of examples) {
        expect(example.outcome).toBe(outcome);
        const line = example.observation.log.at(-1) as string;
        expect(example.candidates).toEqual(enumerateCandidates(decodeRequest(line.slice("|request|".length))).candidates);
        expect(example.candidates?.map((c) => c.choice)).toContain(example.choice);
        // The CloneJudge's record of the same request agrees on which candidates are pending, but is not the source.
        const legal = (record.requests[player].find((r) => r.request_seq === example.observation.identity.request_seq) as { legal: { choices: string[]; pending: string[] } }).legal;
        expect(example.candidates?.filter((c) => !c.pending).map((c) => c.choice).sort()).toEqual([...legal.choices].sort());
        expect(example.candidates?.filter((c) => c.pending).map((c) => c.choice).sort()).toEqual([...legal.pending].sort());
      }
      expect(JSON.stringify(examples)).not.toContain("hidden_verdicts");
    }
  });

  it("scores a tied game as a tie", async () => {
    const tied = mutate(p1, (e) => {
      e.channel = e.channel.map((line) => (line.startsWith("|win|") ? "|tie" : line)) as Episode["channel"];
      e.result.winner = null;
    });
    expect(new Set((await load([tied])).examples.map((e) => e.outcome))).toEqual(new Set(["tie"]));
  });

  it("refuses an Episode of another policy, so only the trained policy's seats train", async () => {
    const other = await episodeFromRecord(record, "p2", {
      team,
      opponent_team: team,
      known_synthetic: KNOWN,
      behaviour_policy_id: "policy:frozen",
      collection: league(record),
    });
    await refusal(load([p1, other]), "Episode 1 was played by policy:frozen, not first-legal: only the trained policy's own seats train");
    const { set } = await loadTrainingSet({ episodes: [other], evaluation_manifests: [], known_synthetic: KNOWN, behaviour_policy_id: "policy:frozen" });
    expect(set.behaviour_policy_id).toBe("policy:frozen");
  });

  it("orders Episodes by hash, whatever order they are given in", async () => {
    const forward = await load([p1, p2]);
    const backward = await load([p2, p1]);
    expect(backward.stream).toBe(forward.stream);
    expect(forward.set.episode_hashes).toEqual([...forward.set.episode_hashes].sort());
    expect(forward.set.episode_hashes).toHaveLength(2);
  });

  it("records the manifests it checked, sorted, and leaves the Episodes unchanged", async () => {
    const before = jcs(p1 as unknown as Json);
    const { set } = await load([p1], [manifest(["f".repeat(64)]), manifest(["e".repeat(64)], "eval-2", false)]);
    expect(set.evaluation_manifest_hashes).toHaveLength(2);
    expect(set.evaluation_manifest_hashes).toEqual([...set.evaluation_manifest_hashes].sort());
    expect(jcs(p1 as unknown as Json)).toBe(before);
  });

  it("refuses an evaluation Episode, sealed or not, by its label", async () => {
    for (const sealed of [true, false]) {
      const evaluation = await write("p1", { kind: "evaluation", run_id: "eval-1", sealed, seed: record.seed });
      await refusal(load([evaluation]), "Episode 0 was collected as evaluation: only a league Episode trains");
    }
  });

  it("refuses a sparring Episode, which is private", async () => {
    const sparring = mutate(p1, (e) => {
      e.data_class = "private";
      e.rights = "owner-private";
      e.provenance.collection = forged({ kind: "sparring", session_id: "s-1" });
    });
    await refusal(load([sparring]), "Episode 0 was collected as sparring: only a league Episode trains");
  });

  it("refuses an Episode with no collection record", async () => {
    const plain = await episodeFromRecord(record, "p1", { team, opponent_team: team, known_synthetic: KNOWN, behaviour_policy_id: "first-legal" });
    await refusal(load([plain]), "Episode 0 has no collection record");
  });

  it("refuses private data, even in a league Episode", async () => {
    const sets = (syntheticTeam() as PrivateTeamSpec["team"]).map((s, i) => (i === 0 ? { ...s, ivs: { ...s.ivs, atk: 30 } } : s));
    const own = stampTeam(sets as PrivateTeamSpec["team"], KNOWN);
    expect(own.data_class).toBe("private");
    const privateRecord = { ...record, team_hashes: { ...record.team_hashes } };
    privateRecord.team_hashes.p1 = teamHash(own.team);
    const episode = await episodeFromRecord(privateRecord, "p1", {
      team: own,
      opponent_team: team,
      known_synthetic: KNOWN,
      behaviour_policy_id: "first-legal",
      collection: league(record),
    });
    expect(episode.data_class).toBe("private");
    await refusal(load([episode]), "Episode 0 is private with owner-private rights: only synthetic, cc0-synthetic data is loaded");
  });

  it("refuses an Episode relabelled to a lower class, through readEpisode", async () => {
    const relabelled = mutate(p1, (e) => {
      (e.provenance.collection as { opponent_team_data_class: string }).opponent_team_data_class = "private";
    });
    await refusal(load([relabelled]), /Episode 0 is refused: .*opponent's team is private: the class must be private/);
    const sparringRelabelled = mutate(p1, (e) => {
      e.provenance.collection = forged({ kind: "sparring", session_id: "s-1" });
    });
    await refusal(load([sparringRelabelled]), "Episode 0 is refused: a sparring Episode is private");
    const unknownTeam = mutate(p1, (e) => {
      // An IV changes the team's hash, not the stats the engine shows.
      (e.team.team[0] as { ivs: { atk: number } }).ivs.atk = 30;
    });
    await refusal(load([unknownTeam]), "Episode 0 is refused: a team labelled synthetic is not a known synthetic team");
  });

  it("refuses an Episode that an EvaluationRunManifest lists, which is an evaluation Episode under a wrong label", async () => {
    const hash = await episodeHash(p1, nodeSha256);
    await refusal(
      load([p2, p1], [manifest([hash])]),
      `Episode 1 (${hash}) is listed in an EvaluationRunManifest: evaluation games never train`,
    );
    // The label of a relabelled copy changes its bytes, and so its hash: only the storage partition catches that copy.
    const evaluation = await write("p1", { kind: "evaluation", run_id: "eval-1", sealed: true, seed: record.seed });
    const relabelled = mutate(evaluation, (e) => {
      e.provenance.collection = { ...league(record), opponent_team_data_class: "synthetic" };
    });
    const listed = manifest([await episodeHash(evaluation, nodeSha256)]);
    const loaded = await load([relabelled], [listed]);
    expect(loaded.set.episode_hashes).toEqual([await episodeHash(relabelled, nodeSha256)]);
    await refusal(load([evaluation], [listed]), "was collected as evaluation");
  });

  it("refuses a repeated Episode or manifest, an invalid manifest, and an empty load", async () => {
    await refusal(load([p1, p1]), "is given twice");
    const same = manifest(["a".repeat(64)]);
    await refusal(load([p1], [same, structuredClone(same)]), "the EvaluationRunManifest of run eval-1 is given twice");
    await refusal(load([p1], [{ ...same, sealed: "yes" }]), "invalid EvaluationRunManifest");
    await refusal(load([p1], [manifest(["b".repeat(64), "a".repeat(64)])]), "sorted and unique");
    await refusal(load([]), "no Episode to load");
    await refusal(load([{ schema_version: 2 }]), /Episode 0 is refused: no reader/);
  });

  it("keeps the cause of a refusal", async () => {
    const error = await load([{ schema_version: 2 }]).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TrainingLoaderError);
    expect((error as Error).cause).toBeInstanceOf(Error);
  });
});

describe("evaluationRunManifest", () => {
  const evaluation = (player: "p1" | "p2", run_id = "eval-1", sealed = true) =>
    write(player, { kind: "evaluation", run_id, sealed, seed: record.seed });

  it("lists the run's Episode hashes sorted and unique, and nothing else, even for a sealed run", async () => {
    const [a, b] = [await evaluation("p1"), await evaluation("p2")];
    const forward = await evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [a, b] });
    const backward = await evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [b, a] });
    expect(backward).toEqual(forward);
    expect(forward.episode_hashes).toEqual([await episodeHash(a, nodeSha256), await episodeHash(b, nodeSha256)].sort());
    expect(Object.keys(forward).sort()).toEqual(["episode_hashes", "run_id", "schema_version", "sealed"]);
    expect((await evaluationRunManifest({ run_id: "eval-2", sealed: false, episodes: [] })).episode_hashes).toEqual([]);
  });

  it("makes the loader refuse the run's Episodes", async () => {
    const a = await evaluation("p1");
    const m = await evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [a] });
    const asLeague = await write("p1", league(record));
    await refusal(load([asLeague], [{ ...m, episode_hashes: [await episodeHash(asLeague, nodeSha256)] }]), "evaluation games never train");
  });

  it("refuses a repeated Episode, a league one, and one of another run or sealed flag", async () => {
    const a = await evaluation("p1");
    await refusal(evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [a, a] }), "is given twice");
    await refusal(evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [p1] }), "Episode 0 is not an evaluation Episode of run eval-1");
    await refusal(evaluationRunManifest({ run_id: "eval-9", sealed: true, episodes: [a] }), "run eval-9");
    await refusal(evaluationRunManifest({ run_id: "eval-1", sealed: false, episodes: [a] }), "sealed false");
  });
});

describe("readEvaluationRunManifest and readTrainingSet", () => {
  it("read a valid manifest as a copy, empty or not", () => {
    const m = manifest(["a".repeat(64), "b".repeat(64)]);
    expect(readEvaluationRunManifest(m)).toEqual(m);
    expect(readEvaluationRunManifest(m)).not.toBe(m);
    expect(readEvaluationRunManifest(manifest([], "eval-2", false)).episode_hashes).toEqual([]);
  });

  it("refuse unsorted and repeated hashes, which the schema cannot", () => {
    for (const hashes of [["b".repeat(64), "a".repeat(64)], ["a".repeat(64), "a".repeat(64)]]) {
      expect(() => readEvaluationRunManifest(manifest(hashes, "eval-9"))).toThrow(/run eval-9 must list its Episode hashes sorted and unique/);
    }
  });

  it("refuse a learner_ready TrainingSet that names no policy, and a learner_ready that is not true", async () => {
    const { set } = await load([p1]);
    expect(readTrainingSet(set)).toEqual(set);
    const { behaviour_policy_id: _dropped, ...unnamed } = set;
    expect(() => readTrainingSet(unnamed)).toThrow("a learner_ready TrainingSet must name its behaviour_policy_id");
    expect(() => readTrainingSet({ ...set, learner_ready: false })).toThrow("invalid TrainingSet");
    expect(() => readTrainingSet({ ...set, behaviour_policy_id: "" })).toThrow("invalid TrainingSet");
    // The legacy shape stays readable: neither property.
    const { learner_ready: _ready, ...legacy } = unnamed;
    expect(readTrainingSet(legacy)).toEqual(legacy);
  });

  it("refuse a TrainingSet that is unsorted, repeated or off the schema", async () => {
    const { set } = await load([p1, p2]);
    const [a, b] = set.episode_hashes as [string, string];
    expect(() => readTrainingSet({ ...set, episode_hashes: [b, a] })).toThrow("episode_hashes must be sorted and unique");
    expect(() => readTrainingSet({ ...set, evaluation_manifest_hashes: [a, a] })).toThrow("evaluation_manifest_hashes must be sorted and unique");
    expect(() => readTrainingSet({ ...set, examples: 0 })).toThrow("invalid TrainingSet");
  });
});

describe("trainingExample", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const request = (edit: (r: { [key: string]: any }) => void) => {
    const preview = structuredClone(JSON.parse((p1Observation.log.at(-1) as string).slice("|request|".length)));
    edit(preview);
    return `|request|${JSON.stringify(preview)}`;
  };
  let p1Observation: TrainingExample["observation"];
  const input = (log: string, choice = "team 1234") => ({
    observation: { ...p1Observation, log: [...p1Observation.log.slice(0, -1), log] } as unknown as TrainingExample["observation"],
    choice,
    outcome: "win" as const,
    episode_hash: "a".repeat(64),
  });
  beforeAll(async () => {
    p1Observation = (await load([p1])).examples[0]?.observation as TrainingExample["observation"];
  });

  it("builds a learner-ready example from the request that ends the log", () => {
    const example = trainingExample(input(p1Observation.log.at(-1) as string));
    expect(example.candidates).toHaveLength(360);
    expect(example.outcome).toBe("win");
  });

  it("refuses a choice that is not among the candidates", () => {
    expect(() => trainingExample(input(p1Observation.log.at(-1) as string, "team 1235 "))).toThrow(/is not among its 360 candidates/);
    expect(() => trainingExample(input(p1Observation.log.at(-1) as string, "move 1"))).toThrow(TrainingLoaderError);
  });

  it("refuses a request the decoder or the enumerator does not understand", () => {
    expect(() => trainingExample(input(request((r) => (r.unknownField = 1))))).toThrow(/has no candidates: request does not match/);
    expect(() => trainingExample(input(request((r) => (r.maxChosenTeamSize = 0))))).toThrow(/has no candidates: .*maxChosenTeamSize/);
  });

  it("refuses a request that has more candidates than N_max", () => {
    const wide = request((r) => {
      r.side.pokemon = [...r.side.pokemon, ...r.side.pokemon.slice(0, 3)];
    });
    expect(() => trainingExample(input(wide))).toThrow(/invalid TrainingExample.*maxItems/);
  });
});

describe("writeStoredTrainingStream (#1069)", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "areneon-stored-"));
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const store = async (episode: Episode, name?: string): Promise<{ hash: string; path: string }> => {
    const hash = await episodeHash(episode, nodeSha256);
    const path = join(dir, `${name ?? hash}.json`);
    writeFileSync(path, `${jcs(episode as unknown as Json)}\n`);
    return { hash, path };
  };
  const stored = (episodes: { hash: string; path: string }[], evaluation_manifests: unknown[] = []) => ({ episodes, evaluation_manifests, known_synthetic: KNOWN, behaviour_policy_id: POLICY });

  it("writes the bytes and the TrainingSet of the in-memory writer, whatever order the Episodes are given in", async () => {
    const [a, b] = [await store(p1), await store(p2)];
    const reference = join(dir, "reference.jsonl");
    const refSet = await writeTrainingStream({ episodes: [p1, p2], evaluation_manifests: [], known_synthetic: KNOWN, behaviour_policy_id: POLICY }, { path: reference });
    for (const [name, order] of [["forward", [a, b]], ["backward", [b, a]]] as const) {
      const out = join(dir, `${name}.jsonl`);
      const set = await writeStoredTrainingStream(stored([...order]), { path: out });
      expect(set).toEqual(refSet);
      expect(readFileSync(out, "utf8")).toBe(readFileSync(reference, "utf8"));
    }
  });

  it("refuses what the in-memory loader refuses: none, a repeat, a listed Episode, a bad Episode, and a file that is not the Episode its name says", async () => {
    const a = await store(p1);
    const out = join(dir, "refused.jsonl");
    await refusal(writeStoredTrainingStream(stored([]), { path: out }), "no Episode to load");
    await refusal(writeStoredTrainingStream(stored([a, a]), { path: out }), "is given twice");
    await refusal(writeStoredTrainingStream(stored([a], [manifest([a.hash])]), { path: out }), "evaluation games never train");
    const bad = await store(forged(mutate(p1, (c) => void delete (c.provenance as { collection?: unknown }).collection)) as Episode, "bad");
    await refusal(writeStoredTrainingStream(stored([{ hash: a.hash, path: bad.path }]), { path: out }), /Episode 0 is refused|no collection record/);
    await refusal(writeStoredTrainingStream(stored([{ hash: a.hash, path: join(dir, "missing.json") }]), { path: out }), /cannot be read from its file/);
    const other = await store(p2);
    await refusal(writeStoredTrainingStream(stored([{ hash: a.hash, path: other.path }]), { path: out }), "stored under a hash that is not its own");
  });
});
