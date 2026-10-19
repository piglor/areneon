import { beforeAll, describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { goldenBattle, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import {
  type BattleRecord,
  type CollectionInput,
  type Episode,
  episodeFromRecord,
  episodeHash,
  evaluationRunManifest,
  loadEngine,
  loadTrainingSet,
  nodeSha256,
  readEvaluationRunManifest,
  readTrainingSet,
  stampTeam,
  teamHash,
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
  loadTrainingSet({ episodes, evaluation_manifests, known_synthetic: KNOWN });
const refusal = (promise: Promise<unknown>, message: string | RegExp) => expect(promise).rejects.toThrow(message);

describe("loadTrainingSet", () => {
  it("writes one TrainingExample per decision, with no provenance, and a TrainingSet that matches the stream", async () => {
    const { examples, stream, set } = await load([p1]);
    expect(examples.map((e) => e.choice)).toEqual(p1.decisions.map((d) => d.choice));
    expect(examples.map((e) => e.observation.identity.request_seq)).toEqual(p1.decisions.map((d) => d.identity.request_seq));
    expect(examples[0]?.observation.kind).toBe("preview");
    const hash = await episodeHash(p1, nodeSha256);
    for (const example of examples) {
      expect(Object.keys(example).sort()).toEqual(["choice", "episode_hash", "observation", "schema_version"]);
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
      examples: examples.length,
      stream_sha256: await nodeSha256(new TextEncoder().encode(stream)),
    });
    expect(readTrainingSet(set)).toEqual(set);
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
    const sets = (syntheticTeam() as PrivateTeamSpec["team"]).map((s, i) => (i === 0 ? { ...s, evs: { ...s.evs, hp: 8 } } : s));
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
      (e.team.team[0] as { evs: { hp: number } }).evs.hp = 8;
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

  it("refuse a TrainingSet that is unsorted, repeated or off the schema", async () => {
    const { set } = await load([p1, p2]);
    const [a, b] = set.episode_hashes as [string, string];
    expect(() => readTrainingSet({ ...set, episode_hashes: [b, a] })).toThrow("episode_hashes must be sorted and unique");
    expect(() => readTrainingSet({ ...set, evaluation_manifest_hashes: [a, a] })).toThrow("evaluation_manifest_hashes must be sorted and unique");
    expect(() => readTrainingSet({ ...set, examples: 0 })).toThrow("invalid TrainingSet");
  });
});
