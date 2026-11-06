import { beforeAll, describe, expect, it } from "vitest";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { firstLegal, goldenBattle, GOLDEN_FORMAT, syntheticFixtureHashes, syntheticTeam } from "../src/golden.ts";
import {
  type BattleRecord,
  type CollectionInput,
  type Episode,
  episodeFromRecord,
  episodeHash,
  evaluationRunManifest,
  loadEngine,
  loadPrivateTrainingSet,
  loadTrainingSet,
  nodeSha256,
  PRIVATE_LOADER_REFUSALS,
  type PrivateLoaderRefusal,
  PrivateTrainingLoaderError,
  readTrainingSet,
  stampTeam,
  teamHash,
  TrainingLoaderError,
} from "../src/index.ts";
import type { PokemonSet } from "../src/engine.ts";
import { runBattle } from "../src/runner.ts";
import { canonicalTeamHash } from "../src/team-records.ts";
import { expectNoLeak, findLeaks, makeCanaryWorld, observe } from "./support/private-canary.ts";

const engine = loadEngine();
const KNOWN = syntheticFixtureHashes(engine);
const RUN = "private-run-1";
const POLICY = "first-legal";
const synthetic = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], KNOWN);
const league = (record: BattleRecord, run_id = RUN): CollectionInput => ({ kind: "league", run_id, seed: record.seed, opponent_policy_id: "first-legal" });
const OPPONENT = "a".repeat(64);

let record: BattleRecord;
let p1: Episode;
let p2: Episode;
const write = (player: "p1" | "p2", collection: CollectionInput, class_floor?: "private") =>
  episodeFromRecord(record, player, { team: synthetic, opponent_team: synthetic, known_synthetic: KNOWN, behaviour_policy_id: POLICY, collection, ...(class_floor && { class_floor }) });
const item = (episode: unknown, opponent_team_hash = OPPONENT) => ({ episode, opponent_team_hash });
const load = (episodes: ReturnType<typeof item>[], extra: { manifests?: unknown[]; holdout?: string[]; run_key?: string; policy?: string } = {}) =>
  loadPrivateTrainingSet({
    episodes,
    evaluation_manifests: extra.manifests ?? [],
    known_synthetic: KNOWN,
    behaviour_policy_id: extra.policy ?? POLICY,
    run_key: extra.run_key ?? RUN,
    holdout_team_hashes: new Set(extra.holdout ?? []),
  });
const refused = async (promise: Promise<unknown>, code: PrivateLoaderRefusal, position: number | null = null): Promise<void> => {
  const error = await promise.then(
    () => null,
    (e: unknown) => e,
  );
  expect(error).toBeInstanceOf(PrivateTrainingLoaderError);
  const refusal = error as PrivateTrainingLoaderError;
  expect(refusal.code).toBe(code);
  expect(refusal.position).toBe(position);
  expect(refusal.message).toBe(position === null ? PRIVATE_LOADER_REFUSALS[code] : `${PRIVATE_LOADER_REFUSALS[code]} (position ${position})`);
};

beforeAll(async () => {
  record = await goldenBattle(engine);
  p1 = await write("p1", league(record), "private");
  p2 = await write("p2", league(record), "private");
});

describe("the class floor (ADR-015 erratum E1 item 5)", () => {
  it("makes a private-run Episode private although both teams are synthetic, and never lowers a class", async () => {
    expect(p1.data_class).toBe("private");
    expect(p1.rights).toBe("owner-private");
    expect(p1.team.data_class).toBe("synthetic");
    expect((await write("p1", league(record))).data_class).toBe("synthetic");
  });
});

describe("loadTrainingSet stays synthetic-only", () => {
  it("refuses every private Episode, a floor-raised one included", async () => {
    const plain = (episodes: unknown[]) => loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: KNOWN, behaviour_policy_id: POLICY });
    await expect(plain([p1])).rejects.toThrow("Episode 0 carries a class floor");
    await expect(plain([p2])).rejects.toThrow("carries a class floor");
    const error = await plain([p1]).then(() => null, (e: unknown) => e as TrainingLoaderError);
    expect(error?.code).toBe("class_floor");
    // The case the floor was not stored for: a floor-raised Episode relabelled synthetic is refused too.
    const relabelled = { ...structuredClone(p1), data_class: "synthetic", rights: "cc0-synthetic" };
    await expect(plain([relabelled])).rejects.toThrow("Episode 0 is refused: the Episode is labelled synthetic, but its collection record's class floor is private");
  });
});

describe("loadPrivateTrainingSet", () => {
  it("writes a private, learner-ready TrainingSet whose stream matches", async () => {
    const { examples, stream, set } = await load([item(p2), item(p1)]);
    expect(set.data_class).toBe("private");
    expect(set.learner_ready).toBe(true);
    expect(set.behaviour_policy_id).toBe(POLICY);
    expect(set.episode_hashes).toEqual([await episodeHash(p1, nodeSha256), await episodeHash(p2, nodeSha256)].sort());
    expect(examples.length).toBe(p1.decisions.length + p2.decisions.length);
    expect(examples.every((e) => e.observation.data_class === "private")).toBe(true);
    expect(set.stream_sha256).toBe(await nodeSha256(new TextEncoder().encode(stream)));
    expect(set.examples).toBe(examples.length);
    expect(readTrainingSet(set)).toEqual(set);
  });

  it("accepts a hold-out set that holds neither team", async () => {
    const { set } = await load([item(p1)], { holdout: ["b".repeat(64), teamHash([{ species: "x" }])] });
    expect(set.episode_hashes).toHaveLength(1);
  });

  const cases: [string, () => Promise<unknown>, PrivateLoaderRefusal, number | null][] = [];
  const add = (name: string, run: () => Promise<unknown>, code: PrivateLoaderRefusal, position: number | null = 0): void => void cases.push([name, run, code, position]);
  add("a synthetic Episode", async () => load([item(await write("p1", league(record)))]), "not_private");
  add("an evaluation Episode", async () => load([item(await write("p1", { kind: "evaluation", run_id: RUN, sealed: true, seed: record.seed }, "private"))]), "evaluation_episode");
  add("a sparring Episode", async () => load([item(await write("p1", { kind: "sparring", session_id: "s-1" }, "private"))]), "sparring_episode");
  add("no collection record", async () => load([item(await episodeFromRecord(record, "p1", { team: synthetic, opponent_team: synthetic, known_synthetic: KNOWN, behaviour_policy_id: POLICY, class_floor: "private" }))]), "no_collection");
  add("another run", async () => load([item(p1)], { run_key: "private-run-2" }), "other_run");
  add("another policy", async () => load([item(p1)], { policy: "policy:other" }), "other_policy");
  add("an Episode the reader refuses", async () => load([item({ ...p1, data_class: "synthetic" })]), "episode_invalid");
  add("junk instead of an Episode", async () => load([item(42)]), "episode_invalid");
  add("a bad opponent hash", async () => load([item(p1, "not-a-hash")]), "opponent_hash_invalid");
  add("a missing opponent hash", async () => load([{ episode: p1 } as never]), "opponent_hash_invalid");
  add("a repeated Episode", async () => load([item(p1), item(p1)]), "repeated_episode", 1);
  add("an own team in the hold-out set", async () => load([item(p1)], { holdout: [teamHash(syntheticTeam())] }), "holdout_team");
  add("an own team in the hold-out set by its canonical hash", async () => load([item(p1)], { holdout: [canonicalTeamHash(syntheticTeam())] }), "holdout_team");
  add("an opponent team in the hold-out set", async () => load([item(p1, "c".repeat(64))], { holdout: ["c".repeat(64)] }), "holdout_team");
  add("a listed hash", async () => {
    const manifest = await evaluationRunManifest({ run_id: "eval-1", sealed: true, episodes: [await write("p1", { kind: "evaluation", run_id: "eval-1", sealed: true, seed: record.seed }, "private")] });
    const listed = { ...manifest, episode_hashes: [await episodeHash(p1, nodeSha256)] };
    return load([item(p1)], { manifests: [listed] });
  }, "listed_by_manifest");
  add("an invalid manifest", async () => load([item(p1)], { manifests: [{ nope: 1 }] }), "manifest_invalid");
  add("a manifest given twice", async () => {
    const m = { schema_version: 1, run_id: "eval-1", sealed: true, episode_hashes: ["d".repeat(64)] };
    return load([item(p1)], { manifests: [m, m] });
  }, "manifest_repeated", 1);
  add("no Episode", async () => load([]), "no_episode", null);
  add("a bad run key", async () => load([item(p1)], { run_key: "a b" }), "run_key_invalid", null);

  it.each(cases)("refuses %s with a fixed message", async (_name, run, code, position) => {
    await refused(run(), code, position);
  });

  it("refuses a stream too long for one string", async () => {
    const longest = Math.max(...(await load([item(p1)])).stream.split("\n").map((line) => line.length)) + 1;
    await refused(
      loadPrivateTrainingSet({ episodes: [item(p1)], evaluation_manifests: [], known_synthetic: KNOWN, behaviour_policy_id: POLICY, run_key: RUN, holdout_team_hashes: new Set() }, { max_string_chars: longest }),
      "stream_too_long",
    );
  });

  it("has a refusal for every code and nothing else", () => {
    const covered = new Set(cases.map(([, , code]) => code));
    covered.add("stream_too_long");
    expect([...covered].sort()).toEqual(Object.keys(PRIVATE_LOADER_REFUSALS).sort());
  });

  it("refuses an Episode whose class was flipped below its stored floor (the label flipped in code)", async () => {
    const down = { ...structuredClone(p1), data_class: "synthetic", rights: "cc0-synthetic" };
    await refused(load([item(down)]), "episode_invalid", 0);
    // Without a stored floor (an Episode of an older writer) the loader's own class rule refuses a synthetic one.
    const unfloored = await write("p1", league(record));
    await refused(load([item(unfloored)]), "not_private", 0);
  });
});

describe("the canary: no team content in the loader's errors or output", () => {
  it("shows none in a TrainingSet, in any refusal, or on the console", async () => {
    const world = makeCanaryWorld(engine, 2, { legal: true });
    try {
      const a = world.teams[0] as { sets: PokemonSet[] };
      const b = world.teams[1] as { sets: PokemonSet[] };
      const result = await runBattle({
        engine,
        formatId: GOLDEN_FORMAT,
        seed: "sodium,0f0e0d0c0b0a09080706050403020100",
        teams: { p1: a.sets, p2: b.sets },
        actors: { p1: firstLegal, p2: firstLegal },
      });
      const none = new Set<string>();
      const teamOf = (sets: unknown) => stampTeam(sets as PrivateTeamSpec["team"], none);
      const own = teamOf(a.sets);
      const rival = teamOf(b.sets);
      expect(own.data_class).toBe("private");
      const episode = await episodeFromRecord(result, "p1", { team: own, opponent_team: rival, known_synthetic: KNOWN, behaviour_policy_id: POLICY, collection: league(result), class_floor: "private" });
      const holdout = [teamHash(a.sets), canonicalTeamHash(b.sets)];
      const all = (extra: Parameters<typeof load>[1]) => load([item(episode, teamHash(b.sets))], extra);

      const seen = await observe(world, async () => {
        const surfaces: Record<string, string> = {};
        const good = await all({});
        surfaces.training_set = JSON.stringify(good.set);
        for (const [name, extra] of Object.entries({
          holdout: { holdout },
          other_run: { run_key: "private-run-9" },
          other_policy: { policy: "policy:other" },
          junk: { manifests: [{ nope: 1 }] },
        })) {
          const error = await all(extra).then(() => null, (e: unknown) => e as Error);
          expect(error, name).toBeInstanceOf(PrivateTrainingLoaderError);
          surfaces[`refusal_${name}`] = `${(error as Error).stack}\n${JSON.stringify(error, Object.getOwnPropertyNames(error))}`;
        }
        const unreadable = await load([item({ ...episode, team: { ...episode.team, team: [] } })]).then(() => null, (e: unknown) => e as Error);
        expect(unreadable).toBeInstanceOf(PrivateTrainingLoaderError);
        surfaces.refusal_unreadable = `${(unreadable as Error).stack}\n${JSON.stringify(unreadable, Object.getOwnPropertyNames(unreadable))}`;
        return { surfaces };
      });
      expect(seen.thrown).toBeNull();
      expectNoLeak(world, seen);

      // Negative controls: the scan flags a leaking refusal, and the stream (the private data itself) does hold the markers.
      const marker = world.markers[0] as string;
      expect(findLeaks(world, { ...seen, surfaces: { refusal: `bad ${marker}` } }).length).toBeGreaterThan(0);
      const { stream } = await all({});
      expect(world.markers.some((m) => stream.includes(m))).toBe(true);
    } finally {
      world.cleanup();
    }
  }, 120_000);
});
