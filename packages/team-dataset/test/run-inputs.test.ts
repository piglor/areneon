/* eslint-disable @typescript-eslint/no-explicit-any */
import { existsSync, linkSync, mkdirSync, readdirSync, readFileSync, symlinkSync, truncateSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { InMemoryObjectStore, ObjectStoreError } from "@areneon/object-store";
import { jcs, planGeneration, readTrainingRunConfig, runInputManifestHash } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { datasetHash, readDataset } from "../src/dataset-reader.ts";
import { loadFixtureInputs } from "../src/fixtures.ts";
import { datasetPoolDetail } from "../src/loop-pool.ts";
import {
  buildRunInputs,
  DATASET_DIR,
  fetchRunInputs,
  fileKey,
  filesPrefix,
  inputsPrefix,
  manifestKey,
  RUN_CONFIG_FILE,
  RUN_INPUT_REFUSALS,
  RunInputError,
  type RunInputRefusal,
  runInputLayout,
  uploadRunInputs,
} from "../src/run-inputs.ts";
import { COMMIT, configFor, configText, DATASET_HASH, datasetFiles, publish, removeDir, RUN_KEY, scratchDir, SOURCE } from "./run-inputs-support.ts";
import { engine, FORMAT } from "./support.ts";

const scratch = scratchDir();
afterAll(() => removeDir(scratch));
let counter = 0;
const fresh = (): string => join(scratch, `out-${counter++}`);

const head = (commit = COMMIT) => () => commit;
const fetchFrom = (store: InMemoryObjectStore, extra: Partial<Parameters<typeof fetchRunInputs>[0]> = {}, runKey = RUN_KEY) =>
  fetchRunInputs({ store, engine, runKey, outDir: fresh(), headCommit: head(), ...extra });

/** The refusal's code, after checking the failure leaves no directory and no partial sibling behind. */
async function refused(run: (out: string) => Promise<unknown>): Promise<string> {
  const out = fresh();
  try {
    await run(out);
  } catch (error) {
    expect(existsSync(out), "the output path must not exist after a refusal").toBe(false);
    expect(readdirSync(scratch).filter((name) => name.includes(".partial-")), "no partial directory is left").toEqual([]);
    expect((error as Error).message).not.toMatch(/run-inputs-|manifest\.json|\.jsonl|[0-9a-f]{64}/);
    return (error as RunInputError).code;
  }
  throw new Error("expected a refusal");
}

describe("the run-input layout", () => {
  it("admits the manifest and files under files/ and nothing else, with no overwrite and one listing prefix", () => {
    const layout = runInputLayout(RUN_KEY);
    expect(layout.allowsKey(manifestKey(RUN_KEY))).toBe(true);
    expect(layout.allowsKey(fileKey(RUN_KEY, "dataset/manifest.json"))).toBe(true);
    for (const key of [
      `${inputsPrefix(RUN_KEY)}other.json`,
      `${filesPrefix(RUN_KEY)}../manifest.json`,
      `${filesPrefix(RUN_KEY)}`,
      `${filesPrefix(RUN_KEY)}a//b`,
      `runs/other-run/inputs/manifest.json`,
      "current.json",
      "runs/",
    ]) expect(layout.allowsKey(key), key).toBe(false);
    expect(layout.allowsListPrefix(filesPrefix(RUN_KEY))).toBe(true);
    expect(layout.allowsListPrefix(inputsPrefix(RUN_KEY))).toBe(false);
    expect(layout.allowsListPrefix("runs/")).toBe(false);
    expect(layout.overwriteKey).toBeNull();
  });

  it("refuses a run key that is not a run key", () => {
    for (const key of ["", "-a", "a b", "a/b", "a".repeat(129)]) expect(() => runInputLayout(key), key).toThrow(RunInputError);
  });

  it("keeps each user's store to its own layout: another run's keys and a foreign key are refused before any read", async () => {
    const { store } = await publish();
    await expect(store.get("runs/other-run/inputs/manifest.json")).rejects.toThrow(ObjectStoreError);
    await expect(store.get(`${filesPrefix(RUN_KEY)}../../x`)).rejects.toThrow("not part of this store's layout");
    await expect(store.putIfAbsent(manifestKey(RUN_KEY), new Uint8Array())).rejects.toMatchObject({ code: "exists" });
    await expect(store.overwriteCurrent(new Uint8Array())).rejects.toMatchObject({ code: "key_invalid" });
  });
});

describe("build, upload and fetch (ADR-024 section 3.5)", () => {
  it("reproduces the dataset directory byte for byte and checks every hash", async () => {
    const dir = join(scratch, "owner");
    mkdirSync(dir);
    const configFile = join(dir, "config.json");
    writeFileSync(configFile, configText(configFor(SOURCE)));
    const inputs = await buildRunInputs({ engine, configFile, datasetDir: SOURCE, commit: COMMIT });
    const { manifest } = inputs;
    expect(manifest).toMatchObject({ data_class: "synthetic", run_key: RUN_KEY, commit: COMMIT, dataset_hash: DATASET_HASH, config_file: RUN_CONFIG_FILE, dataset_dir: DATASET_DIR });
    expect(manifest.files.map((f) => f.path)).toEqual([RUN_CONFIG_FILE, ...[...datasetFiles().keys()].map((n) => `${DATASET_DIR}/${n}`)].sort());

    const store = new InMemoryObjectStore(runInputLayout(RUN_KEY));
    const hash = await uploadRunInputs(store, inputs);
    expect(hash).toBe(await runInputManifestHash(manifest));
    expect(store.writes.at(-1)).toBe(manifestKey(RUN_KEY)); // the manifest is the last object written
    await expect(uploadRunInputs(store, inputs)).rejects.toMatchObject({ code: "exists" }); // nothing is overwritten

    const out = fresh();
    const fetched = await fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir: out, headCommit: head(), expectedManifestHash: hash });
    expect(fetched).toMatchObject({ dir: out, datasetDir: join(out, DATASET_DIR), configFile: join(out, RUN_CONFIG_FILE), manifestHash: hash, datasetHash: DATASET_HASH, files: manifest.files.length });
    for (const [name, bytes] of datasetFiles()) expect(readFileSync(join(out, DATASET_DIR, name)).equals(Buffer.from(bytes)), name).toBe(true);
    expect(readFileSync(join(out, RUN_CONFIG_FILE), "utf8")).toBe(readFileSync(configFile, "utf8"));
    expect(readdirSync(scratch).filter((name) => name.includes(".partial-"))).toEqual([]);
  });

  it("gives a run on the fetched directory the same dataset hash and the same first-generation plan as on the source", async () => {
    const { store } = await publish();
    const fetched = await fetchFrom(store);
    const source = readTrainingRunConfig(configFor(SOURCE));
    const copy = readTrainingRunConfig(configFor(fetched.datasetDir));
    const a = datasetPoolDetail(engine, source);
    const b = datasetPoolDetail(engine, copy);
    expect(b.dataset_hash).toBe(a.dataset_hash);
    expect(datasetHash(readDataset(fetched.datasetDir, loadFixtureInputs(engine, FORMAT, fetched.datasetDir)))).toBe(DATASET_HASH);
    expect(jcs(b.pool.teams as never)).toBe(jcs(a.pool.teams as never));
    expect(jcs(planGeneration(copy, 1, b.pool.teams.length) as never)).toBe(jcs(planGeneration(source, 1, a.pool.teams.length) as never));
  });

  it("refuses a missing manifest, another run's manifest, a pin that differs and a manifest that is not canonical", async () => {
    expect(await refused((outDir) => fetchRunInputs({ store: new InMemoryObjectStore(runInputLayout(RUN_KEY)), engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_missing");
    const { store, manifestHash } = await publish();
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head(), expectedManifestHash: "0".repeat(64) }))).toBe("manifest_hash_mismatch");
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head(), expectedManifestHash: "short" }))).toBe("manifest_hash_mismatch");
    await expect(fetchFrom(store, { expectedManifestHash: manifestHash })).resolves.toBeDefined();
    const other = await publish({ manifest: (m) => (m.run_key = "another-run") });
    expect(await refused((outDir) => fetchRunInputs({ store: other.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_other_run");
    const pretty = await publish({ manifestBytes: (bytes) => new TextEncoder().encode(JSON.stringify(JSON.parse(new TextDecoder().decode(bytes)), null, 2)) });
    expect(await refused((outDir) => fetchRunInputs({ store: pretty.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_not_canonical");
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: "bad key!", outDir, headCommit: head() }))).toBe("run_key_invalid");
  });

  it("refuses a manifest that is not synthetic, not JSON or not a RunInputManifest, with a fixed code", async () => {
    const notSynthetic = await publish({ manifest: (m) => (m.data_class = "public") });
    expect(await refused((outDir) => fetchRunInputs({ store: notSynthetic.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_not_synthetic");
    for (const mutate of [
      (m: Record<string, any>) => (m.files[0].path = "../escape.json"),
      (m: Record<string, any>) => (m.files[0].path = "/absolute.json"),
      (m: Record<string, any>) => (m.extra = 1),
      (m: Record<string, any>) => delete m.data_class,
      (m: Record<string, any>) => (m.files = m.files.reverse()),
    ]) {
      const bad = await publish({ manifest: mutate as never });
      expect(await refused((outDir) => fetchRunInputs({ store: bad.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_invalid");
    }
    const junk = await publish({ manifestBytes: () => new Uint8Array([0xff, 0xfe]) });
    expect(await refused((outDir) => fetchRunInputs({ store: junk.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_invalid");
    const notJson = await publish({ manifestBytes: () => new TextEncoder().encode("not json") });
    expect(await refused((outDir) => fetchRunInputs({ store: notJson.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_invalid");
    const scalar = await publish({ manifestBytes: () => new TextEncoder().encode("7") });
    expect(await refused((outDir) => fetchRunInputs({ store: scalar.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("manifest_invalid");
  });

  it("refuses a manifest whose commit is not the checked-out repository's", async () => {
    const { store } = await publish();
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head("f".repeat(40)) }))).toBe("commit_mismatch");
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head("") }))).toBe("commit_mismatch");
  });

  it("refuses a missing object, an extra object, a changed size and a changed hash, and leaves no directory", async () => {
    const dataset = `${DATASET_DIR}/manifest.json`;
    const missing = await publish({ leaveOut: dataset });
    expect(await refused((outDir) => fetchRunInputs({ store: missing.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("object_missing");

    const extra = await publish();
    extra.store.seed(fileKey(RUN_KEY, "dataset/extra.jsonl"), new Uint8Array([1]));
    expect(await refused((outDir) => fetchRunInputs({ store: extra.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("object_extra");

    const longer = await publish();
    longer.store.seed(fileKey(RUN_KEY, dataset), Buffer.concat([readFileSync(join(SOURCE, "manifest.json")), Buffer.from(" ")]));
    expect(await refused((outDir) => fetchRunInputs({ store: longer.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("size_mismatch");

    const flipped = await publish();
    const original = Buffer.from(readFileSync(join(SOURCE, "manifest.json")));
    original[0] = original[0] === 0x7b ? 0x20 : 0x7b;
    flipped.store.seed(fileKey(RUN_KEY, dataset), original);
    expect(await refused((outDir) => fetchRunInputs({ store: flipped.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("hash_mismatch");

    const gone = await publish();
    const store: InMemoryObjectStore = gone.store;
    const realGet = store.get.bind(store);
    store.get = async (key) => (key === fileKey(RUN_KEY, dataset) ? null : realGet(key));
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("object_missing");
  });

  it("refuses a manifest that names another config, another dataset hash or another run than the config", async () => {
    const config = await publish({ manifest: (m) => (m.run_config_hash = "c".repeat(64)) });
    expect(await refused((outDir) => fetchRunInputs({ store: config.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("config_hash_mismatch");

    const dataset = await publish({ manifest: (m) => (m.dataset_hash = "d".repeat(64)) });
    expect(await refused((outDir) => fetchRunInputs({ store: dataset.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("dataset_hash_mismatch");

    // A config that names another dataset hash than the directory holds: the manifest is consistent with the config, the directory is not.
    const configDataset = await publish({ config: (raw) => (raw.team_dataset.dataset_hash = "e".repeat(64)), manifest: (m) => (m.dataset_hash = "e".repeat(64)) });
    expect(await refused((outDir) => fetchRunInputs({ store: configDataset.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("dataset_hash_mismatch");

    // The manifest is for run RUN_KEY and consistent with its config, but the config names another run.
    const otherRun = await publish({ config: (raw) => (raw.seeds.run_key = "someone-else") });
    expect(await refused((outDir) => fetchRunInputs({ store: otherRun.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("config_other_run");
  });

  it("refuses a config that is not a TrainingRunConfig or names no dataset, and a dataset that does not read", async () => {
    const encode = (text: string): Uint8Array => new TextEncoder().encode(text);
    const garbage = await publish({ files: (files) => files.set(RUN_CONFIG_FILE, encode("{}")) });
    expect(await refused((outDir) => fetchRunInputs({ store: garbage.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("config_invalid");
    const text = await publish({ files: (files) => files.set(RUN_CONFIG_FILE, encode("not json")) });
    expect(await refused((outDir) => fetchRunInputs({ store: text.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("config_invalid");

    const noDataset = await publish({ config: (raw) => delete raw.team_dataset });
    expect(await refused((outDir) => fetchRunInputs({ store: noDataset.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("dataset_hash_mismatch");

    // The manifest and the bucket agree with each other; the dataset reader does not accept the records.
    const doctored = await publish({
      files: (files) => files.set(`${DATASET_DIR}/teams-dev.jsonl`, encode(`${new TextDecoder().decode(files.get(`${DATASET_DIR}/teams-dev.jsonl`)).split("\n")[0]}\n`)),
    });
    expect(await refused((outDir) => fetchRunInputs({ store: doctored.store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("dataset_invalid");
  });

  it("writes only to a fresh path and reports a disk that refuses", async () => {
    const { store } = await publish();
    const taken = join(scratch, "taken");
    mkdirSync(taken);
    expect(await refused(() => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir: taken, headCommit: head() }))).toBe("output_exists");
    expect(readdirSync(taken)).toEqual([]);
    const blocker = join(scratch, "blocker");
    writeFileSync(blocker, "x");
    expect(await refused(() => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir: join(blocker, "child", "out"), headCommit: head() }))).toBe("write_failed");
    const parentFile = join(scratch, "parent-is-a-file");
    writeFileSync(parentFile, "x");
    expect(await refused(() => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir: join(parentFile, "out"), headCommit: head() }))).toBe("write_failed");
  });

  it("leaves nothing at the output path when the rename finds the path taken meanwhile, or the disk fails mid-way", async () => {
    const { store } = await publish();
    const out = fresh();
    const realGet = store.get.bind(store);
    store.get = async (key) => {
      const bytes = await realGet(key);
      if (key === fileKey(RUN_KEY, `${DATASET_DIR}/manifest.json`)) mkdirSync(out); // another process takes the path while the files arrive
      return bytes;
    };
    await expect(fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir: out, headCommit: head() })).rejects.toMatchObject({ code: "output_exists" });
    expect(readdirSync(out)).toEqual([]);
    expect(readdirSync(scratch).filter((name) => name.includes(".partial-"))).toEqual([]);
  });

  it("passes a store failure through with its own fixed code", async () => {
    const { store } = await publish();
    store.list = () => Promise.reject(new ObjectStoreError("denied"));
    expect(await refused((outDir) => fetchRunInputs({ store, engine, runKey: RUN_KEY, outDir, headCommit: head() }))).toBe("denied");
  });

  it("holds a fixed message for every code and quotes nothing", () => {
    for (const code of RUN_INPUT_REFUSALS as readonly RunInputRefusal[]) {
      const error = new RunInputError(code);
      expect(error.message).toMatch(/^run inputs refused: [A-Za-z0-9 ,;'()_-]+$/);
      expect(error.code).toBe(code);
    }
  });
});

describe("the owner's builder", () => {
  const owner = (name: string): string => {
    const dir = join(scratch, `owner-${name}`);
    mkdirSync(dir);
    return dir;
  };
  const build = (configFile: string, datasetDir = SOURCE) => buildRunInputs({ engine, configFile, datasetDir, commit: COMMIT });
  const codeOf = async (run: () => Promise<unknown>): Promise<string> => {
    try {
      await run();
    } catch (error) {
      return (error as RunInputError).code;
    }
    throw new Error("expected a refusal");
  };

  it("refuses a config that is not a run config, names no dataset or names another dataset hash", async () => {
    const dir = owner("config");
    expect(await codeOf(() => build(join(dir, "missing.json")))).toBe("config_invalid");
    const bad = join(dir, "bad.json");
    writeFileSync(bad, "{}");
    expect(await codeOf(() => build(bad))).toBe("config_invalid");
    const none = join(dir, "none.json");
    writeFileSync(none, configText(configFor(SOURCE, (raw) => delete raw.team_dataset)));
    expect(await codeOf(() => build(none))).toBe("config_invalid");
    const wrong = join(dir, "wrong.json");
    writeFileSync(wrong, configText(configFor(SOURCE, (raw) => (raw.team_dataset.dataset_hash = "9".repeat(64)))));
    expect(await codeOf(() => build(wrong))).toBe("dataset_hash_mismatch");
  });

  it("refuses a directory the dataset reader does not accept and a commit that is not a commit", async () => {
    const dir = owner("dataset");
    const configFile = join(dir, "config.json");
    writeFileSync(configFile, configText(configFor(SOURCE)));
    expect(await codeOf(() => build(configFile, dir))).toBe("dataset_invalid");
    expect(await codeOf(() => buildRunInputs({ engine, configFile, datasetDir: SOURCE, commit: "HEAD" }))).toBe("manifest_invalid");
  });

  it("refuses a link or an oversize file among the inputs", async () => {
    const dir = owner("files");
    const data = join(dir, "dataset");
    mkdirSync(data);
    for (const [name, bytes] of datasetFiles()) writeFileSync(join(data, name), bytes);
    const configFile = join(dir, "config.json");
    writeFileSync(configFile, configText(configFor(data)));
    await expect(build(configFile, data)).resolves.toBeDefined();
    symlinkSync(join(SOURCE, "manifest.json"), join(data, "link.json"));
    expect(await codeOf(() => build(configFile, data))).toBe("input_unsupported");

    const hard = owner("hard");
    const hardData = join(hard, "dataset");
    mkdirSync(hardData);
    for (const [name, bytes] of datasetFiles()) writeFileSync(join(hardData, name), bytes);
    const hardConfig = join(hard, "config.json");
    writeFileSync(hardConfig, configText(configFor(hardData)));
    linkSync(join(hardData, "manifest.json"), join(hard, "second-name"));
    const target = join(hard, "target.json");
    writeFileSync(target, configText(configFor(hardData)));
    const linked = join(hard, "config-link.json");
    symlinkSync(target, linked);
    expect(await codeOf(() => build(linked, hardData))).toBe("input_unsupported");

    const big = owner("big");
    const bigData = join(big, "dataset");
    mkdirSync(bigData);
    for (const [name, bytes] of datasetFiles()) writeFileSync(join(bigData, name), bytes);
    const bigConfig = join(big, "config.json");
    writeFileSync(bigConfig, configText(configFor(bigData)));
    writeFileSync(join(bigData, "huge.bin"), "");
    truncateSync(join(bigData, "huge.bin"), 256 * 1024 * 1024 + 1);
    expect(await codeOf(() => build(bigConfig, bigData))).toBe("input_too_large");
  });
});
