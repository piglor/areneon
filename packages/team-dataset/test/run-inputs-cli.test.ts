import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { InMemoryObjectStore, type ObjectLayout, type ObjectStore, ObjectStoreError, type R2Settings } from "@areneon/object-store";
import { afterAll, describe, expect, it } from "vitest";
import { type RunInputsDeps, runInputsCli } from "../src/run-inputs-cli.ts";
import { DATASET_DIR, RUN_CONFIG_FILE, runInputLayout } from "../src/run-inputs.ts";
import { COMMIT, configFor, configText, removeDir, RUN_KEY, scratchDir, SOURCE } from "./run-inputs-support.ts";
import { engine } from "./support.ts";

const scratch = scratchDir();
afterAll(() => removeDir(scratch));

const KEY_ID = "CliCanaryAccessKeyId7abcd";
const MARK_S = "CliCanaryMarkerS7/".repeat(2);
const SESSION = "CliCanarySessionToken7/abcdef+ghijkl==";
const ENV = { AWS_ACCESS_KEY_ID: KEY_ID, AWS_SECRET_ACCESS_KEY: MARK_S, AWS_ENDPOINT_URL: "https://clicanaryacct7.r2.cloudflarestorage.com", AWS_DEFAULT_REGION: "auto", AWS_SESSION_TOKEN: SESSION };
const MARKERS = [KEY_ID, MARK_S, SESSION, "clicanaryacct7", "cli-canary-bucket"];
const BUCKET = "cli-canary-bucket";

interface Run {
  code: number;
  out: string[];
  err: string[];
  layouts: ObjectLayout[];
}

function harness(store: ObjectStore = new InMemoryObjectStore(runInputLayout(RUN_KEY)), head = COMMIT) {
  const run = async (argv: string[], env: RunInputsDeps["env"] = ENV): Promise<Run> => {
    const out: string[] = [];
    const err: string[] = [];
    const layouts: ObjectLayout[] = [];
    const code = await runInputsCli(argv, {
      env,
      engine: () => engine,
      openStore: (_settings: R2Settings, layout) => (layouts.push(layout), store),
      headCommit: () => head,
      out: (line) => out.push(line),
      err: (line) => err.push(line),
    });
    return { code, out, err, layouts };
  };
  return { store, run };
}
const noMarkers = (result: Run): void => {
  const text = [...result.out, ...result.err].join("\n");
  for (const marker of MARKERS) expect(text, marker).not.toContain(marker);
};

describe("the run-inputs command", () => {
  const configFile = join(scratch, "config.json");
  writeFileSync(configFile, configText(configFor(SOURCE)));
  const publishArgs = ["publish", "--run-config", configFile, "--dataset-dir", SOURCE, "--commit", COMMIT, "--bucket", BUCKET];

  it("publishes the inputs, then fetches them into a fresh directory, and prints one JSON line each", async () => {
    const { run, store } = harness();
    const published = await run(publishArgs);
    expect(published.code).toBe(0);
    const summary = JSON.parse(published.out[0] as string) as Record<string, unknown>;
    expect(summary).toMatchObject({ run_key: RUN_KEY, manifest_hash: expect.stringMatching(/^[0-9a-f]{64}$/) });
    expect(published.layouts[0]?.allowsKey(`runs/${RUN_KEY}/inputs/manifest.json`)).toBe(true);
    expect(published.layouts[0]?.overwriteKey).toBeNull();
    noMarkers(published);
    expect((store as InMemoryObjectStore).keys().length).toBe((summary.files as number) + 1);

    const out = join(scratch, "fetched");
    const fetched = await run(["fetch", "--run-key", RUN_KEY, "--out", out, "--bucket", BUCKET, "--manifest-hash", summary.manifest_hash as string, "--repo", "somewhere"]);
    expect(fetched.code).toBe(0);
    expect(JSON.parse(fetched.out[0] as string)).toMatchObject({ manifest_hash: summary.manifest_hash, dataset_dir: join(out, DATASET_DIR), files: summary.files });
    expect(existsSync(join(out, RUN_CONFIG_FILE))).toBe(true);
    noMarkers(fetched);
  });

  it("exits 1 with the code of a refusal and a fixed message, and leaves no directory", async () => {
    const { run } = harness();
    const out = join(scratch, "refused");
    const missing = await run(["fetch", "--run-key", RUN_KEY, "--out", out, "--bucket", BUCKET]);
    expect(missing).toMatchObject({ code: 1, out: [] });
    expect(missing.err).toEqual(["run-inputs: manifest_missing: run inputs refused: the inputs bucket holds no run-input manifest for this run"]);
    expect(existsSync(out)).toBe(false);
    noMarkers(missing);

    const { run: wrongHead } = harness(undefined, "f".repeat(40));
    await wrongHead(publishArgs);
    const stale = await wrongHead(["fetch", "--run-key", RUN_KEY, "--out", out, "--bucket", BUCKET]);
    expect(stale.code).toBe(1);
    expect(stale.err[0]).toMatch(/^run-inputs: commit_mismatch:/);
    expect(existsSync(out)).toBe(false);

    const bad = join(scratch, "bad.json");
    writeFileSync(bad, "{}");
    const unreadable = await run(["publish", "--run-config", bad, "--dataset-dir", SOURCE, "--commit", COMMIT, "--bucket", BUCKET]);
    expect(unreadable.err[0]).toMatch(/^run-inputs: config_invalid:/);
    expect(unreadable.code).toBe(1);
  });

  it("reports a store failure by its own code and any other failure without detail", async () => {
    const denied: ObjectStore = { ...new InMemoryObjectStore(runInputLayout(RUN_KEY)), get: () => Promise.reject(new ObjectStoreError("denied")) } as unknown as ObjectStore;
    const result = await harness(denied).run(["fetch", "--run-key", RUN_KEY, "--out", join(scratch, "denied"), "--bucket", BUCKET]);
    expect(result).toMatchObject({ code: 1, err: ["run-inputs: store: denied: object store refused: the store refused the credentials or their permission"] });
    const broken: ObjectStore = { ...new InMemoryObjectStore(runInputLayout(RUN_KEY)), get: () => Promise.reject(new Error(`boom ${MARK_S}`)) } as unknown as ObjectStore;
    const other = await harness(broken).run(["fetch", "--run-key", RUN_KEY, "--out", join(scratch, "broken"), "--bucket", BUCKET]);
    expect(other).toMatchObject({ code: 1, err: ["run-inputs: failed"] });
    noMarkers(other);
  });

  it("refuses credentials on the command line before reading the environment", async () => {
    const { run } = harness();
    for (const extra of [["--access-key", "x"], ["--secret", "x"], ["--session-token", "x"], ["--endpoint", "https://a.r2.cloudflarestorage.com"], ["--run-key", SESSION]]) {
      const result = await run(["fetch", "--bucket", BUCKET, "--out", join(scratch, "argv"), ...extra, ...(extra[0] === "--run-key" ? [] : ["--run-key", RUN_KEY])]);
      expect(result.code, extra.join(" ")).toBe(2);
      expect(result.err[0]).toMatch(/credentials_on_argv/);
      noMarkers(result);
    }
    const missing = await run(["fetch", "--run-key", RUN_KEY, "--out", join(scratch, "env"), "--bucket", BUCKET], {});
    expect(missing).toMatchObject({ code: 2 });
    expect(missing.err[0]).toMatch(/credentials_missing/);
  });

  it("exits 2 on a usage error", async () => {
    const { run } = harness();
    for (const argv of [[], ["upload"], ["fetch", "--bucket", BUCKET], ["fetch", "--run-key", RUN_KEY, "--bucket", BUCKET], ["fetch", "--nonsense", "x", "--bucket", BUCKET], ["publish", "--bucket", BUCKET], ["fetch", "--run-key"], ["fetch", "extra", "--bucket", BUCKET]]) {
      const result = await run(argv);
      expect(result.code, argv.join(" ")).toBe(2);
      expect(result.err[0]).toMatch(/^run-inputs: /);
    }
    mkdirSync(join(scratch, "unused"), { recursive: true });
    expect(readFileSync(configFile, "utf8")).toContain(RUN_KEY);
  });
});
