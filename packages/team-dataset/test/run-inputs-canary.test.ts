import { inspect } from "node:util";
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { R2ObjectStore, R2Settings } from "@areneon/object-store";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { buildRunInputs, fetchRunInputs, runInputLayout, uploadRunInputs } from "../src/run-inputs.ts";
import { COMMIT, configFor, configText, FakeR2, removeDir, RUN_KEY, scratchDir, SOURCE } from "./run-inputs-support.ts";
import { engine } from "./support.ts";

// Credential canary over the fetch and publish paths (ADR-024 section 3.6,
// ADR-015 section 1): the access key id, the secret, the session token, the
// endpoint (it holds the account id) and the bucket name are markers. None may
// appear in an error (message, name, stack, fields, inspect), in what the
// process printed, in a URL beyond the request it must send, or in a file the
// fetch wrote; the secret and the token appear in no URL. The negative
// controls show the scan can fail.

const KEY_ID = "RunInputsCanaryKeyId7abcd";
const MARK_S = "RunInputsCanaryS7/".repeat(2);
const SESSION = "RunInputsCanarySession7/abcdef+ghijkl==";
const ACCOUNT = "runinputsacct7";
const ENDPOINT = `https://${ACCOUNT}.r2.cloudflarestorage.com`;
const BUCKET = "run-inputs-canary-7";
const MARKERS = [KEY_ID, MARK_S, SESSION, ACCOUNT, ENDPOINT, BUCKET];
const ENV = { AWS_ACCESS_KEY_ID: KEY_ID, AWS_SECRET_ACCESS_KEY: MARK_S, AWS_ENDPOINT_URL: ENDPOINT, AWS_DEFAULT_REGION: "auto", AWS_SESSION_TOKEN: SESSION };
const ECHO = MARKERS.join(" | ");

const scratch = scratchDir();
afterAll(() => removeDir(scratch));
let counter = 0;
const fresh = (): string => join(scratch, `out-${counter++}`);

const leaks = (text: string): string[] => MARKERS.filter((marker) => text.includes(marker));
const surface = (error: unknown): string => {
  const e = error as Error & { code?: string };
  return [String(error), e.message, e.name, e.stack ?? "", JSON.stringify(error, Object.getOwnPropertyNames(error as object)), inspect(error, { depth: 8, showHidden: true })].join("\n");
};

let printed: string[] = [];
vi.spyOn(process.stdout, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true));
vi.spyOn(process.stderr, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true));
afterEach(() => {
  printed = [];
});

const settings = (): R2Settings => R2Settings.fromEnvironment(ENV, [], BUCKET);
const storeOn = (fake: FakeR2): R2ObjectStore => new R2ObjectStore(settings(), runInputLayout(RUN_KEY), { fetch: fake.fetch });
const head = () => COMMIT;

let configFileOfLast = "";
/** A fake R2 holding a run's published inputs; `target` is where the config says the fetch will write. */
async function seeded(target = join(scratch, "not-fetched-here")): Promise<FakeR2> {
  const fake = new FakeR2(BUCKET);
  const dir = mkdtempSync(join(scratch, "owner-"));
  configFileOfLast = join(dir, "config.json");
  writeFileSync(configFileOfLast, configText(configFor(join(target, "dataset"))));
  await uploadRunInputs(storeOn(fake), await buildRunInputs({ engine, configFile: configFileOfLast, datasetDir: SOURCE, commit: COMMIT }));
  return fake;
}

async function refusal(run: () => Promise<unknown>): Promise<Error> {
  try {
    await run();
  } catch (error) {
    expect(leaks(surface(error)), `leaked from ${String((error as { code?: string }).code)}`).toEqual([]);
    expect(leaks(printed.join("")), "printed").toEqual([]);
    return error as Error;
  }
  throw new Error("expected a refusal");
}

describe("negative controls: the scan can fail", () => {
  it("flags an error or output that repeats a marker, and a URL that carries one", () => {
    for (const marker of MARKERS) {
      expect(leaks(surface(new Error(`failed for ${marker}`)))).toContain(marker);
      expect(leaks(surface(Object.assign(new Error("x"), { cause: new Error(marker) })))).toContain(marker);
    }
    process.stderr.write(`oops ${KEY_ID}\n`);
    expect(leaks(printed.join(""))).toContain(KEY_ID);
    expect(leaks(`https://${ACCOUNT}.r2.cloudflarestorage.com/${BUCKET}/x`)).toEqual(expect.arrayContaining([ACCOUNT, BUCKET]));
  });
});

describe("the credential canary over publish and fetch", () => {
  it("leaks nothing on a good fetch: the result, the output, the URLs and the written files carry no marker, and the secret and token are in no URL", async () => {
    const out = fresh();
    const fake = await seeded(out);
    const fetched = await fetchRunInputs({ store: storeOn(fake), engine, runKey: RUN_KEY, outDir: out, headCommit: head });
    expect(leaks(JSON.stringify(fetched).replaceAll(out, ""))).toEqual([]);
    expect(leaks(printed.join(""))).toEqual([]);
    for (const url of fake.seen) for (const marker of [KEY_ID, MARK_S, SESSION]) expect(url).not.toContain(marker);
    for (const name of readdirSync(join(out, "dataset"))) expect(leaks(readFileSync(join(out, "dataset", name), "utf8")), name).toEqual([]);
    expect(leaks(readFileSync(join(out, "run-config.json"), "utf8"))).toEqual([]);
  });

  it.each([
    ["denied", 403],
    ["unavailable", 500],
    ["unavailable", 429],
    ["failed", 400],
  ])("refuses a %s answer that echoes the markers with a fixed code (status %i)", async (code, status) => {
    for (const phase of ["manifest", "list", "object"] as const) {
      const fake = await seeded();
      fake.override = (request) => {
        const url = new URL(request.url);
        const isList = url.pathname.endsWith(`/${BUCKET}`) || url.pathname.endsWith(`/${BUCKET}/`);
        const hit = phase === "manifest" ? url.pathname.endsWith("/manifest.json") && !url.pathname.includes("/files/") : phase === "list" ? isList : url.pathname.includes("/files/dataset/manifest.json");
        return hit ? new Response(`<Error>${ECHO}</Error>`, { status, headers: { "x-echo": ECHO } }) : undefined;
      };
      const error = (await refusal(() => fetchRunInputs({ store: storeOn(fake), engine, runKey: RUN_KEY, outDir: fresh(), headCommit: head }))) as Error & { code: string };
      expect(error.code, phase).toBe(code);
    }
  });

  it("refuses a transport that throws the markers, a publish into a store that refuses, and a response that is not a listing", async () => {
    const throwing = await seeded();
    throwing.override = () => {
      throw new Error(`connect failed: ${ECHO}`);
    };
    await refusal(() => fetchRunInputs({ store: storeOn(throwing), engine, runKey: RUN_KEY, outDir: fresh(), headCommit: head }));

    const inputs = await buildRunInputs({ engine, configFile: configFileOfLast, datasetDir: SOURCE, commit: COMMIT });
    const full = await seeded();
    const changed = await buildRunInputs({ engine, configFile: configFileOfLast, datasetDir: SOURCE, commit: COMMIT });
    full.objects.set([...full.objects.keys()][0] as string, new TextEncoder().encode(ECHO)); // an object of another content is already there
    await refusal(() => uploadRunInputs(storeOn(full), changed));
    expect(inputs.manifest.files.length).toBeGreaterThan(0);

    const garbled = await seeded();
    garbled.override = (request) => (new URL(request.url).searchParams.has("list-type") ? new Response(ECHO, { status: 200 }) : undefined);
    await refusal(() => fetchRunInputs({ store: storeOn(garbled), engine, runKey: RUN_KEY, outDir: fresh(), headCommit: head }));
  });

  it("makes no request for a key outside the layout, and a tampered object leaves no marker behind", async () => {
    const fake = await seeded();
    const before = fake.seen.length;
    await refusal(() => storeOn(fake).get("runs/another/inputs/manifest.json"));
    await refusal(() => storeOn(fake).get(`runs/${RUN_KEY}/inputs/files/../manifest.json`));
    expect(fake.seen.length).toBe(before);

    const key = [...fake.objects.keys()].find((k) => k.endsWith("dataset/manifest.json")) as string;
    fake.objects.set(key, new TextEncoder().encode(ECHO));
    await refusal(() => fetchRunInputs({ store: storeOn(fake), engine, runKey: RUN_KEY, outDir: fresh(), headCommit: head }));
    expect(readdirSync(scratch).filter((name) => name.includes(".partial-"))).toEqual([]);
  });
});
