import { inspect } from "node:util";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ObjectStoreError } from "../src/object-store.ts";
import { type FetchLike, R2ObjectStore } from "../src/r2-client.ts";
import { CredentialError, R2Settings } from "../src/r2-settings.ts";
import { publishSnapshot } from "../src/publish.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { writePrivateSnapshot } from "../src/snapshot-writer.ts";
import { ACCOUNT, BUCKET, ENDPOINT, ENV, FakeR2, HASH_A, KEY_ID, manifestKey, SECRET, SECRETS, settings } from "./r2-support.ts";
import { engine, format, input, labelSet, records, workDir } from "./snapshot-support.ts";
import { dirname, join } from "node:path";

vi.setConfig({ testTimeout: 60_000 });

// Credential canary tests (ADR-015 section 1, "Credential handling"): the
// access key id, the secret, the endpoint (it holds the account id) and the
// bucket name are markers. None may appear in an error (message, name, stack,
// fields, inspect), in what the process printed, in a URL the client builds
// beyond the request it must send, or, for the secret, anywhere in a request.
// The negative controls show the scan can fail.

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);
const ECHO = SECRETS.join(" | ");

const surface = (error: unknown): string => {
  const e = error as Error & { code?: string };
  return [
    String(error),
    e.message,
    e.name,
    e.stack ?? "",
    JSON.stringify(error, Object.getOwnPropertyNames(error as object)),
    inspect(error, { depth: 8, showHidden: true }),
    JSON.stringify({ code: e.code }),
  ].join("\n");
};
const leaks = (text: string): string[] => SECRETS.filter((s) => text.includes(s));

let printed: string[] = [];
vi.spyOn(process.stdout, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true));
vi.spyOn(process.stderr, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true));
afterEach(() => {
  printed = [];
});

const refusal = async (run: () => unknown): Promise<Error> => {
  try {
    await run();
  } catch (error) {
    expect(leaks(surface(error)), `leaked from ${String((error as { code?: string }).code)}`).toEqual([]);
    expect(leaks(printed.join("")), "printed").toEqual([]);
    return error as Error;
  }
  throw new Error("expected a refusal");
};

describe("negative controls: the scan can fail", () => {
  it("flags an error that repeats a credential, the endpoint or the bucket", () => {
    for (const marker of SECRETS) {
      expect(leaks(surface(new Error(`failed for ${marker}`)))).toContain(marker);
      expect(leaks(surface(Object.assign(new Error("x"), { cause: new Error(marker) })))).toContain(marker);
      expect(leaks(surface(Object.assign(new Error("x"), { detail: marker })))).toContain(marker);
    }
    expect(leaks(surface(new ObjectStoreError("failed")))).toEqual([]);
  });

  it("flags process output that carries a marker", () => {
    process.stderr.write(`oops ${KEY_ID}\n`);
    process.stdout.write(`oops ${SECRET}\n`);
    expect(leaks(printed.join(""))).toEqual(expect.arrayContaining([KEY_ID, SECRET]));
  });

  it("flags a client that put the URL in its error", async () => {
    const request = await settings().sign({ method: "GET", key: manifestKey(HASH_A) });
    const leaky = new Error(`request to ${request.url} failed`);
    expect(leaks(surface(leaky))).toEqual(expect.arrayContaining([ACCOUNT, BUCKET]));
  });
});

describe("the settings", () => {
  it("never repeat a value in a refusal", async () => {
    const bad: Record<string, string>[] = [
      { ...ENV, AWS_ACCESS_KEY_ID: `${KEY_ID}!` },
      { ...ENV, AWS_SECRET_ACCESS_KEY: `${SECRET} x` },
      { ...ENV, AWS_ENDPOINT_URL: `${ENDPOINT}/${BUCKET}` },
      { ...ENV, AWS_ENDPOINT_URL: `http://${ACCOUNT}.r2.cloudflarestorage.com` },
      { ...ENV, AWS_DEFAULT_REGION: ECHO },
      { ...ENV, AWS_DEFAULT_REGION: "" },
      { AWS_ACCESS_KEY_ID: KEY_ID },
    ];
    for (const env of bad) {
      const error = await refusal(() => R2Settings.fromEnvironment(env, [], BUCKET));
      expect(error).toBeInstanceOf(CredentialError);
    }
    expect(await refusal(() => R2Settings.fromEnvironment(ENV, [], `${BUCKET}!`))).toBeInstanceOf(CredentialError);
  });

  it("refuse an attempt to give them on the command line without printing them", async () => {
    for (const argv of [[`--access-key-id=${KEY_ID}`], ["--secret", SECRET], [ENDPOINT], [`--bucket=${BUCKET}`, `${ACCOUNT}`], [`AWS_SECRET_ACCESS_KEY=${SECRET}`]]) {
      const error = await refusal(() => R2Settings.fromEnvironment(ENV, argv, BUCKET));
      expect(error).toBeInstanceOf(CredentialError);
      expect((error as CredentialError).code).toBe("credentials_on_argv");
    }
  });

  it("leave the environment as it was and are not a value anyone can print", () => {
    const before = JSON.stringify(process.env);
    const s = settings();
    expect(JSON.stringify(process.env)).toBe(before);
    expect(leaks(`${String(s)}${JSON.stringify(s)}${inspect(s, { depth: 9, showHidden: true })}${JSON.stringify(Object.getOwnPropertyNames(s))}`)).toEqual([]);
  });
});

describe("the R2 client", () => {
  const echoing = (status: number): FetchLike => () =>
    Promise.resolve(new Response(`<Error>${ECHO}</Error>`, { status, headers: { "x-echo": ECHO, "content-length": String(ECHO.length + 14) } }));
  const ops = (store: R2ObjectStore): (() => Promise<unknown>)[] => [
    () => store.putIfAbsent(manifestKey(HASH_A), bytes("x")),
    () => store.get(manifestKey(HASH_A)),
    () => store.head(manifestKey(HASH_A)),
    () => store.list("snapshots/"),
    () => store.overwriteCurrent(bytes("x")),
  ];

  it("repeats nothing of a response that echoes the credentials, whatever the status", async () => {
    for (const status of [200, 201, 301, 400, 401, 403, 404, 409, 412, 429, 500, 503]) {
      const store = new R2ObjectStore(settings(), { fetch: echoing(status) });
      for (const run of ops(store)) {
        try {
          await run();
        } catch (error) {
          expect(error).toBeInstanceOf(ObjectStoreError);
          expect(leaks(surface(error)), `status ${status}`).toEqual([]);
        }
      }
    }
    expect(leaks(printed.join(""))).toEqual([]);
  });

  it("repeats nothing of a transport failure or a body that fails, even when they quote the request", async () => {
    const throwing: FetchLike = (request) => Promise.reject(Object.assign(new TypeError(`fetch failed: ${request.url} ${ECHO}`), { cause: new Error(ECHO) }));
    for (const run of ops(new R2ObjectStore(settings(), { fetch: throwing }))) {
      const error = await refusal(run);
      expect((error as ObjectStoreError).code).toBe("unavailable");
    }
    const midBody = { status: 200, headers: new Headers(), body: null, arrayBuffer: () => Promise.reject(new Error(ECHO)) } as unknown as Response;
    expect((await refusal(() => new R2ObjectStore(settings(), { fetch: () => Promise.resolve(midBody) }).get(manifestKey(HASH_A)))).message).not.toContain(KEY_ID);
  });

  it("sends the secret nowhere: it signs with it and never puts it in a URL, header or body", async () => {
    const fake = new FakeR2();
    const store = new R2ObjectStore(settings(), { fetch: fake.fetch });
    await store.putIfAbsent(manifestKey(HASH_A), bytes("x"));
    await store.get(manifestKey(HASH_A));
    await store.list("snapshots/");
    await store.overwriteCurrent(bytes("x"));
    for (const seen of fake.seen) {
      const text = [seen.url, [...seen.headers.entries()].flat().join(" "), new TextDecoder().decode(seen.body)].join("\n");
      expect(text).not.toContain(SECRET);
      expect(new URL(seen.url).origin).toBe(ENDPOINT);
      expect(new URL(seen.url).search).not.toMatch(/X-Amz-(Credential|Signature)/i);
    }
  });

  it("repeats nothing through a publish against a bucket that echoes the credentials", async () => {
    const dir = join(dirname(workDir()), "snapshots");
    const written = await writePrivateSnapshot(input({ records: records(5), shardRows: 4 }));
    const common = { engine, labelSet, written, expected: { manifest: format.manifest, vocabularyHash: "ee".repeat(32) }, expectedCurrent: null, snapshotsDir: dir, shardRows: 4 };
    for (const status of [403, 412, 500]) {
      const error = await refusal(() => publishSnapshot({ ...common, store: new R2ObjectStore(settings(), { fetch: echoing(status) }) }));
      expect(error).toBeInstanceOf(Error);
    }
    // A bucket that accepts the writes but serves back something else on the download.
    const fake = new FakeR2();
    fake.override = (request) => (request.method === "GET" ? new Response(ECHO, { status: 200 }) : undefined);
    const error = await refusal(() => publishSnapshot({ ...common, store: new R2ObjectStore(settings(), { fetch: fake.fetch }) }));
    expect(error).toBeInstanceOf(SnapshotIoError);
  });
});
