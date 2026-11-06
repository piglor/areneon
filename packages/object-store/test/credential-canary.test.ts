import { inspect } from "node:util";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ObjectStoreError } from "../src/object-store.ts";
import { type FetchLike, R2ObjectStore } from "../src/r2-client.ts";
import { CredentialError, R2Settings } from "../src/r2-settings.ts";
import { ACCOUNT, BUCKET, ENDPOINT, ENV, ENV_TOKEN, FakeR2, KEY_ID, LAYOUT, objectKey, SECRET, SECRETS, SESSION_TOKEN, settings } from "./support.ts";

// Credential canary tests (ADR-015 section 1, ADR-024 section 3.6): the access
// key id, the secret, the session token, the endpoint (it holds the account
// id) and the bucket name are markers; the region is the fifth variable and
// must equal "auto", so a bad region carries the echo instead. None may appear
// in an error (message, name, stack, fields, inspect), in what the process
// printed, or in a URL the client builds beyond the request it must send; the
// secret appears in no request, and the session token in none but the signed
// `x-amz-security-token` header. The negative controls show the scan can fail.

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
  it("flags an error that repeats any of the markers, in the message, a cause or a field", () => {
    for (const marker of SECRETS) {
      expect(leaks(surface(new Error(`failed for ${marker}`)))).toContain(marker);
      expect(leaks(surface(Object.assign(new Error("x"), { cause: new Error(marker) })))).toContain(marker);
      expect(leaks(surface(Object.assign(new Error("x"), { detail: marker })))).toContain(marker);
    }
    expect(leaks(surface(new ObjectStoreError("failed")))).toEqual([]);
  });

  it("flags process output that carries a marker", () => {
    process.stderr.write(`oops ${KEY_ID}\n`);
    process.stdout.write(`oops ${SESSION_TOKEN}\n`);
    expect(leaks(printed.join(""))).toEqual(expect.arrayContaining([KEY_ID, SESSION_TOKEN]));
  });

  it("flags a client that put the URL in its error", async () => {
    const request = await settings().sign({ method: "GET", key: objectKey("a") });
    expect(leaks(surface(new Error(`request to ${request.url} failed`)))).toEqual(expect.arrayContaining([ACCOUNT, BUCKET]));
  });

  it("flags a request that carries the token outside its header, or the secret anywhere", async () => {
    const fake = new FakeR2();
    await new R2ObjectStore(settings(true), LAYOUT, { fetch: fake.fetch }).get(objectKey("a"));
    const seen = fake.seen[0];
    if (seen === undefined) throw new Error("no request");
    const headersText = [...seen.headers.entries()].flat().join(" ");
    expect(headersText).toContain(SESSION_TOKEN); // the signed header is the one place the token is
    expect(`${seen.url}?x=${SESSION_TOKEN}`).toContain(SESSION_TOKEN);
    expect(`${headersText} ${SECRET}`).toContain(SECRET);
  });
});

describe("the settings", () => {
  it("never repeat a value in a refusal", async () => {
    const bad: Record<string, string>[] = [
      { ...ENV, AWS_ACCESS_KEY_ID: `${KEY_ID}!` },
      { ...ENV, AWS_SECRET_ACCESS_KEY: `${SECRET} x` },
      { ...ENV_TOKEN, AWS_SESSION_TOKEN: `${SESSION_TOKEN} x` },
      { ...ENV, AWS_ENDPOINT_URL: `${ENDPOINT}/${BUCKET}` },
      { ...ENV, AWS_ENDPOINT_URL: `http://${ACCOUNT}.r2.cloudflarestorage.com` },
      { ...ENV, AWS_DEFAULT_REGION: ECHO },
      { ...ENV, AWS_DEFAULT_REGION: "" },
      { AWS_ACCESS_KEY_ID: KEY_ID, AWS_SESSION_TOKEN: SESSION_TOKEN },
    ];
    for (const env of bad) expect(await refusal(() => R2Settings.fromEnvironment(env, [], BUCKET))).toBeInstanceOf(CredentialError);
    expect(await refusal(() => R2Settings.fromEnvironment(ENV_TOKEN, [], `${BUCKET}!`))).toBeInstanceOf(CredentialError);
  });

  it("refuse an attempt to give them on the command line without printing them", async () => {
    for (const argv of [
      [`--access-key-id=${KEY_ID}`],
      ["--secret", SECRET],
      ["--session-token", SESSION_TOKEN],
      [`AWS_SESSION_TOKEN=${SESSION_TOKEN}`],
      ["x", SESSION_TOKEN],
      [ENDPOINT],
      [`--bucket=${BUCKET}`, `${ACCOUNT}`],
      [`AWS_SECRET_ACCESS_KEY=${SECRET}`],
    ]) {
      const error = await refusal(() => R2Settings.fromEnvironment(ENV_TOKEN, argv, BUCKET));
      expect((error as CredentialError).code).toBe("credentials_on_argv");
    }
  });

  it("leave the environment as it was and are not a value anyone can print", () => {
    const before = JSON.stringify(process.env);
    const s = settings(true);
    expect(JSON.stringify(process.env)).toBe(before);
    expect(leaks(`${String(s)}${JSON.stringify(s)}${inspect(s, { depth: 9, showHidden: true })}${JSON.stringify(Object.getOwnPropertyNames(s))}`)).toEqual([]);
  });
});

describe("the R2 client", () => {
  const echoing = (status: number): FetchLike => () =>
    Promise.resolve(new Response(`<Error>${ECHO}</Error>`, { status, headers: { "x-echo": ECHO, "content-length": String(ECHO.length + 14) } }));
  const ops = (store: R2ObjectStore): (() => Promise<unknown>)[] => [
    () => store.putIfAbsent(objectKey("a"), bytes("x")),
    () => store.get(objectKey("a")),
    () => store.head(objectKey("a")),
    () => store.list("objects/"),
    () => store.overwriteCurrent(bytes("x")),
    () => store.get("not-in-the-layout"),
  ];

  it("repeats nothing of a response that echoes the credentials, whatever the status", async () => {
    for (const status of [200, 201, 301, 400, 401, 403, 404, 409, 412, 429, 500, 503]) {
      const store = new R2ObjectStore(settings(true), LAYOUT, { fetch: echoing(status) });
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
    for (const run of ops(new R2ObjectStore(settings(true), LAYOUT, { fetch: throwing })).slice(0, 5)) {
      expect(((await refusal(run)) as ObjectStoreError).code).toBe("unavailable");
    }
    const midBody = { status: 200, headers: new Headers(), body: null, arrayBuffer: () => Promise.reject(new Error(ECHO)) } as unknown as Response;
    expect((await refusal(() => new R2ObjectStore(settings(true), LAYOUT, { fetch: () => Promise.resolve(midBody) }).get(objectKey("a")))).message).not.toContain(KEY_ID);
  });

  it("sends the secret nowhere and the token only in its signed header", async () => {
    const fake = new FakeR2();
    const store = new R2ObjectStore(settings(true), LAYOUT, { fetch: fake.fetch });
    await store.putIfAbsent(objectKey("a"), bytes("x"));
    await store.get(objectKey("a"));
    await store.list("objects/");
    await store.overwriteCurrent(bytes("x"));
    expect(fake.seen).toHaveLength(4);
    for (const seen of fake.seen) {
      const headers = [...seen.headers.entries()].filter(([name]) => name !== "x-amz-security-token");
      const text = [seen.url, headers.flat().join(" "), new TextDecoder().decode(seen.body)].join("\n");
      expect(text).not.toContain(SECRET);
      expect(text).not.toContain(SESSION_TOKEN);
      expect(seen.headers.get("x-amz-security-token")).toBe(SESSION_TOKEN);
      expect(new URL(seen.url).origin).toBe(ENDPOINT);
      expect(new URL(seen.url).search).not.toMatch(/X-Amz-(Credential|Signature|Security-Token)/i);
    }
  });
});
