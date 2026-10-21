import { describe, expect, it, vi } from "vitest";
import { MAX_OBJECT_BYTES, ObjectStoreError } from "../src/object-store.ts";
import { type FetchLike, R2ObjectStore } from "../src/r2-client.ts";
import { BUCKET, ENDPOINT, FakeR2, HASH_A, KEY_ID, manifestKey, type Seen, settings, sha256, shardKey } from "./r2-support.ts";

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text);
const refusal = async (run: () => Promise<unknown>): Promise<ObjectStoreError> => {
  try {
    await run();
  } catch (error) {
    expect(error).toBeInstanceOf(ObjectStoreError);
    return error as ObjectStoreError;
  }
  throw new Error("expected a refusal");
};
const first = (fake: FakeR2, k = 0): Seen => {
  const seen = fake.seen[k];
  if (seen === undefined) throw new Error(`no request ${k}`);
  return seen;
};
const open = (fake = new FakeR2(), timeoutMs?: number): { fake: FakeR2; store: R2ObjectStore } => ({
  fake,
  store: new R2ObjectStore(settings(), { fetch: fake.fetch, ...(timeoutMs === undefined ? {} : { timeoutMs }) }),
});

describe("the request shape of a conditional write", () => {
  it("is a path-style PUT with If-None-Match, the body's SHA-256 and a SigV4 signature that signs both", async () => {
    const { fake, store } = open();
    const body = bytes("shard bytes");
    await store.putIfAbsent(shardKey(HASH_A, 3), body);
    const seen = first(fake);
    expect(seen.method).toBe("PUT");
    expect(seen.url).toBe(`${ENDPOINT}/${BUCKET}/${shardKey(HASH_A, 3)}`);
    expect(seen.headers.get("if-none-match")).toBe("*");
    expect(seen.headers.get("x-amz-content-sha256")).toBe(sha256(body));
    expect(seen.headers.get("x-amz-date")).toMatch(/^\d{8}T\d{6}Z$/);
    expect(Array.from(seen.body)).toEqual(Array.from(body));
    const auth = seen.headers.get("authorization") ?? "";
    expect(auth).toMatch(new RegExp(`^AWS4-HMAC-SHA256 Credential=${KEY_ID}/\\d{8}/auto/s3/aws4_request, SignedHeaders=`));
    const signed = /SignedHeaders=([^,]+),/.exec(auth)?.[1]?.split(";") ?? [];
    expect(signed).toEqual(expect.arrayContaining(["host", "if-none-match", "x-amz-content-sha256", "x-amz-date"]));
    expect(auth).toMatch(/Signature=[0-9a-f]{64}$/);
    expect(seen.redirect).toBe("error");
    expect(seen.hasSignal).toBe(true);
  });

  it("signs a read with the hash of the empty body and no If-None-Match", async () => {
    const { fake, store } = open();
    await store.get(manifestKey(HASH_A));
    await store.head(manifestKey(HASH_A));
    expect(fake.seen.map((s) => s.method)).toEqual(["GET", "HEAD"]);
    for (const seen of fake.seen) {
      expect(seen.headers.get("x-amz-content-sha256")).toBe(sha256(new Uint8Array()));
      expect(seen.headers.has("if-none-match")).toBe(false);
    }
  });

  it("overwrites current.json without a condition", async () => {
    const { fake, store } = open();
    await store.overwriteCurrent(bytes("{}"));
    const seen = first(fake);
    expect(seen.url).toBe(`${ENDPOINT}/${BUCKET}/current.json`);
    expect(seen.headers.has("if-none-match")).toBe(false);
    expect(seen.headers.get("x-amz-content-sha256")).toBe(sha256(bytes("{}")));
  });

  it("lists with a signed ListObjectsV2 query and follows the continuation token", async () => {
    const { fake, store } = open();
    fake.pageSize = 2;
    for (let k = 0; k < 5; k += 1) fake.objects.set(shardKey(HASH_A, k), bytes("x"));
    fake.objects.set(`snapshots/${HASH_A}/odd&name`, bytes("x"));
    expect(await store.list("snapshots/")).toEqual([`snapshots/${HASH_A}/odd&name`, shardKey(HASH_A, 0), shardKey(HASH_A, 1), shardKey(HASH_A, 2), shardKey(HASH_A, 3), shardKey(HASH_A, 4)]);
    expect(fake.seen).toHaveLength(3);
    const firstUrl = new URL(first(fake).url);
    expect(firstUrl.pathname).toBe(`/${BUCKET}`);
    expect(firstUrl.searchParams.get("list-type")).toBe("2");
    expect(firstUrl.searchParams.get("prefix")).toBe("snapshots/");
    expect(new URL(first(fake, 1).url).searchParams.get("continuation-token")).toBe("2");
  });
});

describe("status mapping", () => {
  const answer = (status: number, body = "body that must not be repeated", headers: Record<string, string> = {}): FetchLike => () => Promise.resolve(new Response(body, { status, headers }));
  const storeFor = (fetch: FetchLike): R2ObjectStore => new R2ObjectStore(settings(), { fetch });

  it("maps 412 and 409 on a conditional write to exists and success to success", async () => {
    for (const status of [412, 409]) expect((await refusal(() => storeFor(answer(status)).putIfAbsent(manifestKey(HASH_A), bytes("x")))).code).toBe("exists");
    for (const status of [200, 201]) await storeFor(answer(status)).putIfAbsent(manifestKey(HASH_A), bytes("x"));
  });

  it.each([
    [401, "denied"],
    [403, "denied"],
    [429, "unavailable"],
    [500, "unavailable"],
    [503, "unavailable"],
    [400, "failed"],
    [404, "failed"],
    [301, "failed"],
  ])("maps status %i on every operation to %s with a fixed message", async (status, code) => {
    const store = storeFor(answer(status));
    const runs: (() => Promise<unknown>)[] = [
      () => store.putIfAbsent(manifestKey(HASH_A), bytes("x")),
      () => store.overwriteCurrent(bytes("x")),
      () => store.list("snapshots/"),
      () => store.head(manifestKey(HASH_A)),
      () => store.get(manifestKey(HASH_A)),
    ];
    for (const run of runs) {
      // A 404 is "absent" for get and head; the other statuses are refusals for all.
      if (status === 404 && (run === runs[3] || run === runs[4])) continue;
      const error = await refusal(run);
      expect(error.code).toBe(code);
      expect(error.message).not.toContain("body that must not");
    }
  });

  it("treats 404 as absent for get and head", async () => {
    expect(await storeFor(answer(404)).get(manifestKey(HASH_A))).toBeNull();
    expect(await storeFor(answer(404)).head(manifestKey(HASH_A))).toBeNull();
  });

  it("refuses an answer that is not in the expected form", async () => {
    expect((await refusal(() => storeFor(answer(200, "<html>not a listing</html>")).list("snapshots/"))).code).toBe("response_invalid");
    expect((await refusal(() => storeFor(answer(200, "<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>")).list("snapshots/"))).code).toBe("response_invalid");
    expect((await refusal(() => storeFor(() => Promise.resolve(new Response(null, { status: 200 }))).head(manifestKey(HASH_A)))).code).toBe("response_invalid");
    const endless: FetchLike = () => Promise.resolve(new Response("<ListBucketResult><IsTruncated>true</IsTruncated><NextContinuationToken>t</NextContinuationToken></ListBucketResult>", { status: 200 }));
    expect((await refusal(() => storeFor(endless).list("snapshots/"))).code).toBe("response_invalid");
  });

  it("drops a listed key with an entity it does not know and keeps the others", async () => {
    const page = `<ListBucketResult><IsTruncated>false</IsTruncated><Contents><Key>snapshots/a&#38;b</Key></Contents><Contents><Key>snapshots/ok&lt;1</Key></Contents><Contents><Key>elsewhere/x</Key></Contents></ListBucketResult>`;
    expect(await storeFor(answer(200, page)).list("snapshots/")).toEqual(["snapshots/ok<1"]);
  });

  it("refuses an object larger than the limit, by its declared size and by its real size", async () => {
    const declared = new Response("small", { status: 200, headers: { "content-length": String(MAX_OBJECT_BYTES + 1) } });
    expect((await refusal(() => storeFor(() => Promise.resolve(declared)).get(manifestKey(HASH_A)))).code).toBe("too_large");
    expect((await refusal(() => storeFor(answer(200)).putIfAbsent(manifestKey(HASH_A), { byteLength: MAX_OBJECT_BYTES + 1 } as Uint8Array))).code).toBe("too_large");
    const lying = { status: 200, headers: new Headers(), body: null, arrayBuffer: () => Promise.resolve(new ArrayBuffer(MAX_OBJECT_BYTES + 1)) } as unknown as Response;
    expect((await refusal(() => storeFor(() => Promise.resolve(lying)).get(manifestKey(HASH_A)))).code).toBe("too_large");
  });

  it("maps a body that fails mid-read, a network error and a timeout to unavailable, whatever they say", async () => {
    const broken = { status: 200, headers: new Headers(), body: null, arrayBuffer: () => Promise.reject(new Error(`read failed at ${ENDPOINT}`)) } as unknown as Response;
    expect((await refusal(() => storeFor(() => Promise.resolve(broken)).get(manifestKey(HASH_A)))).code).toBe("unavailable");
    const network = (): Promise<Response> => Promise.reject(new TypeError(`fetch failed ${ENDPOINT}/${BUCKET} ${KEY_ID}`));
    const error = await refusal(() => storeFor(network).get(manifestKey(HASH_A)));
    expect(error.code).toBe("unavailable");
    expect(error.message + (error.stack ?? "")).not.toMatch(/canary/i);
    const slow = (request: Request): Promise<Response> => new Promise((_, reject) => request.signal.addEventListener("abort", () => reject(new Error("aborted"))));
    expect((await refusal(() => open(Object.assign(new FakeR2(), { fetch: slow }), 20).store.get(manifestKey(HASH_A)))).code).toBe("unavailable");
  });

  it("drops a response body it cannot cancel", async () => {
    const stubborn = { status: 404, headers: new Headers(), body: { cancel: () => Promise.reject(new Error("no")) } } as unknown as Response;
    expect(await storeFor(() => Promise.resolve(stubborn)).get(manifestKey(HASH_A))).toBeNull();
  });

  it("uses the global fetch when none is given", async () => {
    const calls: string[] = [];
    vi.stubGlobal("fetch", (request: Request) => (calls.push(request.method), Promise.resolve(new Response(null, { status: 404 }))));
    try {
      expect(await new R2ObjectStore(settings()).get(manifestKey(HASH_A))).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
    expect(calls).toEqual(["GET"]);
  });

  it("refuses layout-invalid keys before any request", async () => {
    const { fake, store } = open();
    await refusal(() => store.putIfAbsent("snapshots/_lock-check/x.txt", bytes("x")));
    await refusal(() => store.list("snapshots/_lock-check/"));
    expect(fake.seen).toEqual([]);
  });
});
