// What the object store tests share: made-up R2 settings (the values are
// markers, not credentials), the snapshot-shaped keys and a fake R2 behind
// `fetch`. The fake keeps its objects in a map, honours `If-None-Match: *`
// and lists in pages; `override` lets a test answer a request itself. It
// records every request it sees (method, URL, headers, body) so the tests can
// check the request shape. No test opens a socket.

import { createHash } from "node:crypto";
import { R2Settings } from "../src/r2-settings.ts";

export const KEY_ID = "CanaryAccessKeyId7abcd";
export const SECRET = "CanaryMarkerS7/".repeat(2);
export const ACCOUNT = "canaryaccount7";
export const ENDPOINT = `https://${ACCOUNT}.r2.cloudflarestorage.com`;
export const BUCKET = "canary-bucket-7";
export const ENV = { AWS_ACCESS_KEY_ID: KEY_ID, AWS_SECRET_ACCESS_KEY: SECRET, AWS_ENDPOINT_URL: ENDPOINT, AWS_DEFAULT_REGION: "auto" } as const;
export const SECRETS = [KEY_ID, SECRET, ACCOUNT, ENDPOINT, BUCKET];

export const settings = (): R2Settings => R2Settings.fromEnvironment(ENV, [], BUCKET);

export const HASH_A = "a1".repeat(32);
export const HASH_B = "b2".repeat(32);
export const manifestKey = (hash: string): string => `snapshots/${hash}/manifest.json`;
export const shardKey = (hash: string, index: number): string => `snapshots/${hash}/shards/${index}.parquet`;

export interface Seen {
  readonly method: string;
  readonly url: string;
  readonly headers: Headers;
  readonly body: Uint8Array;
  readonly redirect: string;
  readonly hasSignal: boolean;
}

export const sha256 = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
const xml = (text: string, status = 200): Response => new Response(text, { status, headers: { "content-type": "application/xml" } });

export class FakeR2 {
  readonly objects = new Map<string, Uint8Array>();
  readonly seen: Seen[] = [];
  pageSize = 1000;
  override: ((request: Request, seen: Seen) => Response | Promise<Response> | undefined) | undefined;

  readonly fetch = async (request: Request): Promise<Response> => {
    const seen: Seen = {
      method: request.method,
      url: request.url,
      headers: new Headers(request.headers),
      body: new Uint8Array(await request.clone().arrayBuffer()),
      redirect: request.redirect,
      hasSignal: request.signal !== undefined,
    };
    this.seen.push(seen);
    const forced = await this.override?.(request, seen);
    if (forced !== undefined) return forced;
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts[0] !== BUCKET) return xml("<Error>NoSuchBucket</Error>", 404);
    const key = parts.slice(1).join("/");
    if (key === "") return this.list(url);
    if (request.method === "PUT") {
      if (request.headers.get("if-none-match") === "*" && this.objects.has(key)) return xml("<Error>PreconditionFailed</Error>", 412);
      this.objects.set(key, seen.body);
      return new Response(null, { status: 200 });
    }
    const found = this.objects.get(key);
    if (found === undefined) return xml("<Error>NoSuchKey</Error>", 404);
    if (request.method === "HEAD") return new Response(null, { status: 200, headers: { "content-length": String(found.byteLength) } });
    return new Response(found as unknown as BodyInit, { status: 200 });
  };

  private list(url: URL): Response {
    const prefix = url.searchParams.get("prefix") ?? "";
    const all = [...this.objects.keys()].filter((k) => k.startsWith(prefix)).sort();
    const from = Number(url.searchParams.get("continuation-token") ?? 0);
    const page = all.slice(from, from + this.pageSize);
    const more = from + this.pageSize < all.length;
    const contents = page.map((k) => `<Contents><Key>${k.replaceAll("&", "&amp;")}</Key><Size>1</Size></Contents>`).join("");
    return xml(`<?xml version="1.0"?><ListBucketResult><Name>${BUCKET}</Name><IsTruncated>${more}</IsTruncated>${more ? `<NextContinuationToken>${from + this.pageSize}</NextContinuationToken>` : ""}${contents}</ListBucketResult>`);
  }
}
