// The R2 implementation of the object store port (ADR-015 section 1): the
// only module of the private store that touches the network, through
// `aws4fetch` for the SigV4 signature and a `fetch` the caller may replace
// (tests give a fake; nothing in CI opens a socket).
//
// Requests are path-style (`<endpoint>/<bucket>/<key>`, the endpoint from the
// environment), signed with the real SHA-256 of the body in
// `x-amz-content-sha256`, never followed through a redirect, never retried
// (a retried conditional write could turn a success into a conflict) and
// bounded by a time limit. A conditional write carries `If-None-Match: *`; 412
// (and 409, a concurrent conditional write) is `exists`. Every other failure
// maps to a fixed `ObjectStoreError` by status class: the response body, the
// URL, the headers and whatever `fetch` threw are dropped, never put in a
// message, because any of them can hold the endpoint or an echo of a secret.

import {
  checkListPrefix,
  checkObjectKey,
  CURRENT_KEY,
  MAX_OBJECT_BYTES,
  type ObjectHead,
  type ObjectStore,
  ObjectStoreError,
} from "./object-store.ts";
import { type R2Settings, type SignedRequestInit } from "./r2-settings.ts";

export type FetchLike = (request: Request) => Promise<Response>;

export interface R2ClientOptions {
  /** Replaces the global `fetch` (tests). */
  readonly fetch?: FetchLike;
  /** Time limit of one request, in ms; the default is 120000. */
  readonly timeoutMs?: number;
}

const MAX_LIST_BYTES = 64 * 1024 * 1024;
const MAX_LIST_PAGES = 1000;

/** Drop a response body unread; a failure to do so is not worth reporting. */
async function discard(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // Nothing to report: the status is all that matters here.
  }
}

/** The refusal for a status the caller has no special meaning for. */
function refusalFor(status: number): ObjectStoreError {
  if (status === 401 || status === 403) return new ObjectStoreError("denied");
  if (status === 429 || status >= 500) return new ObjectStoreError("unavailable");
  return new ObjectStoreError("failed");
}

async function readBody(response: Response, limit: number): Promise<Uint8Array> {
  const declared = response.headers.get("content-length");
  if (declared !== null && Number(declared) > limit) {
    await discard(response);
    throw new ObjectStoreError("too_large");
  }
  let buffer: ArrayBuffer;
  try {
    buffer = await response.arrayBuffer();
  } catch {
    throw new ObjectStoreError("unavailable");
  }
  if (buffer.byteLength > limit) throw new ObjectStoreError("too_large");
  return new Uint8Array(buffer);
}

const XML_ENTITIES = /&(amp|lt|gt|quot|apos);/g;
const ENTITY_VALUES: Readonly<Record<string, string>> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };

interface ListPage {
  readonly keys: string[];
  readonly next: string | null;
}

/** One ListObjectsV2 answer. Keys with an entity beyond the five named ones cannot be ours and are dropped. */
function parseListPage(text: string): ListPage {
  if (!text.includes("<ListBucketResult")) throw new ObjectStoreError("response_invalid");
  const keys: string[] = [];
  for (const match of text.matchAll(/<Contents>[\s\S]*?<Key>([^<]*)<\/Key>[\s\S]*?<\/Contents>/g)) {
    const raw = match[1] as string;
    const key = raw.replace(XML_ENTITIES, (_, name: string) => ENTITY_VALUES[name] as string);
    if (!raw.replace(XML_ENTITIES, "").includes("&")) keys.push(key);
  }
  if (!/<IsTruncated>true<\/IsTruncated>/.test(text)) return { keys, next: null };
  const token = /<NextContinuationToken>([^<]+)<\/NextContinuationToken>/.exec(text);
  if (token === null) throw new ObjectStoreError("response_invalid");
  return { keys, next: (token[1] as string).replace(XML_ENTITIES, (_, name: string) => ENTITY_VALUES[name] as string) };
}

export class R2ObjectStore implements ObjectStore {
  readonly #settings: R2Settings;
  readonly #fetch: FetchLike;
  readonly #timeoutMs: number;

  constructor(settings: R2Settings, options: R2ClientOptions = {}) {
    this.#settings = settings;
    this.#fetch = options.fetch ?? ((request) => fetch(request));
    this.#timeoutMs = options.timeoutMs ?? 120_000;
  }

  async #send(init: Omit<SignedRequestInit, "signal">): Promise<Response> {
    try {
      const request = await this.#settings.sign({ ...init, signal: AbortSignal.timeout(this.#timeoutMs) });
      return await this.#fetch(request);
    } catch {
      // The error of a failed fetch or signature can quote the URL (the endpoint holds the account id).
      throw new ObjectStoreError("unavailable");
    }
  }

  async putIfAbsent(key: string, body: Uint8Array): Promise<void> {
    checkObjectKey(key);
    if (key === CURRENT_KEY) throw new ObjectStoreError("key_invalid");
    if (body.byteLength > MAX_OBJECT_BYTES) throw new ObjectStoreError("too_large");
    const response = await this.#send({ method: "PUT", key, body, headers: { "if-none-match": "*" } });
    await discard(response);
    if (response.status === 200 || response.status === 201) return;
    if (response.status === 412 || response.status === 409) throw new ObjectStoreError("exists");
    throw refusalFor(response.status);
  }

  async get(key: string): Promise<Uint8Array | null> {
    checkObjectKey(key);
    const response = await this.#send({ method: "GET", key });
    if (response.status === 404) {
      await discard(response);
      return null;
    }
    if (response.status !== 200) {
      await discard(response);
      throw refusalFor(response.status);
    }
    return readBody(response, MAX_OBJECT_BYTES);
  }

  async head(key: string): Promise<ObjectHead | null> {
    checkObjectKey(key);
    const response = await this.#send({ method: "HEAD", key });
    await discard(response);
    if (response.status === 404) return null;
    if (response.status !== 200) throw refusalFor(response.status);
    const length = response.headers.get("content-length");
    if (length === null || !/^(0|[1-9][0-9]{0,15})$/.test(length)) throw new ObjectStoreError("response_invalid");
    return { bytes: Number(length) };
  }

  async list(prefix: string): Promise<string[]> {
    checkListPrefix(prefix);
    const found = new Set<string>();
    let token: string | null = null;
    for (let page = 0; page < MAX_LIST_PAGES; page += 1) {
      const response: Response = await this.#send({
        method: "GET",
        key: null,
        query: { "list-type": "2", prefix, "max-keys": "1000", ...(token === null ? {} : { "continuation-token": token }) },
      });
      if (response.status !== 200) {
        await discard(response);
        throw refusalFor(response.status);
      }
      const parsed = parseListPage(new TextDecoder("utf-8", { fatal: false }).decode(await readBody(response, MAX_LIST_BYTES)));
      for (const key of parsed.keys) if (key.startsWith(prefix)) found.add(key);
      if (parsed.next === null) return [...found].sort();
      token = parsed.next;
    }
    throw new ObjectStoreError("response_invalid");
  }

  async overwriteCurrent(body: Uint8Array): Promise<void> {
    const response = await this.#send({ method: "PUT", key: CURRENT_KEY, body });
    await discard(response);
    if (response.status !== 200 && response.status !== 201) throw refusalFor(response.status);
  }
}
