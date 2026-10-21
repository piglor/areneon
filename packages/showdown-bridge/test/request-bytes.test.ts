import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { canonicalJson } from "../src/canonical-json.ts";
import { GOLDEN_DIR, type RequestFixture } from "../src/golden.ts";
import { nodeSha256 } from "../src/player-channel.ts";
import { hashRequest, requestIdentityBytes } from "../src/request-bytes.ts";
import { decodeRequest, parseRequest, requestHash } from "../src/index.ts";
import { webSha256 } from "../src/web-sha256.ts";

const files = ["preview.request.json", "turn.request.json", "force-switch.request.json"];
const fixture = (file: string): RequestFixture =>
  JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as RequestFixture;
const src = (file: string): string => readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");

describe("parseRequest", () => {
  it.each(files)("parses %s without hashing and carries the identity bytes", (file) => {
    const { request } = fixture(file);
    const parsed = parseRequest(JSON.stringify(request));
    expect(parsed).not.toHaveProperty("request_hash");
    expect(parsed.identity_bytes).toEqual(requestIdentityBytes(request));
    const { identity_bytes: _bytes, ...rest } = parsed;
    expect(decodeRequest(JSON.stringify(request))).toEqual({ ...rest, request_hash: requestHash(request) });
  });

  it("imports no hash function, and the shared bytes module imports only canonical JSON", () => {
    // request.ts reaches Node's hash only through decodeRequest's call of request-identity.
    expect(src("request.ts")).not.toMatch(/createHash|node:crypto|webSha256|subtle/);
    expect(src("request-bytes.ts")).not.toMatch(/node:|createHash|crypto/);
  });
});

describe("the request identity has one definition", () => {
  it.each(files)("Node and WebCrypto ports give the digest of requestHash for %s", async (file) => {
    const { request, request_hash } = fixture(file);
    const node = await hashRequest(request, nodeSha256);
    expect(await hashRequest(request, webSha256)).toBe(node);
    expect(node).toBe(requestHash(request));
    expect(node).toBe(request_hash);
  });

  it("ignores rqid and key order", async () => {
    const { request } = fixture("turn.request.json");
    const base = await hashRequest(request, nodeSha256);
    expect(await hashRequest({ rqid: 9, ...request }, nodeSha256)).toBe(base);
    expect(await hashRequest(Object.fromEntries(Object.entries(request).reverse()), nodeSha256)).toBe(base);
  });

  it("negative control: a different byte definition changes the digest", async () => {
    const { request } = fixture("turn.request.json");
    const right = await hashRequest(request, nodeSha256);
    // rqid kept, pretty-printed and plain JSON.stringify are all other definitions.
    const withRqid = new TextEncoder().encode(canonicalJson({ ...request, rqid: 1 }));
    const pretty = new TextEncoder().encode(JSON.stringify(request, null, 1));
    expect(await nodeSha256(withRqid)).not.toBe(right);
    expect(await nodeSha256(pretty)).not.toBe(right);
  });

  it("no module but request-bytes.ts turns a request into hashed bytes", () => {
    const dirs = ["../src/", "../../bot-client/src/"];
    const all = dirs.flatMap((d) =>
      readdirSync(new URL(d, import.meta.url))
        .filter((f) => f.endsWith(".ts"))
        .map((f) => [`${d}${f}`, readFileSync(new URL(`${d}${f}`, import.meta.url), "utf8")] as const),
    );
    expect(all.some(([f]) => f.includes("bot-client"))).toBe(true);
    for (const [file, source] of all) {
      if (file === "../src/request-bytes.ts") continue;
      expect(source, file).not.toMatch(/const \{ rqid,? \.\.\./);
      if (file === "../src/observation.ts") expect(source).not.toMatch(/canonicalJson|TextEncoder/);
    }
  });
});
