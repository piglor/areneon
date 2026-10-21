import { createHash } from "node:crypto";
import { type Json } from "./canonical-json.ts";
import { requestIdentityBytes } from "./request-bytes.ts";

export { canonicalJson, type Json } from "./canonical-json.ts";

/**
 * Areneon's request identity (ADR-008 §4): SHA-256 of the bytes
 * `requestIdentityBytes` defines (the canonical request JSON without `rqid`),
 * hashed synchronously with Node's `createHash`.
 */
export function requestHash(request: { [key: string]: Json }): string {
  return nodeSha256Hex(requestIdentityBytes(request));
}

/** SHA-256 of `bytes` as lowercase hex, synchronously on Node. */
export function nodeSha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}
