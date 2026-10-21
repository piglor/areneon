// The one definition of which bytes a request identity hashes (ADR-008 §4, §5).
// Engine-free and free of built-in imports: it imports only canonical JSON, so
// the ObservationBuilder (WebCrypto) and the Node decoder share it.

import { canonicalJson, type Json } from "./canonical-json.ts";

/** SHA-256 of some bytes as lowercase hex: a Node or a WebCrypto implementation behind one port. */
export type Sha256 = (bytes: Uint8Array) => Promise<string>;

/**
 * The bytes `request_hash` is the SHA-256 of: the UTF-8 canonical JSON of the
 * request without `rqid`, which only the Showdown server assigns, so the sim
 * and server paths hash alike. Pure; it hashes nothing.
 */
export function requestIdentityBytes(request: { [key: string]: Json }): Uint8Array {
  return new TextEncoder().encode(requestIdentityText(request));
}

/** The same bytes as text: the canonical request JSON without `rqid`. */
export function requestIdentityText(request: { [key: string]: Json }): string {
  const { rqid, ...rest } = request;
  return canonicalJson(rest);
}

/** `request_hash` through an injected `Sha256` port. */
export function hashRequest(request: { [key: string]: Json }, sha256: Sha256): Promise<string> {
  return sha256(requestIdentityBytes(request));
}
