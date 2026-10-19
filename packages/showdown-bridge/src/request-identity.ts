import { createHash } from "node:crypto";
import { canonicalJson, type Json } from "./canonical-json.ts";

export { canonicalJson, type Json } from "./canonical-json.ts";

/**
 * Areneon's request identity (ADR-008 §4): SHA-256 of the canonical request
 * JSON. `rqid` is excluded because only the Showdown server assigns it, so
 * the same request hashes identically on the sim and server paths.
 */
export function requestHash(request: { [key: string]: Json }): string {
  const { rqid, ...rest } = request;
  return createHash("sha256").update(canonicalJson(rest)).digest("hex");
}
