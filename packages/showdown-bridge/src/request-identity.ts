import { createHash } from "node:crypto";

/** A JSON value as found in a Showdown `|request|` payload. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * Serialise `value` deterministically: object keys sorted, no whitespace.
 * Two requests with the same content always produce the same string,
 * whichever path (headless sim or browser client) observed them.
 */
export function canonicalJson(value: Json): string {
  if (typeof value === "number" && !Number.isFinite(value)) {
    // JSON.stringify would turn NaN/Infinity into null and collide with it.
    throw new RangeError(`canonicalJson: non-finite number ${value}`);
  }
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k] as Json)}`).join(",")}}`;
}

/**
 * Areneon's request identity (ADR-004 r2): SHA-256 of the canonical request
 * JSON. `rqid` is excluded because only the Showdown server assigns it, so
 * the same request hashes identically on the sim and server paths.
 */
export function requestHash(request: { [key: string]: Json }): string {
  const { rqid, ...rest } = request;
  return createHash("sha256").update(canonicalJson(rest)).digest("hex");
}
