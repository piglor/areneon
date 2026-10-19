// Canonical JSON and RFC 8785 (JCS), with no imports at all (ADR-008 §5):
// the browser-bundled ObservationBuilder and `scripts/rules-hash.ts` (rerun
// from bare archives of old commits) can both depend on it. A test keeps it
// import-free.

/** A JSON value as found in a Showdown `|request|` payload. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * Serialise `value` deterministically: object keys sorted, no whitespace.
 * Two requests with the same content always produce the same string,
 * whichever path (in-process sim or server connection) observed them.
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

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

function assertIJson(value: Json): void {
  if (typeof value === "string") {
    if (LONE_SURROGATE.test(value)) throw new RangeError("jcs: lone surrogate in string");
  } else if (Array.isArray(value)) {
    value.forEach(assertIJson);
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      assertIJson(key);
      assertIJson(item);
    }
  }
}

/**
 * RFC 8785 (JCS) canonical form. `canonicalJson` already sorts keys by UTF-16
 * code units and prints numbers the ECMAScript way; this adds the I-JSON
 * check it lacks: no lone surrogate, in keys or values. Every language
 * reproduces fixtures/contracts/jcs-vectors.json. An integer literal beyond
 * ±(2^53−1) cannot be held exactly by a JS number at all; contract schemas
 * bound every integer inside that range and only validated data is hashed.
 * `canonicalJson` is left unchanged so `request_hash` and `ruleset_hash`
 * keep their bytes.
 */
export function jcs(value: Json): string {
  assertIJson(value);
  return canonicalJson(value);
}
