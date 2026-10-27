// Canonical JSON (RFC 8785, the subset this package needs) for the hashes of the sampler and the belief
// identity: strings, safe integers, booleans, null, arrays and objects. The package is engine-free and
// cannot import the bridge's `jcs`, so it carries its own; a test holds the two equal on the same
// values. Object keys sort by UTF-16 code unit as the RFC says; strings serialise as ECMAScript JSON
// does (RFC 8785 section 3.2.2.2). A float, `undefined`, a bigint, a function or a lone surrogate is
// refused: a hash input must be exact.

import { sha256Hex } from "./sha256.ts";

export type Json = null | boolean | number | string | readonly Json[] | { readonly [key: string]: Json };

export function jcs(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new TypeError("canonical JSON here holds integers only");
    return Object.is(value, -0) ? "0" : String(value);
  }
  if (typeof value === "string") {
    if (!isWellFormed(value)) throw new TypeError("canonical JSON refuses a lone surrogate");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(jcs).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${jcs(key)}:${jcs(record[key])}`).join(",")}}`;
  }
  throw new TypeError("canonical JSON refuses this value");
}

function isWellFormed(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = text.charCodeAt(i + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return false;
      i += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) return false;
  }
  return true;
}

/** The value with every bigint written as its decimal string (an exact, portable form for a hash input). */
export function plain(value: unknown): unknown {
  if (typeof value === "bigint") return value.toString();
  if (Array.isArray(value)) return value.map(plain);
  if (value !== null && typeof value === "object") {
    if (Object.getPrototypeOf(value) !== Object.prototype) throw new TypeError("only plain objects are hashed");
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key, plain(v)]));
  }
  return value;
}

/** SHA-256 of the canonical JSON of `value` (bigints as decimal strings), in lower-case hexadecimal. */
export const digest = (value: unknown): string => sha256Hex(jcs(plain(value)));
