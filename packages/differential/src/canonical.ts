// RFC 8785 (JCS) for the values the harness hashes and compares: strings, safe integers, booleans, null, arrays and objects.
// The harness holds no float (ADR-030 rule 3: integers only, no tolerance): a non-integer number is refused here, so a backend
// answer that carries one can never compare equal by accident of printing. The bridge's `canonical-json.ts` and this give the
// same bytes on every integer value (the joining package's test holds them equal); this one adds the integer rule.

/** A JSON value the harness compares. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export class CanonicalError extends Error {
  override name = "CanonicalError";
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

function text(value: string): string {
  if (LONE_SURROGATE.test(value)) throw new CanonicalError("jcs: lone surrogate in a string");
  return JSON.stringify(value);
}

/** The JCS bytes of `value`, refusing a number that is not a safe integer, a lone surrogate and any non-JSON value. */
export function jcs(value: Json): string {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new CanonicalError("jcs: the harness holds integers only");
    return JSON.stringify(value);
  }
  if (typeof value === "string") return text(value);
  if (Array.isArray(value)) return `[${value.map(jcs).join(",")}]`;
  if (typeof value !== "object") throw new CanonicalError("jcs: not a JSON value");
  const keys = Object.keys(value).sort();
  return `{${keys.map((key) => `${text(key)}:${jcs(value[key] as Json)}`).join(",")}}`;
}
