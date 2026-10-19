/**
 * The bot client connects only to a Showdown server on this machine (ADR-005:
 * A2 is a server bound to 127.0.0.1). `assertLocal` is the guard every
 * transport calls before it connects.
 */

/** The only hosts accepted, exactly as WHATWG `URL` serializes them. */
const LOCAL_HOSTNAMES: ReadonlySet<string> = new Set(["127.0.0.1", "[::1]", "localhost"]);

const SCHEMES: ReadonlySet<string> = new Set(["ws:", "wss:"]);

/** The scheme and authority as written, before any `URL` normalization. */
const RAW_AUTHORITY = /^([a-z]+):\/\/([^/?#]*)/;

/** A URL refused by `assertLocal`. */
export class LocalUrlError extends Error {
  override name = "LocalUrlError";
}

/**
 * Returns the parsed URL if it is a ws or wss URL whose host is exactly
 * 127.0.0.1, [::1] or localhost, and throws a `LocalUrlError` otherwise.
 *
 * The authority must already be in canonical form: `URL` rewrites
 * `2130706433`, `0x7f.1`, `127.1` and `[0:0:0:0:0:0:0:1]` to a loopback host,
 * so the text as written must equal what `URL` serializes. That also refuses
 * userinfo, upper case, a default port spelled out, a trailing dot and every
 * other loopback address (127.0.0.2, [::ffff:127.0.0.1], 0.0.0.0). IPv6 zone
 * ids do not parse at all. A fragment, even an empty one, is refused.
 */
export function assertLocal(url: string): URL {
  // `URL` reports an empty fragment as "", so look for the "#" itself.
  if (url.includes("#")) {
    throw new LocalUrlError("fragment refused: a WebSocket URL has none");
  }
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new LocalUrlError(`not a valid URL: ${JSON.stringify(url)}`);
  }
  if (!SCHEMES.has(parsed.protocol)) {
    throw new LocalUrlError(`scheme ${parsed.protocol} refused: only ws: and wss: are allowed`);
  }
  if (parsed.username !== "" || parsed.password !== "") {
    throw new LocalUrlError("userinfo refused: a local server URL has none");
  }
  if (!LOCAL_HOSTNAMES.has(parsed.hostname)) {
    throw new LocalUrlError(`host ${parsed.hostname} refused: only 127.0.0.1, [::1] and localhost are allowed`);
  }
  const raw = RAW_AUTHORITY.exec(url);
  if (raw === null || `${raw[1]}:` !== parsed.protocol || raw[2] !== parsed.host) {
    throw new LocalUrlError(
      `${JSON.stringify(url)} refused: write the scheme and host exactly as ${parsed.protocol}//${parsed.host}`,
    );
  }
  return parsed;
}
