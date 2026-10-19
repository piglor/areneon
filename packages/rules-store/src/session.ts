// Opens the rules store session a command runs on (ADR-007). The URL carries
// every connection setting, TLS included: production uses
// `sslmode=verify-full`, so the server's certificate and host name are
// checked. The URL is never logged.

import pg from "pg";
import { parse } from "pg-connection-string";

/** Server-side limits on the session, in milliseconds; each must be positive. */
export interface SessionTimeouts {
  /** Cancel any statement running longer than this. */
  readonly statementMs: number;
  /** Cancel any statement waiting longer than this for a lock, the per-channel import lock included. */
  readonly lockMs: number;
  /** End a session left idle inside an open transaction for longer than this. */
  readonly idleInTransactionMs: number;
}

export interface SessionOptions {
  readonly applicationName?: string;
  readonly timeouts?: SessionTimeouts;
}

/**
 * The defaults: 300 s for a statement and for a lock wait (generous for the
 * import's per-channel lock behind another import), 60 s idle in a
 * transaction.
 */
export const DEFAULT_TIMEOUTS: SessionTimeouts = { statementMs: 300_000, lockMs: 300_000, idleInTransactionMs: 60_000 };

/** A connection the session's settings would not hold on: the caller's configuration is at fault. */
export class SessionConfigError extends Error {
  override name = "SessionConfigError";
}

const ignore = (): void => {};

/** Each timeout's server setting (milliseconds in pg_settings). */
const SETTING: Readonly<Record<keyof SessionTimeouts, string>> = {
  statementMs: "statement_timeout",
  lockMs: "lock_timeout",
  idleInTransactionMs: "idle_in_transaction_session_timeout",
};
/**
 * URL query keys that would override the timeouts: pg lays the URL's
 * parameters over the explicit config, and `options=-c …` sets any setting.
 * Keys are compared in lower case after the parser's percent-decoding. This
 * early refusal only gives a clear message; the real guard is the
 * pg_settings read-back after connecting, which catches any route.
 */
const OVERRIDING_KEYS = [...Object.values(SETTING), "options"];

/**
 * Connect a `pg` client to `url`, with the timeouts set as startup
 * parameters for the whole session, then read them back. A statement that
 * exceeds one fails, so applyImport rolls its transaction back. Refuses with
 * a SessionConfigError, before connecting, a timeout that is not a positive
 * integer or a URL whose query sets a timeout or `options`; and, after
 * connecting (the client is then closed), a session whose settings differ
 * from the requested ones (as a pooler that drops startup parameters could
 * make them). The client fits
 * applyImport's ImportClient; the caller ends it. A dropped idle connection
 * would emit 'error' and, unheard, crash the process; the next query reports
 * it instead.
 */
export async function connect(url: string, options: SessionOptions = {}): Promise<pg.Client> {
  const timeouts = options.timeouts ?? DEFAULT_TIMEOUTS;
  for (const [name, value] of Object.entries(timeouts)) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new SessionConfigError(`timeout ${name} must be a positive integer`);
    }
  }
  let keys: string[];
  try {
    keys = Object.keys(parse(url)).map((key) => key.toLowerCase());
  } catch {
    // The parser's message could quote the URL.
    throw new SessionConfigError("the database URL is not a connection string");
  }
  const overriding = OVERRIDING_KEYS.filter((key) => keys.includes(key));
  if (overriding.length > 0) {
    throw new SessionConfigError(`the database URL must not set ${overriding.join(", ")}: the session's timeouts are fixed`);
  }
  const client = new pg.Client({
    connectionString: url,
    application_name: options.applicationName ?? "areneon-rules-store",
    connectionTimeoutMillis: 10_000,
    statement_timeout: timeouts.statementMs,
    lock_timeout: timeouts.lockMs,
    idle_in_transaction_session_timeout: timeouts.idleInTransactionMs,
  });
  client.on("error", ignore);
  await client.connect();
  try {
    const { rows } = await client.query<{ name: string; setting: string }>(
      "SELECT name, setting FROM pg_settings WHERE name = ANY($1)",
      [Object.values(SETTING)],
    );
    const actual = new Map(rows.map((row) => [row.name, row.setting]));
    for (const [key, name] of Object.entries(SETTING) as [keyof SessionTimeouts, string][]) {
      if (actual.get(name) !== String(timeouts[key])) {
        throw new SessionConfigError(`${name} is ${String(actual.get(name))} ms, not the requested ${timeouts[key]} ms`);
      }
    }
  } catch (error) {
    await client.end().catch(ignore);
    throw error;
  }
  return client;
}
