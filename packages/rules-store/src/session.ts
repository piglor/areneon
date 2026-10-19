// Opens the rules store session a command runs on (ADR-007). The URL carries
// every connection setting, TLS included: production uses
// `sslmode=verify-full`, so the server's certificate and host name are
// checked. The URL is never logged.

import pg from "pg";

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

const ignore = (): void => {};

/**
 * Connect a `pg` client to `url`, with the timeouts set as startup
 * parameters for the whole session. A statement that exceeds one fails, so
 * applyImport rolls its transaction back. The client fits applyImport's
 * ImportClient; the caller ends it. A dropped idle connection would emit
 * 'error' and, unheard, crash the process; the next query reports it instead.
 */
export async function connect(url: string, options: SessionOptions = {}): Promise<pg.Client> {
  const timeouts = options.timeouts ?? DEFAULT_TIMEOUTS;
  for (const [name, value] of Object.entries(timeouts)) {
    if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`timeout ${name} must be a positive integer`);
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
  return client;
}
