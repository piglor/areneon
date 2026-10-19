// Opens the rules store session a command runs on (ADR-007). The URL carries
// every connection setting, TLS included: production uses
// `sslmode=verify-full`, so the server's certificate and host name are
// checked. The URL is never logged.

import pg from "pg";

const ignore = (): void => {};

/**
 * Connect a `pg` client to `url`. The client fits `applyImport`'s
 * ImportClient; the caller ends it. A dropped idle connection would emit
 * 'error' and, unheard, crash the process; the next query reports it instead.
 */
export async function connect(url: string, applicationName = "areneon-rules-store"): Promise<pg.Client> {
  const client = new pg.Client({
    connectionString: url,
    application_name: applicationName,
    connectionTimeoutMillis: 10_000,
  });
  client.on("error", ignore);
  await client.connect();
  return client;
}
