// connect (ADR-007) against a real Postgres 18: `make db-up` first.

import { loadRulesSnapshot, PINNED_SHOWDOWN_COMMIT, rulesSnapshotHash } from "@areneon/showdown-bridge";
import pg from "pg";
import { afterAll, describe, expect, it } from "vitest";
import { applyImport, connect, DEFAULT_TIMEOUTS, type ImportClient } from "../src/index.ts";

const HOST = process.env.RULES_DB_HOST ?? "127.0.0.1:55432";
// The throwaway local password `make db-up` sets (never the Coolify one).
const IMPORTER_URL = `postgres://importer:local-importer@${HOST}/areneon`;

const open: { end(): Promise<void> }[] = [];
afterAll(async () => {
  await Promise.all(open.map((client) => client.end()));
});

const SETTINGS = `SELECT current_setting('statement_timeout') AS statement,
  current_setting('lock_timeout') AS lock,
  current_setting('idle_in_transaction_session_timeout') AS idle`;

const NESTED: Record<string, string> = {
  BEGIN: "SAVEPOINT s",
  COMMIT: "RELEASE SAVEPOINT s",
  ROLLBACK: "ROLLBACK TO SAVEPOINT s",
};

describe("connect", () => {
  it("opens a session as the URL's role, under the given application name", async () => {
    const client = await connect(IMPORTER_URL, { applicationName: "rules-store-test" });
    open.push(client);
    const { rows } = await client.query("SELECT current_user AS u, current_setting('application_name') AS app");
    expect(rows).toEqual([{ u: "importer", app: "rules-store-test" }]);
  });

  it("sets the default timeouts on the session", async () => {
    expect(DEFAULT_TIMEOUTS).toEqual({ statementMs: 300_000, lockMs: 300_000, idleInTransactionMs: 60_000 });
    const client = await connect(IMPORTER_URL);
    open.push(client);
    expect((await client.query(SETTINGS)).rows).toEqual([{ statement: "5min", lock: "5min", idle: "1min" }]);
  });

  it("sets the given timeouts on the session", async () => {
    const timeouts = { statementMs: 1500, lockMs: 2000, idleInTransactionMs: 2500 };
    const client = await connect(IMPORTER_URL, { timeouts });
    open.push(client);
    expect((await client.query(SETTINGS)).rows).toEqual([{ statement: "1500ms", lock: "2s", idle: "2500ms" }]);
  });

  it.each([0, -1, 1.5, Number.NaN, 2 ** 53])("refuses a timeout of %s before connecting", async (value) => {
    const timeouts = { ...DEFAULT_TIMEOUTS, lockMs: value };
    await expect(connect(IMPORTER_URL, { timeouts })).rejects.toThrow("timeout lockMs must be a positive integer");
  });

  it("refuses an import that waits on a channel lock past lock_timeout, and rolls it back", async () => {
    const holder = new pg.Client({ connectionString: IMPORTER_URL });
    await holder.connect();
    const client = await connect(IMPORTER_URL, { timeouts: { ...DEFAULT_TIMEOUTS, lockMs: 200 } });
    open.push(client);
    // The import's own transaction nests in one that is always rolled back,
    // so nothing could stay behind even if the lock were granted.
    const nested: ImportClient = { query: (text, values) => client.query(NESTED[text] ?? text, values) };
    const hash = rulesSnapshotHash(loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT));
    const step = { kind: "channel", channel: "stable", seq: 1, hash, source_commit: "a".repeat(40), bytes: new Uint8Array() } as const;
    try {
      await holder.query("BEGIN");
      await holder.query("SELECT pg_advisory_xact_lock(hashtext('areneon.rules_release:stable'))");
      await client.query("BEGIN");
      await expect(applyImport(nested, { steps: [step] })).rejects.toThrow("canceling statement due to lock timeout");
      // Rolled back to before the import: the session is usable and holds no lock.
      const locks = await client.query(
        "SELECT count(*)::int AS n FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()",
      );
      expect(locks.rows).toEqual([{ n: 0 }]);
    } finally {
      await client.query("ROLLBACK");
      await holder.query("ROLLBACK");
      await holder.end();
    }
  });

  it("survives an error on an idle connection; the next query reports it", async () => {
    const client = await connect(IMPORTER_URL);
    open.push(client);
    expect(client.emit("error", new Error("connection dropped"))).toBe(true);
    const { rows } = await client.query("SELECT current_setting('application_name') AS app");
    expect(rows).toEqual([{ app: "areneon-rules-store" }]);
  });

  it("rejects when the server refuses the login", async () => {
    await expect(connect(`postgres://importer:not-the-password@${HOST}/areneon`)).rejects.toThrow(
      /password authentication failed/,
    );
  });
});
