// connect (ADR-007) against a real Postgres 18: `make db-up` first.

import { loadRulesSnapshot, PINNED_SHOWDOWN_COMMIT, rulesSnapshotHash } from "@areneon/showdown-bridge";
import pg from "pg";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";
import { applyImport, connect, DEFAULT_TIMEOUTS, type ImportClient, SessionConfigError } from "../src/index.ts";

const HOST = process.env.RULES_DB_HOST ?? "127.0.0.1:55432";
// The throwaway local password `make db-up` sets (never the Coolify one).
const IMPORTER_URL = `postgres://importer:local-importer@${HOST}/areneon`;

const open: { end(): Promise<void> }[] = [];
afterAll(async () => {
  await Promise.all(open.map((client) => client.end()));
});

afterEach(() => {
  vi.restoreAllMocks();
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
    const refusal = connect(IMPORTER_URL, { timeouts });
    await expect(refusal).rejects.toBeInstanceOf(SessionConfigError);
    await expect(refusal).rejects.toThrow("timeout lockMs must be a positive integer");
  });

  it.each([
    ["?statement_timeout=0", "statement_timeout"],
    ["?lock_timeout=0", "lock_timeout"],
    ["?idle_in_transaction_session_timeout=0", "idle_in_transaction_session_timeout"],
    ["?options=-c%20statement_timeout%3D0", "options"],
    ["?sslmode=disable&lock_timeout=1&options=x", "lock_timeout, options"],
    ["?STATEMENT_TIMEOUT=0", "statement_timeout"],
    ["?%73tatement_timeout=0", "statement_timeout"],
    ["?Options=-c%20lock_timeout%3D0", "options"],
  ])("refuses a URL ending %s before connecting", async (query, keys) => {
    const connectSpy = vi.spyOn(pg.Client.prototype, "connect");
    const refusal = connect(`${IMPORTER_URL}${query}`);
    await expect(refusal).rejects.toBeInstanceOf(SessionConfigError);
    await expect(refusal).rejects.toThrow(`the database URL must not set ${keys}: the session's timeouts are fixed`);
    expect(connectSpy).not.toHaveBeenCalled();
  });

  it("accepts a URL query that sets no timeout", async () => {
    const client = await connect(`${IMPORTER_URL}?sslmode=disable&application_name=x`);
    open.push(client);
    expect((await client.query(SETTINGS)).rows).toEqual([{ statement: "5min", lock: "5min", idle: "1min" }]);
  });

  it("refuses a URL the parser cannot read, without quoting it", async () => {
    const refusal = connect("postgres://importer:pw@[::1/areneon");
    await expect(refusal).rejects.toThrow(/^the database URL is not a connection string$/);
  });

  it("refuses and closes a session whose settings differ from the requested ones", async () => {
    // As behind a pooler that drops startup parameters: the server reports its own value.
    vi.spyOn(pg.Client.prototype, "query").mockImplementationOnce((async () => {
      return {
        rows: [
          { name: "statement_timeout", setting: "300000" },
          { name: "lock_timeout", setting: "0" },
          { name: "idle_in_transaction_session_timeout", setting: "60000" },
        ],
      };
    }) as unknown as typeof pg.Client.prototype.query);
    const end = vi.spyOn(pg.Client.prototype, "end");
    const refusal = connect(IMPORTER_URL);
    await expect(refusal).rejects.toBeInstanceOf(SessionConfigError);
    await expect(refusal).rejects.toThrow("lock_timeout is 0 ms, not the requested 300000 ms");
    expect(end).toHaveBeenCalledTimes(1);
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
