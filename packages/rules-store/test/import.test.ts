// applyImport (ADR-007) against a real Postgres 18: `make db-up` first.
// The tables are append-only and schema.test.ts expects them empty, so every
// test runs inside an outer transaction it rolls back; applyImport's own
// BEGIN / COMMIT / ROLLBACK become a savepoint, its release and its rollback.

import { createHash } from "node:crypto";
import { loadRulesSnapshot, PINNED_SHOWDOWN_COMMIT, rulesSnapshotHash, snapshotBytes } from "@areneon/showdown-bridge";
import pg from "pg";
import { afterAll, describe, expect, it } from "vitest";
import { applyImport, type ImportChannel, type ImportClient, ImportError, type ImportPlan, type ImportStep } from "../src/import.ts";

const HOST = process.env.RULES_DB_HOST ?? "127.0.0.1:55432";
// The throwaway local passwords `make db-up` sets (never the Coolify ones).
const importer = new pg.Pool({ connectionString: `postgres://importer:local-importer@${HOST}/areneon` });
const reader = new pg.Pool({ connectionString: `postgres://areneon_reader:local-reader@${HOST}/areneon` });
afterAll(async () => {
  await Promise.all([importer.end(), reader.end()]);
});

const NESTED: Record<string, string> = {
  BEGIN: "SAVEPOINT apply_import",
  COMMIT: "RELEASE SAVEPOINT apply_import",
  ROLLBACK: "ROLLBACK TO SAVEPOINT apply_import",
};

/** Run `body` with a client whose transactions nest in one that is always rolled back. */
async function rolledBack(pool: pg.Pool, body: (client: ImportClient) => Promise<void>): Promise<void> {
  const client = await pool.connect();
  const nested: ImportClient = {
    query: (text, values) => client.query(NESTED[text] ?? text, values),
  };
  try {
    await client.query("BEGIN");
    await body(nested);
  } finally {
    await client.query("ROLLBACK");
    client.release();
  }
}

// The repository's own snapshot, plus a second, synthetic one: the database
// checks only its hash and the two keys it extracts.
const realBytes = new Uint8Array(snapshotBytes(loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT)));
const realHash = rulesSnapshotHash(loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT));
const otherBytes = new TextEncoder().encode(`{"schema_version":1,"showdown_commit":"${"f".repeat(40)}","synthetic":true}`);
const otherHash = createHash("sha256").update(otherBytes).digest("hex");
const SOURCE = "a".repeat(40);
const CHANNEL_BYTES = new TextEncoder().encode("{}");

const snapshot = (hash: string, bytes: Uint8Array, source_commit = SOURCE): ImportStep => ({
  kind: "snapshot",
  hash,
  source_commit,
  bytes,
});
const channel = (seq: number, hash: string, source_commit: string): ImportChannel => ({
  kind: "channel",
  channel: "stable",
  seq,
  hash,
  source_commit,
  bytes: CHANNEL_BYTES,
});
const plan = (...steps: ImportStep[]): ImportPlan => ({ steps });

async function stored(client: ImportClient): Promise<{ snapshots: unknown[]; releases: unknown[] }> {
  const snapshots = await client.query(
    "SELECT hash, source_commit, schema_version FROM rules.rules_snapshot ORDER BY hash",
  );
  const releases = await client.query("SELECT channel, seq, hash, source_commit FROM rules.rules_release ORDER BY seq");
  return { snapshots: snapshots.rows, releases: releases.rows };
}

describe("applyImport", () => {
  it("writes a plan once and reports each step; re-running it is a no-op", async () => {
    await rolledBack(importer, async (client) => {
      const first = plan(snapshot(realHash, realBytes), channel(1, realHash, SOURCE));
      expect(await applyImport(client, first)).toEqual({
        steps: [
          { kind: "snapshot", hash: realHash, status: "inserted" },
          { kind: "channel", channel: "stable", seq: 1, hash: realHash, status: "inserted" },
        ],
      });
      const after = await stored(client);
      expect(after.releases).toEqual([{ channel: "stable", seq: 1, hash: realHash, source_commit: SOURCE }]);
      expect(after.snapshots).toEqual([{ hash: realHash, source_commit: SOURCE, schema_version: 1 }]);

      // The stored snapshot keeps its first source_commit.
      const again = plan(snapshot(realHash, realBytes, "b".repeat(40)), channel(1, realHash, SOURCE));
      expect(await applyImport(client, again)).toEqual({
        steps: [
          { kind: "snapshot", hash: realHash, status: "already-present" },
          { kind: "channel", channel: "stable", seq: 1, hash: realHash, status: "already-present" },
        ],
      });
      expect(await stored(client)).toEqual(after);
    });
  });

  it("refuses a stale seq and writes it nowhere, while the rest of the plan lands", async () => {
    await rolledBack(importer, async (client) => {
      await applyImport(client, plan(snapshot(realHash, realBytes), channel(1, realHash, SOURCE)));
      const result = await applyImport(
        client,
        plan(snapshot(otherHash, otherBytes), channel(1, otherHash, "c".repeat(40)), channel(2, otherHash, "d".repeat(40))),
      );
      expect(result.steps.map((s) => s.status)).toEqual(["inserted", "stale", "inserted"]);
      expect((await stored(client)).releases).toEqual([
        { channel: "stable", seq: 1, hash: realHash, source_commit: SOURCE },
        { channel: "stable", seq: 2, hash: otherHash, source_commit: "d".repeat(40) },
      ]);
    });
  });

  it("reports stale for a seq below the channel's maximum", async () => {
    await rolledBack(importer, async (client) => {
      await applyImport(
        client,
        plan(snapshot(realHash, realBytes), channel(1, realHash, SOURCE), channel(3, realHash, "e".repeat(40))),
      );
      const result = await applyImport(client, plan(channel(2, realHash, "c".repeat(40))));
      expect(result.steps).toEqual([{ kind: "channel", channel: "stable", seq: 2, hash: realHash, status: "stale" }]);
      expect((await stored(client)).releases).toHaveLength(2);
    });
  });

  it.each<[string, ImportPlan, RegExp]>([
    [
      "a channel naming a snapshot that is not stored",
      plan(snapshot(realHash, realBytes), channel(1, otherHash, SOURCE)),
      /release_snapshot_exists/,
    ],
    [
      "a snapshot whose bytes are not its hash",
      plan(snapshot(realHash, realBytes), snapshot(otherHash, realBytes), channel(1, realHash, SOURCE)),
      /snapshot_hash_is_sha256_of_body/,
    ],
    [
      "a channel outside the enum",
      plan(snapshot(realHash, realBytes), { ...channel(1, realHash, SOURCE), channel: "beta" }),
      /release_channel_in_enum/,
    ],
  ])("rolls the whole plan back on %s", async (_label, failing, error) => {
    await rolledBack(importer, async (client) => {
      await expect(applyImport(client, failing)).rejects.toThrow(error);
      expect(await stored(client)).toEqual({ snapshots: [], releases: [] });
    });
  });

  it("waits for a concurrent import of the same channel to end before reading its seqs", async () => {
    // The other import holds the channel's lock in an open transaction.
    const other = await importer.connect();
    try {
      await other.query("BEGIN");
      await other.query("SELECT pg_advisory_xact_lock(hashtext('areneon.rules_release:' || $1))", ["stable"]);
      let settled = false;
      const waiting = rolledBack(importer, async (client) => {
        const result = await applyImport(client, plan(snapshot(realHash, realBytes), channel(1, realHash, SOURCE)));
        expect(result.steps.map((s) => s.status)).toEqual(["inserted", "inserted"]);
      }).finally(() => {
        settled = true;
      });
      const blocked = "SELECT count(*)::int AS n FROM pg_locks WHERE locktype = 'advisory' AND NOT granted";
      const deadline = Date.now() + 5000;
      while ((await other.query(blocked)).rows[0].n !== 1) {
        expect(Date.now()).toBeLessThan(deadline);
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      expect(settled).toBe(false);
      await other.query("ROLLBACK");
      await waiting;
    } finally {
      other.release();
    }
  });

  it("refuses to run as any role but importer, and writes nothing", async () => {
    await rolledBack(reader, async (client) => {
      const run = applyImport(client, plan(snapshot(realHash, realBytes)));
      await expect(run).rejects.toThrow(ImportError);
      await expect(applyImport(client, plan())).rejects.toThrow(
        "the rules import runs as importer, not areneon_reader",
      );
    });
  });
});
