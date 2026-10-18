// The rules store's database contract (ADR-007), against a real Postgres 18:
// `make db-up` first. Every test runs in a transaction it rolls back, since
// the tables are append-only.

import { readFileSync } from "node:fs";
import { loadRulesSnapshot, PINNED_SHOWDOWN_COMMIT, rulesSnapshotHash, snapshotBytes } from "@areneon/showdown-bridge";
import pg from "pg";
import { afterAll, describe, expect, it } from "vitest";

const HOST = process.env.RULES_DB_HOST ?? "127.0.0.1:55432";
// The throwaway local passwords `make db-up` sets (never the Coolify ones).
const ROLES = { migrator: "rules_migrator", importer: "importer", reader: "areneon_reader" } as const;
const PASSWORDS: Record<keyof typeof ROLES, string> = {
  migrator: "local-migrator",
  importer: "local-importer",
  reader: "local-reader",
};

const pools: pg.Pool[] = [];
function pool(role: keyof typeof ROLES): pg.Pool {
  const p = new pg.Pool({ connectionString: `postgres://${ROLES[role]}:${PASSWORDS[role]}@${HOST}/areneon` });
  pools.push(p);
  return p;
}
const db = { migrator: pool("migrator"), importer: pool("importer"), reader: pool("reader") };
afterAll(async () => {
  await Promise.all(pools.map((p) => p.end()));
});

/** Run `body` in a transaction that is always rolled back. */
async function rolledBack(role: keyof typeof db, body: (client: pg.PoolClient) => Promise<void>): Promise<void> {
  const client = await db[role].connect();
  try {
    await client.query("BEGIN");
    await body(client);
  } finally {
    await client.query("ROLLBACK");
    client.release();
  }
}

/** Expect the statement to fail inside a savepoint, so the transaction lives on. */
async function refused(client: pg.PoolClient, sql: string, values: unknown[], error: RegExp): Promise<void> {
  await client.query("SAVEPOINT probe");
  await expect(client.query(sql, values)).rejects.toThrow(error);
  await client.query("ROLLBACK TO SAVEPOINT probe");
}

// The snapshot the repository's rules/ records build, validated by the loader.
const snapshot = loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT);
const body = snapshotBytes(snapshot);
const hash = rulesSnapshotHash(snapshot);
const SOURCE = "a".repeat(40);
const INSERT_SNAPSHOT = `INSERT INTO rules.rules_snapshot
  (hash, body, payload, schema_version, showdown_commit, source_commit, published_at)
  VALUES ($1, $2, $3, $4, $5, $6, now())`;
const snapshotRow = (overrides: Partial<Record<"hash" | "schema_version" | "showdown_commit" | "source_commit", unknown>> = {}) => [
  overrides.hash ?? hash,
  body,
  JSON.parse(body.toString("utf8")),
  overrides.schema_version ?? snapshot.schema_version,
  overrides.showdown_commit ?? snapshot.showdown_commit,
  overrides.source_commit ?? SOURCE,
];
const INSERT_RELEASE = "INSERT INTO rules.rules_release (channel, seq, hash, source_commit) VALUES ($1, $2, $3, $4)";

describe("the importer role", () => {
  it("inserts a snapshot and its release, idempotently", async () => {
    await rolledBack("importer", async (client) => {
      await client.query(INSERT_SNAPSHOT, snapshotRow());
      await client.query(`${INSERT_SNAPSHOT} ON CONFLICT DO NOTHING`, snapshotRow());
      await client.query(INSERT_RELEASE, ["stable", 1, hash, SOURCE]);
      await client.query(`${INSERT_RELEASE} ON CONFLICT DO NOTHING`, ["stable", 1, hash, SOURCE]);
      const { rows } = await client.query("SELECT seq, hash FROM rules.rules_release");
      expect(rows).toEqual([{ seq: 1, hash }]);
      const stored = await client.query("SELECT encode(sha256(body), 'hex') AS h FROM rules.rules_snapshot");
      expect(stored.rows[0].h).toBe(hash);
    });
  });

  it("may not update, delete or truncate", async () => {
    await rolledBack("importer", async (client) => {
      await client.query(INSERT_SNAPSHOT, snapshotRow());
      await refused(client, "UPDATE rules.rules_snapshot SET source_commit = $1", ["b".repeat(40)], /permission denied/);
      await refused(client, "DELETE FROM rules.rules_snapshot", [], /permission denied/);
      await refused(client, "TRUNCATE rules.rules_release", [], /permission denied/);
    });
  });
});

describe("the append-only triggers", () => {
  it("refuse UPDATE, DELETE and TRUNCATE even for the owner", async () => {
    await rolledBack("migrator", async (client) => {
      await client.query("SET LOCAL ROLE rules_owner");
      await client.query(INSERT_SNAPSHOT, snapshotRow());
      await client.query(INSERT_RELEASE, ["stable", 1, hash, SOURCE]);
      for (const table of ["rules_snapshot", "rules_release"]) {
        await refused(client, `UPDATE rules.${table} SET source_commit = $1`, ["b".repeat(40)], /append-only: UPDATE/);
        await refused(client, `DELETE FROM rules.${table}`, [], /append-only: DELETE/);
      }
      await refused(client, "TRUNCATE rules.rules_release", [], /append-only: TRUNCATE on rules_release/);
      await refused(client, "TRUNCATE rules.rules_snapshot CASCADE", [], /append-only: TRUNCATE/);
    });
  });
});

describe("the constraints", () => {
  it.each<[string, unknown[], RegExp]>([
    ["a hash that is not sha256(body)", snapshotRow({ hash: "0".repeat(64) }), /snapshot_hash_is_sha256_of_body/],
    ["a schema_version that disagrees with body", snapshotRow({ schema_version: 99 }), /snapshot_schema_version_from_payload/],
    ["a showdown_commit that disagrees with body", snapshotRow({ showdown_commit: "f".repeat(40) }), /snapshot_showdown_commit_from_payload/],
    ["a malformed source_commit", snapshotRow({ source_commit: "main" }), /snapshot_source_commit_is_sha1/],
  ])("refuse a snapshot with %s", async (_label, row, error) => {
    await rolledBack("importer", async (client) => {
      await refused(client, INSERT_SNAPSHOT, row, error);
    });
  });

  it("refuse a payload that is not body's JSON, or a body that is not UTF-8 JSON", async () => {
    await rolledBack("importer", async (client) => {
      // Each row's hash is consistent with its body, so only the payload ties fail.
      const forged = Buffer.from('{"a":1}');
      const forgedHash = (await client.query("SELECT encode(sha256($1::bytea), 'hex') AS h", [forged])).rows[0].h;
      const row = (bytes: Buffer, h: string, payload: unknown) => [h, bytes, payload, 7, "whatever", SOURCE];
      await refused(client, INSERT_SNAPSHOT, row(forged, forgedHash, { x: 1 }), /snapshot_payload_is_body/);
      // payload equals body here, but the extracted columns name keys body lacks.
      await refused(client, INSERT_SNAPSHOT, row(forged, forgedHash, { a: 1 }), /snapshot_schema_version_from_payload/);
      const notUtf8 = Buffer.from([0xff]);
      const notUtf8Hash = (await client.query("SELECT encode(sha256($1::bytea), 'hex') AS h", [notUtf8])).rows[0].h;
      await refused(client, INSERT_SNAPSHOT, row(notUtf8, notUtf8Hash, {}), /invalid byte sequence for encoding "UTF8"/);
    });
  });

  it("refuse a release that breaks its channel's history", async () => {
    await rolledBack("importer", async (client) => {
      await client.query(INSERT_SNAPSHOT, snapshotRow());
      await client.query(INSERT_RELEASE, ["stable", 1, hash, SOURCE]);
      await refused(client, INSERT_RELEASE, ["stable", 1, hash, "c".repeat(40)], /release_seq_unique_per_channel/);
      await refused(client, INSERT_RELEASE, ["stable", 2, hash, SOURCE], /release_source_commit_unique_per_channel/);
      await refused(client, INSERT_RELEASE, ["beta", 2, hash, "c".repeat(40)], /release_channel_in_enum/);
      await refused(client, INSERT_RELEASE, ["stable", 0, hash, "c".repeat(40)], /release_seq_positive/);
      await refused(client, INSERT_RELEASE, ["stable", 2, "e".repeat(64), "c".repeat(40)], /release_snapshot_exists/);
    });
  });

  it("keep the channel CHECK equal to the channel schema's closed enum", async () => {
    const { rows } = await db.reader.query<{ def: string }>(
      "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint WHERE conname = 'release_channel_in_enum'",
    );
    const inDb = [...(rows[0]?.def ?? "").matchAll(/'([^']+)'::text/g)].map((m) => m[1]).sort();
    const schema = JSON.parse(
      readFileSync(new URL("../../schema/rules/v1/channel.schema.json", import.meta.url), "utf8"),
    ) as { properties: { channel: { enum: string[] } } };
    expect(inDb).toEqual([...schema.properties.channel.enum].sort());
  });
});

describe("the reader role", () => {
  it("reads but never writes", async () => {
    await rolledBack("reader", async (client) => {
      const { rows } = await client.query("SELECT count(*)::int AS n FROM rules.rules_snapshot");
      expect(rows[0].n).toBeGreaterThanOrEqual(0);
      await refused(client, INSERT_SNAPSHOT, snapshotRow(), /permission denied/);
    });
  });
});

describe("the roles", () => {
  it("give the migrator the owner's rights only after SET ROLE", async () => {
    await rolledBack("migrator", async (client) => {
      await refused(client, "SELECT count(*) FROM rules.rules_snapshot", [], /permission denied/);
      await client.query("SET LOCAL ROLE rules_owner");
      await client.query("SELECT count(*) FROM rules.rules_snapshot");
    });
  });

  it("leave no TEMP privilege on the database to the login roles", async () => {
    for (const role of ["importer", "reader"] as const) {
      await rolledBack(role, async (client) => {
        await refused(client, "CREATE TEMP TABLE probe (x int)", [], /permission denied/);
      });
    }
  });
});

describe("the migration", () => {
  it("rolls back cleanly", async () => {
    const sql = readFileSync(new URL("../../../db/migrations/20261019000000_rules_store.sql", import.meta.url), "utf8");
    const down = sql.slice(sql.indexOf("-- migrate:down") + "-- migrate:down".length);
    await rolledBack("migrator", async (client) => {
      await client.query(down);
      const { rows } = await client.query("SELECT to_regclass('rules.rules_snapshot') AS t, current_user AS u");
      expect(rows[0]).toEqual({ t: null, u: "rules_migrator" });
    });
  });
});
