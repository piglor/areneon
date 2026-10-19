import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { ImportClient, ImportResult, SessionTimeouts, StepOutcome } from "@areneon/rules-store";
import type { ImportPlan, ReleaseAsset, ReleaseSource } from "@areneon/showdown-bridge";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type DbSession, EXIT_CONFIG, EXIT_OK, EXIT_REFUSED, runImportCommand, TIMEOUT_ENV } from "../src/index.ts";
import { FIXTURES } from "./fake-gh.ts";

const SNAPSHOT_HASH = "7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8";
const SNAPSHOT_TAG = `rules-snapshot-${SNAPSHOT_HASH}`;
const CHANNEL_TAG = "rules-channel-stable-b226c3bfe889f8cf775933b0a964a44ef6d69913";
const asset = async (tag: string, commit: string): Promise<ReleaseAsset> => ({
  tag,
  name: `${tag}.json`,
  commit,
  bytes: new Uint8Array(await readFile(join(FIXTURES, `${tag}.json`))),
});
const SNAPSHOT = await asset(SNAPSHOT_TAG, "35f9c5f04aecbe1dfc38e55d2599b1013f1b43b7");
const CHANNEL = await asset(CHANNEL_TAG, "b226c3bfe889f8cf775933b0a964a44ef6d69913");
const sourceOf = (...assets: ReleaseAsset[]): ReleaseSource => ({ list: async () => assets });

const PASSWORD = "s3cret-Pw";
const URL_ = `postgres://importer:${PASSWORD}@db.internal:5432/areneon?sslmode=verify-full`;

interface FakeDb {
  login?: unknown;
  effective?: unknown;
  hashes?: unknown[];
  endFails?: boolean;
}

/** A session answering the command's own two queries; applyImport is injected separately. */
function fakeSession({ login = "importer", effective = "importer", hashes = [], endFails = false }: FakeDb = {}) {
  const queries: string[] = [];
  let ended = 0;
  const session: DbSession = {
    query: async (text) => {
      queries.push(text);
      if (text.includes("session_user")) return { rows: [{ login, effective }], rowCount: 1 };
      if (text === "SELECT hash FROM rules.rules_snapshot") {
        return { rows: hashes.map((hash) => ({ hash })), rowCount: hashes.length };
      }
      throw new Error(`unexpected query ${text}`);
    },
    end: async () => {
      ended++;
      if (endFails) throw new Error("already closed");
    },
  };
  return { session, queries, ended: () => ended };
}

const inserted = (plan: ImportPlan, status: StepOutcome["status"] = "inserted"): ImportResult => ({
  steps: plan.steps.map((step): StepOutcome =>
    step.kind === "snapshot"
      ? { kind: "snapshot", hash: step.hash, status: status === "stale" ? "already-present" : status }
      : { kind: "channel", channel: step.channel, seq: step.seq, hash: step.hash, status },
  ),
});

async function run(options: {
  env?: Record<string, string | undefined>;
  source?: ReleaseSource;
  db?: FakeDb;
  connect?: (url: string, timeouts: SessionTimeouts) => Promise<DbSession>;
  apply?: (client: ImportClient, plan: ImportPlan) => Promise<ImportResult>;
}) {
  const lines: string[] = [];
  const db = fakeSession(options.db);
  const connected: string[] = [];
  const timeouts: SessionTimeouts[] = [];
  const plans: ImportPlan[] = [];
  const code = await runImportCommand({
    env: options.env ?? { DATABASE_URL: URL_ },
    source: options.source ?? sourceOf(SNAPSHOT, CHANNEL),
    connect:
      options.connect ??
      (async (url, given) => {
        connected.push(url);
        timeouts.push(given);
        return db.session;
      }),
    apply:
      options.apply ??
      (async (client, plan) => {
        expect(client).toBe(db.session);
        plans.push(plan);
        return inserted(plan);
      }),
    out: (line) => lines.push(line),
  });
  expect(lines).toHaveLength(1);
  const output = lines[0] as string;
  expect(output).not.toContain(PASSWORD);
  expect(output).not.toContain("db.internal");
  return { code, summary: JSON.parse(output) as Record<string, unknown>, db, connected, plans, timeouts };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("runImportCommand", () => {
  it("plans every released artifact, applies it and prints the outcomes", async () => {
    const { code, summary, db, connected, plans } = await run({});
    expect(code).toBe(EXIT_OK);
    expect(connected).toEqual([URL_]);
    expect(plans[0]?.steps.map((step) => step.kind)).toEqual(["snapshot", "channel"]);
    expect(summary).toEqual({
      ok: true,
      steps: [
        { kind: "snapshot", hash: SNAPSHOT_HASH, status: "inserted" },
        { kind: "channel", channel: "stable", seq: 5, hash: SNAPSHOT_HASH, status: "inserted" },
      ],
    });
    expect(db.ended()).toBe(1);
  });

  it("connects with the default timeouts", async () => {
    const { timeouts } = await run({});
    expect(timeouts).toEqual([{ statementMs: 300_000, lockMs: 300_000, idleInTransactionMs: 60_000 }]);
  });

  it("takes each timeout from its environment variable", async () => {
    expect(TIMEOUT_ENV).toEqual({
      statementMs: "RULES_IMPORT_STATEMENT_TIMEOUT_MS",
      lockMs: "RULES_IMPORT_LOCK_TIMEOUT_MS",
      idleInTransactionMs: "RULES_IMPORT_IDLE_IN_TRANSACTION_TIMEOUT_MS",
    });
    const env = {
      DATABASE_URL: URL_,
      RULES_IMPORT_STATEMENT_TIMEOUT_MS: "1",
      RULES_IMPORT_LOCK_TIMEOUT_MS: "999999999",
      RULES_IMPORT_IDLE_IN_TRANSACTION_TIMEOUT_MS: "45000",
    };
    const { code, timeouts } = await run({ env });
    expect(code).toBe(EXIT_OK);
    expect(timeouts).toEqual([{ statementMs: 1, lockMs: 999_999_999, idleInTransactionMs: 45_000 }]);
  });

  it.each(
    Object.values(TIMEOUT_ENV).flatMap((name) =>
      ["", "0", "-1", "1.5", "1e3", " 5", "05", "1000000000", "abc"].map((value) => [name, value] as const),
    ),
  )("refuses %s=%j without connecting", async (name, value) => {
    const { code, summary, connected } = await run({ env: { DATABASE_URL: URL_, [name]: value } });
    expect(code).toBe(EXIT_CONFIG);
    expect(summary).toEqual({
      ok: false,
      error: `ConfigError: ${name} must be a whole number of milliseconds from 1 to 999999999`,
    });
    expect(connected).toEqual([]);
  });

  it("refuses an import a timeout cancelled (applyImport has rolled it back)", async () => {
    const { code, summary, db } = await run({
      apply: async () => {
        throw Object.assign(new Error("canceling statement due to lock timeout"), { name: "error", code: "55P03" });
      },
    });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary).toEqual({ ok: false, error: "error: canceling statement due to lock timeout" });
    expect(db.ended()).toBe(1);
  });

  it("lets a channel name a snapshot that is already stored", async () => {
    const { code, plans } = await run({ source: sourceOf(CHANNEL), db: { hashes: [SNAPSHOT_HASH] } });
    expect(code).toBe(EXIT_OK);
    expect(plans[0]?.steps.map((step) => step.kind)).toEqual(["channel"]);
  });

  it("fails a stale channel step, after the transaction", async () => {
    const { code, summary } = await run({ apply: async (_, plan) => inserted(plan, "stale") });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary.ok).toBe(false);
    expect(summary.steps).toHaveLength(2);
  });

  it.each([
    ["unset", {}],
    ["empty", { DATABASE_URL: "" }],
  ])("refuses a DATABASE_URL that is %s, without connecting", async (_, env) => {
    const { code, summary, connected } = await run({ env });
    expect(code).toBe(EXIT_CONFIG);
    expect(summary).toEqual({ ok: false, error: "ConfigError: DATABASE_URL is not set" });
    expect(connected).toEqual([]);
  });

  it.each([
    ["another login role", { login: "areneon_reader", effective: "areneon_reader" }, "not session areneon_reader / current areneon_reader"],
    ["a role set after login", { effective: "rules_owner" }, "not session importer / current rules_owner"],
    ["a login set into importer", { login: "postgres" }, "not session postgres / current importer"],
  ])("refuses %s before reading anything", async (_, db, message) => {
    const { code, summary, db: fake, plans } = await run({ db });
    expect(code).toBe(EXIT_CONFIG);
    expect(summary).toEqual({ ok: false, error: `ConfigError: the rules import runs as importer, ${message}` });
    expect(fake.queries).toHaveLength(1);
    expect(plans).toEqual([]);
    expect(fake.ended()).toBe(1);
  });

  it("refuses a plan the planner refuses, without applying it", async () => {
    const { code, summary, plans } = await run({ source: sourceOf(CHANNEL) });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary.error).toMatch(/^ImportPlanError: channel stable seq 5 names snapshot 7c618d78/);
    expect(plans).toEqual([]);
  });

  it("refuses when the source fails", async () => {
    const source: ReleaseSource = { list: () => Promise.reject(new Error("gh offline")) };
    const { code, summary } = await run({ source });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary.error).toBe("ImportPlanError: could not list the released rules artifacts: gh offline");
  });

  it("refuses a stored hash that is not text", async () => {
    const { code, summary } = await run({ db: { hashes: [Buffer.from("x")] } });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary.error).toBe("Error: rules.rules_snapshot returned a non-text hash");
  });

  it("reports a database refusal and still closes the session, even if closing fails", async () => {
    const { code, summary, db } = await run({
      db: { endFails: true },
      apply: async () => {
        throw Object.assign(new Error("violates check constraint"), { name: "error" });
      },
    });
    expect(code).toBe(EXIT_REFUSED);
    expect(summary.error).toBe("error: violates check constraint");
    expect(db.ended()).toBe(1);
  });

  it("reports a thrown non-Error", async () => {
    const { summary } = await run({ connect: () => Promise.reject("boom" as unknown as Error) });
    expect(summary.error).toBe("Error: boom");
  });

  it.each([
    ["the URL", URL_],
    ["its password", PASSWORD],
    ["its percent-decoded password", "p@ss/word"],
  ])("redacts %s from an error message", async (_, secret) => {
    const url = secret === "p@ss/word" ? "postgres://importer:p%40ss%2Fword@db.internal/areneon" : URL_;
    const { summary } = await run({
      env: { DATABASE_URL: url },
      connect: async () => {
        throw new Error(`cannot reach ${secret}`);
      },
    });
    expect(summary.error).toBe("Error: cannot reach [redacted]");
  });

  it("redacts a password whose percent escape is malformed", async () => {
    const { summary } = await run({
      env: { DATABASE_URL: "postgres://importer:p%zzw@db.internal/areneon" },
      connect: async () => {
        throw new Error("auth failed for p%zzw");
      },
    });
    expect(summary.error).toBe("Error: auth failed for [redacted]");
  });

  it("refuses a URL that sets a session timeout as a configuration error, before connecting", async () => {
    const code = await runImportCommand({
      env: { DATABASE_URL: `postgres://importer:${PASSWORD}@127.0.0.1:1/areneon?statement_timeout=0` },
      source: sourceOf(),
      out: (line) => {
        expect(JSON.parse(line)).toEqual({
          ok: false,
          error: "SessionConfigError: the database URL must not set statement_timeout: the session's timeouts are fixed",
        });
      },
    });
    expect(code).toBe(EXIT_CONFIG);
  });

  it("redacts a URL that is not WHATWG-parsable as a whole", async () => {
    const url = "host=db password=xyz";
    const { summary } = await run({
      env: { DATABASE_URL: url },
      connect: async () => {
        throw new Error(`bad ${url}`);
      },
    });
    expect(summary.error).toBe("Error: bad [redacted]");
  });

  it("connects with pg by default and prints to stdout", async () => {
    const write = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    // Port 1 refuses at once; the password must not reach the output.
    const code = await runImportCommand({
      env: { DATABASE_URL: `postgres://importer:${PASSWORD}@127.0.0.1:1/areneon` },
      source: sourceOf(),
    });
    expect(code).toBe(EXIT_REFUSED);
    expect(write).toHaveBeenCalledTimes(1);
    const line = String(write.mock.calls[0]?.[0]);
    expect(line).toMatch(/^\{"ok":false,"error":".*ECONNREFUSED.*"\}\n$/);
    expect(line).not.toContain(PASSWORD);
  });
});
