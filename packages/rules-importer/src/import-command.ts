// The rules import command (ADR-007 §6), minus its composition root (cli.ts):
// connect as `importer`, read which snapshots are stored, plan the import of
// every released artifact, apply it in one transaction, and print a one-line
// JSON summary. The database URL comes only from the environment and never
// reaches the output; a refusal anywhere gives a non-zero exit code.

import {
  applyImport,
  connect,
  DEFAULT_TIMEOUTS,
  type ImportClient,
  type ImportResult,
  SessionConfigError,
  type SessionTimeouts,
} from "@areneon/rules-store";
import { planImport, type ImportPlan, type ReleaseSource } from "@areneon/showdown-bridge";

/** The login role the import must run as (INSERT and SELECT on the rules tables only). */
export const IMPORTER_ROLE = "importer";

/** Exit codes: success, a refusal during the import, a configuration error. */
export const EXIT_OK = 0;
export const EXIT_REFUSED = 1;
export const EXIT_CONFIG = 2;

/** A connected database session; `end` closes it. */
export interface DbSession extends ImportClient {
  end(): Promise<void>;
}

export interface ImportCommandOptions {
  /** Read for `DATABASE_URL` and the `RULES_IMPORT_*_TIMEOUT_MS` settings (TIMEOUT_ENV) only. */
  readonly env: Readonly<Record<string, string | undefined>>;
  readonly source: ReleaseSource;
  /** Opens the session with the given timeouts; default the rules store's `pg` connect. */
  readonly connect?: (url: string, timeouts: SessionTimeouts) => Promise<DbSession>;
  /** Applies the plan; default the rules store's applyImport. */
  readonly apply?: (client: ImportClient, plan: ImportPlan) => Promise<ImportResult>;
  /** Receives the one-line summary; default stdout. */
  readonly out?: (line: string) => void;
}

class ConfigError extends Error {
  override name = "ConfigError";
}

const connectImporter = (url: string, timeouts: SessionTimeouts): Promise<DbSession> =>
  connect(url, { applicationName: "areneon-rules-importer", timeouts });

/**
 * The environment variables that override the session's timeouts, in
 * milliseconds (defaults: 300000, 300000, 60000). A statement that exceeds
 * one fails, and the import rolls back and exits EXIT_REFUSED.
 */
export const TIMEOUT_ENV: Readonly<Record<keyof SessionTimeouts, string>> = {
  statementMs: "RULES_IMPORT_STATEMENT_TIMEOUT_MS",
  lockMs: "RULES_IMPORT_LOCK_TIMEOUT_MS",
  idleInTransactionMs: "RULES_IMPORT_IDLE_IN_TRANSACTION_TIMEOUT_MS",
};
// 1 ms to 999999999 ms (about 11.5 days), inside Postgres' integer range.
const MILLISECONDS = /^[1-9][0-9]{0,8}$/;

function timeoutsFrom(env: ImportCommandOptions["env"]): SessionTimeouts {
  const read = (key: keyof SessionTimeouts): number => {
    const value = env[TIMEOUT_ENV[key]];
    if (value === undefined) return DEFAULT_TIMEOUTS[key];
    if (!MILLISECONDS.test(value)) {
      throw new ConfigError(`${TIMEOUT_ENV[key]} must be a whole number of milliseconds from 1 to 999999999`);
    }
    return Number(value);
  };
  return { statementMs: read("statementMs"), lockMs: read("lockMs"), idleInTransactionMs: read("idleInTransactionMs") };
}

/**
 * Run one import. Returns the exit code: EXIT_OK when every step was inserted,
 * already present or referenced, EXIT_REFUSED when anything refused (the plan, the
 * attestation, the database) or a channel step was stale, EXIT_CONFIG when
 * DATABASE_URL is missing or sets a session timeout, a timeout setting is
 * invalid, or the session is not the importer role.
 */
export async function runImportCommand(options: ImportCommandOptions): Promise<number> {
  const out = options.out ?? ((line: string) => process.stdout.write(`${line}\n`));
  const url = options.env.DATABASE_URL;
  if (url === undefined || url === "") {
    out(JSON.stringify({ ok: false, error: "ConfigError: DATABASE_URL is not set" }));
    return EXIT_CONFIG;
  }
  let session: DbSession | undefined;
  try {
    const timeouts = timeoutsFrom(options.env);
    session = await (options.connect ?? connectImporter)(url, timeouts);
    await requireImporter(session);
    const known = await storedSnapshots(session);
    const plan = await planImport(options.source, { snapshots: known });
    const { steps } = await (options.apply ?? applyImport)(session, plan);
    const ok = steps.every((step) => step.status !== "stale");
    out(JSON.stringify({ ok, steps }));
    return ok ? EXIT_OK : EXIT_REFUSED;
  } catch (error) {
    const { name, message } = error instanceof Error ? error : new Error(String(error));
    out(JSON.stringify({ ok: false, error: `${name}: ${redact(message, url)}` }));
    return error instanceof ConfigError || error instanceof SessionConfigError ? EXIT_CONFIG : EXIT_REFUSED;
  } finally {
    await session?.end().catch(() => {});
  }
}

async function requireImporter(session: ImportClient): Promise<void> {
  const { rows } = await session.query("SELECT session_user AS login, current_user AS effective");
  const { login, effective } = rows[0] ?? {};
  if (login !== IMPORTER_ROLE || effective !== IMPORTER_ROLE) {
    throw new ConfigError(
      `the rules import runs as ${IMPORTER_ROLE}, not session ${String(login)} / current ${String(effective)}`,
    );
  }
}

async function storedSnapshots(session: ImportClient): Promise<Set<string>> {
  const { rows } = await session.query("SELECT hash FROM rules.rules_snapshot");
  return new Set(
    rows.map(({ hash }) => {
      if (typeof hash !== "string") throw new Error("rules.rules_snapshot returned a non-text hash");
      return hash;
    }),
  );
}

/** Remove the URL and its password from a message, should a driver ever echo them. */
function redact(message: string, url: string): string {
  const secrets = [url];
  let password = "";
  try {
    ({ password } = new URL(url));
  } catch {
    // Not WHATWG-parsable; the whole URL is still redacted.
  }
  if (password !== "") {
    secrets.push(password);
    try {
      secrets.push(decodeURIComponent(password));
    } catch {
      // A malformed escape: the raw form is redacted.
    }
  }
  return secrets.reduce((text, secret) => text.replaceAll(secret, "[redacted]"), message);
}
