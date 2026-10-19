// The rules import command (ADR-007 §6), minus its composition root (cli.ts):
// connect as `importer`, read which snapshots are stored, plan the import of
// every released artifact, apply it in one transaction, and print a one-line
// JSON summary. The database URL comes only from the environment and never
// reaches the output; a refusal anywhere gives a non-zero exit code.

import { applyImport, connect, type ImportClient, type ImportResult } from "@areneon/rules-store";
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
  /** Read for `DATABASE_URL` only. */
  readonly env: Readonly<Record<string, string | undefined>>;
  readonly source: ReleaseSource;
  /** Opens the session; default the rules store's `pg` connect. */
  readonly connect?: (url: string) => Promise<DbSession>;
  /** Applies the plan; default the rules store's applyImport. */
  readonly apply?: (client: ImportClient, plan: ImportPlan) => Promise<ImportResult>;
  /** Receives the one-line summary; default stdout. */
  readonly out?: (line: string) => void;
}

class ConfigError extends Error {
  override name = "ConfigError";
}

const connectImporter = (url: string): Promise<DbSession> => connect(url, "areneon-rules-importer");

/**
 * Run one import. Returns the exit code: EXIT_OK when every step was inserted
 * or already present, EXIT_REFUSED when anything refused (the plan, the
 * attestation, the database) or a channel step was stale, EXIT_CONFIG when
 * DATABASE_URL is missing or the session is not the importer role.
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
    session = await (options.connect ?? connectImporter)(url);
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
    return error instanceof ConfigError ? EXIT_CONFIG : EXIT_REFUSED;
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
  try {
    const { password } = new URL(url);
    if (password !== "") secrets.push(password, decodeURIComponent(password));
  } catch {
    // Not WHATWG-parsable; the whole URL is still redacted.
  }
  return secrets.reduce((text, secret) => text.replaceAll(secret, "[redacted]"), message);
}
