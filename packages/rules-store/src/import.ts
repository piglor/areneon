// The rules store's writer (ADR-007): applies an import plan in one
// transaction, as the `importer` role (INSERT and SELECT only). Re-running a
// plan is a no-op: a snapshot already stored is left alone, and a channel step
// is written only when its `seq` is above the channel's current maximum.
// A snapshot the plan names by reference (`stored-snapshot`, one the release
// source did not download again) must be stored with its hash the sha256 of
// its stored body. Anything refused (a body that is not its hash, a channel
// naming a snapshot that is not stored, a referenced snapshot that is not
// stored) rolls the whole plan back. Each channel the plan names is locked for
// the transaction, in sorted order, before its maximum seq is read, so
// concurrent imports of one channel run one after the other and cannot commit
// their seqs out of order.

import type { ImportChannel, ImportPlan, ImportSnapshot, ImportStoredSnapshot } from "@areneon/showdown-bridge";

/**
 * What happened to one step: written, already stored, or below the channel's
 * history; for a snapshot planned by reference, `referenced`: not downloaded,
 * and found stored with a body that hashes to it.
 */
export type StepStatus = "inserted" | "already-present" | "stale" | "referenced";
export type StepOutcome =
  | { kind: "snapshot"; hash: string; status: Exclude<StepStatus, "stale"> }
  | { kind: "channel"; channel: string; seq: number; hash: string; status: Exclude<StepStatus, "referenced"> };
export interface ImportResult {
  steps: StepOutcome[];
}

/** The part of a `pg` client this needs; a pg.Client or pg.PoolClient fits. */
export interface ImportClient {
  query(text: string, values?: unknown[]): Promise<{ rows: Record<string, unknown>[]; rowCount: number | null }>;
}

export class ImportError extends Error {
  override name = "ImportError";
}

const IMPORTER = "importer";

// Payload and the extracted columns are derived from the body in SQL, so the
// row's CHECKs bind them; only `hash` comes from the plan, and
// snapshot_hash_is_sha256_of_body checks it.
const INSERT_SNAPSHOT = `INSERT INTO rules.rules_snapshot
  (hash, body, payload, schema_version, showdown_commit, source_commit, published_at)
  SELECT $1, $2::bytea, p, (p ->> 'schema_version')::integer, p ->> 'showdown_commit', $3, now()
  FROM (SELECT convert_from($2::bytea, 'UTF8')::jsonb AS p) AS parsed
  ON CONFLICT (hash) DO NOTHING`;
// The row must exist and its body must hash to it (the CHECK holds this
// already; reading it again costs one digest and trusts no earlier read).
const STORED_SNAPSHOT = `SELECT 1 FROM rules.rules_snapshot
  WHERE hash = $1 AND hash = encode(sha256(body), 'hex')`;
// Held to the end of the transaction; the key names the table and channel.
const LOCK_CHANNEL = "SELECT pg_advisory_xact_lock(hashtext('areneon.rules_release:' || $1))";
const MAX_SEQ = "SELECT max(seq) AS seq FROM rules.rules_release WHERE channel = $1";
const SAME_RELEASE = `SELECT 1 FROM rules.rules_release
  WHERE channel = $1 AND seq = $2 AND hash = $3 AND source_commit = $4`;
const INSERT_RELEASE = "INSERT INTO rules.rules_release (channel, seq, hash, source_commit) VALUES ($1, $2, $3, $4)";

/**
 * Apply `plan` in one transaction on `client`, which must be connected as
 * `importer`. Returns each step's outcome in plan order; on any failure it
 * rolls back, so nothing of the plan is written, and rethrows.
 *
 * Snapshots are inserted in plan order, which the planner fixes by hash, so
 * two importers insert overlapping snapshots in the same order. A channel step
 * above the channel's maximum whose source_commit another seq of that channel
 * already uses fails release_source_commit_unique_per_channel and rolls the
 * plan back (fail closed).
 */
export async function applyImport(client: ImportClient, plan: ImportPlan): Promise<ImportResult> {
  await client.query("BEGIN");
  try {
    // A fail-closed guard matching the spec; the GRANTs are the real enforcement.
    const { rows } = await client.query("SELECT current_user AS u");
    if (rows[0]?.u !== IMPORTER) {
      throw new ImportError(`the rules import runs as ${IMPORTER}, not ${String(rows[0]?.u)}`);
    }
    const channels = [...new Set(plan.steps.flatMap((step) => (step.kind === "channel" ? [step.channel] : [])))].sort();
    for (const channel of channels) {
      await client.query(LOCK_CHANNEL, [channel]);
    }
    const steps: StepOutcome[] = [];
    for (const step of plan.steps) {
      if (step.kind === "snapshot") steps.push(await importSnapshot(client, step));
      else if (step.kind === "stored-snapshot") steps.push(await checkStoredSnapshot(client, step));
      else steps.push(await importChannel(client, step));
    }
    await client.query("COMMIT");
    return { steps };
  } catch (error) {
    // A dead connection must not mask the original error.
    await client.query("ROLLBACK").catch(() => {});
    throw error;
  }
}

async function importSnapshot(client: ImportClient, step: ImportSnapshot): Promise<StepOutcome> {
  const { rowCount } = await client.query(INSERT_SNAPSHOT, [step.hash, Buffer.from(step.bytes), step.source_commit]);
  return { kind: "snapshot", hash: step.hash, status: rowCount === 1 ? "inserted" : "already-present" };
}

async function checkStoredSnapshot(client: ImportClient, step: ImportStoredSnapshot): Promise<StepOutcome> {
  const { rows } = await client.query(STORED_SNAPSHOT, [step.hash]);
  if (rows.length !== 1) {
    throw new ImportError(`snapshot ${step.hash} is planned as stored, but no stored row has that hash`);
  }
  return { kind: "snapshot", hash: step.hash, status: "referenced" };
}

async function importChannel(client: ImportClient, step: ImportChannel): Promise<StepOutcome> {
  const outcome = (status: Exclude<StepStatus, "referenced">): StepOutcome => ({
    kind: "channel",
    channel: step.channel,
    seq: step.seq,
    hash: step.hash,
    status,
  });
  const { rows } = await client.query(MAX_SEQ, [step.channel]);
  const max = rows[0]?.seq;
  if (typeof max === "number" && step.seq <= max) {
    const same = await client.query(SAME_RELEASE, [step.channel, step.seq, step.hash, step.source_commit]);
    return outcome(same.rows.length === 1 ? "already-present" : "stale");
  }
  await client.query(INSERT_RELEASE, [step.channel, step.seq, step.hash, step.source_commit]);
  return outcome("inserted");
}
