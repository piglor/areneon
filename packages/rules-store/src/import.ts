// The rules store's writer (ADR-007): applies an import plan in one
// transaction, as the `importer` role (INSERT and SELECT only). Re-running a
// plan is a no-op: a snapshot already stored is left alone, and a channel step
// is written only when its `seq` is above the channel's current maximum.
// Anything the database refuses (a body that is not its hash, a channel naming
// a snapshot that is not stored) rolls the whole plan back. Each channel the
// plan names is locked for the transaction, in sorted order, before its
// maximum seq is read, so concurrent imports of one channel run one after the
// other and cannot commit their seqs out of order.

// These mirror the planner's ImportPlan (`rules-import-plan.ts` in
// @areneon/showdown-bridge); the import CLI switches to importing it.
export interface ImportSnapshot {
  kind: "snapshot";
  hash: string;
  /** The commit the snapshot's Release tag points at (40 hex). */
  source_commit: string;
  bytes: Uint8Array;
}
export interface ImportChannel {
  kind: "channel";
  channel: string;
  seq: number;
  hash: string;
  source_commit: string;
  bytes: Uint8Array;
}
export type ImportStep = ImportSnapshot | ImportChannel;
/** Snapshots first, then channels by (channel, seq). */
export interface ImportPlan {
  steps: ImportStep[];
}

/** What happened to one step: written, already stored, or below the channel's history. */
export type StepStatus = "inserted" | "already-present" | "stale";
export type StepOutcome =
  | { kind: "snapshot"; hash: string; status: Exclude<StepStatus, "stale"> }
  | { kind: "channel"; channel: string; seq: number; hash: string; status: StepStatus };
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
 */
export async function applyImport(client: ImportClient, plan: ImportPlan): Promise<ImportResult> {
  await client.query("BEGIN");
  try {
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
      steps.push(step.kind === "snapshot" ? await importSnapshot(client, step) : await importChannel(client, step));
    }
    await client.query("COMMIT");
    return { steps };
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  }
}

async function importSnapshot(client: ImportClient, step: ImportSnapshot): Promise<StepOutcome> {
  const { rowCount } = await client.query(INSERT_SNAPSHOT, [step.hash, Buffer.from(step.bytes), step.source_commit]);
  return { kind: "snapshot", hash: step.hash, status: rowCount === 1 ? "inserted" : "already-present" };
}

async function importChannel(client: ImportClient, step: ImportChannel): Promise<StepOutcome> {
  const outcome = (status: StepStatus): StepOutcome => ({
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
