// The pure import planner of the private team store (ADR-015 section 3): staged
// files plus the parent snapshot's records in, the next record list plus a
// counts-only summary out, or a typed refusal. No file system, no network, no
// key: the hold-out flag, the labels and fresh team ids come through ports the
// caller passes (the snapshot layer holds the hold-out key).
//
// Rules (ADR-015 section 3; the two marked * go beyond its text):
//   - a file with a `team_id` is a new revision of that parent record (no
//     change if its paste and tags are unchanged and the record is active; * a
//     tag-only edit or a Pokemon-order-only edit is a new revision; a
//     retired record becomes active again); an id that is not in the parent is
//     refused, never created;
//   - a file without one is a new team with a fresh id, unless its canonical
//     hash equals an active record's (or an earlier file's): it then changes
//     nothing and is counted as a duplicate;
//   - `canonical_team_hash` is unique among active records of the result;
//   - a parent record absent from staging is unchanged, never dropped;
//   - retiring is explicit (`retire`; retiring a retired record is a no-op); a result that lacks an active parent
//     record without a matching retirement is refused;
//   - every staged team goes through the pinned engine's import round trip and
//     validator.
//
// Nothing in a refusal or in the summary is taken from an input (see
// import-error.ts); the summary holds four counts.

import {
  canonicalTeamHash,
  checkPrivateRecordSet,
  type Engine,
  exportPaste,
  importPaste,
  type PokemonSet,
  type PrivateTeamRecord,
  readPrivateTeamRecord,
  recordHash,
  type ResolvedFormat,
  teamHash,
  validateTeam,
} from "@areneon/showdown-bridge";
import { ImportPlanError } from "./import-error.ts";
import { parseStagedFile, type StagedFile } from "./staged-file.ts";

export interface ImportPlanInput {
  readonly engine: Engine;
  /** The resolved format every staged team is validated for. */
  readonly format: ResolvedFormat;
  readonly staged: readonly StagedFile[];
  /** The parent snapshot's records, or null for `--first`. */
  readonly parent: readonly PrivateTeamRecord[] | null;
  /** `--retire` team_ids. */
  readonly retire: readonly string[];
  /** The label values of a team (the committed label set and labeller, passed in). */
  readonly labels: (sets: PokemonSet[]) => PrivateTeamRecord["labels"];
  /** The hold-out flag of a team_id (HMAC with the snapshot's key; the snapshot layer plugs it in). */
  readonly holdout: (teamId: string) => boolean;
  /** A fresh random team_id (32 lowercase hex); asked once per new team. */
  readonly newTeamId: () => string;
}

/**
 * Counts only. `unchanged` counts the parent records the result carries as they
 * were; `duplicate` counts the id-less files that repeat an active team.
 */
export interface ChangeSummary {
  readonly new: number;
  readonly duplicate: number;
  readonly revised: number;
  readonly unchanged: number;
  readonly retired: number;
}

export interface ImportPlan {
  /** The next record list: one record per team_id, sorted by team_id. */
  readonly records: readonly PrivateTeamRecord[];
  readonly summary: ChangeSummary;
}

const TEAM_ID = /^[0-9a-f]{32}$/;
const codeUnits = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const sameList = (a: readonly string[], b: readonly string[]): boolean => a.length === b.length && a.every((v, k) => v === b[k]);

/**
 * The invariant of section 3: every active parent record is, in the result,
 * still active or retired by an explicit `--retire`. The planner builds its
 * result from the parent so this holds by construction; it is checked anyway,
 * and exported so a test can show it fails.
 */
export function assertNoActiveRecordLost(
  parent: readonly PrivateTeamRecord[],
  result: readonly PrivateTeamRecord[],
  retire: ReadonlySet<string>,
): void {
  const byId = new Map(result.map((r) => [r.team_id, r]));
  for (const record of parent) {
    if (record.status !== "active") continue;
    const next = byId.get(record.team_id);
    if (next === undefined || (next.status !== "active" && !retire.has(record.team_id))) {
      throw new ImportPlanError("active_record_lost");
    }
  }
}

/** Run a caller's hook; whatever it throws is replaced by a fixed refusal (the text may hold a paste). */
function hook<T>(run: () => T): T {
  try {
    return run();
  } catch {
    throw new ImportPlanError("hook_failed");
  }
}

/** Plan an import. Throws `ImportPlanError` and nothing else for a refusal. */
export function planImport(input: ImportPlanInput): ImportPlan {
  const { engine, format } = input;
  const parent = input.parent ?? [];
  try {
    for (const record of parent) readPrivateTeamRecord(engine, record);
    checkPrivateRecordSet(parent);
  } catch {
    throw new ImportPlanError("parent_invalid");
  }
  for (const record of parent) {
    if (hook(() => input.holdout(record.team_id)) !== record.holdout) throw new ImportPlanError("holdout_mismatch");
  }
  const parentById = new Map(parent.map((r) => [r.team_id, r]));

  // Retirements first: they are explicit and independent of the staged files.
  const retire = new Set<string>();
  const toRetire = new Set<string>(); // the active ones: retiring a retired record changes nothing
  input.retire.forEach((id, k) => {
    const at = { subject: "retire", index: k + 1 } as const;
    if (!TEAM_ID.test(id)) throw new ImportPlanError("retire_id_malformed", at);
    if (retire.has(id)) throw new ImportPlanError("retire_repeated", at);
    const record = parentById.get(id);
    if (record === undefined) throw new ImportPlanError("retire_unknown", at);
    retire.add(id);
    if (record.status === "active") toRetire.add(id);
  });

  // The next record per team_id, the staged file that made it (null if carried over).
  const next = new Map<string, { record: PrivateTeamRecord; file: number | null }>();
  for (const record of parent) next.set(record.team_id, { record, file: null });
  let created = 0;
  let revised = 0;
  let duplicate = 0;
  const stagedIds = new Set<string>();

  const make = (file: number | null, base: Omit<PrivateTeamRecord, "holdout" | "schema_version" | "data_class">): PrivateTeamRecord => {
    const record: PrivateTeamRecord = { schema_version: 1, data_class: "private", ...base, holdout: hook(() => input.holdout(base.team_id)) };
    try {
      return readPrivateTeamRecord(engine, record);
    } catch {
      throw new ImportPlanError("record_invalid", file === null ? null : { subject: "file", index: file });
    }
  };

  // Files in name order, so fresh ids are issued the same way on every run.
  const order = input.staged.map((f, k) => ({ f, file: k + 1 })).sort((a, b) => codeUnits(a.f.name, b.f.name) || a.file - b.file);
  const fresh: { file: number; sets: PokemonSet[]; paste: string; tags: readonly string[]; canonical: string }[] = [];

  for (const { f, file } of order) {
    const at = { subject: "file", index: file } as const;
    const parsed = parseStagedFile(f.text, file);
    let sets: PokemonSet[];
    let paste: string;
    try {
      sets = importPaste(engine, parsed.paste);
      paste = exportPaste(engine, sets);
    } catch {
      throw new ImportPlanError("import_refused", at);
    }
    // The validator normalises sets in place: judge a copy.
    if (!validateTeam(engine, format, structuredClone(sets)).valid) throw new ImportPlanError("team_illegal", at);

    if (parsed.team_id === null) {
      fresh.push({ file, sets, paste, tags: parsed.tags ?? [], canonical: canonicalTeamHash(sets) });
      continue;
    }
    const id = parsed.team_id;
    if (stagedIds.has(id)) throw new ImportPlanError("team_id_repeated", at);
    stagedIds.add(id);
    if (retire.has(id)) throw new ImportPlanError("retire_and_staged", at);
    const before = parentById.get(id);
    if (before === undefined) throw new ImportPlanError("team_id_unknown", at);
    const tags = parsed.tags ?? before.tags;
    if (before.status === "active" && before.paste === paste && sameList(before.tags, tags)) continue;
    next.set(id, {
      file,
      record: make(file, {
        team_id: id,
        revision: before.revision + 1,
        supersedes_record_hash: recordHash(before),
        team_hash: teamHash(sets),
        canonical_team_hash: canonicalTeamHash(sets),
        paste,
        status: "active",
        tags: [...tags],
        labels: hook(() => input.labels(sets)),
      }),
    });
    revised++;
  }

  for (const id of toRetire) {
    const before = parentById.get(id) as PrivateTeamRecord;
    next.set(id, { file: null, record: make(null, { ...before, revision: before.revision + 1, supersedes_record_hash: recordHash(before), status: "retired" }) });
  }

  // A file without an id: new, unless it is an active record's team already.
  const held = new Set([...next.values()].filter((e) => e.record.status === "active").map((e) => e.record.canonical_team_hash));
  for (const item of fresh) {
    const at = { subject: "file", index: item.file } as const;
    if (held.has(item.canonical)) {
      duplicate++;
      continue;
    }
    const id = hook(() => input.newTeamId());
    if (!TEAM_ID.test(id) || next.has(id)) throw new ImportPlanError("id_source_invalid", at);
    held.add(item.canonical);
    next.set(id, {
      file: item.file,
      record: make(item.file, {
        team_id: id,
        revision: 1,
        supersedes_record_hash: null,
        team_hash: teamHash(item.sets),
        canonical_team_hash: item.canonical,
        paste: item.paste,
        status: "active",
        tags: [...item.tags],
        labels: hook(() => input.labels(item.sets)),
      }),
    });
    created++;
  }

  // One active record per canonical hash (a revision may have taken another's team).
  const seen = new Map<string, number | null>();
  for (const { record, file } of [...next.values()].sort((a, b) => codeUnits(a.record.team_id, b.record.team_id))) {
    if (record.status !== "active") continue;
    if (seen.has(record.canonical_team_hash)) {
      const where = file ?? seen.get(record.canonical_team_hash) ?? null;
      throw new ImportPlanError("duplicate_canonical_hash", where === null ? null : { subject: "file", index: where });
    }
    seen.set(record.canonical_team_hash, file);
  }

  const records = [...next.values()].map((e) => e.record).sort((a, b) => codeUnits(a.team_id, b.team_id));
  try {
    checkPrivateRecordSet(records);
  } catch {
    throw new ImportPlanError("result_invalid");
  }
  assertNoActiveRecordLost(parent, records, retire);
  if (records.length === 0) throw new ImportPlanError("no_teams");
  return { records, summary: { new: created, duplicate, revised, unchanged: parent.length - revised - toRetire.size, retired: toRetire.size } };
}
