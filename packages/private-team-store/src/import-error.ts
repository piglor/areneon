// The typed refusal of the import planner. A refusal is a code and a position
// (the 1-based place of a staged file or of a `--retire` entry in the lists the
// caller passed), never text taken from an input: not a paste, a file name, a
// tag, a header value, a species or move name, or an id. The messages below are
// fixed strings, so an error can be printed or logged without leaking a team.

export const IMPORT_REFUSALS = [
  "header_malformed",
  "comment_line",
  "import_refused",
  "team_illegal",
  "record_invalid",
  "team_id_unknown",
  "team_id_repeated",
  "duplicate_canonical_hash",
  "retire_id_malformed",
  "retire_repeated",
  "retire_unknown",
  "retire_and_staged",
  "parent_invalid",
  "holdout_mismatch",
  "active_record_lost",
  "result_invalid",
  "id_source_invalid",
  "hook_failed",
  "no_teams",
] as const;
export type ImportRefusal = (typeof IMPORT_REFUSALS)[number];

const MESSAGES: Readonly<Record<ImportRefusal, string>> = {
  header_malformed: "the header line is not exactly `#! team_id=<32 hex> tags=<tag>[,<tag>...]` (either key optional, tags sorted-unique, at most 16)",
  comment_line: "a line starting with `#` other than the one header line is not allowed",
  import_refused: "the pinned engine's import refuses the paste",
  team_illegal: "the pinned validator refuses the team",
  record_invalid: "the team does not make a valid private record",
  team_id_unknown: "the header's team_id is not in the parent snapshot",
  team_id_repeated: "two staged files name the same team_id",
  duplicate_canonical_hash: "the result would hold the same team in two active records (a revision or a revival takes another active record's team)",
  retire_id_malformed: "a --retire value is not a 32 lowercase hex team_id",
  retire_repeated: "a --retire value is given twice",
  retire_unknown: "a --retire team_id is not in the parent snapshot",
  retire_and_staged: "a team is both staged and retired in this import",
  parent_invalid: "the parent records are not a valid private record set",
  holdout_mismatch: "a parent record's hold-out flag differs from the hold-out rule's value for its team_id",
  result_invalid: "the result is not a valid private record set",
  hook_failed: "a labels, hold-out or team_id hook failed",
  active_record_lost: "an active parent record is missing from the result without --retire",
  id_source_invalid: "the team_id source returned an id that is malformed or already used",
  no_teams: "the import would produce a snapshot with no records",
};

export interface RefusalPosition {
  /** `file`: the staged files as passed to the planner; `retire`: the --retire list. */
  readonly subject: "file" | "retire";
  /** 1-based. */
  readonly index: number;
}

export class ImportPlanError extends Error {
  override name = "ImportPlanError";
  readonly code: ImportRefusal;
  readonly position: RefusalPosition | null;

  constructor(code: ImportRefusal, position: RefusalPosition | null = null) {
    super(position === null ? `import refused: ${MESSAGES[code]}` : `import refused (${position.subject} ${position.index}): ${MESSAGES[code]}`);
    this.code = code;
    this.position = position;
  }
}
