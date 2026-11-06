// Format binding of the engine-sent team sheet (ADR-027 section 5, #1150). In an open-sheet
// format the engine prints one `|showteam|<side>|<packed>` line per side before Team Preview; in
// a closed-sheet format it never does. A channel is judged against the regime of the format its
// version manifest names, in both directions: a `|showteam|` line in a closed-format channel is a
// spliced or mislabelled channel, and an open-format channel without exactly one per side is not
// read as a closed game. Whether a format is open is the rules snapshot's `team_sheets`, never a
// format id written here.

import { loadRulesSnapshot } from "./rules-load.ts";
import { type RulesSnapshot, teamSheetsOf } from "./rules.ts";
import { ObservationError } from "./observation.ts";
import { parseRequest } from "./request.ts";

export type TeamSheets = "closed" | "open";

const SHEET = "|showteam";
const REQUEST = "|request|";

const isSheet = (line: string): boolean => line === SHEET || line.startsWith(`${SHEET}|`);

/** The regime of a format in a rules snapshot; a format the snapshot lacks is `closed` (the strict reading). */
export const sheetRegimeIn = (snapshot: RulesSnapshot, formatId: string): TeamSheets => {
  const spec = snapshot.formats[formatId];
  return spec === undefined ? "closed" : teamSheetsOf(spec);
};

const snapshots = new Map<string, RulesSnapshot>();

/**
 * The regime of `formatId` in the rules snapshot built for the engine `commit` (cached per commit).
 * Rules that cannot be read are a defect and propagate; a format the snapshot lacks reads as `closed`.
 */
export function sheetRegimeOf(commit: string, formatId: string): TeamSheets {
  let snapshot = snapshots.get(commit);
  if (snapshot === undefined) {
    snapshot = loadRulesSnapshot(commit);
    snapshots.set(commit, snapshot);
  }
  return sheetRegimeIn(snapshot, formatId);
}

/**
 * Refuse a channel whose `|showteam|` lines do not fit the format's regime.
 * - `closed`: no `|showteam|` line at all.
 * - `open`: at most one per side, before `|start` and before the first request; and when the first
 *   request is Team Preview, exactly one for each of `p1` and `p2` before it. (A rejoin channel
 *   alone has no Team Preview request; `buildObservation` refuses it on its own.)
 */
export function assertSheetBinding(lines: readonly string[], regime: TeamSheets): void {
  const sheets = lines.filter(isSheet);
  if (regime === "closed") {
    if (sheets.length > 0) throw new ObservationError("a closed-sheet format's channel holds a |showteam| line");
    return;
  }
  const sides = new Map<string, number>();
  for (const line of sheets) {
    const side = line.split("|")[2] ?? "";
    if (side !== "p1" && side !== "p2") throw new ObservationError("an open-sheet channel's |showteam| line names no side");
    sides.set(side, (sides.get(side) ?? 0) + 1);
  }
  if ([...sides.values()].some((n) => n > 1)) throw new ObservationError("an open-sheet channel holds a second |showteam| line for one side");
  const start = lines.indexOf("|start");
  const firstRequest = lines.findIndex((line) => line.startsWith(REQUEST));
  const last = lines.findLastIndex(isSheet);
  if (last >= 0 && ((start >= 0 && last > start) || (firstRequest >= 0 && last > firstRequest))) {
    throw new ObservationError("an open-sheet channel's |showteam| line is not before Team Preview");
  }
  const first = firstRequest < 0 ? undefined : lines[firstRequest];
  if (first !== undefined && parseRequest(first.slice(REQUEST.length)).phase === "preview" && sides.size !== 2) {
    throw new ObservationError("an open-sheet channel's Team Preview request is not preceded by one |showteam| line per side");
  }
}
