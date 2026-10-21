// The staged-file format of the private import (ADR-015 section 3): a PokePaste,
// optionally preceded by one header line
//
//   #! team_id=<32 lowercase hex> tags=<tag>[,<tag>...]
//
// Either key may be left out; the order is fixed. Any other line that starts
// with `#` is refused. This module is pure: it reads text already in memory
// and never repeats that text in an error (the refusal carries a code only).

import { ImportPlanError } from "./import-error.ts";

export interface StagedFile {
  /** Only used to order the files, so ids are issued in the same order on every run. Never echoed. */
  readonly name: string;
  readonly text: string;
}

export interface ParsedStagedFile {
  /** The header's `team_id`, or null when the header has none (or there is no header). */
  readonly team_id: string | null;
  /** The header's tags, sorted and unique; `[]` for an empty `tags=` (clear the tags); null when the header has no `tags=` (tags are then left as they are). */
  readonly tags: readonly string[] | null;
  /** The PokePaste with the header removed. */
  readonly paste: string;
}

export const MAX_TAGS = 16;
const TAG = /^[a-z0-9][a-z0-9-]{0,31}$/;
const HEADER = /^#! (?:team_id=([0-9a-f]{32})(?: tags=(\S*))?|tags=(\S*))$/;

const fail = (file: number, code: "header_malformed" | "comment_line"): never => {
  throw new ImportPlanError(code, { subject: "file", index: file });
};

/**
 * Split `text` into its header (if any) and its paste. `file` is the 1-based
 * position of the file in the caller's list, used only to report where.
 */
export function parseStagedFile(text: string, file: number): ParsedStagedFile {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let team_id: string | null = null;
  let tags: string[] | null = null;
  let body = lines;
  if ((lines[0] ?? "").startsWith("#")) {
    const match = HEADER.exec(lines[0] as string);
    if (match === null) return fail(file, "header_malformed");
    team_id = match[1] ?? null;
    const rawTags = match[2] ?? match[3];
    if (rawTags !== undefined) {
      const list = rawTags === "" ? [] : rawTags.split(",");
      const sorted = [...list].sort();
      if (list.length > MAX_TAGS || !list.every((t) => TAG.test(t)) || sorted.some((t, k) => k > 0 && t === sorted[k - 1])) {
        return fail(file, "header_malformed");
      }
      tags = sorted;
    }
    body = lines.slice(1);
  }
  // Any other `#` line, however indented, is refused (including a second header).
  if (body.some((line) => line.trimStart().startsWith("#"))) return fail(file, "comment_line");
  return { team_id, tags, paste: body.join("\n") };
}
