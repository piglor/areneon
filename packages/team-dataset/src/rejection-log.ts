// The rejection log (ADR-013 section 2): every rejection, duplicate and
// overflow, by canonical hash, with its reason and the counter k, one JCS
// line each. It is written as the run goes (an overflow is the usual fate of
// most draws, so the log can be much longer than the dataset) and hashed on
// the way, so the manifest can name it by SHA-256.

import { createHash } from "node:crypto";
import { closeSync, openSync, writeSync } from "node:fs";
import { jcs } from "@areneon/showdown-bridge";

/**
 * `near_duplicate` is L2 (a sealed set only: within `d_team` of a known team). `near_reference` is the
 * evaluation-only exclusion of a public dataset (ADR-020 section 7): within `d_team` of a reference team.
 */
export type LogReason = "invalid" | "round_trip" | "duplicate" | "near_duplicate" | "near_reference" | "overflow";

export interface LogEntry {
  k: number;
  reason: LogReason;
  canonical_team_hash: string;
  /** The stratum, for a duplicate or an overflow. */
  stratum?: number;
  /** The engine's first complaint, for a rejection. */
  detail?: string;
}

export interface LogSummary {
  log_sha256: string;
  rejections: number;
  duplicates: number;
  /** L2 refusals; 0 for a public dataset. */
  near_duplicates: number;
  /** Evaluation-only exclusions; 0 for an ordinary dataset. */
  near_references: number;
  overflows: number;
}

const FLUSH_AT = 1 << 20;

export class RejectionLog {
  readonly #fd: number;
  readonly #hash = createHash("sha256");
  #pending: string[] = [];
  #size = 0;
  #rejections = 0;
  #duplicates = 0;
  #nearDuplicates = 0;
  #nearReferences = 0;
  #overflows = 0;

  constructor(path: string) {
    this.#fd = openSync(path, "w");
  }

  write(entry: LogEntry): void {
    if (entry.reason === "duplicate") this.#duplicates++;
    else if (entry.reason === "near_duplicate") this.#nearDuplicates++;
    else if (entry.reason === "near_reference") this.#nearReferences++;
    else if (entry.reason === "overflow") this.#overflows++;
    else this.#rejections++;
    const line = `${jcs(entry as never)}\n`;
    this.#pending.push(line);
    this.#size += line.length;
    if (this.#size >= FLUSH_AT) this.#flush();
  }

  #flush(): void {
    const text = this.#pending.join("");
    this.#pending = [];
    this.#size = 0;
    this.#hash.update(text, "utf8");
    writeSync(this.#fd, text);
  }

  close(): LogSummary {
    this.#flush();
    closeSync(this.#fd);
    return { log_sha256: this.#hash.digest("hex"), rejections: this.#rejections, duplicates: this.#duplicates, near_duplicates: this.#nearDuplicates, near_references: this.#nearReferences, overflows: this.#overflows };
  }
}
