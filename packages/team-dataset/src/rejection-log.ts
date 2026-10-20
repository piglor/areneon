// The rejection log (ADR-013 section 2): every rejection, duplicate and
// overflow, by canonical hash, with its reason and the counter k, one JCS
// line each. It is written as the run goes (an overflow is the usual fate of
// most draws, so the log can be much longer than the dataset) and hashed on
// the way, so the manifest can name it by SHA-256.

import { createHash } from "node:crypto";
import { closeSync, openSync, writeSync } from "node:fs";
import { jcs } from "@areneon/showdown-bridge";

export type LogReason = "invalid" | "round_trip" | "duplicate" | "overflow";

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
  #overflows = 0;

  constructor(path: string) {
    this.#fd = openSync(path, "w");
  }

  write(entry: LogEntry): void {
    if (entry.reason === "duplicate") this.#duplicates++;
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
    return { log_sha256: this.#hash.digest("hex"), rejections: this.#rejections, duplicates: this.#duplicates, overflows: this.#overflows };
  }
}
