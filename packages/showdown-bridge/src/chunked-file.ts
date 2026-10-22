// Large files are written and read in chunks, never as one string (#998):
// Node's string limit is about 512 MB, and a generation's TrainingExample
// stream (every example carries the log up to its request) passes it at about
// 480 games. A writer buffers whole lines up to a chunk of characters, writes
// the chunk and flushes at the end; a reader hands back one line at a time.
// The bytes on disk are the ones a single `writeFileSync` of the joined text
// would give.

import { closeSync, fsyncSync, openSync, readSync, writeSync } from "node:fs";
import { StringDecoder } from "node:string_decoder";

/** Characters buffered before a write: about 1 MiB. */
export const DEFAULT_CHUNK_CHARS = 1 << 20;

/** The limits that keep every string small. Both are injectable so a test can use a tiny one. */
export interface ChunkLimits {
  /** Characters buffered before a write (default `DEFAULT_CHUNK_CHARS`). */
  chunk_chars?: number;
  /** No single string made on the way (a line, a chunk, an in-memory result) may be longer than this; the check throws. Default: none. */
  max_string_chars?: number;
}

/** Throws `RangeError` when `text` is longer than the limit. */
export function checkStringLength(text: string, limits: ChunkLimits, what: string): void {
  if (limits.max_string_chars !== undefined && text.length > limits.max_string_chars) {
    throw new RangeError(`${what} is ${text.length} characters, over the limit of ${limits.max_string_chars}`);
  }
}

/** A file written as a series of text pieces, each flushed in chunks; `finish` makes it durable. */
export class ChunkedFileWriter {
  readonly #descriptor: number;
  readonly #limits: ChunkLimits;
  #parts: string[] = [];
  #chars = 0;
  #written = 0;
  #open = true;

  constructor(path: string, limits: ChunkLimits = {}) {
    this.#limits = limits;
    this.#descriptor = openSync(path, "w");
  }

  /** Bytes written to the file so far (the buffer excluded until it is flushed). */
  get bytes(): number {
    return this.#written;
  }

  write(text: string): void {
    checkStringLength(text, this.#limits, "a piece of a chunked file");
    this.#parts.push(text);
    this.#chars += text.length;
    if (this.#chars >= (this.#limits.chunk_chars ?? DEFAULT_CHUNK_CHARS)) this.#flush();
  }

  #flush(): void {
    if (this.#parts.length === 0) return;
    const chunk = this.#parts.join("");
    this.#parts = [];
    this.#chars = 0;
    checkStringLength(chunk, this.#limits, "a chunk of a chunked file");
    const bytes = Buffer.from(chunk, "utf8");
    let offset = 0;
    while (offset < bytes.length) offset += writeSync(this.#descriptor, bytes, offset);
    this.#written += bytes.length;
  }

  /** Flush the rest, fsync and close. */
  finish(): void {
    try {
      this.#flush();
      fsyncSync(this.#descriptor);
    } finally {
      this.close();
    }
  }

  /** Close without flushing (after a failure). */
  close(): void {
    if (!this.#open) return;
    this.#open = false;
    closeSync(this.#descriptor);
  }
}

/**
 * The lines of a file (without their `\n`), read `read_bytes` at a time. A final
 * line without a newline is returned too. A file with no line yields nothing.
 */
export function* readFileLines(path: string, read_bytes: number = DEFAULT_CHUNK_CHARS): Generator<string> {
  const descriptor = openSync(path, "r");
  try {
    const buffer = Buffer.alloc(read_bytes);
    const decoder = new StringDecoder("utf8");
    let pending = "";
    for (;;) {
      const count = readSync(descriptor, buffer, 0, buffer.length, null);
      if (count === 0) break;
      const text = pending + decoder.write(buffer.subarray(0, count));
      const lines = text.split("\n");
      pending = lines.pop() as string;
      yield* lines;
    }
    pending += decoder.end();
    if (pending !== "") yield pending;
  } finally {
    closeSync(descriptor);
  }
}
