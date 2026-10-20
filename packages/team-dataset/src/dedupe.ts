// Dedupe against every earlier released dataset (ADR-013 section 2). A team
// is a duplicate when its `canonical_team_hash` is already known. The known
// set can be large (about 3.2 GB of hashes at 10^8 teams), so the test has two
// stages:
//   1. an in-memory sorted array of the 64-bit prefixes of the known hashes
//      (8 bytes each: about 0.8 GB at 10^8); a prefix that is absent decides
//      "not known" without touching the disk;
//   2. on a prefix hit, an exact comparison against the full 32-byte hashes,
//      held in sorted order and read by binary search.
// The second stage is a file of sorted 32-byte records, read with positional
// reads, or the same bytes in memory.

import { createHash } from "node:crypto";
import { closeSync, openSync, readSync, writeFileSync } from "node:fs";

export class DedupeError extends Error {
  override name = "DedupeError";
}

const HASH = /^[0-9a-f]{64}$/;

function checked(hash: string): string {
  if (!HASH.test(hash)) throw new DedupeError(`${hash.slice(0, 80)} is not a 64-digit lowercase hex hash`);
  return hash;
}

/** The first 64 bits of a hash. */
export const prefixOf = (hash: string): bigint => BigInt(`0x${hash.slice(0, 16)}`);

/** The sorted, unique hashes of a pool. */
export function sortedUnique(hashes: Iterable<string>): string[] {
  // sort, then drop neighbours that repeat: a Set would stop at 2^24 entries
  const sorted = [...hashes].map(checked).sort();
  return sorted.filter((h, k) => k === 0 || h !== sorted[k - 1]);
}

/** SHA-256 of the JCS bytes of the sorted unique hash list: the `reference_pool_hash`. */
export function referencePoolHash(sorted: readonly string[]): string {
  // JCS of an array of hex strings is `["a","b"]`: write it piece by piece, so no copy of a huge list is made.
  const hash = createHash("sha256");
  hash.update("[");
  sorted.forEach((h, k) => hash.update(k === 0 ? `"${h}"` : `,"${h}"`));
  hash.update("]");
  return hash.digest("hex");
}

/** The second stage: exact membership in a sorted set of full hashes. */
export interface ExactStage {
  has(hash: string): boolean;
  close(): void;
}

const RECORD = 32;

/** The hashes as consecutive 32-byte records (a joined string would pass the engine's string limit at ten million hashes). */
function pack(sorted: readonly string[]): Buffer {
  const bytes = Buffer.alloc(sorted.length * RECORD);
  sorted.forEach((hash, k) => bytes.write(hash, k * RECORD, RECORD, "hex"));
  return bytes;
}

/** Sorted 32-byte records in memory. */
export class MemoryStage implements ExactStage {
  readonly #bytes: Buffer;

  constructor(sorted: readonly string[]) {
    this.#bytes = pack(sorted);
  }

  has(hash: string): boolean {
    const want = Buffer.from(checked(hash), "hex");
    let low = 0;
    let high = this.#bytes.length / RECORD;
    while (low < high) {
      const mid = (low + high) >>> 1;
      const order = this.#bytes.compare(want, 0, RECORD, mid * RECORD, (mid + 1) * RECORD);
      if (order === 0) return true;
      if (order > 0) high = mid;
      else low = mid + 1;
    }
    return false;
  }

  close(): void {}
}

/** Sorted 32-byte records in a file, read by binary search with positional reads. */
export class FileStage implements ExactStage {
  readonly #fd: number;
  readonly #count: number;
  readonly #buffer = Buffer.alloc(RECORD);

  /** Write `sorted` to `path` and open it. */
  static write(path: string, sorted: readonly string[]): FileStage {
    writeFileSync(path, pack(sorted));
    return new FileStage(path, sorted.length);
  }

  constructor(path: string, count: number) {
    this.#fd = openSync(path, "r");
    this.#count = count;
  }

  has(hash: string): boolean {
    const want = Buffer.from(checked(hash), "hex");
    let low = 0;
    let high = this.#count;
    while (low < high) {
      const mid = (low + high) >>> 1;
      readSync(this.#fd, this.#buffer, 0, RECORD, mid * RECORD);
      const order = this.#buffer.compare(want);
      if (order === 0) return true;
      if (order > 0) high = mid;
      else low = mid + 1;
    }
    return false;
  }

  close(): void {
    closeSync(this.#fd);
  }
}

/** Known hashes: prefix array first, exact stage on a prefix hit. */
export class KnownHashes {
  readonly #prefixes: BigUint64Array;
  readonly #exact: ExactStage;
  /** How many known hashes there are. */
  readonly size: number;
  /** Lookups the prefix stage answered "no" to, and ones it passed to the exact stage. */
  prefixMisses = 0;
  exactChecks = 0;

  /** `sorted` must be sorted and unique (`sortedUnique`). `stage` defaults to memory. */
  constructor(sorted: readonly string[], stage?: ExactStage) {
    this.size = sorted.length;
    this.#prefixes = new BigUint64Array(sorted.length);
    sorted.forEach((hash, k) => (this.#prefixes[k] = prefixOf(hash)));
    this.#prefixes.sort();
    this.#exact = stage ?? new MemoryStage(sorted);
  }

  /** Bytes of the in-memory prefix array: 8 per known hash. */
  get prefixBytes(): number {
    return this.#prefixes.byteLength;
  }

  has(hash: string): boolean {
    const want = prefixOf(checked(hash));
    let low = 0;
    let high = this.#prefixes.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if ((this.#prefixes[mid] as bigint) < want) low = mid + 1;
      else high = mid;
    }
    if (low === this.#prefixes.length || this.#prefixes[low] !== want) {
      this.prefixMisses++;
      return false;
    }
    this.exactChecks++;
    return this.#exact.has(hash);
  }

  close(): void {
    this.#exact.close();
  }
}
