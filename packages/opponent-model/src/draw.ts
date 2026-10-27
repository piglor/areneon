// The exact integer draw of the sampler (ADR-025 section 4, review finding N-A). The totals are products
// of many integer weights and can exceed 2^64, so an integer below T is drawn by rejection with no float
// and no reduced weight: take L = bitlen(T) + 64 bits from the stream, accept when n < floor(2^L / T) * T
// and return n mod T, else take the next L bits. The acceptance probability exceeds 1 - 2^-64 for any T
// and the result is exactly uniform.
//
// The stream is the successive 256-bit blocks SHA-256(JCS({counter: j, purpose, stream: <hex>})) for j
// counting up from 0, read as one bit string (big-endian) with the leftover bits of a block carried to
// the next draw. A world's draws all read one such string; `restart` throws the leftover bits away so the
// next draw begins on the next block (a legality redraw starts "from the next counter block").

import { jcs } from "./canonical.ts";
import { sha256, toHex, utf8 } from "./sha256.ts";

export const WORLD_PURPOSE = "opponent-world/1";
export const INDEX_PURPOSE = "opponent-world-index/1";
export const STREAM_BYTES = 32;
const BLOCK_BITS = 256n;
export const EXTRA_BITS = 64n;

/** The block at counter `j`, as an unsigned 256-bit integer. */
export type BlockSource = (counter: number) => bigint;

export function checkStream(stream: Uint8Array): void {
  if (!(stream instanceof Uint8Array) || stream.length !== STREAM_BYTES) throw new RangeError(`a stream is ${STREAM_BYTES} bytes`);
}

/** The block source of a stream under the purpose of the world draws. */
export function blocksOf(stream: Uint8Array): BlockSource {
  checkStream(stream);
  const hex = toHex(stream);
  return (counter) => BigInt(`0x${toHex(sha256(utf8(jcs({ counter, purpose: WORLD_PURPOSE, stream: hex }))))}`);
}

/** The stream of world `index` of a consumer's stream: SHA-256 of the JCS of {stream, index, purpose}. */
export function worldStream(stream: Uint8Array, index: number): Uint8Array {
  checkStream(stream);
  if (!Number.isSafeInteger(index) || index < 0) throw new RangeError("a world index is a non-negative integer");
  return sha256(utf8(jcs({ index, purpose: INDEX_PURPOSE, stream: toHex(stream) })));
}

const bitLength = (value: bigint): bigint => BigInt(value.toString(2).length);

export class BitStream {
  readonly #source: BlockSource;
  #counter = 0;
  #buffer = 0n;
  #held = 0n;

  constructor(source: BlockSource) {
    this.#source = source;
  }

  /** The number of blocks read so far (the next counter). */
  get counter(): number {
    return this.#counter;
  }

  /** Drop the bits left over from the last block: the next draw starts on the next block. */
  restart(): void {
    this.#buffer = 0n;
    this.#held = 0n;
  }

  #bits(count: bigint): bigint {
    while (this.#held < count) {
      this.#buffer = (this.#buffer << BLOCK_BITS) | this.#source(this.#counter);
      this.#counter += 1;
      this.#held += BLOCK_BITS;
    }
    const left = this.#held - count;
    const out = this.#buffer >> left;
    this.#buffer &= (1n << left) - 1n;
    this.#held = left;
    return out;
  }

  /** An integer uniform in [0, total). A total of 1 has one outcome and reads nothing. */
  below(total: bigint): bigint {
    if (total < 1n) throw new RangeError("an integer is drawn below a positive total");
    if (total === 1n) return 0n;
    const length = bitLength(total) + EXTRA_BITS;
    const limit = ((1n << length) / total) * total;
    for (;;) {
      const n = this.#bits(length);
      if (n < limit) return n % total;
    }
  }

  /**
   * An index into `weights`, chosen with probability proportional to its weight. Zero weights are never
   * chosen; a single positive weight is the only outcome and reads nothing.
   */
  choose(weights: readonly bigint[]): number {
    let only = -1;
    let positive = 0;
    let total = 0n;
    weights.forEach((w, i) => {
      if (w < 0n) throw new RangeError("a weight is not negative");
      if (w > 0n) {
        positive += 1;
        only = i;
        total += w;
      }
    });
    if (positive === 0) throw new RangeError("a draw needs a positive weight");
    if (positive === 1) return only;
    const n = this.below(total);
    let upTo = 0n;
    let pick = only;
    // The pick is the first index whose running sum from the front exceeds n.
    for (let i = 0; i < weights.length; i += 1) {
      upTo += weights[i] as bigint;
      if (n < upTo) {
        pick = i;
        break;
      }
    }
    return pick;
  }
}
