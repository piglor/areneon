import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { digest, jcs, plain } from "../src/canonical.ts";
import { BitStream, blocksOf, worldStream } from "../src/draw.ts";
import { fromHex, sha256, sha256Hex, toHex, utf8 } from "../src/sha256.ts";

const STREAM = Uint8Array.from({ length: 32 }, (_, i) => i);

describe("SHA-256", () => {
  it("equals the platform's on inputs of every length across several blocks", () => {
    for (let length = 0; length <= 300; length += 1) {
      const data = Uint8Array.from({ length }, (_, i) => (i * 7 + length) & 255);
      expect(toHex(sha256(data)), `${length} bytes`).toBe(createHash("sha256").update(data).digest("hex"));
    }
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("reads and writes lower-case hexadecimal of whole bytes only", () => {
    expect(toHex(fromHex("00ff10"))).toBe("00ff10");
    expect(fromHex("")).toEqual(new Uint8Array(0));
    for (const bad of ["0", "FF", "zz", "0g"]) expect(() => fromHex(bad)).toThrow(RangeError);
    expect(utf8("é")).toEqual(Uint8Array.from([0xc3, 0xa9]));
  });
});

describe("canonical JSON", () => {
  it("sorts keys, writes integers and escapes strings as RFC 8785 does (a vector made with Python)", () => {
    expect(jcs({ b: [1, "é \"\\\n", null, true], a: { z: -5, y: 0 } })).toBe('{"a":{"y":0,"z":-5},"b":[1,"é \\"\\\\\\n",null,true]}');
    expect(jcs(-0)).toBe("0");
  });

  it("refuses what a hash input must not hold", () => {
    for (const bad of [1.5, Number.NaN, undefined, 1n, () => 1, "\ud800", "\udc00x"]) expect(() => jcs(bad)).toThrow(TypeError);
    expect(() => plain(new Map())).toThrow(TypeError);
    expect(() => jcs({ a: undefined })).toThrow(TypeError);
  });

  it("writes a bigint as its decimal string", () => {
    expect(jcs(plain({ n: 2n ** 70n, list: [{ num: -3n }] }))).toBe('{"list":[{"num":"-3"}],"n":"1180591620717411303424"}');
    expect(digest({ a: 1n })).toBe(sha256Hex('{"a":"1"}'));
    expect(jcs("😀")).toBe('"😀"');
  });
});

describe("the exact integer draw (ADR-025 section 4)", () => {
  it("reads blocks SHA-256(JCS({counter, purpose, stream})) and draws the vector made with Python, T above 2^64 included", () => {
    const bits = new BitStream(blocksOf(STREAM));
    const totals = [3n, 10n, 2n ** 64n + 13n, 2n ** 200n + 7n, 5n, 1n, 2n ** 70n];
    expect(totals.map((t) => bits.below(t).toString())).toEqual(["0", "4", "1227234455201723852", "851520130221711426797538192192953103706496167564437876321124", "1", "0", "278507010866276798595"]);
    // The total of 1 read nothing; the three blocks cover all the other draws.
    expect(bits.counter).toBe(3);
    expect(blocksOf(STREAM)(0).toString(16)).toBe("9e10067730bd057353f0747504ab313b10f07722d2f23c4c4a581ccb2e89e64a");
  });

  it("derives the stream of world i from the consumer's stream (a vector made with Python)", () => {
    expect(toHex(worldStream(STREAM, 3))).toBe("221ff75fd0e6929ec3b8f1d7d504a013cb1a395f1dcfbe5d542a6bdc9a5c079a");
    expect(() => worldStream(STREAM, -1)).toThrow(RangeError);
    expect(() => worldStream(STREAM, 1.5)).toThrow(RangeError);
    expect(() => worldStream(new Uint8Array(31), 0)).toThrow(RangeError);
    expect(() => blocksOf(new Uint8Array(33))).toThrow(RangeError);
  });

  it("rejects a candidate at or above floor(2^L / T) * T and takes the next L bits", () => {
    // T = 3, L = 66: the largest accepted candidate is below floor(2^66 / 3) * 3 = 2^66 - 2. A block of all ones
    // gives candidate 2^66 - 1, which is rejected; the next bits give the answer.
    const ones = (1n << 256n) - 1n;
    const blocks = [ones, 5n << 190n, 0n];
    const bits = new BitStream((j) => blocks[j] as bigint);
    // After the first 66 bits (all ones) the next 66 bits are the top bits of the same block: still ones, rejected again;
    // 256 bits hold three such candidates and then a fourth read needs the second block.
    const n = bits.below(3n);
    expect(n).toBeLessThan(3n);
    expect(bits.counter).toBeGreaterThanOrEqual(2);
  });

  it("restart drops the leftover bits: the next draw begins on the next block", () => {
    const seen: number[] = [];
    const bits = new BitStream((j) => {
      seen.push(j);
      return BigInt(j + 1) << 200n;
    });
    bits.below(10n);
    bits.below(10n);
    expect(seen).toEqual([0]);
    bits.restart();
    bits.below(10n);
    expect(seen).toEqual([0, 1]);
  });

  it("is uniform: the frequencies of a draw below 7 over many streams are close to 1/7", () => {
    const counts = new Array<number>(7).fill(0);
    for (let i = 0; i < 7000; i += 1) {
      const bits = new BitStream(blocksOf(worldStream(STREAM, i)));
      const k = Number(bits.below(7n));
      counts[k] = (counts[k] ?? 0) + 1;
    }
    for (const c of counts) expect(Math.abs(c - 1000)).toBeLessThan(150);
  });

  it("choose never picks a zero weight, reads nothing for a single positive weight and refuses the rest", () => {
    const bits = new BitStream(() => {
      throw new Error("no block should be read");
    });
    expect(bits.choose([0n, 0n, 4n])).toBe(2);
    expect(bits.below(1n)).toBe(0n);
    expect(() => bits.choose([0n, 0n])).toThrow(RangeError);
    expect(() => bits.choose([1n, -1n])).toThrow(RangeError);
    expect(() => bits.below(0n)).toThrow(RangeError);
    const real = new BitStream(blocksOf(STREAM));
    const picks = new Set(Array.from({ length: 200 }, () => real.choose([0n, 3n, 0n, 1n])));
    expect([...picks].sort()).toEqual([1, 3]);
  });
});
