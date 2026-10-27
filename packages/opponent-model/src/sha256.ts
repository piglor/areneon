// SHA-256 (FIPS 180-4) on bytes, written out so the package needs no Node built-in (it is engine-free and
// runs wherever plain TypeScript does). A test holds it equal to the platform's SHA-256 on inputs of every
// length up to several blocks.

const K = Uint32Array.from([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));

/** The 32-byte SHA-256 digest of `data`. */
export function sha256(data: Uint8Array): Uint8Array {
  const h = Uint32Array.from([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const padded = new Uint8Array(((data.length + 9 + 63) >>> 6) << 6);
  padded.set(data);
  padded[data.length] = 0x80;
  const view = new DataView(padded.buffer);
  // The bit length as two words (an input is far below 2^32 bytes): high word, then low word.
  view.setUint32(padded.length - 8, data.length >>> 29, false);
  view.setUint32(padded.length - 4, (data.length << 3) >>> 0, false);
  const w = new Uint32Array(64);
  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let t = 0; t < 16; t += 1) w[t] = view.getUint32(offset + t * 4, false);
    for (let t = 16; t < 64; t += 1) {
      const a = w[t - 15] as number;
      const b = w[t - 2] as number;
      const s0 = rotr(a, 7) ^ rotr(a, 18) ^ (a >>> 3);
      const s1 = rotr(b, 17) ^ rotr(b, 19) ^ (b >>> 10);
      w[t] = ((w[t - 16] as number) + s0 + (w[t - 7] as number) + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h as unknown as [number, number, number, number, number, number, number, number];
    for (let t = 0; t < 64; t += 1) {
      const t1 = (hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + (K[t] as number) + (w[t] as number)) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    h[0] = ((h[0] as number) + a) >>> 0;
    h[1] = ((h[1] as number) + b) >>> 0;
    h[2] = ((h[2] as number) + c) >>> 0;
    h[3] = ((h[3] as number) + d) >>> 0;
    h[4] = ((h[4] as number) + e) >>> 0;
    h[5] = ((h[5] as number) + f) >>> 0;
    h[6] = ((h[6] as number) + g) >>> 0;
    h[7] = ((h[7] as number) + hh) >>> 0;
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  h.forEach((word, i) => outView.setUint32(i * 4, word, false));
  return out;
}

export const toHex = (bytes: Uint8Array): string => Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");

export function fromHex(text: string): Uint8Array {
  if (!/^(?:[0-9a-f]{2})*$/.test(text)) throw new RangeError("not lower-case hexadecimal of whole bytes");
  return Uint8Array.from(text.match(/../g) ?? [], (pair) => parseInt(pair, 16));
}

const encoder = new TextEncoder();
/** SHA-256 of the UTF-8 bytes of `text`, in lower-case hexadecimal. */
export const sha256Hex = (text: string): string => toHex(sha256(encoder.encode(text)));
export const utf8 = (text: string): Uint8Array => encoder.encode(text);
