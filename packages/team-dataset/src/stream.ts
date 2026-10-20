// The draw stream of ADR-013 section 2: draw k of a plan has its own seed, so
// any draw can be computed alone, in any order, on any worker.
//
//   seed_k = "sodium," + hex32(HMAC-SHA256(generation_key, JCS(preimage)))
//
// `hex32` keeps the first 32 hex digits (the form protocol D9 and the runner
// use). The preimage is a JSON object, so JCS fixes its bytes:
//   a draw   {config_hash, k, plan_id, purpose: "draw"}
//   a pilot  {config_hash, k, purpose: "pilot"}
// The pilot does not carry the plan id: it measures the configured proposal,
// not one plan, and one recorded pilot serves every plan (and re-plan) of a
// config. The two purposes never share a preimage, so their streams are
// disjoint.

import { createHmac } from "node:crypto";
import { type Engine, jcs } from "@areneon/showdown-bridge";

export class StreamError extends Error {
  override name = "StreamError";
}

const HEX_KEY = /^[0-9a-f]{64}$/;

/** A 256-bit key as 64 lowercase hex digits; anything else is refused. */
export function checkKey(key: string, what: string): string {
  if (!HEX_KEY.test(key)) throw new StreamError(`${what} must be 64 lowercase hex digits`);
  return key;
}

export interface StreamIdentity {
  /** SHA-256 of the JCS bytes of the generator config. */
  config_hash: string;
  /** The plan whose stream this is; a re-plan has a fresh one. */
  plan_id: string;
}

/** The PRNG seed of draw `k`. */
export function drawSeed(key: string, stream: StreamIdentity, k: number): string {
  if (!Number.isSafeInteger(k) || k < 0) throw new StreamError(`draw counter ${k} is not a non-negative safe integer`);
  const preimage = jcs({ config_hash: stream.config_hash, k, plan_id: stream.plan_id, purpose: "draw" });
  return seedOf(key, preimage);
}

/** The PRNG seed of pilot draw `k`. */
export function pilotSeed(key: string, configHash: string, k: number): string {
  if (!Number.isSafeInteger(k) || k < 0) throw new StreamError(`draw counter ${k} is not a non-negative safe integer`);
  return seedOf(key, jcs({ config_hash: configHash, k, purpose: "pilot" }));
}

function seedOf(key: string, preimage: string): string {
  const digest = createHmac("sha256", Buffer.from(checkKey(key, "the key"), "hex")).update(preimage, "utf8").digest("hex");
  return `sodium,${digest.slice(0, 32)}`;
}

/** Whole numbers from the engine's seeded generator, with no modulo bias. */
export interface Rng {
  /** A uniform integer in [0, n) for 1 <= n <= 2^32. */
  below(n: number): number;
}

const WORD = 2 ** 32;

/**
 * An `Rng` over the pinned engine's PRNG. The engine yields a 32-bit word
 * divided by 2^32, which is exact in a double, so multiplying by 2^32 gives
 * the word back. A pick rejects the words that would make some outcomes more
 * likely than others (those at or above the largest multiple of n), so every
 * outcome has probability exactly 1/n.
 */
export function engineRng(engine: Engine, seed: string): Rng {
  const prng = new engine.sim.PRNG(seed);
  const word = (): number => prng.random() * WORD;
  return {
    below(n: number): number {
      if (!Number.isInteger(n) || n < 1 || n > WORD) throw new StreamError(`cannot pick below ${n}`);
      const limit = WORD - (WORD % n);
      for (;;) {
        const x = word();
        if (x < limit) return x % n;
      }
    },
  };
}
