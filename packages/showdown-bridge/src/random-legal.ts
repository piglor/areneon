import { NoCandidateError } from "./actor-rule.ts";
import type { Actor } from "./runner.ts";
import type { LegalChoiceSet } from "@areneon/schema/types/v1/legal-choice-set.ts";

/** The PRNG state: one unsigned 32-bit integer. */
export type RandomLegalState = number;

/** The state is not an unsigned 32-bit integer. */
export class RandomStateError extends RangeError {
  override name = "RandomStateError";
}

/**
 * One step of Mulberry32 (Tommy Ettinger's public-domain 32-bit generator):
 * add the Weyl constant 0x6D2B79F5 to the state, then mix it. Only integer
 * operations (`Math.imul`, shifts, xor) are used, so the stream is the same
 * on every JavaScript engine and Node version. Returns the next state and a
 * uniform unsigned 32-bit output.
 */
function mulberry32(state: RandomLegalState): { state: RandomLegalState; value: number } {
  const next = (state + 0x6d2b79f5) >>> 0;
  let t = Math.imul(next ^ (next >>> 15), next | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return { state: next, value: (t ^ (t >>> 14)) >>> 0 };
}

/** A uniform integer in `[0, n)` from 32-bit draws; rejection removes modulo bias. */
function below(n: number, state: RandomLegalState): { state: RandomLegalState; value: number } {
  const limit = 2 ** 32 - (2 ** 32 % n);
  let draw = mulberry32(state);
  while (draw.value >= limit) draw = mulberry32(draw.state);
  return { state: draw.state, value: draw.value % n };
}

/**
 * The random-legal baseline (evaluation protocol, pool B): pick uniformly
 * among the non-pending candidates of the legal choice set. The pending-only
 * fallback mirrors `actorRule`: pending candidates are played only when no
 * non-pending one exists, and then uniformly among the pending ones, so a
 * hidden-information verdict never matters. An empty set throws
 * `NoCandidateError`.
 *
 * Pure: the randomness is the explicit `state` (a uint32, e.g. a seed), and
 * the next state is returned; pass it to the next call. Exactly one draw (or
 * more only on a rejected draw) is consumed per call, whatever the set size.
 * Never `Math.random` or the clock. The same state and set give the same pick.
 */
export function randomLegal(
  legal: Pick<LegalChoiceSet, "choices" | "pending">,
  state: RandomLegalState,
): { choice: string; state: RandomLegalState } {
  checkState(state);
  const pool = legal.choices.length > 0 ? legal.choices : legal.pending;
  if (pool.length === 0) throw new NoCandidateError("the legal choice set has no candidate");
  const draw = below(pool.length, state);
  return { choice: pool[draw.value] as string, state: draw.state };
}

/** Throw `RandomStateError` unless `state` is an unsigned 32-bit integer. */
function checkState(state: RandomLegalState): void {
  if (!Number.isInteger(state) || state < 0 || state > 0xffffffff) {
    throw new RandomStateError("the random state must be an unsigned 32-bit integer");
  }
}

/**
 * A random-legal `Actor` for the A1 runner: it keeps the PRNG state between
 * requests, starting from `seed`, which is checked now (a bad seed throws
 * `RandomStateError` here, not at the first request). The A2 choice step
 * would thread `randomLegal`'s state through its own battle state instead.
 *
 * Seeding (protocol D9): the caller reduces the HMAC for the seat and policy
 * label to a uint32, e.g. `parseInt(h.slice(0, 8), 16)` on the hex digest. A
 * raw 128-bit value must not go through `Number()` or `>>> 0`: the first
 * loses low bits, the second keeps only the low 32 of an already rounded
 * double. The label must include the seat and the policy id, so streams in
 * one battle differ. The seed space is 2^32, so about 10^5 streams are
 * enough for two to collide (birthday bound), and every stream is the same
 * 2^32-cycle read from a different offset.
 */
export function randomLegalActor(seed: RandomLegalState): Actor {
  checkState(seed);
  let state = seed;
  return (request) => {
    if (request.legal === undefined) {
      throw new NoCandidateError(`${request.player}: request ${request.request_seq} has no legal choice set`);
    }
    const pick = randomLegal(request.legal, state);
    state = pick.state;
    return pick.choice;
  };
}
