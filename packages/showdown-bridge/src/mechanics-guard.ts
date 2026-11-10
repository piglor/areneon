// The random-source guards the scenario probes share (ADR-016 section 2). A probe replaces the engine's random
// source before it declares any state: every method of the replacement refuses, so a draw ends the probe with an
// `unsupported` result (`random_effect`) instead of a number computed from an answer the probe made up.

import { MechanicsProbeError } from "./mechanics-error.ts";

/** Thrown by the replacement random source; a probe turns it into `unsupported`. */
export class RandomDraw extends Error {}

export const draw = (): never => {
  throw new RandomDraw("the engine drew a random number");
};

/**
 * The source target resolution answers from: the first of any list, zero for a
 * number, nothing else. A move without a declared target (a spread move, or a
 * target the query does not give) is resolved by the engine's own
 * `getRandomTarget`, which draws; the target never enters the order keys, so
 * it is answered with the first candidate instead of ending the probe.
 */
export const firstTarget = { sample: <T>(items: readonly T[]): T => items[0] as T, random: (): number => 0, randomChance: draw, shuffle: draw, getSeed: draw, setSeed: draw, clone: draw };

/** The battle's random source with every method replaced by one that refuses. */
export const refusingSource = (source: object): object => new Proxy(source, { get: (target, key) => (typeof Reflect.get(target, key) === "function" ? draw : Reflect.get(target, key)) });

/** A declared state or action the engine refuses is a `MechanicsProbeError` (the caller's `ProbeRefusal` path), never a crash. */
export const need = (ok: boolean, what: string): void => {
  if (!ok) throw new MechanicsProbeError(what);
};
