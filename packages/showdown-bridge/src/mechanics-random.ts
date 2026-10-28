// The recording random source of the damage probe (ADR-016 section 2). The
// engine draws from `battle.prng`; the probe replaces it on a clone with this
// object, which answers `random(16)` (the damage roll) with the roll index it
// was given, answers a crit chance with a fixed outcome, and records every call.
// It never guesses: any other call, or any other argument, is recorded as such
// and `acceptedDraws` refuses the run, so the result is `unsupported`
// (`random_effect`) rather than a number computed from an answer the probe made
// up. It imports nothing, so the source can be tested without an engine.

/** One call the engine made on its random source. */
export type Draw =
  | { kind: "random"; args: number[] }
  | { kind: "randomChance"; numerator: number; denominator: number }
  /** Any other member of the engine's PRNG (`sample`, `shuffle`, `next`, ...). */
  | { kind: "other"; name: string };

export interface RecordingRandomOptions {
  /** The index (0 to 15) `random(16)` answers with. */
  roll: number;
  /** What `randomChance` answers with, whatever its arguments. */
  chance: boolean;
}

/** The recorder: the engine-facing members and the log. */
export interface RecordingRandom {
  readonly draws: readonly Draw[];
  random(m?: number, n?: number): number;
  randomChance(numerator: number, denominator: number): boolean;
  /** Change what the next calls answer (the one-clone path asks for each roll in turn on one battle). */
  answer(roll: number, chance: boolean): void;
}

/**
 * A recording source. `random(16)` returns `roll`; any other `random` call
 * returns 0 (in range for every argument the engine uses) and is recorded
 * with its arguments, as is every other member of the engine's PRNG.
 */
export function recordingRandom(options: RecordingRandomOptions): RecordingRandom {
  const check = (roll: number): void => {
    if (!Number.isInteger(roll) || roll < 0 || roll > 15) throw new RangeError("roll must be an integer from 0 to 15");
  };
  check(options.roll);
  const current = { roll: options.roll, chance: options.chance };
  const draws: Draw[] = [];
  const known: RecordingRandom = {
    draws,
    random(m?: number, n?: number): number {
      const args = [m, n].filter((value): value is number => value !== undefined);
      draws.push({ kind: "random", args });
      return args.length === 1 && args[0] === 16 ? current.roll : 0;
    },
    randomChance(numerator: number, denominator: number): boolean {
      draws.push({ kind: "randomChance", numerator, denominator });
      return current.chance;
    },
    answer(roll: number, chance: boolean): void {
      check(roll);
      current.roll = roll;
      current.chance = chance;
    },
  };
  // Any other member the engine reaches for is a draw it was not expected to make.
  return new Proxy(known, {
    get(target, name) {
      if (name in target) return Reflect.get(target, name) as unknown;
      return (): number => {
        draws.push({ kind: "other", name: String(name) });
        return 0;
      };
    },
  });
}

export const isRoll16 = (draw: Draw | undefined): boolean => draw?.kind === "random" && draw.args.length === 1 && draw.args[0] === 16;

/**
 * Whether a run drew exactly what the damage probe expects: the damage roll
 * `random(16)` at most once, and, when `allowCrit` (the run that observes the
 * crit chance), one `randomChance` before it. A move that deals a fixed amount
 * or nothing draws nothing, which is also accepted. Anything else (an ability
 * or item with a random effect, a random base power, a second roll) is not.
 */
export function acceptedDraws(draws: readonly Draw[], allowCrit: boolean): boolean {
  const rest = allowCrit && draws[0]?.kind === "randomChance" ? draws.slice(1) : draws;
  return rest.length === 0 || (rest.length === 1 && isRoll16(rest[0]));
}

/** The crit draw of a run, when it made one (`acceptedDraws` has judged the run). */
export function critDraw(draws: readonly Draw[]): { numerator: number; denominator: number } | null {
  const first = draws[0];
  return first?.kind === "randomChance" ? { numerator: first.numerator, denominator: first.denominator } : null;
}
