// The batch seam of a game process (ADR-030 step 1, #1455): the interface between the decision pipeline of many games and
// the backends that answer its questions. Two parts.
//
// 1. Mechanics queries as data. A feature builder is written as a generator (`Gen`): wherever it needs the probe it
//    `yield`s the parsed query and receives the result (or has the probe's refusal thrown into it). Run inline by
//    `driveSync`, with the probe called for each query as it is yielded, it is exactly the builder that called the probe
//    in line; run by a `WaveSeam`, the queries of every build of a wave are collected and answered by ONE call of a
//    `BatchRunner`, then handed back. The only implementation of the batch is the pinned engine's own probe, so the
//    answers are the probe's, in query order, whatever the batching: a query's answer is a pure function of the query.
//
// 2. The wave. A game process plays `n_parallel` games at once; the `WaveSeam` is the one place where they meet. A game's
//    decision does `await seam.build(gen)` (features) and `await seam.infer(ask)` (the policy query). The seam waits until
//    every game that can run has reached one of the two (macrotask quiescence: the games advance only through microtasks
//    between seam points), then answers all the builds together (round after round while they still have queries), then
//    issues all the policy queries together. The worker (batch 1 or batched, #1195) answers them as it always did.
//    The seam changes the ORDER of work across games, never an answer: every file a generation writes is the same bytes
//    with it and without it (the gate of the ticket, `test/batch-seam.test.ts` and the tiny-loop comparison).
//
// Nothing here reads a team, a log or a seed: the seam sees queries and thunks.
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "./mechanics-query.ts";
import type { MechanicsResult } from "./mechanics-result.ts";

/** What a builder asks the backend: one parsed mechanics query. */
export type ProbeCall = ParsedMechanicsQuery;

/** A computation that asks mechanics queries: yields a `ProbeCall`, receives its `MechanicsResult` (or the probe's error is thrown in). */
export type Gen<T> = Generator<ProbeCall, T, MechanicsResult>;

/** The probe answering one query, as the inline path calls it. Throws what the probe throws. */
export type ProbeRunner = (query: ProbeCall) => MechanicsResult;

/** One query's outcome in a batch: its result, or what the probe threw for it (a refusal is part of the answer). */
export type Settled = { ok: true; result: MechanicsResult } | { ok: false; error: unknown };

/** The batched mechanics call: the outcomes of `queries`, in query order. */
export type BatchRunner = (queries: readonly ProbeCall[]) => Settled[];

/** A computation that asks no query and returns `value` (a builder step whose inputs are all at hand, in the generator form of the builders). */
// eslint-disable-next-line require-yield -- by definition: no query is asked
export function* pure<T>(value: T): Gen<T> {
  return value;
}

/** Run `gen` to its end, answering each yielded query with `run`. A throw of `run` is thrown into the generator, which may catch it as the inline call's `try` did. */
export function driveSync<T>(gen: Gen<T>, run: ProbeRunner): T {
  let step = gen.next();
  while (step.done !== true) {
    let result: MechanicsResult;
    try {
      result = run(step.value);
    } catch (error) {
      step = gen.throw(error);
      continue;
    }
    step = gen.next(result);
  }
  return step.value;
}

const known = new WeakMap<object, string>();

/** Remember the JCS hash a caller already computed for `query` (the view's memo key), so the batch does not canonicalise and hash it again. */
export function noteQueryHash(query: ProbeCall, hash: string): void {
  known.set(query, hash);
}

/**
 * The batched call whose implementation is `run`, one query after another in query order. Equal queries of a batch (the same
 * JCS hash) are run once and share the outcome: the answer is a pure function of the query, so no result differs.
 */
export function batchOf(run: ProbeRunner): BatchRunner {
  return (queries) => {
    const seen = new Map<string, Settled>();
    return queries.map((query) => {
      const key = known.get(query) ?? mechanicsQueryHash(query);
      let settled = seen.get(key);
      if (settled === undefined) {
        try {
          settled = { ok: true, result: run(query) };
        } catch (error) {
          settled = { ok: false, error };
        }
        seen.set(key, settled);
      }
      return settled;
    });
  };
}

/** Sizes of the batches a seam made, as `size -> count` tables (integers; counts only, never a query). */
export interface SeamStats {
  /** Builds (feature decisions) answered. */
  builds: number;
  /** Rounds of the mechanics batch (each: one batched call). */
  rounds: number;
  /** Mechanics queries sent to the batch, and those the batch ran (after the within-batch dedup is not visible here: it is the runner's). */
  queries: number;
  /** Builds per flush (the games that reached the mechanics seam together), by size. */
  builds_per_flush: Record<string, number>;
  /** Queries per round, by size. */
  queries_per_round: Record<string, number>;
  /** Policy queries issued together, by size. */
  asks_per_issue: Record<string, number>;
  /** Times a build flush went ahead of an outstanding policy query that had run past the hold. */
  held_forced: number;
}

export const emptySeamStats = (): SeamStats => ({ builds: 0, rounds: 0, queries: 0, builds_per_flush: {}, queries_per_round: {}, asks_per_issue: {}, held_forced: 0 });

/** The statistics of several waves added up (size tables merged). */
export function sumSeamStats(list: readonly SeamStats[]): SeamStats {
  const total = emptySeamStats();
  const merge = (into: Record<string, number>, from: Record<string, number>): void => {
    for (const [size, count] of Object.entries(from)) into[size] = (into[size] ?? 0) + count;
  };
  for (const each of list) {
    total.builds += each.builds;
    total.rounds += each.rounds;
    total.queries += each.queries;
    total.held_forced += each.held_forced;
    merge(total.builds_per_flush, each.builds_per_flush);
    merge(total.queries_per_round, each.queries_per_round);
    merge(total.asks_per_issue, each.asks_per_issue);
  }
  return total;
}

const bump = (table: Record<string, number>, size: number): void => {
  const key = String(size);
  table[key] = (table[key] ?? 0) + 1;
};

/** How long builds wait for outstanding policy queries before going ahead (a query a game gave up on, past its budget, stays outstanding for good). */
export const SEAM_HOLD_MS = 5_000;

export interface WaveSeamOptions {
  /** The mechanics batch; without one `build` is refused (a run with no feature set has none to answer). */
  batch?: BatchRunner;
  /** The hold (default `SEAM_HOLD_MS`). */
  hold_ms?: number;
  /** Test seams: the clock, and how the flush is scheduled after the microtasks (default `Date.now` and `setImmediate`). */
  now?: () => number;
  defer?: (callback: () => void) => void;
  /** Runs `callback` after `ms` (default: an unref'd `setTimeout`); the hold timer. */
  after?: (callback: () => void, ms: number) => void;
}

interface Build {
  gen: Gen<unknown>;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  /** What the last round's batch said to this build's last query. */
  answer?: Settled | undefined;
}

interface Ask {
  call: () => Promise<unknown>;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
}

export class SeamError extends Error {
  override name = "SeamError";
}

/**
 * The wave of one game process (see the header). `build(gen)` and `infer(call)` return promises the game's actor awaits; the
 * seam answers them in flushes, builds first, then the policy queries.
 */
export class WaveSeam {
  readonly #batch: BatchRunner | undefined;
  readonly #hold: number;
  readonly #now: () => number;
  readonly #defer: (callback: () => void) => void;
  readonly #after: (callback: () => void, ms: number) => void;
  readonly #builds: Build[] = [];
  readonly #asks: Ask[] = [];
  /** Policy queries issued and not yet answered, by the time they were issued. */
  readonly #outstanding = new Map<number, number>();
  #next = 0;
  #scheduled = false;
  #armed = false;
  readonly #stats = emptySeamStats();

  constructor(options: WaveSeamOptions = {}) {
    this.#batch = options.batch;
    this.#hold = options.hold_ms ?? SEAM_HOLD_MS;
    this.#now = options.now ?? Date.now;
    this.#defer = options.defer ?? ((callback) => void setImmediate(callback));
    this.#after = options.after ?? ((callback, ms) => setTimeout(callback, ms).unref());
  }

  /** What the seam did so far (a copy). */
  stats(): SeamStats {
    return structuredClone(this.#stats);
  }

  /** The value of `gen`, whose mechanics queries are answered by the wave's batch. Rejects with what `gen` throws. */
  build<T>(gen: Gen<T>): Promise<T> {
    if (this.#batch === undefined) throw new SeamError("this seam has no mechanics batch");
    return new Promise<T>((resolve, reject) => {
      this.#builds.push({ gen: gen as Gen<unknown>, resolve: resolve as (value: unknown) => void, reject });
      this.#schedule();
    });
  }

  /** What `call()` returns, issued together with the other policy queries of the wave. */
  infer<T>(call: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.#asks.push({ call, resolve: resolve as (value: unknown) => void, reject });
      this.#schedule();
    });
  }

  #schedule(): void {
    if (this.#scheduled) return;
    this.#scheduled = true;
    this.#defer(() => {
      this.#scheduled = false;
      this.#flush();
    });
  }

  /** Whether a policy query issued within the hold is still unanswered (the lockstep: builds wait for it). */
  #blocking(): boolean {
    const now = this.#now();
    for (const issued of this.#outstanding.values()) if (now - issued < this.#hold) return true;
    return false;
  }

  #flush(): void {
    if (this.#builds.length > 0) {
      if (!this.#blocking()) {
        if (this.#outstanding.size > 0) this.#stats.held_forced += 1;
        this.#answerBuilds();
        // The games whose builds just ended run on in microtasks and reach their next seam point; the next flush sees them all.
        this.#schedule();
        return;
      }
      this.#armHold();
    }
    if (this.#asks.length > 0) this.#issueAsks();
  }

  /** The builds wait for the outstanding policy queries; a timer lets them go ahead once the oldest has outlasted the hold. */
  #armHold(): void {
    if (this.#armed) return;
    this.#armed = true;
    // The oldest query still inside the hold: an expired one no longer blocks, and counting it would make the timer fire at once and again.
    const now = this.#now();
    let oldest = Infinity;
    for (const issued of this.#outstanding.values()) if (now - issued < this.#hold) oldest = Math.min(oldest, issued);
    this.#after(() => {
      this.#armed = false;
      this.#schedule();
    }, Math.max(1, oldest + this.#hold - now));
  }

  #issueAsks(): void {
    const asks = this.#asks.splice(0);
    bump(this.#stats.asks_per_issue, asks.length);
    for (const ask of asks) {
      const id = this.#next++;
      this.#outstanding.set(id, this.#now());
      let called: Promise<unknown>;
      try {
        called = ask.call();
      } catch (error) {
        called = Promise.reject(error);
      }
      called.then(
        (value) => {
          this.#settleAsk(id);
          ask.resolve(value);
        },
        (error) => {
          this.#settleAsk(id);
          ask.reject(error);
        },
      );
    }
  }

  #settleAsk(id: number): void {
    this.#outstanding.delete(id);
    // The last answer of a wave lets the builds that wait for it go ahead.
    if (!this.#blocking() && this.#builds.length > 0) this.#schedule();
  }

  /** Answer every pending build, round after round: each round collects the next query of every build that has one and asks the batch once. */
  #answerBuilds(): void {
    const batch = this.#batch as BatchRunner;
    let live = this.#builds.splice(0);
    bump(this.#stats.builds_per_flush, live.length);
    while (live.length > 0) {
      const owners: Build[] = [];
      const calls: ProbeCall[] = [];
      for (const build of live) {
        const { answer } = build;
        build.answer = undefined;
        let step: IteratorResult<ProbeCall, unknown>;
        try {
          step = answer === undefined ? build.gen.next() : answer.ok ? build.gen.next(answer.result) : build.gen.throw(answer.error);
        } catch (error) {
          build.reject(error);
          continue;
        }
        if (step.done === true) {
          this.#stats.builds += 1;
          build.resolve(step.value);
        } else {
          owners.push(build);
          calls.push(step.value);
        }
      }
      if (calls.length === 0) break;
      this.#stats.rounds += 1;
      this.#stats.queries += calls.length;
      bump(this.#stats.queries_per_round, calls.length);
      let settled: Settled[];
      try {
        settled = batch(calls);
      } catch (error) {
        // The batch itself failed (not a query's refusal): every build waiting on it fails with it.
        for (const build of owners) build.reject(error);
        break;
      }
      owners.forEach((build, index) => void (build.answer = settled[index] as Settled));
      live = owners;
    }
  }
}
