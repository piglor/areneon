// Request-by-request comparison of a battle backend with a trace of the pinned engine (ADR-028 section 5, protocol path). At
// every step, in this order: (1) each player's channel lines other than requests, byte-equal; (2) each request, its line in the
// channel byte-equal and its request_hash and number equal; (3) every candidate's verdict, pending ones included; (4) the draw
// requests (method and arguments) in order, the whole joint sequence; the engine state hash when the backend has one; (5) the
// end, the winner and the turn count. The first differing item is reported with fixed words, never a line of the battle.
import { type BattleBackend, type BattleStepOutput, type Draw, type Seat, SEATS, type StepRequest } from "./battle-backend.ts";
import { type BattleTrace, endStep, type Step } from "./battle-trace.ts";
import type { Settled } from "./backend.ts";
import { checkProjection, type ProjectionOracle } from "./battle-projection.ts";

export type BattleDifferenceKind = "channel" | "request" | "mask" | "draws" | "state" | "outcome" | "refusal" | "count" | "projection";

export interface BattleDifference {
  kind: BattleDifferenceKind;
  /** The step (0: the start). */
  step: number;
  /** The seat whose view differs, null for the draws, the state and the outcome. */
  seat: Seat | null;
  /** Fixed words naming the first difference. */
  where: string;
}

const REQUEST = "|request|";
const isRequest = (line: string): boolean => line.startsWith(REQUEST);

/** The first index where two lists differ, or -1 when they are equal. */
function firstIndex<T>(a: readonly T[], b: readonly T[], same: (x: T, y: T) => boolean): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (!same(a[i] as T, b[i] as T)) return i;
  return a.length === b.length ? -1 : n;
}

const eq = (x: string, y: string): boolean => x === y;
const drawKey = (d: Draw): string => JSON.stringify([d.method, d.args]);

/** Compare one step's output with the trace's step. `ends` says whether the trace's battle ends at this step. */
export function compareStep(expected: Step, got: BattleStepOutput, index: number, ends: boolean, outcome: BattleTrace["outcome"]): BattleDifference | null {
  const at = (kind: BattleDifferenceKind, seat: Seat | null, where: string): BattleDifference => ({ kind, step: index, seat, where });
  for (const seat of SEATS) {
    const i = firstIndex(expected.channels[seat].filter((l) => !isRequest(l)), got.channels[seat].filter((l) => !isRequest(l)), eq);
    if (i >= 0) return at("channel", seat, `player-view line ${i} of the step (requests aside) differs`);
  }
  for (const seat of SEATS) {
    const i = firstIndex(expected.channels[seat].filter(isRequest), got.channels[seat].filter(isRequest), eq);
    if (i >= 0) return at("request", seat, `request line ${i} of the step differs`);
    const want = expected.requests[seat];
    const have = got.requests[seat];
    const j = firstIndex(want, have, (x, y) => x.request_seq === y.request_seq && x.request_hash === y.request_hash);
    if (j >= 0) return at("request", seat, `request ${j} of the step has another number or request_hash`);
  }
  for (const seat of SEATS) {
    for (const [j, want] of expected.requests[seat].entries()) {
      const have = (got.requests[seat][j] as StepRequest).mask;
      if (want.mask === undefined || have === undefined) {
        if (want.mask !== have) return at("mask", seat, `request ${j} of the step has a mask on one side only`);
        continue;
      }
      const k = firstIndex(want.mask, have, (x, y) => x.choice === y.choice && x.verdict === y.verdict);
      if (k >= 0) return at("mask", seat, `candidate ${k} of request ${j} of the step differs`);
    }
  }
  const d = firstIndex(expected.draws.map(drawKey), got.draws.map(drawKey), eq);
  if (d >= 0) return at("draws", null, `draw request ${d} of the step differs`);
  if (got.state_sha256 !== undefined && got.state_sha256 !== expected.state_sha256) return at("state", null, "the engine state after the step differs");
  if (got.ended !== ends) return at("outcome", null, ends ? "the battle does not end where the trace's does" : "the battle ends before the trace's does");
  if (ends && (got.winner !== outcome.winner || got.turns !== outcome.turns)) return at("outcome", null, got.winner !== outcome.winner ? "the winner differs" : "the turn count differs");
  return null;
}

/** What comparing one battle gave: how many steps were compared (equal ones, plus the first difference) and that difference. */
export interface BattleComparison {
  battle_id: string;
  steps: number;
  compared: number;
  difference: BattleDifference | null;
}

export class BattleHarnessError extends Error {
  override name = "BattleHarnessError";
}

/** Called once per compared step, in each battle's step order (the corpus runner tallies from it). */
export type StepListener = (trace: number, step: number, difference: BattleDifference | null) => void;

const settledDifference = (outcome: Settled<unknown> | undefined, index: number, count: boolean): BattleDifference | null => {
  if (count) return { kind: "count", step: index, seat: null, where: "the batch came back with another number of outcomes than inputs" };
  return outcome !== undefined && outcome.ok ? null : { kind: "refusal", step: index, seat: null, where: "the backend refused the step" };
};

/**
 * Drive `traces` through `backend` in batches of `batchSize` battles, stepping every live battle of a batch together (a
 * battle leaves its batch at its first difference or its last step), and compare at every step. Results in trace order.
 * With a `projection` oracle (G4, #1464), a step whose five protocol items agree is then compared by the backend's player-view
 * projection of each seat (`backend.project`) against the oracle's views of the step (`checkProjection`, kind `projection`); a
 * step the oracle has no views for is not projected.
 */
export function compareBattles(backend: BattleBackend, traces: readonly BattleTrace[], batchSize = 64, listen?: StepListener, projection?: ProjectionOracle): BattleComparison[] {
  if (!Number.isSafeInteger(batchSize) || batchSize < 1) throw new BattleHarnessError("a batch holds at least one battle");
  const project = backend.project?.bind(backend);
  if (projection !== undefined && project === undefined) throw new BattleHarnessError("a projection oracle needs a backend with a projection");
  const projected = (n: number, step: number, handle: string | undefined): BattleDifference | null => {
    const want = projection?.(n, step) ?? null;
    if (want === null || project === undefined || handle === undefined) return null;
    return checkProjection({ p1: project(handle, "p1"), p2: project(handle, "p2") }, want, step);
  };
  const results: BattleComparison[] = traces.map((t) => {
    if (endStep(t) !== t.steps.length - 1) throw new BattleHarnessError("a trace must end at its last step (read it with readBattleTrace)");
    return { battle_id: t.battle_id, steps: t.steps.length, compared: 0, difference: null };
  });
  for (let first = 0; first < traces.length; first += batchSize) {
    const members = traces.slice(first, first + batchSize).map((_, i) => first + i);
    const handles = new Map<number, string>();
    const record = (n: number, step: number, difference: BattleDifference | null): void => {
      const result = results[n] as BattleComparison;
      result.compared += 1;
      result.difference = difference;
      listen?.(n, step, difference);
    };
    const check = (n: number, step: number, out: Settled<BattleStepOutput> | undefined, count: boolean): boolean => {
      const trace = traces[n] as BattleTrace;
      const refused = settledDifference(out, step, count);
      const difference = refused ?? compareStep(trace.steps[step] as Step, (out as { result: BattleStepOutput }).result, step, step === trace.steps.length - 1, trace.outcome) ?? projected(n, step, handles.get(n));
      record(n, step, difference);
      return difference === null && step < trace.steps.length - 1;
    };
    const starts = backend.start(
      members.map((n) => {
        const t = traces[n] as BattleTrace;
        return { battle_id: t.battle_id, format_id: t.format_id, seed: t.seed, teams: { p1: [...t.teams.p1], p2: [...t.teams.p2] }, draws: (t.steps[0] as Step).draws };
      }),
    );
    let live = members.filter((n, i) => {
      const out = starts[i];
      const mapped = out?.ok === true ? { ok: true as const, result: out.result.output } : out;
      if (out?.ok === true) handles.set(n, out.result.handle);
      return check(n, 0, mapped as Settled<BattleStepOutput> | undefined, starts.length !== members.length);
    });
    for (let step = 1; live.length > 0; step++) {
      const outs = backend.step(
        live.map((n) => {
          const next = (traces[n] as BattleTrace).steps[step] as Step;
          return { handle: handles.get(n) as string, actions: next.actions, draws: next.draws };
        }),
      );
      const count = outs.length !== live.length;
      live = live.filter((n, i) => check(n, step, outs[i], count));
    }
    backend.release([...handles.values()]);
  }
  return results;
}
