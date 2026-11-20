// The request-by-request comparison and the corpus run on hand-made traces, against an echo backend (it answers a trace with the
// trace's own steps) and one fault per changed echo: every kind with its step and seat, refusals, a dropped result, the batch.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { BattleBackend, BattleStepOutput } from "../src/battle-backend.ts";
import { BattleHarnessError, compareBattles, compareStep } from "../src/battle-compare.ts";
import { addBattles, newBattleCorpus } from "../src/battle-corpus.ts";
import { readInteractionClasses } from "../src/battle-classes.ts";
import { battleCorpusSummary } from "../src/battle-report.ts";
import { type BattleTrace, readBattleTrace } from "../src/battle-trace.ts";

const sample = (name: string): BattleTrace => readBattleTrace(JSON.parse(readFileSync(fileURLToPath(new URL(`../../../fixtures/contracts/v1/battle-trace/valid/${name}.json`, import.meta.url)), "utf8")));
const minimal = sample("minimal");
const tie = sample("forced-tie");

/** A trace of `n` steps: the minimal one with its middle step repeated (requests renumbered). */
function longer(n: number): BattleTrace {
  const t = structuredClone(minimal);
  const [start, end] = t.steps;
  const middle = Array.from({ length: n - 2 }, (_, i) => ({ ...structuredClone(end as BattleTrace["steps"][number]), omniscient: ["|turn|" + String(i + 1)], channels: { p1: ["|turn|" + String(i + 1)], p2: ["|turn|" + String(i + 1)] } }));
  t.steps = [start, ...middle, end] as BattleTrace["steps"];
  t.battle_id = `sample/longer-${n}`;
  return readBattleTrace(t);
}

const outputOf = (t: BattleTrace, k: number): BattleStepOutput => {
  const s = t.steps[k] as BattleTrace["steps"][number];
  const last = k === t.steps.length - 1;
  return { channels: structuredClone(s.channels), requests: structuredClone(s.requests), draws: structuredClone(s.draws), ended: last, winner: last ? t.outcome.winner : null, turns: last ? t.outcome.turns : 0, state_sha256: s.state_sha256 };
};

/** Answers every battle with its own trace; `change` alters one output. */
function echo(traces: readonly BattleTrace[], change?: (o: BattleStepOutput, id: string, step: number) => BattleStepOutput, refuse?: (id: string, step: number) => boolean): BattleBackend & { released: string[] } {
  const at = new Map<string, number>();
  const out = (id: string, k: number): BattleStepOutput => {
    const t = traces.find((x) => x.battle_id === id) as BattleTrace;
    const o = outputOf(t, k);
    return change === undefined ? o : change(o, id, k);
  };
  const released: string[] = [];
  return {
    id: "echo",
    backend_format: 1,
    backend_commit: "e".repeat(40),
    released,
    start: (batch) => batch.map((b) => (refuse?.(b.battle_id, 0) === true ? { ok: false, error: new Error("no") } : (at.set(b.battle_id, 0), { ok: true, result: { handle: b.battle_id, output: out(b.battle_id, 0) } }))),
    step: (batch) =>
      batch.map((b) => {
        const k = (at.get(b.handle) as number) + 1;
        at.set(b.handle, k);
        return refuse?.(b.handle, k) === true ? { ok: false, error: new Error("no") } : { ok: true, result: out(b.handle, k) };
      }),
    release: (handles) => released.push(...handles),
  };
}

const one = (backend: BattleBackend, t: BattleTrace = minimal) => (compareBattles(backend, [t])[0] as { difference: unknown; compared: number }).difference;

describe("compareBattles", () => {
  it("finds no difference when the backend answers the trace itself, in any batch size, and releases every battle", () => {
    const traces = [minimal, tie, longer(5)];
    for (const size of [1, 2, 64]) {
      const backend = echo(traces);
      const results = compareBattles(backend, traces, size);
      expect(results.map((r) => [r.battle_id, r.compared, r.steps, r.difference])).toEqual([
        ["sample/minimal", 2, 2, null],
        ["sample/forced-tie", 2, 2, null],
        ["sample/longer-5", 5, 5, null],
      ]);
      expect(backend.released.length).toBe(3);
    }
  });

  it("reports each kind at its step and seat", () => {
    const cases: [string, (o: BattleStepOutput, k: number) => BattleStepOutput, unknown][] = [
      ["a channel line", (o, k) => (k === 1 ? { ...o, channels: { ...o.channels, p2: ["|start", "|win|p2"] } } : o), { kind: "channel", step: 1, seat: "p2", where: "player-view line 1 of the step (requests aside) differs" }],
      ["a missing channel line", (o, k) => (k === 1 ? { ...o, channels: { ...o.channels, p1: ["|start"] } } : o), { kind: "channel", step: 1, seat: "p1", where: "player-view line 1 of the step (requests aside) differs" }],
      ["a request line", (o, k) => (k === 0 ? { ...o, channels: { ...o.channels, p1: ["|teampreview", '|request|{"x":1}'] } } : o), { kind: "request", step: 0, seat: "p1", where: "request line 0 of the step differs" }],
      ["a request hash", (o, k) => (k === 0 ? { ...o, requests: { ...o.requests, p2: [{ ...(o.requests.p2[0] as object), request_hash: "f".repeat(64) } as never] } } : o), { kind: "request", step: 0, seat: "p2", where: "request 0 of the step has another number or request_hash" }],
      ["a request number", (o, k) => (k === 0 ? { ...o, requests: { ...o.requests, p1: [{ ...(o.requests.p1[0] as object), request_seq: 2 } as never] } } : o), { kind: "request", step: 0, seat: "p1", where: "request 0 of the step has another number or request_hash" }],
      ["a verdict", (o, k) => (k === 0 ? { ...o, requests: { ...o.requests, p1: [{ ...(o.requests.p1[0] as object), mask: [{ choice: "team 1234", verdict: "pending_legal" }] } as never] } } : o), { kind: "mask", step: 0, seat: "p1", where: "candidate 0 of request 0 of the step differs" }],
      ["a mask on one side", (o, k) => (k === 0 ? { ...o, requests: { ...o.requests, p2: [{ request_seq: 1, request_hash: "a".repeat(64) }] } } : o), { kind: "mask", step: 0, seat: "p2", where: "request 0 of the step has a mask on one side only" }],
      ["a draw request", (o, k) => (k === 1 ? { ...o, draws: o.draws.slice(1) } : o), { kind: "draws", step: 1, seat: null, where: "draw request 0 of the step differs" }],
      ["an extra draw request", (o, k) => (k === 0 ? { ...o, draws: [...o.draws, { method: "random", args: [], result: ["0"] }] } : o), { kind: "draws", step: 0, seat: null, where: "draw request 1 of the step differs" }],
      ["the state", (o, k) => (k === 1 ? { ...o, state_sha256: "0".repeat(64) } : o), { kind: "state", step: 1, seat: null, where: "the engine state after the step differs" }],
      ["an early end", (o, k) => (k === 0 ? { ...o, ended: true } : o), { kind: "outcome", step: 0, seat: null, where: "the battle ends before the trace's does" }],
      ["no end", (o, k) => (k === 1 ? { ...o, ended: false } : o), { kind: "outcome", step: 1, seat: null, where: "the battle does not end where the trace's does" }],
      ["the winner", (o, k) => (k === 1 ? { ...o, winner: "p2" } : o), { kind: "outcome", step: 1, seat: null, where: "the winner differs" }],
      ["the turns", (o, k) => (k === 1 ? { ...o, turns: 9 } : o), { kind: "outcome", step: 1, seat: null, where: "the turn count differs" }],
    ];
    for (const [what, change, expected] of cases) expect(one(echo([minimal], (o, _id, k) => change(o, k))), what).toEqual(expected);
  });

  it("a request without a mask on both sides (a wait request) compares equal", () => {
    const t = structuredClone(minimal);
    for (const seat of ["p1", "p2"] as const) delete (t.steps[0] as BattleTrace["steps"][number]).requests[seat][0]?.mask;
    expect(one(echo([t]), t)).toBeNull();
  });

  it("a state hash the backend does not give is not compared", () => {
    expect(one(echo([minimal], (o) => ({ ...o, state_sha256: undefined }) as never))).toBeNull();
  });

  it("stops a battle at its first difference and compares the others on", () => {
    const a = longer(4);
    const b = structuredClone(tie);
    const results = compareBattles(echo([a, b], (o, id, k) => (id === a.battle_id && k === 1 ? { ...o, draws: [] } : o)), [a, b], 2);
    expect(results.map((r) => [r.compared, (r.difference as { kind: string } | null)?.kind ?? null])).toEqual([
      [2, "draws"],
      [2, null],
    ]);
  });

  it("a refused start or step is a refusal; a batch with a dropped result is a count difference for each battle", () => {
    expect(one(echo([minimal], undefined, (_id, k) => k === 0))).toEqual({ kind: "refusal", step: 0, seat: null, where: "the backend refused the step" });
    expect(one(echo([minimal], undefined, (_id, k) => k === 1))).toMatchObject({ kind: "refusal", step: 1 });
    const pair = [minimal, tie];
    const dropping = echo(pair);
    const base = dropping.step.bind(dropping);
    const results = compareBattles({ ...dropping, step: (batch) => base(batch).slice(1) }, pair, 2);
    for (const r of results) expect(r.difference).toMatchObject({ kind: "count", step: 1 });
    const startDropping = echo(pair);
    const startBase = startDropping.start.bind(startDropping);
    expect(compareBattles({ ...startDropping, start: (batch) => startBase(batch).slice(0, 1) }, pair, 2).map((r) => r.difference)).toEqual([
      { kind: "count", step: 0, seat: null, where: "the batch came back with another number of outcomes than inputs" },
      { kind: "count", step: 0, seat: null, where: "the batch came back with another number of outcomes than inputs" },
    ]);
  });

  it("refuses a batch size below one and a trace that does not end at its last step", () => {
    expect(() => compareBattles(echo([minimal]), [minimal], 0)).toThrow(BattleHarnessError);
    const open = structuredClone(minimal);
    open.steps = [open.steps[0]] as BattleTrace["steps"];
    expect(() => compareBattles(echo([open]), [open])).toThrow("must end at its last step");
  });

  it("compareStep compares one step alone", () => {
    expect(compareStep(minimal.steps[0] as BattleTrace["steps"][number], outputOf(minimal, 0), 0, false, minimal.outcome)).toBeNull();
  });
});

describe("a corpus run", () => {
  const classes = readInteractionClasses({ schema_version: 1, data_class: "synthetic", classes: [{ id: "shuffled", description: "a speed tie", all: [[{ draw: "shuffle" }]] }, { id: "won", description: "a win line", all: [[{ line: "win" }]] }] });
  const spec = { id: "toy", kind: "regression" as const, seed: "a".repeat(64), pool: "b".repeat(64) };

  it("tallies every compared step with its entities, tags its classes and keeps each first difference", () => {
    const corpus = newBattleCorpus(spec, classes);
    const a = longer(3);
    addBattles(corpus, classes, echo([a]), [a]);
    addBattles(corpus, classes, echo([tie], (o, _id, k) => (k === 1 ? { ...o, turns: 1 } : o)), [tie]);
    addBattles(corpus, classes, echo([minimal], (o, _id, k) => (k === 1 ? { ...o, turns: 2 } : o)), [minimal]);
    expect(corpus.battles).toBe(3);
    expect(corpus.steps).toBe(7);
    expect(corpus.tally.items).toBe(7);
    expect(corpus.unreached).toBe(0);
    expect(corpus.differences).toEqual([
      { battle_id: "sample/forced-tie", kind: "outcome", step: 1, seat: null, where: "the turn count differs" },
      { battle_id: "sample/minimal", kind: "outcome", step: 1, seat: null, where: "the turn count differs" },
    ]);
    expect(corpus.classes.get("shuffled")).toEqual({ tagged: 4, equal: 2 });
    expect(corpus.classes.get("won")).toEqual({ tagged: 2, equal: 1 });
    const summary = battleCorpusSummary(corpus) as Record<string, unknown>;
    expect(summary.differences_by_kind).toEqual({ outcome: 2 });
    expect(summary.compared_steps).toBe(7);
    expect((summary.entities as { id: string; exercised: number }[]).some((e) => e.exercised === 0)).toBe(true);
    expect((summary.entity_kinds as Record<string, { entities: number }>).species?.entities).toBeGreaterThan(0);
    expect(summary.tally_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("counts the steps after a first difference as unreached and a refusal as a settled difference", () => {
    const corpus = newBattleCorpus(spec, classes);
    const a = longer(4);
    addBattles(corpus, classes, echo([a], undefined, (_id, k) => k === 1), [a]);
    expect(corpus.unreached).toBe(2);
    expect(corpus.tally.differences.map((d) => d.reason)).toEqual(["settled"]);
    const counted = newBattleCorpus(spec, classes);
    const pair = [minimal, tie];
    const e = echo(pair);
    const base = e.step.bind(e);
    addBattles(counted, classes, { ...e, step: (batch) => base(batch).slice(1) }, pair);
    expect(counted.tally.differences.map((d) => d.reason)).toEqual(["count", "count"]);
  });
});
