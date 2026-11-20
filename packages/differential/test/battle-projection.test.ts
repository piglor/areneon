// The `projection` comparison kind (ADR-030 step 5 groundwork G4, #1464): a backend's player-view projection of its battle-state
// tensors against the reference projection of the trace's step, per seat, field by field. Driven here with hand-made views and
// an echo backend (it answers a trace with the trace's own steps); the reference views of a real battle come from the bridge's
// reference packer and projection (`battle-state-projection.python.test.ts`).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { BattleBackend, BattleStepOutput, Seat } from "../src/battle-backend.ts";
import { BattleHarnessError, compareBattles } from "../src/battle-compare.ts";
import { checkProjection, ProjectionError, type SeatView } from "../src/battle-projection.ts";
import { addBattles, newBattleCorpus } from "../src/battle-corpus.ts";
import { readInteractionClasses } from "../src/battle-classes.ts";
import { type BattleTrace, readBattleTrace } from "../src/battle-trace.ts";

const sample = (name: string): BattleTrace => readBattleTrace(JSON.parse(readFileSync(fileURLToPath(new URL(`../../../fixtures/contracts/v1/battle-trace/valid/${name}.json`, import.meta.url)), "utf8")));
const minimal = sample("minimal");
const SPEC = "a".repeat(64);

const view = (seat: Seat, fields: SeatView["fields"], spec_hash = SPEC): SeatView => ({ seat, spec_hash, fields });
/** The reference views of a step: a turn counter, an own hit-point list and the opponent's shown species with an unknown entry. */
const reference = (step: number): Record<Seat, SeatView> => ({
  p1: view("p1", { "battle.turn": step, "seat_roster.hp_shown@own": [100, 87], "seat_roster.shown_species@opponent": [5, null] }),
  p2: view("p2", { "battle.turn": step, "seat_roster.hp_shown@own": [64, 64], "seat_roster.shown_species@opponent": [null, null] }),
});

describe("checkProjection", () => {
  it("finds no difference between equal views, whatever the key order", () => {
    const want = reference(3);
    const got = { p1: view("p1", { "seat_roster.shown_species@opponent": [5, null], "seat_roster.hp_shown@own": [100, 87], "battle.turn": 3 }), p2: structuredClone(want.p2) };
    expect(checkProjection(got, want, 3)).toBeNull();
  });

  it("names the first differing field of the first differing seat, never a value", () => {
    const want = reference(2);
    const got = structuredClone(want);
    (got.p2.fields["seat_roster.hp_shown@own"] as number[])[1] = 63;
    expect(checkProjection(got, want, 2)).toEqual({ kind: "projection", step: 2, seat: "p2", where: "projected field seat_roster.hp_shown@own differs" });
    // An unknown entry (null) is a value like any other: projecting a value where the reference has the unknown marker is a leak.
    const leak = structuredClone(want);
    (leak.p1.fields["seat_roster.shown_species@opponent"] as (number | null)[])[1] = 9;
    expect(checkProjection(leak, want, 2)).toEqual({ kind: "projection", step: 2, seat: "p1", where: "projected field seat_roster.shown_species@opponent differs" });
  });

  it("flags a field present on one side only (an extra field is a field the class does not let the seat read)", () => {
    const want = reference(1);
    const extra = structuredClone(want);
    extra.p1.fields["roster.hp@opponent"] = [1, 2];
    expect(checkProjection(extra, want, 1)).toEqual({ kind: "projection", step: 1, seat: "p1", where: "projected field roster.hp@opponent is present on one side only" });
    const missing = structuredClone(want);
    delete missing.p2.fields["battle.turn"];
    expect(checkProjection(missing, want, 1)).toEqual({ kind: "projection", step: 1, seat: "p2", where: "projected field battle.turn is present on one side only" });
  });

  it("flags a view of another seat or another spec, and a missing view", () => {
    const want = reference(4);
    expect(checkProjection({ p1: want.p2, p2: want.p2 }, want, 4)).toEqual({ kind: "projection", step: 4, seat: "p1", where: "the projection is of another seat or another spec" });
    expect(checkProjection({ p1: want.p1, p2: view("p2", want.p2.fields, "b".repeat(64)) }, want, 4)).toEqual({ kind: "projection", step: 4, seat: "p2", where: "the projection is of another seat or another spec" });
    expect(checkProjection({ p1: undefined, p2: want.p2 }, want, 4)).toEqual({ kind: "projection", step: 4, seat: "p1", where: "the projection is not a seat view" });
  });

  it("never quotes a field name the spec could not have (fixed words), and refuses a float", () => {
    const want = reference(0);
    const odd = structuredClone(want);
    odd.p1.fields["|switch|p1a: Secret"] = 1;
    expect(checkProjection(odd, want, 0)).toEqual({ kind: "projection", step: 0, seat: "p1", where: "the projection names a field outside the spec" });
    const float = structuredClone(want);
    float.p1.fields["battle.turn"] = 0.5;
    expect(checkProjection(float, want, 0)).toEqual({ kind: "projection", step: 0, seat: "p1", where: "projected field battle.turn differs" });
    expect(() => checkProjection(want, { p1: view("p1", { "battle.turn": 0.5 }), p2: want.p2 }, 0)).toThrow(ProjectionError);
  });
});

// ---- the kind inside the battle comparison ----

const outputOf = (t: BattleTrace, k: number): BattleStepOutput => {
  const s = t.steps[k] as BattleTrace["steps"][number];
  const last = k === t.steps.length - 1;
  return { channels: structuredClone(s.channels), requests: structuredClone(s.requests), draws: structuredClone(s.draws), ended: last, winner: last ? t.outcome.winner : null, turns: last ? t.outcome.turns : 0, state_sha256: s.state_sha256 };
};

/** An echo backend whose `project` gives the reference views of its step, changed by `change`. */
function projecting(change?: (v: Record<Seat, SeatView>, step: number) => Record<Seat, SeatView>, withProject = true): BattleBackend {
  let at = 0;
  const backend: BattleBackend = {
    id: "echo",
    backend_format: 1,
    backend_commit: "e".repeat(40),
    start: (batch) => batch.map((b) => ((at = 0), { ok: true, result: { handle: b.battle_id, output: outputOf(minimal, 0) } })),
    step: (batch) => batch.map(() => ((at += 1), { ok: true, result: outputOf(minimal, at) })),
    release: () => undefined,
  };
  if (withProject) backend.project = (_handle, seat) => (change === undefined ? reference(at) : change(reference(at), at))[seat];
  return backend;
}

describe("compareBattles with a projection oracle", () => {
  it("compares every step's views after the five protocol items and finds none differing for a faithful backend", () => {
    const [result] = compareBattles(projecting(), [minimal], 64, undefined, (_trace, step) => reference(step));
    expect(result).toMatchObject({ compared: 2, difference: null });
  });

  it("reports the projection kind at the step and seat where the backend's view differs", () => {
    const backend = projecting((v, step) => {
      if (step === 1) (v.p1.fields["seat_roster.hp_shown@own"] as number[])[0] = 1;
      return v;
    });
    expect(compareBattles(backend, [minimal], 64, undefined, (_trace, step) => reference(step))[0]?.difference).toEqual({ kind: "projection", step: 1, seat: "p1", where: "projected field seat_roster.hp_shown@own differs" });
  });

  it("skips a step the oracle has no views for, and refuses a projection oracle for a backend without projection", () => {
    const backend = projecting(() => ({ p1: view("p1", {}), p2: view("p2", {}) }));
    expect(compareBattles(backend, [minimal], 64, undefined, () => null)[0]?.difference).toBeNull();
    expect(() => compareBattles(projecting(undefined, false), [minimal], 64, undefined, (_t, s) => reference(s))).toThrow(BattleHarnessError);
    // Without an oracle, a backend's projection is never asked.
    expect(compareBattles(projecting(() => { throw new Error("asked"); }), [minimal])[0]?.difference).toBeNull();
  });

  it("a corpus run tallies a projection difference as a value difference", () => {
    const classes = readInteractionClasses({ schema_version: 1, data_class: "synthetic", classes: [{ id: "won", description: "a win line", all: [[{ line: "win" }]] }] });
    const corpus = newBattleCorpus({ id: "toy", kind: "regression", seed: "a".repeat(64), pool: "b".repeat(64) }, classes);
    const backend = projecting((v, step) => {
      if (step === 0) v.p2.fields["battle.turn"] = 99;
      return v;
    });
    addBattles(corpus, classes, backend, [minimal], 64, (_trace, step) => reference(step));
    expect(corpus.differences).toEqual([{ battle_id: "sample/minimal", kind: "projection", step: 0, seat: "p2", where: "projected field battle.turn differs" }]);
    expect(corpus.tally.differences.map((d) => [d.reason, d.field])).toEqual([["value", "projection:p2"]]);
  });
});
