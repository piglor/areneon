/* eslint-disable @typescript-eslint/no-explicit-any */
import { beforeAll, describe, expect, it } from "vitest";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { firstDifference, plainRequest, v2Candidates, v2Request, v4Conditions, v4Durations } from "../src/world-checks.ts";
import { type WorldBuilder, type WorldInput, type WorldResult } from "../src/world-builder.ts";
import { Refuse } from "../src/world-refusal.ts";
import { createRedrawer } from "../src/world-redraw.ts";
import { cases } from "./world-builder-support.ts";

let request: EngineRequest;
let input: WorldInput;
beforeAll(() => {
  input = (cases().find((c) => (c.input.request.active ?? []).length === 2 && c.input.request.side.pokemon.length === 4) as { input: WorldInput }).input;
  request = input.request;
});

const copy = (change: (r: any) => void): EngineRequest => {
  const r = structuredClone(request) as any;
  change(r);
  return r;
};

describe("V2: the world's own request equals the real one", () => {
  it("ignores what differs by transport (rqid, the side's name) and finds the first difference otherwise", () => {
    const named = copy((r) => {
      r.rqid = 7;
      r.side.name = "Somebody";
    });
    expect(plainRequest(named)).toEqual(plainRequest(request));
    expect(v2Request(named, request)).toBeNull();
    expect(firstDifference({ a: [1, 2] }, { a: [1, 2] })).toBeNull();
    expect(firstDifference({ a: [1, 2] }, { a: { 0: 1 } })).toBe("/a");
    expect(firstDifference({ a: 1 }, { b: 1 })).toBe("/a");
    expect(firstDifference(1, 2)).toBe("/");
    expect(firstDifference({ a: { b: 1 } }, { a: { b: 2 } })).toBe("/a/b");
    expect(firstDifference({ a: null }, { a: { b: 1 } })).toBe("/a");
  });

  it.each([
    ["moves and PP", (r: any) => (r.active[0].moves[0].pp -= 1), { item: null, state_class: "E", redraw: false }],
    ["stats", (r: any) => (r.side.pokemon[0].stats.atk += 1), { item: null, state_class: "E", redraw: false }],
    ["conditions", (r: any) => (r.side.pokemon[0].condition = "1/2"), { item: null, state_class: "E", redraw: false }],
    ["Mega Evolution", (r: any) => (r.active[0].canMegaEvo = true), { item: null, state_class: "E", redraw: false }],
    ["disabled moves", (r: any) => (r.active[0].moves[0].disabled = true), { item: "choice_lock", state_class: "W", redraw: false }],
    ["trapping flags", (r: any) => (r.active[0].trapped = true), { item: null, state_class: null, redraw: true }],
    ["trapping flags", (r: any) => (r.active[1].maybeTrapped = true), { item: null, state_class: null, redraw: true }],
    ["a request field", (r: any) => (r.side.pokemon[0].details += "x"), { item: null, state_class: "E", redraw: false }],
    ["a request field", (r: any) => (r.side.pokemon[0].item = ""), { item: "item_removed", state_class: "W", redraw: false }],
  ])("names %s and what a redraw can do about it", (words, change, expected) => {
    const failure = v2Request(copy(change), request);
    expect(failure).toMatchObject({ code: "v2_request", ...expected });
    expect(failure?.detail).toContain(words);
  });

  it("compares the candidate list with its pending flags", () => {
    expect(v2Candidates(request, request)).toBeNull();
    const trapped = copy((r) => (r.active[0].maybeTrapped = true));
    expect(v2Candidates(trapped, request)).toMatchObject({ code: "v2_candidates", item: null, state_class: null, redraw: true });
    // the other way round too: the list differs in length or in a flag
    expect(v2Candidates(request, trapped)?.code).toBe("v2_candidates");
    const locked = copy((r) => (r.active[0].moves[0].disabled = true));
    expect(v2Candidates(locked, request)?.code).toBe("v2_candidates");
  });
});

describe("V4: conditions equal and consistent with the log", () => {
  const shown = { weather: "raindance", terrain: null, field: ["trickroom", "gravity"], own: { reflect: 1 }, opponent: { spikes: 2 } };
  it("passes equal conditions in any order", () => {
    expect(v4Conditions({ ...shown, field: ["gravity", "trickroom"] }, shown)).toBeNull();
  });

  it.each([
    ["weather", { weather: null }],
    ["terrain", { terrain: "grassyterrain" }],
    ["field", { field: ["trickroom"] }],
    ["own side", { own: { reflect: 2 } }],
    ["own side ids", { own: { lightscreen: 1 } }],
    ["opponent side", { opponent: {} }],
  ])("fails when the %s differs", (_name, change) => {
    expect(v4Conditions({ ...shown, ...change }, shown)).toMatchObject({ code: "v4_conditions", item: "condition_duration", state_class: "W", redraw: false });
  });

  it("fails an effect the log still shows but that has no turns left, as evidence against the source's sampled item", () => {
    expect(v4Durations([])).toBeNull();
    expect(v4Durations([{ what: "weather", remaining: 1 }, { what: "side", remaining: 3 }])).toBeNull();
    expect(v4Durations([{ what: "weather", remaining: 4 }, { what: "side", remaining: 0 }])).toMatchObject({ code: "v4_duration", item: "condition_duration", state_class: "W", redraw: true });
    expect(v4Durations([{ what: "volatile", remaining: -1 }])).toMatchObject({ code: "v4_duration", item: "volatile_duration" });
  });
});

describe("the bounded redraw interface", () => {
  const refusal = (redraw: boolean, code: "v2_request" | "v1_revealed_fact" = "v2_request") => new Refuse(code, "fixed words", { redraw }).refusal;
  const stub = (results: WorldResult[]): { builder: WorldBuilder; calls: number[] } => {
    const calls: number[] = [];
    return {
      calls,
      builder: {
        build: () => {
          calls.push(calls.length);
          return results[Math.min(calls.length - 1, results.length - 1)] as WorldResult;
        },
        counts: () => {
          throw new Error("not used");
        },
      },
    };
  };
  const world = { kind: "world", world: {} } as unknown as WorldResult;

  it("returns the first world, redrawing after each refusal a redraw can fix, at most R times", () => {
    const { builder, calls } = stub([{ kind: "refused", refusal: refusal(true) }, { kind: "refused", refusal: refusal(true, "v1_revealed_fact") }, world]);
    const redrawer = createRedrawer(builder, 4);
    const drawn: number[] = [];
    const out = redrawer.build((attempt) => {
      drawn.push(attempt);
      return input;
    });
    expect(out).toMatchObject({ attempts: 3, exhausted: false });
    expect(out.result.kind).toBe("world");
    expect(drawn).toEqual([0, 1, 2]);
    expect(calls).toHaveLength(3);
    expect(redrawer.counts()).toEqual({ worlds: 1, attempts: 3, redraws: 2, exhausted: 0, final_refusals: 0, redrawn: { "V2:-": 1, "V1:-": 1 } });
  });

  it("stops at the bound with the last refusal, counted as exhausted", () => {
    const { builder } = stub([{ kind: "refused", refusal: refusal(true) }]);
    const redrawer = createRedrawer(builder, 2);
    const out = redrawer.build(() => input);
    expect(out).toMatchObject({ attempts: 3, exhausted: true });
    expect(out.result.kind).toBe("refused");
    expect(redrawer.counts()).toMatchObject({ worlds: 1, attempts: 3, redraws: 2, exhausted: 1, final_refusals: 0 });
  });

  it("does not redraw a refusal no redraw can fix", () => {
    const { builder } = stub([{ kind: "refused", refusal: refusal(false) }]);
    const redrawer = createRedrawer(builder, 5);
    expect(redrawer.build(() => input)).toMatchObject({ attempts: 1, exhausted: false });
    expect(redrawer.counts()).toMatchObject({ worlds: 1, attempts: 1, redraws: 0, final_refusals: 1 });
  });

  it("with no redraws allowed asks once, and refuses a bound that is not a count", () => {
    const { builder } = stub([{ kind: "refused", refusal: refusal(true) }]);
    expect(createRedrawer(builder, 0).build(() => input)).toMatchObject({ attempts: 1, exhausted: true });
    expect(() => createRedrawer(builder, -1)).toThrow(RangeError);
    expect(() => createRedrawer(builder, 1.5)).toThrow(RangeError);
  });

  it("counts a copy", () => {
    const { builder } = stub([world]);
    const redrawer = createRedrawer(builder, 1);
    redrawer.build(() => input);
    const counts = redrawer.counts();
    counts.worlds = 99;
    expect(redrawer.counts().worlds).toBe(1);
  });
});
