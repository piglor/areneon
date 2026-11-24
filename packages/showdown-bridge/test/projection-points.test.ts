/* eslint-disable @typescript-eslint/no-explicit-any */
// The trace reader of the battle-state projection (#1464): `trimState` keeps what the spec reads and drops everything else, for a
// full engine state and for the sparse and odd shapes a state may have (a missing key, a non-object value, a non-array list).
import { beforeAll, describe, expect, it } from "vitest";
import { type BattleStateSpec, loadBattleStateSpec } from "../src/battle-state-spec.ts";
import { type Engine, loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { SPEC_V2_PATH, trimState } from "../src/projection-points.ts";

let spec: BattleStateSpec;
beforeAll(() => {
  const engine: Engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  spec = loadBattleStateSpec(engine, GOLDEN_FORMAT, new URL(`file://${SPEC_V2_PATH}`));
});

const JUNK = { notAnEngineKey: 1 };

describe("trimState", () => {
  it("keeps the join keys and the keys a field reads, at every level, and drops the rest", () => {
    const state = {
      turn: 3,
      prngSeed: "x",
      ...JUNK,
      field: { weather: "raindance", weatherState: { duration: 4, ...JUNK }, terrainState: { duration: 2, ...JUNK }, pseudoWeather: { trickroom: { duration: 3, ...JUNK } }, ...JUNK },
      sides: [
        {
          id: "p1",
          ...JUNK,
          sideConditions: { tailwind: { duration: 2, ...JUNK } },
          slotConditions: [{ wish: { duration: 1, ...JUNK } }, {}],
          active: [0],
          pokemon: [
            {
              set: { name: "Mon", species: "Rillaboom", evs: { hp: 1 } },
              details: "Rillaboom, L50",
              isActive: true,
              position: 0,
              hp: 100,
              statusState: { time: 2, ...JUNK },
              itemState: { ...JUNK },
              abilityState: { ...JUNK },
              speciesState: { ...JUNK },
              volatiles: { stall: { counter: 3, ...JUNK } },
              moveSlots: [{ id: "protect", pp: 8, ...JUNK }],
              baseMoveSlots: [{ id: "protect", pp: 8, ...JUNK }],
              ...JUNK,
            },
          ],
        },
      ],
    };
    const out = trimState(spec, state);
    expect(JSON.stringify(out)).not.toContain("notAnEngineKey");
    expect(out.turn).toBe(3);
    expect(out.prngSeed).toBeUndefined();
    expect(out.sides[0].id).toBe("p1");
    expect(out.sides[0].active).toEqual([0]);
    expect(out.sides[0].slotConditions).toHaveLength(2);
    expect(out.sides[0].pokemon[0].set).toEqual({ name: "Mon", species: "Rillaboom" });
    expect(out.sides[0].pokemon[0].details).toBe("Rillaboom, L50");
    expect(out.sides[0].pokemon[0].moveSlots[0].id).toBe("protect");
    expect(out.field.pseudoWeather.trickroom).toBeDefined();
  });

  it("copes with a state that lacks the optional keys", () => {
    const out = trimState(spec, { field: {}, sides: [{ id: "p2", pokemon: [{ hp: 5 }], active: [] }] });
    expect(out.field).toEqual({});
    expect(out.sides[0].pokemon).toEqual([{ hp: 5 }].map((m) => ({ ...m })));
    expect(out.sides[0].sideConditions).toBeUndefined();
    expect(out.sides[0].slotConditions).toBeUndefined();
  });

  it("passes a value that is not an object through where an object is expected, and leaves a list that is not an array", () => {
    const out = trimState(spec, {
      field: { weatherState: null, pseudoWeather: null },
      sides: [
        {
          id: "p1",
          sideConditions: null,
          slotConditions: "none",
          active: [],
          pokemon: [{ set: "not an object", volatiles: null, statusState: null, moveSlots: ["odd", 7], baseMoveSlots: "none" }],
        },
      ],
    });
    expect(out.field.weatherState).toBeNull();
    expect(out.field.pseudoWeather).toBeNull();
    expect(out.sides[0].sideConditions).toBeNull();
    expect(out.sides[0].slotConditions).toBe("none");
    expect(out.sides[0].pokemon[0].set).toBe("not an object");
    expect(out.sides[0].pokemon[0].volatiles).toBeNull();
    expect(out.sides[0].pokemon[0].moveSlots).toEqual(["odd", 7]);
    expect(out.sides[0].pokemon[0].baseMoveSlots).toBe("none");
  });
});
