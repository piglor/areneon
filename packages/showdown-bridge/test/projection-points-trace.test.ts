/* eslint-disable @typescript-eslint/no-explicit-any */
// `projectionBattle` on a stand-in recorder (#1464): a request without a mask asks for no decision, and a decision that is not the
// last line its seat got in the step is refused with a `ProjectionPointError` (the real recordings are in the corpus gate).
import { describe, expect, it, vi } from "vitest";

const recorded: { value: any } = { value: undefined };
vi.mock("../src/battle-trace.ts", () => ({
  recordBattleTrace: async (_engine: unknown, _spec: unknown, options: { onStep?: (step: number, battle: unknown) => void }) => {
    recorded.value.steps.forEach((_: unknown, k: number) => options.onStep?.(k, { toJSON: () => ({ sides: [], field: {} }) }));
    return { trace: recorded.value, record: { manifest: {} } };
  },
}));

const { ProjectionPointError, projectionBattle } = await import("../src/projection-points.ts");

const spec: any = { battle_id: "t/1", format_id: "f", seed: "s", teams: { p1: [], p2: [] }, actors: {}, max_turns: 1 };
const noSpec: any = { fields: [], groups: [], omitted: [] };

describe("projectionBattle on a stand-in recorder", () => {
  it("skips a request with no mask and reads no decision from it", async () => {
    const request = { request_seq: 1 };
    recorded.value = { steps: [{ channels: { p1: ["|request|{}"], p2: [] }, requests: { p1: [request], p2: [] } }] };
    const battle = await projectionBattle({} as any, spec, noSpec);
    expect(battle.points).toHaveLength(1);
    expect(battle.points[0]?.decisions).toEqual([]);
    expect(battle.points[0]?.cut).toEqual({ p1: 1, p2: 0 });
  });

  it("refuses a decision whose seat has no channel line yet, and one whose request is not the last line", async () => {
    const request = { request_seq: 1, mask: [] };
    recorded.value = { steps: [{ channels: { p1: [], p2: [] }, requests: { p1: [request], p2: [] } }] };
    await expect(projectionBattle({} as any, spec, noSpec)).rejects.toThrow(ProjectionPointError);
    recorded.value = { steps: [{ channels: { p1: ["|request|{}", "|turn|2"], p2: [] }, requests: { p1: [request], p2: [] } }] };
    await expect(projectionBattle({} as any, spec, noSpec)).rejects.toThrow(/not the last line/);
  });
});
