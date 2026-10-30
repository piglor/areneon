import { describe, expect, it } from "vitest";
import type { BeliefInput } from "../src/index.ts";
import { belief } from "../src/index.ts";
import { input } from "./support.ts";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
const same: Equal<keyof BeliefInput, "prior" | "vocabulary" | "config" | "opponent" | "flags" | "trapping" | "damage" | "speed"> = true;

describe("the type test (ADR-025 section 8)", () => {
  it("the input has no field for the opponent's team, the omniscient log, hidden verdicts, a battle record or the seed", () => {
    expect(same).toBe(true);
    const ok = input();
    // @ts-expect-error: the opponent's real team is not an input
    const a: BeliefInput = { ...ok, team: [] };
    // @ts-expect-error: nor is the omniscient log
    const b: BeliefInput = { ...ok, omniscient_log: [] };
    // @ts-expect-error: nor the hidden verdicts of the clone judge
    const c: BeliefInput = { ...ok, hidden_verdicts: [] };
    // @ts-expect-error: nor a battle record
    const d: BeliefInput = { ...ok, record: {} };
    // @ts-expect-error: nor the seed
    const e: BeliefInput = { ...ok, seed: "sodium,0" };
    // @ts-expect-error: the own half of the reader's state is not an input either
    const f: BeliefInput = { ...ok, own: ok.opponent };
    expect([a, b, c, d, e, f]).toHaveLength(6);
  });

  it("extra fields passed at run time are never read: the belief of the same view is the same", () => {
    const ok = input();
    const smuggled = { ...ok, team: [{ species: "alpha", item: "iy" }], seed: "sodium,0" } as unknown as BeliefInput;
    expect(JSON.stringify(belief(smuggled), (_, v: unknown) => (typeof v === "bigint" ? `${v}` : v))).toBe(JSON.stringify(belief(ok), (_, v: unknown) => (typeof v === "bigint" ? `${v}` : v)));
  });
});
