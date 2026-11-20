import { describe, expect, it } from "vitest";
import { CanonicalError, jcs } from "../src/canonical.ts";
import { compareBatch, compareItem, firstDifference, HarnessError } from "../src/compare.ts";
import { runCorpus, RunError } from "../src/harness.ts";
import { addBatch, newTally, pairKey, TallyError } from "../src/tally.ts";
import { backendOf, FRESH, oracle, REGRESSION, type ToyIn, toyLevel, truth } from "./support.ts";

const input = (n: number): ToyIn => ({ n, entities: [`move:m${Math.abs(n)}`, "kind:damage"], reads: ["kind:damage"] });

describe("jcs", () => {
  it("writes sorted keys, integers, strings, booleans and null", () => {
    expect(jcs({ b: [1, true, null], a: "xé", c: { z: -2, y: 0 } })).toBe('{"a":"xé","b":[1,true,null],"c":{"y":0,"z":-2}}');
  });
  it("refuses a float, a non-finite number, a lone surrogate and a non-JSON value", () => {
    expect(() => jcs(1.5)).toThrow(CanonicalError);
    expect(() => jcs(Number.NaN)).toThrow("integers only");
    expect(() => jcs(2 ** 60)).toThrow("integers only");
    expect(() => jcs("\uD800")).toThrow("lone surrogate");
    expect(() => jcs({ ["\uDC00"]: 1 })).toThrow("lone surrogate");
    expect(() => jcs(undefined as never)).toThrow("not a JSON value");
  });
});

describe("firstDifference", () => {
  it("names the first differing field in sorted key order, or null", () => {
    expect(firstDifference({ a: [1, 2, 3], b: 1 }, { a: [1, 2, 3], b: 1 })).toBeNull();
    expect(firstDifference({ a: [1, 2, 3], b: 1 }, { a: [1, 9, 3], b: 2 })).toBe("$.a[1]");
    expect(firstDifference({ a: [1, 2] }, { a: [1, 2, 3] })).toBe("$.a[2]");
    expect(firstDifference({ a: 1 }, { a: 1, b: 2 })).toBe("$.b");
    expect(firstDifference({ a: 1, c: 2 }, { a: 1 })).toBe("$.c");
    expect(firstDifference({ a: null }, { a: 0 })).toBe("$.a");
    expect(firstDifference([1], { 0: 1 })).toBe("$");
    expect(firstDifference("x", "y")).toBe("$");
    expect(firstDifference(null, null)).toBeNull();
  });
});

describe("compareItem", () => {
  const ok = truth(input(3));
  it("is equal on equal bytes and on the same refusal", () => {
    expect(compareItem(toyLevel, input(3), 0, ok, truth(input(3))).difference).toBeNull();
    expect(compareItem(toyLevel, input(-3), 0, truth(input(-3)), truth(input(-3)))).toMatchObject({ difference: null, exercised: [] });
  });
  it("names each reason and its field", () => {
    const at = (backend: Parameters<typeof compareItem>[4]) => compareItem(toyLevel, input(3), 7, ok, backend).difference;
    expect(at(undefined)).toMatchObject({ reason: "missing", field: "$", position: 7, entities: ["kind:damage", "move:m3"] });
    expect(at({ ok: false, error: new RangeError("refused") })).toMatchObject({ reason: "settled", field: "$" });
    expect(at({ ok: true, result: { for: 4, value: null } })).toMatchObject({ reason: "answer", field: "$" });
    expect(at({ ok: true, result: { for: 3, value: { rolls: [3, 5], kind: "computed" } } })).toMatchObject({ reason: "value", field: "$.value.rolls[1]" });
    expect(compareItem(toyLevel, input(-3), 0, truth(input(-3)), { ok: false, error: new RangeError("other") }).difference).toMatchObject({ reason: "refusal", field: "$error" });
  });
  it("refuses an oracle outcome that does not answer its input", () => {
    expect(() => compareItem(toyLevel, input(3), 0, { ok: true, result: { for: 9, value: 0 } }, ok)).toThrow(HarnessError);
  });
});

describe("compareBatch and runCorpus", () => {
  const inputs = [1, 2, -3, 4].map(input);
  it("finds no difference between the oracle and itself", () => {
    const tally = runCorpus(toyLevel, oracle, oracle, REGRESSION, inputs, 3);
    expect(tally.items).toBe(4);
    expect(tally.differences).toEqual([]);
    expect(tally.pairs.size).toBe(0);
  });
  it("shows a dropped result from the dropped item on, and counts the mismatch", () => {
    const drop = backendOf("drop", (batch) => batch.map(truth).filter((_, i) => i !== 1));
    const result = compareBatch(toyLevel, oracle, drop, [1, 2, 3, 4].map(input));
    expect(result.count_mismatch).toBe(true);
    expect(result.items.map((i) => i.difference?.reason ?? "equal")).toEqual(["count", "answer", "answer", "missing"]);
    const tally = newTally(REGRESSION);
    addBatch(tally, result);
    expect(tally.count_mismatches).toBe(1);
    expect(tally.entities.get("move:m2")).toMatchObject({ differing: 1, first_difference: 1 });
  });
  it("refuses an oracle that returns another number of outcomes, and a batch size below 1", () => {
    expect(() => compareBatch(toyLevel, backendOf("short", () => []), oracle, inputs)).toThrow("the oracle returned another number");
    expect(() => runCorpus(toyLevel, oracle, oracle, REGRESSION, inputs, 0)).toThrow(RunError);
  });
  it("keeps pairs for a fresh corpus only", () => {
    const tally = runCorpus(toyLevel, oracle, oracle, FRESH, [input(1), { ...input(1), n: 2 }]);
    expect(tally.pairs.get(pairKey("move:m1", "kind:damage"))).toEqual({ co_present: 2, co_exercised: 0, co_equal: 0 });
    expect(pairKey("b", "a")).toBe("a|b");
  });
});

describe("tallies", () => {
  it("refuse a bad spec, an item out of order and an exercised entity that is not present", () => {
    const hex = "a".repeat(64);
    expect(() => newTally({ id: "Bad", kind: "regression", seed: hex, pool: hex })).toThrow(TallyError);
    expect(() => newTally({ id: "x".repeat(65), kind: "regression", seed: hex, pool: hex })).toThrow(TallyError);
    expect(() => newTally({ id: "ok", kind: "regression", seed: "A".repeat(64), pool: hex })).toThrow("64 hex");
    expect(() => newTally({ id: "ok", kind: "regression", seed: hex, pool: "short" })).toThrow("64 hex");
    const tally = newTally(REGRESSION);
    expect(() => addBatch(tally, { items: [{ position: 1, key: "k", entities: [], exercised: [], difference: null }], count_mismatch: false })).toThrow("corpus order");
    expect(() => addBatch(tally, { items: [{ position: 0, key: "k", entities: ["a:b"], exercised: ["a:c"], difference: null }], count_mismatch: false })).toThrow("must be present");
  });
});
