import { describe, expect, it } from "vitest";
import { checkOutcomeConfig, loadOutcomeConfig, OutcomeConfigError, outcomeConfigHash, readOutcomeConfig } from "../src/outcome-config.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";

const good = { schema_version: 1, format_id: GOLDEN_FORMAT, assumed_branch: "assumed-set-filler", max_branches: 5, unknown_count_end: 20, unknown_item: "sitrusberry", unknown_move: "tackle", volatiles: "declared" };

describe("the probe-outcome record", () => {
  it("reads the committed file, and its hash is the SHA-256 of its JCS bytes", () => {
    const config = loadOutcomeConfig(GOLDEN_FORMAT);
    expect(config).toEqual(good);
    expect(outcomeConfigHash(config)).toMatch(/^[0-9a-f]{64}$/);
    expect(outcomeConfigHash({ ...config, max_branches: 4 })).not.toBe(outcomeConfigHash(config));
  });
  it("refuses an unknown or missing key, a bad rule, an out-of-range integer and a malformed id", () => {
    for (const bad of [null, [], { ...good, extra: 1 }, { ...good, schema_version: 2 }, { ...good, format_id: "" }, { ...good, assumed_branch: "random" }, { ...good, volatiles: "some" }, { ...good, max_branches: 1 }, { ...good, max_branches: 6 }, { ...good, max_branches: 2.5 }, { ...good, unknown_count_end: 0 }, { ...good, unknown_item: "Sitrus Berry" }, { ...good, unknown_move: 3 }]) {
      expect(() => readOutcomeConfig(bad), JSON.stringify(bad)).toThrow(OutcomeConfigError);
    }
    const { unknown_move: _gone, ...missing } = good;
    expect(() => readOutcomeConfig(missing)).toThrow(/exactly its keys/);
  });
  it("is checked against the engine's data and refuses a record of another format", () => {
    const dex = { items: { get: (id: string) => ({ exists: id === "sitrusberry" }) }, moves: { get: (id: string) => ({ exists: id === "tackle" }) } };
    expect(checkOutcomeConfig(readOutcomeConfig(good), dex)).toEqual([]);
    expect(checkOutcomeConfig(readOutcomeConfig({ ...good, unknown_item: "x", unknown_move: "y" }), dex)).toHaveLength(2);
    expect(() => loadOutcomeConfig(GOLDEN_FORMAT, "/nonexistent")).toThrow();
  });
});
