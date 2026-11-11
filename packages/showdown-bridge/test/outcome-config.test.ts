import { describe, expect, it } from "vitest";
import { checkOutcomeConfig, loadOutcomeConfig, movePathConditions, OutcomeConfigError, outcomeConfigHash, readOutcomeConfig } from "../src/outcome-config.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";

const good = { schema_version: 1, format_id: GOLDEN_FORMAT, assumed_branch: "assumed-set-filler", max_branches: 5, unknown_count_end: 20, unknown_item: "sitrusberry", unknown_move: "tackle", move_path_conditions: ["stall", "stockpile"], unknown_volatile_limit: 2, volatiles: "declared" };

describe("the probe-outcome record", () => {
  it("reads the committed file, and its hash is the SHA-256 of its JCS bytes", () => {
    const config = loadOutcomeConfig(GOLDEN_FORMAT);
    expect(config).toEqual({ ...good, move_path_conditions: config.move_path_conditions });
    expect(config.move_path_conditions.length).toBeGreaterThan(config.unknown_volatile_limit); // so an unsettled Pokemon makes its queries history_unknown
    expect(outcomeConfigHash(config)).toMatch(/^[0-9a-f]{64}$/);
    expect(outcomeConfigHash({ ...config, max_branches: 4 })).not.toBe(outcomeConfigHash(config));
  });
  it("refuses an unknown or missing key, a bad rule, an out-of-range integer and a malformed id", () => {
    for (const bad of [null, [], { ...good, extra: 1 }, { ...good, schema_version: 2 }, { ...good, format_id: "" }, { ...good, assumed_branch: "random" }, { ...good, volatiles: "some" }, { ...good, max_branches: 1 }, { ...good, max_branches: 6 }, { ...good, max_branches: 2.5 }, { ...good, unknown_count_end: 0 }, { ...good, unknown_item: "Sitrus Berry" }, { ...good, unknown_move: 3 }, { ...good, move_path_conditions: [] }, { ...good, move_path_conditions: ["stockpile", "stall"] }, { ...good, move_path_conditions: "stall" }, { ...good, unknown_volatile_limit: -1 }]) {
      expect(() => readOutcomeConfig(bad), JSON.stringify(bad)).toThrow(OutcomeConfigError);
    }
    const { unknown_move: _gone, ...missing } = good;
    expect(() => readOutcomeConfig(missing)).toThrow(/exactly its keys/);
  });
  it("is checked against the engine's data: the move-path conditions are derived from its handlers, never named", () => {
    // a mod with three conditions: two volatiles the move path reads (one defined by a move's `condition`), one it does not, and a status that has the handler
    const handler = () => undefined;
    const bodies: Record<string, Record<string, unknown>> = {
      stall: { exists: true, effectType: "Condition", onBeforeMove: handler },
      stockpile: { exists: true, effectType: "Condition", onModifyMove: handler },
      other: { exists: true, effectType: "Condition", onStart: handler },
      par: { exists: true, effectType: "Status", onBeforeMove: handler },
      encore: { exists: true, effectType: "Move", condition: { onOverrideAction: handler } },
    };
    const dex = {
      data: { Conditions: { stall: {}, stockpile: {}, other: {}, par: {} }, Moves: { encore: { condition: {} } }, Abilities: {}, Items: {} },
      conditions: { get: (id: string) => bodies[id] as never },
      items: { get: (id: string) => ({ exists: id === "sitrusberry" }) },
      moves: { get: (id: string) => ({ exists: id === "tackle" }) },
    } as never;
    expect(movePathConditions(dex)).toEqual(["encore", "stall", "stockpile"]);
    const record = { ...good, move_path_conditions: ["encore", "stall", "stockpile"] };
    expect(checkOutcomeConfig(readOutcomeConfig(record), dex)).toEqual([]);
    // a condition the engine reads and the record lacks, one the record names that the engine does not read, ids the engine lacks, and no engine data at all
    expect(checkOutcomeConfig(readOutcomeConfig({ ...record, move_path_conditions: ["stall", "stockpile", "zzz"], unknown_item: "x", unknown_move: "y" }), dex)).toEqual([
      "unknown_item is not an engine item",
      "unknown_move is not an engine move",
      "the engine's encore reads the move path and is not in move_path_conditions",
      "move_path_conditions names zzz, which the engine's move path does not read",
    ]);
    expect(checkOutcomeConfig(readOutcomeConfig(good), { items: (dex as never as { items: never }).items, moves: (dex as never as { moves: never }).moves })).toEqual(["no engine data to check move_path_conditions against"]);
    expect(() => loadOutcomeConfig(GOLDEN_FORMAT, "/nonexistent")).toThrow();
  });
});
