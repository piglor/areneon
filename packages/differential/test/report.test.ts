import { describe, expect, it } from "vitest";
import { differentialReport, HARNESS_FORMAT, LISTED_DIFFERENCES, renderReport } from "../src/report.ts";
import { FRESH, REGRESSION, tallyOf } from "./support.ts";

describe("the report", () => {
  const fresh = tallyOf(FRESH, [{ count: 20, entities: ["move:a", "ability:b"] }, { count: 3, entities: ["move:a"], reads: [] }]);
  const regression = tallyOf(REGRESSION, [{ count: LISTED_DIFFERENCES + 5, entities: ["move:a", "item:c"], differing: Array.from({ length: LISTED_DIFFERENCES + 5 }, (_, i) => i) }]);
  it("sums entities over corpora, counts differences by reason and field, and lists the first ones", () => {
    const report = differentialReport({ pin: "x" }, [fresh, regression]);
    expect(report.harness_format).toBe(HARNESS_FORMAT);
    expect(report.corpora.map((c) => c.id)).toEqual([REGRESSION.id, FRESH.id]);
    expect(report.corpora[0]).toMatchObject({ differences: 55, by_reason: { value: 55 }, by_field: { "$.x": 55 }, pairs: null });
    expect(report.corpora[1]?.pairs).toEqual({ present: 1, frequent: 1, co_exercised_enough: 1 });
    expect(report.entities.find((e) => e.id === "move:a")).toEqual({ id: "move:a", present: 78, exercised: 75, equal: 20, differing: 55 });
    expect(report.kinds).toEqual({ ability: 1, item: 1, move: 1 });
    expect(report.differences).toHaveLength(LISTED_DIFFERENCES);
    const md = renderReport(report, "Self differential");
    expect(md).toContain("# Self differential");
    expect(md).toContain("| committed | regression | 55 | 55 | 0 | value 55 | $.x 55 | not kept |");
    expect(md).toContain("| move:a | 78 | 75 | 20 | 55 |");
  });
  it("sorts reasons and fields", () => {
    const tally = tallyOf(REGRESSION, [{ count: 1, entities: ["move:a"] }]);
    tally.differences.push({ position: 0, reason: "settled", field: "$", entities: ["move:a"] }, { position: 0, reason: "answer", field: "$.b", entities: ["move:a"] });
    const report = differentialReport({}, [tally, tallyOf({ ...REGRESSION, id: "aaa" }, [{ count: 1, entities: ["move:b"] }])]);
    expect(Object.keys(report.corpora[1]?.by_reason ?? {})).toEqual(["answer", "settled"]);
    expect(Object.keys(report.corpora[1]?.by_field ?? {})).toEqual(["$", "$.b"]);
    expect(report.entities.map((e) => e.id)).toEqual(["move:a", "move:b"]);
  });
  it("says none when nothing differs", () => {
    const md = renderReport(differentialReport({}, [fresh]), "t");
    expect(md).toContain("## Differences\n\nNone.");
    expect(md).toContain("| none | none |");
  });
});
