import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readTrainingExample, readTrainingSet, searchTargetProblem, SEARCH_K_MAX, TrainingLoaderError } from "../src/training-loader.ts";

const load = (name: string): unknown => JSON.parse(readFileSync(new URL(`../../../fixtures/contracts/v1/${name}`, import.meta.url), "utf8"));

describe("search label contracts (ADR-026 erratum E2, #1288)", () => {
  it("reads the valid label example and manifest", () => {
    const example = readTrainingExample(load("training-example/valid/battle-search-target.json"));
    expect(example.search_target?.entries.length).toBeGreaterThanOrEqual(2);
    expect(readTrainingSet(load("training-set/valid/search-labels.json")).search_labels?.labelled).toBe(10);
  });

  it("a plain example has no problem", () => {
    expect(searchTargetProblem(readTrainingExample(load("training-example/valid/battle.json")))).toBeNull();
  });

  it("holds the reader's k_max beside the schema's", () => {
    const example = readTrainingExample(load("training-example/valid/battle-search-target.json"));
    const target = example.search_target as NonNullable<typeof example.search_target>;
    const wide = { ...example, search_target: { ...target, entries: Array.from({ length: SEARCH_K_MAX + 1 }, (_, i) => ({ candidate: i, q_ppm: 1 })) } };
    expect(searchTargetProblem(wide as never)).toContain("more than 16");
  });

  it("refuses a label whose manifest block is wrong, with the loader's error", () => {
    expect(() => readTrainingSet(load("training-set/reader-invalid/search-labels-count-mismatch.json"))).toThrow(TrainingLoaderError);
  });
});
