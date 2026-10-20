import { describe, expect, it } from "vitest";
import { committedDatasetHashes, loadFixtureInputs, trainingPool } from "@areneon/team-dataset";
import { engine, FORMAT } from "./support.ts";

describe("the package entry point", () => {
  it("gives a caller the committed datasets' synthetic hashes", () => {
    const known = committedDatasetHashes(loadFixtureInputs(engine, FORMAT), FORMAT);
    expect(known.size).toBe(240);
    expect(typeof trainingPool).toBe("function");
  });
});
