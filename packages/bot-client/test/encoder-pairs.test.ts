import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { expect, it } from "vitest";
import { encoderPairs } from "./encoder-pairs.ts";

it("matches a fresh capture: re-run scripts/capture-encoder-pairs.ts and review the diff", async () => {
  const engine = loadEngine();
  const fresh = await encoderPairs(engine, loadRulesSnapshot(engine.commit));
  expect(Object.keys(fresh)).toEqual(["gen9championsvgc2026regmc"]);
  for (const [format, document] of Object.entries(fresh)) {
    const path = fileURLToPath(new URL(`../../../fixtures/showdown/${format}/a2-pairs.json`, import.meta.url));
    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual(document);
  }
}, 120_000);
