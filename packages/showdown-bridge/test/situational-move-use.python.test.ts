// The done-when of #1283: the raw counts of a small verified progress root feed the Python summariser, byte-stably.
// Runs where the repository's Python environment exists (the typescript CI job installs it and sets ARENEON_REQUIRE_PYTHON_WORKER).
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { readVerifiedGeneration } from "../src/situational-input.ts";
import { situationalSetup, type StoredGame, tallyGame } from "../src/situational-move-use.ts";
import { goldenWorldFactsData } from "../src/world-facts-data.ts";
import { engine } from "./damage-aware-support.ts";
import { play, storedGame, writeRoot } from "./situational-support.ts";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../../..");
const PYTHON = join(ROOT, ".venv/bin/python");

describe.skipIf(!existsSync(PYTHON) && process.env.ARENEON_REQUIRE_PYTHON_WORKER !== "1")("the raw counts through the Python summariser", () => {
  it("summarises a verified root's counts twice to the same bytes", async () => {
    const { record, teams } = await play(0);
    const root = writeRoot([storedGame(record, teams, 0), storedGame(record, teams, 1)]);
    const { record: evaluation, games } = readVerifiedGeneration(join(root, "g000001"), 1);
    const setup = situationalSetup(engine, GOLDEN_FORMAT, evaluation.plan.max_turns, goldenWorldFactsData());
    const tallies = [];
    for (const game of games) tallies.push(await tallyGame(setup, game as StoredGame));
    const raw = join(root, "raw.json");
    writeFileSync(raw, JSON.stringify({ format: "situational-move-use/1", label: "x", pin: engine.commit, generation: 1, run_id: evaluation.run_id, evaluation_key: null, unreplayable_games: 0, games: tallies }));
    const env = { ...process.env, PYTHONPATH: [ROOT, join(ROOT, "packages/schema/python"), join(ROOT, "ml")].join(":") };
    const summarise = (): string => execFileSync(PYTHON, ["-m", "areneon_ml.report.situational", raw, "--out", join(root, "summary.json")], { env, encoding: "utf8" }) + readFileSync(join(root, "summary.json"), "utf8");
    const first = summarise();
    expect(summarise()).toBe(first);
    const summary = JSON.parse(first.slice(first.indexOf("{"))) as { matches: Record<string, Record<string, { pairs: number; counts: Record<string, number> }>> };
    const match = summary.matches["actor-rule-argmax"] as Record<string, { pairs: number; counts: Record<string, number> }>;
    expect(Object.values(match).every((policy) => policy.pairs === 1)).toBe(true);
    expect(Object.values(match).reduce((n, policy) => n + (policy.counts.decisions ?? 0), 0)).toBeGreaterThan(0);
  }, 180_000);
});
