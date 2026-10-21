import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { interimPoolSource, readTrainingRunConfig } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { enumerateCandidates, type RequestRecord } from "@areneon/showdown-bridge";
import { caseSpec, loadCorpus, replayCase } from "../../showdown-bridge/src/choice-corpus.ts";
import { parseSweepArgs, playSweepBattle, SWEEP_MIXES, SweepUsageError, uniformAllActor } from "../src/reducer-sweep.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const tiny = readTrainingRunConfig(JSON.parse(readFileSync(`${ROOT}fixtures/contracts/v1/training-run-config/valid/tiny.json`, "utf8")));
const pool = interimPoolSource(engine, tiny);
const pendingCase = () => {
  const found = Object.values(loadCorpus()).find((c) => c.candidates.some((candidate) => candidate.pending));
  if (found === undefined) throw new Error("the choice corpus has no pending candidate");
  return { spec: caseSpec(found) };
};
const options = { run_key: "sweep-test", max_turns: 6 };

describe("playSweepBattle", () => {
  it.each(SWEEP_MIXES)("plays %s and records each decision at its request line", async (mix) => {
    const battle = await playSweepBattle(engine, FORMAT, pool, { ...options, mix, index: 0 });
    expect(battle.failure).toBeUndefined();
    expect(battle.seed).toMatch(/^sodium,[0-9a-f]{32}$/);
    for (const player of ["p1", "p2"] as const) {
      expect(battle.decisions[player].length).toBeGreaterThan(1);
      expect(battle.decisions[player][0]?.kind).toBe("preview");
      for (const decision of battle.decisions[player]) {
        expect(battle.channels[player][decision.n - 1]).toMatch(/^\|request\|/);
        expect(decision.candidates.length).toBeGreaterThan(0);
      }
    }
  });

  it("is a pure function of the run key, the mix and the index", async () => {
    const a = await playSweepBattle(engine, FORMAT, pool, { ...options, mix: "uniform-all", index: 3 });
    const b = await playSweepBattle(engine, FORMAT, pool, { ...options, mix: "uniform-all", index: 3 });
    const c = await playSweepBattle(engine, FORMAT, pool, { ...options, mix: "uniform-all", index: 4 });
    expect(b).toEqual(a);
    expect(c.seed).not.toBe(a.seed);
  });

  it("names the seats' actors", async () => {
    const battle = await playSweepBattle(engine, FORMAT, pool, { ...options, mix: "rule-vs-random", index: 1 });
    expect(battle.actors).toEqual({ p1: "actor-rule", p2: "random-legal" });
  });

  it("keeps a battle that failed as a finding, with its kind and a bounded message", async () => {
    const battle = await playSweepBattle(engine, "gen9nosuchformat", pool, { ...options, mix: "random-legal", index: 0 });
    expect(battle.failure?.kind).toBeTruthy();
    expect(battle.failure?.message.length).toBeLessThanOrEqual(400);
    expect(battle.channels).toEqual({ p1: [], p2: [] });
  });
});

describe("uniformAllActor", () => {
  it("draws from every candidate, pending ones included", () => {
    const { spec } = pendingCase();
    const { request } = replayCase(engine, spec);
    const record = { ...request, request_seq: 1 } as RequestRecord;
    const all = enumerateCandidates(record).candidates;
    expect(all.some((candidate) => candidate.pending)).toBe(true);
    const picked = new Set<string>();
    for (let seed = 1; seed <= 400; seed++) picked.add(uniformAllActor(seed)(record) as string);
    expect(picked.size).toBe(all.length);
    expect([...picked].some((choice) => all.find((c) => c.choice === choice)?.pending)).toBe(true);
  });
});

describe("parseSweepArgs", () => {
  const good = ["--config", "c.json", "--out", "o.jsonl", "--mix", "uniform-all", "--battles", "5"];
  it("reads the flags and defaults the rest", () => {
    expect(parseSweepArgs(good)).toEqual({ config: "c.json", out: "o.jsonl", mix: "uniform-all", battles: 5, start: 0, run_key: "sweep-1" });
    expect(parseSweepArgs([...good, "--start", "7", "--run-key", "k"])).toMatchObject({ start: 7, run_key: "k" });
  });
  it.each([
    [["--bogus", "1", ...good], /unknown argument/],
    [[...good, "--start"], /needs a value/],
    [["--config", "--out", "o", "--mix", "uniform-all", "--battles", "5"], /needs a value/],
    [[...good, "--battles", "6"], /twice/],
    [good.slice(0, -2), /--battles is required/],
    [[...good.slice(0, -1), "0"], /at least 1/],
    [[...good.slice(0, -1), "abc"], /whole number/],
    [[...good.slice(0, -1), "2.5"], /whole number/],
    [[...good.slice(0, -1), "-3"], /whole number/],
    [[...good, "--start", "-1"], /at least 0/],
    [good.map((x) => (x === "uniform-all" ? "nope" : x)), /--mix is one of/],
  ])("refuses %j", (argv, message) => {
    expect(() => parseSweepArgs(argv as string[])).toThrow(SweepUsageError);
    expect(() => parseSweepArgs(argv as string[])).toThrow(message as RegExp);
  });
});
