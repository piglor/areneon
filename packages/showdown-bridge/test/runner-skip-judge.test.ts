import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CloneJudge } from "../src/choice-judge.ts";
import { enumerateCandidates } from "../src/choices.ts";
import { guardActor } from "../src/evaluation-harness.ts";
import { firstLegal, GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "../src/golden.ts";
import { type Actor, declareReadsLegal, EnumeratorDefectError, loadEngine, randomLegalActor, runBattle } from "../src/index.ts";

const engine = loadEngine();
const team = syntheticTeam();
const play = (actors: { p1: Actor; p2: Actor }, extra = {}) =>
  runBattle({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: team, p2: team }, actors, ...extra });

// A seat that reads only the candidates enumerated from its own request, never `request.legal` (what `policyActor` does).
const nonReader = (): Actor =>
  declareReadsLegal((request) => {
    const candidates = enumerateCandidates(request).candidates;
    return (candidates.find((c) => !c.pending) ?? (candidates[0] as { choice: string })).choice;
  }, false);

afterEach(() => vi.restoreAllMocks());

describe("skip_unread_judge (ADR-028 F4)", () => {
  it("plays the same battle with and without the judge for non-reading seats", async () => {
    const full = await play({ p1: nonReader(), p2: nonReader() });
    const skipped = await play({ p1: nonReader(), p2: nonReader() }, { skip_unread_judge: true });
    expect(skipped.omniscient_log).toEqual(full.omniscient_log);
    expect(skipped.channels).toEqual(full.channels);
    expect(skipped.choices).toEqual(full.choices);
    expect(skipped.requests).toEqual(full.requests);
    expect(skipped.winner).toBe(full.winner);
    expect(skipped.turns).toBe(full.turns);
    expect(skipped.hidden_verdicts).toEqual([]);
  });

  it("does not construct a CloneJudge for a declared non-reader, and only then", async () => {
    const judge = vi.spyOn(CloneJudge.prototype, "judge");
    await play({ p1: nonReader(), p2: nonReader() }, { skip_unread_judge: true });
    expect(judge).not.toHaveBeenCalled();
    judge.mockClear();
    // Without the option, evaluation and every other caller keep full judging, declaration or not.
    await play({ p1: nonReader(), p2: nonReader() });
    expect(judge).toHaveBeenCalled();
    judge.mockClear();
    // Undeclared and declared readers are judged even with the option.
    await play({ p1: (r) => firstLegal(r), p2: randomLegalActor(7) }, { skip_unread_judge: true });
    expect(judge).toHaveBeenCalled();
  });

  it("judges a reading seat while skipping the other", async () => {
    const full = await play({ p1: firstLegal, p2: nonReader() });
    const mixed = await play({ p1: firstLegal, p2: nonReader() }, { skip_unread_judge: true });
    expect(mixed.omniscient_log).toEqual(full.omniscient_log);
    expect(mixed.hidden_verdicts).toEqual(full.hidden_verdicts.filter((v) => v.player === "p1"));
  });

  it("gives the non-reader no verdict: its request carries the same legal set and no hidden field", async () => {
    const seen: unknown[] = [];
    const spy = declareReadsLegal((request, context) => {
      seen.push(request.legal, context);
      return nonReader()(request, context);
    }, false);
    await play({ p1: spy, p2: nonReader() }, { skip_unread_judge: true });
    for (const legal of seen.filter((_, i) => i % 2 === 0)) {
      if (legal !== undefined) expect(Object.keys(legal as object).sort()).toEqual(["choices", "pending", "request_hash", "request_seq", "schema_version", "showdown_commit"].filter((k) => k in (legal as object)));
    }
    expect(JSON.stringify(seen)).not.toContain("verdict");
  });

  it("still reports a refused non-pending submission as an enumerator defect", async () => {
    const bad = declareReadsLegal((request) => (request.phase === "preview" ? "team 1234" : "move 9 9"), false);
    // The judged path aborts as an engine error; the unjudged path names the defect.
    await expect(play({ p1: bad, p2: nonReader() }, { skip_unread_judge: true })).rejects.toBeInstanceOf(EnumeratorDefectError);
    await expect(play({ p1: bad, p2: nonReader() })).rejects.not.toBeInstanceOf(EnumeratorDefectError);
  });

  it("carries the declaration through the guard and declares the scripted actors", () => {
    const budget = { preview_ms: 1000, battle_ms: 1000 };
    expect(guardActor(nonReader(), { role: "policy", latency_budget: budget }).reads_legal).toBe(false);
    expect(guardActor(firstLegal, { role: "scripted", latency_budget: budget }).reads_legal).toBe(true);
    expect(randomLegalActor(1).reads_legal).toBe(true);
    expect(guardActor((r) => firstLegal(r), { role: "scripted", latency_budget: budget }).reads_legal).toBeUndefined();
  });
});

describe("sources", () => {
  const src = join(import.meta.dirname, "../src");
  it("keeps the non-reader (policyActor) from touching request.legal", () => {
    const text = readFileSync(join(src, "policy-actor.ts"), "utf8");
    expect(text).not.toMatch(/\brequest\.legal\b/);
    expect(text).toContain("declareReadsLegal(actor, preview_actor !== undefined)");
  });

  it("sets skip_unread_judge only in the runner, the policy game and the training generation", () => {
    const users = readdirSync(src)
      .filter((f) => f.endsWith(".ts") && readFileSync(join(src, f), "utf8").includes("skip_unread_judge"))
      .sort();
    expect(users).toEqual(["league-generation.ts", "policy-actor.ts", "runner.ts"]);
  });
});
