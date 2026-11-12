import { describe, expect, it } from "vitest";
import { type CandidateFeaturesOutcome, candidateFeaturesOutcome } from "../src/candidate-features-outcome.ts";
import { createFeatureMemo, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { withMoveOutcome } from "../src/mechanics-config.ts";
import { observationOf } from "../src/mechanics-outcome.ts";
import { loadOutcomeConfig } from "../src/outcome-config.ts";
import { provenPasses, UNTARGETED } from "../src/outcome-skip.ts";
import { playerChannel } from "../src/player-channel.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import { type Actor, runBattle } from "../src/runner.ts";
import { config, corpus, engine, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";

// #1353 (ADR-022 E-22.5.10): the cost reductions of `probe-outcome/1`. Outcome rows the engine's data proves are made without a query
// (`skip`) and the move branches of a part whose `switch` branch never read the queue share its answer (`fold`). Both are exact: the
// rows with both off (every query asked) are the rows with both on, on every request of seeded battles, and the queries are fewer.

const configs = (change: Partial<NonNullable<FeatureConfigs["outcome"]>> = {}): FeatureConfigs => ({ vocabulary, config, mechanics: withMoveOutcome(mechanics), outcome: { config: loadOutcomeConfig(GOLDEN_FORMAT), ...change }, snapshot, data_class: "synthetic" });

interface Point {
  channel: string[];
  team: Parameters<typeof candidateFeaturesOutcome>[3];
  candidates: { choice: string; pending: boolean }[];
}

async function points(index: number): Promise<Point[]> {
  const battle = corpus[index] as (typeof corpus)[number];
  const out: Point[] = [];
  const base = randomLegalActor(1000 + index);
  const p1: Actor = (request, context) => {
    if (request.phase === "move" && request.legal !== undefined && context?.team !== undefined) {
      const legal = request.legal;
      out.push({ channel: [...context.channel], team: context.team, candidates: [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) })) });
    }
    return base(request, context);
  };
  await runBattle({ engine, formatId: GOLDEN_FORMAT, seed: battle.seed, teams: battle.teams, actors: { p1, p2: randomLegalActor(2000 + index) }, maxTurns: 12 });
  return out;
}

const build = (point: Point, change: Partial<NonNullable<FeatureConfigs["outcome"]>>, memo = createFeatureMemo()): { features: CandidateFeaturesOutcome; memo: ReturnType<typeof createFeatureMemo> } => ({
  features: candidateFeaturesOutcome(engine, configs(change), playerChannel({ schema_version: 1, data_class: "synthetic", player: "p1", lines: point.channel }), point.team, point.candidates, memo),
  memo,
});

describe("skipping and folding are exact", () => {
  it("gives the same arrays as asking every query, with fewer queries, on every request of seeded battles", async () => {
    let compared = 0;
    let changed = 0;
    let skipped = 0;
    let folded = 0;
    let queriesOn = 0;
    let queriesOff = 0;
    for (const index of process.env.ARENEON_COST_BATTLES ? Array.from({ length: Math.min(Number(process.env.ARENEON_COST_BATTLES), corpus.length) }, (_, i) => i) : [0, 3, 5, 8]) {
      const list = await points(index);
      for (const point of list.filter((_, i) => process.env.ARENEON_COST_BATTLES !== undefined || i % 2 === 0)) {
        const fast = build(point, {});
        const slow = build(point, { skip: false, fold: false });
        // Every array but `outcomes` is equal bytes. A move-branch row may differ in one way only: the branch could not be asked (a foe move the probe
        // cannot queue, such as Counter; an exact speed tie) and the fast row is the part's `switch` answer, which the engine's run proved the foe's move cannot change.
        const { outcomes: fastOutcomes, ...fastRest } = fast.features;
        const { outcomes: slowOutcomes, ...slowRest } = slow.features;
        expect(jcs(fastRest as unknown as Json)).toBe(jcs(slowRest as unknown as Json));
        expect(fastOutcomes.length).toBe(slowOutcomes.length);
        fastOutcomes.forEach((row, at) => {
          const other = slowOutcomes[at] as (typeof slowOutcomes)[number];
          if (jcs(row as unknown as Json) === jcs(other as unknown as Json)) return;
          const mate = fastOutcomes.find((each) => each.slot === row.slot && each.move === row.move && each.target === row.target && each.mega === row.mega && each.action === "switch") as (typeof fastOutcomes)[number];
          if (row.action !== "move") console.log("SWDIFF", jcs(row as unknown as Json), "|", jcs(other as unknown as Json));
          expect(row.action).toBe("move");
          expect(other.kind).toBe("unsupported");
          expect(["query_kind", "random_effect"]).toContain((other as { reason: string }).reason);
          expect(jcs({ ...row, action: "", foe_move: null, assumed: false } as unknown as Json)).toBe(jcs({ ...mate, action: "", foe_move: null, assumed: false } as unknown as Json));
          changed += 1;
        });
        compared += 1;
        skipped += fast.memo.outcome?.skipped ?? 0;
        folded += fast.memo.outcome?.folded ?? 0;
        queriesOn += fast.memo.counts.queries;
        queriesOff += slow.memo.counts.queries;
      }
    }
    expect(compared).toBeGreaterThan(10);
    expect(folded).toBeGreaterThan(0);
    expect(skipped).toBeGreaterThan(0);
    expect(changed).toBeLessThan(folded);
    expect(queriesOn).toBeLessThan(queriesOff);
  }, 600_000);
});

describe("provenPasses (engine-free)", () => {
  const move = { accuracy: true, target: "self", priority: 0, flags: {} };
  const clean = { status: null, history: undefined };
  const base = { move, ability: undefined, item: undefined, combatants: [clean, clean], battle: undefined, marked: false };
  it("proves an untargeted move with no accuracy roll and no handler", () => {
    expect(provenPasses(base)).toBe(true);
    for (const target of UNTARGETED) expect(provenPasses({ ...base, move: { ...move, target } })).toBe(true);
  });
  it("proves nothing for an accuracy roll, a handler, a target, a priority, a charge, a stall or a one-hit KO move", () => {
    for (const changed of [{ accuracy: 90 }, { onTry: () => true }, { beforeMoveCallback: () => true }, { target: "normal" }, { priority: 1 }, { flags: { charge: 1 } }, { flags: { recharge: 1 } }, { stallingMove: true }, { ohko: true }])
      expect(provenPasses({ ...base, move: { ...move, ...changed } })).toBe(false);
  });
  it("proves nothing while any Pokemon has a status, a declared volatile, an unknown fact or a pending slot condition, or the history is marked", () => {
    expect(provenPasses({ ...base, marked: true })).toBe(false);
    expect(provenPasses({ ...base, combatants: [clean, { status: "slp", history: undefined }] })).toBe(false);
    expect(provenPasses({ ...base, combatants: [{ status: null, history: { volatiles: { kind: "known", value: [{ id: "taunt" }] } } }, clean] })).toBe(false);
    expect(provenPasses({ ...base, combatants: [clean, { status: null, history: { move_actions: { kind: "unknown", ends: [0, 1] } } }] })).toBe(false);
    expect(provenPasses({ ...base, battle: { sides: [{ pending_slots: { kind: "known", value: [{ id: "wish" }] } }, {}] } })).toBe(false);
    expect(provenPasses({ ...base, battle: { turn: 3, sides: [{ pending_slots: { kind: "known", value: [] }, total_fainted: { kind: "known", value: 1 } }, {}] }, combatants: [{ status: null, history: { volatiles: { kind: "known", value: [] }, move_actions: { kind: "known", value: 2 }, ignored: 1 } }, clean] })).toBe(true);
  });
  it("proves nothing when the user's ability or item answers a move-path event, and ignores other handlers", () => {
    expect(provenPasses({ ...base, ability: { onBeforeMove: () => true } })).toBe(false);
    expect(provenPasses({ ...base, item: { onFoeTryMove: () => true } })).toBe(false);
    expect(provenPasses({ ...base, everyone: [{ onFractionalPriority: () => 1 }] })).toBe(false);
    expect(provenPasses({ ...base, everyone: [{ onStart: () => 1 }] })).toBe(true);
    expect(provenPasses({ ...base, ability: { onModifyAccuracy: () => 1, onStart: () => 1 } })).toBe(true);
  });
});

describe("what the probe observes of the queue", () => {
  it("is carried by the answer and absent for an answer built another way", () => {
    expect(observationOf({})).toBeUndefined();
  });
});
