import { describe, expect, it } from "vitest";
import { NoCandidateError } from "../src/actor-rule.ts";
import { EnumeratorDefectError } from "../src/choice-judge.ts";
import {
  guardActor,
  type GuardOptions,
  playGame,
  PolicyWorkerError,
  UnsupportedCandidateError,
} from "../src/evaluation-harness.ts";
import { EpisodeError, episodeFromRecord } from "../src/index.ts";
import { firstLegal, GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "../src/golden.ts";
import { type Actor, BattleAbortedError, ForfeitSignal, loadEngine, type Player, type RequestRecord } from "../src/index.ts";

const engine = loadEngine();
const team = syntheticTeam();
const budget = { preview_ms: 2_000, battle_ms: 2_000 };
const policy: GuardOptions = { role: "policy", latency_budget: budget };
const scripted: GuardOptions = { role: "scripted", latency_budget: budget };
const other = (p: Player): Player => (p === "p1" ? "p2" : "p1");

/** Plays `firstLegal`, but runs `fail` instead on the `at`-th request that asks for a choice. */
const failingAt = (at: number, fail: Actor): Actor => {
  let asked = 0;
  return (request) => (++asked === at ? fail(request) : firstLegal(request));
};

const game = (actors: { p1: Actor; p2: Actor }, extra = {}) =>
  playGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, teams: { p1: team, p2: team }, actors, ...extra });

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

describe("policy forfeits", () => {
  const causes: [string, Actor, string][] = [
    ["crash", () => { throw new Error("segfault"); }, "crash"],
    ["crash (worker)", () => { throw new PolicyWorkerError("crash", "exit 1"); }, "crash"],
    ["out_of_memory", () => { throw new PolicyWorkerError("out_of_memory", "oom"); }, "out_of_memory"],
    [
      "out_of_memory (node worker code)",
      () => { throw Object.assign(new Error("heap"), { code: "ERR_WORKER_OUT_OF_MEMORY" }); },
      "out_of_memory",
    ],
    ["crash (non-error throw)", () => Promise.reject("boom"), "crash"],
    ["invalid_reply (not a candidate)", () => "move 9 9", "invalid_reply"],
    ["invalid_reply (not a string)", () => 7 as unknown as string, "invalid_reply"],
  ];

  for (const [name, fail, cause] of causes) {
    for (const loser of ["p1", "p2"] as const) {
      for (const at of [1, 2]) {
        it(`${name} at decision ${at} of ${loser} is a forfeit loss`, async () => {
          const bad = guardActor(failingAt(at, fail), policy);
          const good = guardActor(firstLegal, policy);
          const { outcome, record } = await game(loser === "p1" ? { p1: bad, p2: good } : { p1: good, p2: bad });
          expect(outcome).toMatchObject({ kind: "forfeit", loser, winner: other(loser), cause });
          if (outcome.kind !== "forfeit" || record === null) throw new Error("expected a forfeit");
          expect(record.winner).toBe(other(loser));
          expect(record.forfeit).toMatchObject({ player: loser, cause, request_seq: outcome.request_seq });
          expect(record.omniscient_log.some((l) => l.startsWith("|win|"))).toBe(true);
          // The failing submission never reached the engine nor the choice record.
          expect(record.choices.some((c) => c.player === loser && c.request_seq === outcome.request_seq)).toBe(false);
        });
      }
    }
  }

  it("a latency overrun forfeits, in the preview budget and in the battle budget", async () => {
    const slow: Actor = async () => { await sleep(300); return "default"; };
    const tight = { role: "policy", latency_budget: { preview_ms: 5_000, battle_ms: 50 } } as const;
    const good = guardActor(firstLegal, policy);
    const turn = await game({ p1: guardActor(failingAt(2, slow), tight), p2: good });
    expect(turn.outcome).toMatchObject({ kind: "forfeit", loser: "p1", cause: "latency_overrun" });
    const previewBudget = { role: "policy", latency_budget: { preview_ms: 50, battle_ms: 5_000 } } as const;
    const preview = await game({ p1: good, p2: guardActor(slow, previewBudget) });
    expect(preview.outcome).toMatchObject({ kind: "forfeit", loser: "p2", cause: "latency_overrun", request_seq: 1 });
  });

  it("an actor that never answers forfeits instead of hanging the run", async () => {
    const never: Actor = () => new Promise<string>(() => {});
    const { outcome } = await game({
      p1: guardActor(never, { role: "policy", latency_budget: { preview_ms: 30, battle_ms: 30 } }),
      p2: guardActor(firstLegal, policy),
    });
    expect(outcome).toMatchObject({ kind: "forfeit", loser: "p1", cause: "latency_overrun" });
  });

  it("when both sides fail, the first failure is the one recorded, and it is no incident", async () => {
    const quick = guardActor(() => "nonsense", policy);
    const later = guardActor(async () => { await sleep(60); return "nonsense"; }, policy);
    const { outcome, record } = await game({ p1: quick, p2: later });
    expect(outcome).toMatchObject({ kind: "forfeit", loser: "p1", winner: "p2" });
    expect(record?.forfeit?.player).toBe("p1");
    expect(record?.omniscient_log.filter((l) => l.startsWith("|win|"))).toHaveLength(1);
  });

  it("another error after a forfeit is recorded still aborts, as an incident (fail closed)", async () => {
    const quick = guardActor(() => "nonsense", policy);
    const broken: Actor = async () => { await sleep(60); throw new Error("harness bug"); };
    const { outcome, record } = await game({ p1: quick, p2: broken });
    expect(outcome).toMatchObject({ kind: "incident", incident: "unclassified" });
    expect(record).toBeNull();
  });

  it("records only the detail, without the signal's prefix", async () => {
    const bad = guardActor(() => { throw new Error("segfault"); }, policy);
    const { record } = await game({ p1: bad, p2: guardActor(firstLegal, policy) });
    expect(record?.forfeit?.detail).toBe("segfault");
  });

  it("a synchronous actor that busy-waits past the budget forfeits though it answered", async () => {
    const busy: Actor = (request) => {
      const end = performance.now() + 40;
      while (performance.now() < end);
      return firstLegal(request);
    };
    const tight = { role: "policy", latency_budget: { preview_ms: 20, battle_ms: 20 } } as const;
    const { outcome } = await game({ p1: guardActor(busy, tight), p2: guardActor(firstLegal, policy) });
    expect(outcome).toMatchObject({ kind: "forfeit", loser: "p1", cause: "latency_overrun" });
  });

  it("a forfeited game is refused as an Episode", async () => {
    const bad = guardActor(() => "nonsense", policy);
    const { record } = await game({ p1: bad, p2: guardActor(firstLegal, policy) });
    await expect(episodeFromRecord(record as never, "p2", {} as never)).rejects.toBeInstanceOf(EpisodeError);
  });

  it("a forfeit does not stop the next game of the run", async () => {
    const bad = guardActor(() => { throw new Error("x"); }, policy);
    const good = guardActor(firstLegal, policy);
    const rows = [await game({ p1: bad, p2: good }), await game({ p1: good, p2: good })];
    expect(rows.map((r) => r.outcome.kind)).toEqual(["forfeit", "played"]);
  });

  it("a clean game is played, with the winner and turns of the record", async () => {
    const good = guardActor(firstLegal, policy);
    const { outcome, record } = await game({ p1: good, p2: good });
    expect(outcome).toEqual({ kind: "played", winner: record?.winner, turns: record?.turns });
    expect(record?.forfeit).toBeUndefined();
  });
});

describe("harness incidents", () => {
  const good = guardActor(firstLegal, policy);
  const incidents: [string, Actor, string][] = [
    ["unsupported_candidate", () => { throw new UnsupportedCandidateError("shape"); }, "unsupported_candidate"],
    ["no_candidate", () => { throw new NoCandidateError("empty"); }, "no_candidate"],
    ["enumerator_defect", () => { throw new EnumeratorDefectError([]); }, "enumerator_defect"],
    ["engine_error", () => { throw new BattleAbortedError("engine"); }, "engine_error"],
  ];
  for (const [name, fail, incident] of incidents) {
    it(`${name} from a policy is an incident, never a forfeit`, async () => {
      const { outcome, record } = await game({ p1: guardActor(fail, policy), p2: good });
      expect(outcome).toMatchObject({ kind: "incident", incident });
      expect(record).toBeNull();
    });
  }

  it("any failure of a scripted baseline is an incident", async () => {
    for (const fail of [
      () => { throw new Error("baseline bug"); },
      () => "nonsense",
      () => sleep(300).then(() => "default"),
    ]) {
      const { outcome } = await game({
        p1: guardActor(fail, { role: "scripted", latency_budget: { preview_ms: 50, battle_ms: 50 } }),
        p2: good,
      });
      expect(outcome).toMatchObject({ kind: "incident", incident: "baseline_failure" });
    }
  });

  it("a forfeit signal from a scripted actor, or one wrapping a policy actor, is an incident", async () => {
    const wrapped = guardActor(() => "nonsense", policy);
    for (const fail of [() => { throw new ForfeitSignal("crash", "x"); }, wrapped]) {
      const { outcome } = await game({ p1: guardActor(fail, scripted), p2: good });
      expect(outcome).toMatchObject({ kind: "incident", incident: "baseline_failure" });
    }
  });

  it("an invalid team is an incident row", async () => {
    const bad = team.map((set) => ({ ...set, moves: ["Splash"] }));
    const { outcome } = await game({ p1: good, p2: good }, { teams: { p1: bad, p2: team } });
    expect(outcome).toMatchObject({ kind: "incident", incident: "invalid_team" });
  });

  it("an unexpected runner error is an unclassified incident", async () => {
    const { outcome } = await game({ p1: good, p2: good }, { players: { p1: { name: "a" }, p2: { name: "a" } } });
    expect(outcome).toMatchObject({ kind: "incident", incident: "unclassified" });
  });

  it("a non-error rejection is reported as text", async () => {
    const { outcome } = await game({ p1: good, p2: good }, { formatId: "gen9nonexistent" });
    expect(outcome.kind).toBe("incident");
  });
});

describe("guardActor", () => {
  const request = (legal?: RequestRecord["legal"]): RequestRecord =>
    ({ phase: "move", legal }) as unknown as RequestRecord;

  it("refuses a non-positive or non-finite budget", () => {
    for (const ms of [0, -1, Number.NaN, Infinity]) {
      expect(() => guardActor(firstLegal, { role: "policy", latency_budget: { preview_ms: ms, battle_ms: 1 } })).toThrow(RangeError);
    }
  });

  it("passes a reply from the pending candidates and refuses a request without a legal set", async () => {
    const legal = { choices: [], pending: ["move 1"] } as unknown as RequestRecord["legal"];
    expect(await guardActor(() => "move 1", policy)(request(legal))).toBe("move 1");
    await expect(guardActor(() => "move 1", policy)(request())).rejects.toMatchObject({ cause_class: "invalid_reply" });
    await expect(guardActor(() => "move 1", scripted)(request())).rejects.toMatchObject({ name: "BaselineFailureError" });
  });
});
