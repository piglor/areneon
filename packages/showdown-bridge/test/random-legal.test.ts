import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { type ChoiceCase, loadCorpus } from "../src/choice-corpus.ts";
import { firstLegal, GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "../src/golden.ts";
import {
  type BattleRecord,
  loadEngine,
  NoCandidateError,
  type RequestRecord,
  randomBelow,
  randomLegal,
  randomLegalActor,
  RandomStateError,
  runBattle,
} from "../src/index.ts";

const set = (choices: string[], pending: string[] = []) => ({ choices, pending });
const TEN = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/** Draw `count` picks, threading the state. */
const picks = (legal: ReturnType<typeof set>, seed: number, count: number) => {
  const out: string[] = [];
  let state = seed;
  for (let i = 0; i < count; i++) {
    const pick = randomLegal(legal, state);
    out.push(pick.choice);
    state = pick.state;
  }
  return out;
};

describe("randomLegal (baseline actor, evaluation protocol pool B)", () => {
  it("gives the same picks from the same seed, pinned against a PRNG change", () => {
    expect(picks(set(TEN), 12345, 5)).toEqual(["9", "4", "0", "2", "6"]);
    expect(picks(set(TEN), 12345, 5)).toEqual(picks(set(TEN), 12345, 5));
    expect(picks(set(TEN), 0, 5)).toEqual(["8", "7", "6", "4", "6"]);
    expect(picks(set(TEN), 12345, 5)).not.toEqual(picks(set(TEN), 12346, 5));
  });

  it("returns the next state without touching the input", () => {
    const legal = set(["a", "b", "c"]);
    const first = randomLegal(legal, 7);
    expect(first.state).toBe((7 + 0x6d2b79f5) >>> 0);
    expect(randomLegal(legal, 7)).toEqual(first);
    expect(legal).toEqual(set(["a", "b", "c"]));
  });

  it("is uniform over the non-pending candidates within five binomial standard deviations", () => {
    const choices = ["a", "b", "c", "d", "e", "f"];
    const seeds = 60_000;
    const counts = new Map<string, number>(choices.map((c) => [c, 0]));
    for (let seed = 0; seed < seeds; seed++) {
      const { choice } = randomLegal(set(choices), seed);
      counts.set(choice, (counts.get(choice) as number) + 1);
    }
    const p = 1 / choices.length;
    const sd = Math.sqrt(seeds * p * (1 - p)); // binomial standard deviation of one count
    for (const count of counts.values()) expect(Math.abs(count - seeds * p)).toBeLessThan(5 * sd);
  });

  it("is uniform along one stream too", () => {
    const stream = picks(set(["a", "b", "c"]), 99, 30_000);
    const sd = Math.sqrt(30_000 * (1 / 3) * (2 / 3));
    for (const c of ["a", "b", "c"]) {
      expect(Math.abs(stream.filter((x) => x === c).length - 10_000)).toBeLessThan(5 * sd);
    }
  });

  it("removes modulo bias by redrawing values past the largest multiple of n", () => {
    // n = 2^31 + 1 rejects about half of all 32-bit draws. Only `length` is
    // read, so no array is allocated.
    const huge = { choices: { length: 2 ** 31 + 1 } as unknown as string[], pending: [] };
    let redrawn = 0;
    for (let seed = 0; seed < 200; seed++) {
      const { state } = randomLegal(huge, seed);
      if (state !== (seed + 0x6d2b79f5) >>> 0) redrawn++;
    }
    expect(redrawn).toBeGreaterThan(50);
    expect(redrawn).toBeLessThan(150);
  });

  it("plays a pending candidate only when no non-pending one exists, uniformly among them", () => {
    for (let seed = 0; seed < 200; seed++) {
      expect(["a", "b"]).toContain(randomLegal(set(["a", "b"], ["x"]), seed).choice);
    }
    const seen = new Set(picks(set([], ["x", "y", "z"]), 1, 100));
    expect([...seen].sort()).toEqual(["x", "y", "z"]);
    expect(randomLegal(set([], ["only"]), 5).choice).toBe("only");
    expect(randomLegal(set(["only"], ["x"]), 5).choice).toBe("only");
  });

  it("refuses an empty set with NoCandidateError and a bad state with RandomStateError", () => {
    expect(() => randomLegal(set([]), 1)).toThrow(NoCandidateError);
    for (const bad of [-1, 1.5, 2 ** 32, Number.NaN]) {
      expect(() => randomLegal(set(["a"]), bad)).toThrow(RandomStateError);
    }
  });

  it("only picks candidates of the request, on every corpus case", () => {
    const corpus = loadCorpus();
    for (const file of Object.keys(corpus)) {
      const { candidates } = corpus[file] as ChoiceCase;
      const legal = set(
        candidates.filter((c) => !c.pending).map((c) => c.choice),
        candidates.filter((c) => c.pending).map((c) => c.choice),
      );
      const all = candidates.map((c) => c.choice);
      for (const choice of picks(legal, 1, 25)) {
        expect(all).toContain(choice);
        if (legal.choices.length > 0) expect(legal.choices).toContain(choice);
      }
    }
  });

  it("only picks candidates of the request, on the golden request fixtures", () => {
    const dir = new URL("../../../fixtures/showdown/gen9championsvgc2026regmc/", import.meta.url);
    for (const name of ["preview", "turn", "force-switch"]) {
      const text = readFileSync(new URL(`${name}.request.json`, dir), "utf8");
      const legal = (JSON.parse(text) as { legal?: { choices: string[]; pending: string[] } }).legal;
      if (legal === undefined) continue;
      for (const choice of picks(legal, 3, 25)) expect([...legal.choices, ...legal.pending]).toContain(choice);
    }
  });
});

describe("randomLegalActor", () => {
  const request = (legal?: ReturnType<typeof set>) =>
    ({ phase: "move", player: "p1", request_seq: 4, request_hash: "0".repeat(64), request: {}, legal }) as unknown as RequestRecord;

  it("threads its state: it equals the pure function's sequence", async () => {
    const legal = set(TEN);
    const actor = randomLegalActor(12345);
    const got = [];
    for (let i = 0; i < 5; i++) got.push(await actor(request(legal)));
    expect(got).toEqual(picks(legal, 12345, 5));
  });

  it("refuses a bad seed at construction, not at the first request", () => {
    for (const bad of [-1, 1.5, 2 ** 32, Number.NaN]) {
      expect(() => randomLegalActor(bad)).toThrow(RandomStateError);
    }
  });

  it("refuses a request without a legal choice set", () => {
    expect(() => randomLegalActor(1)(request())).toThrow(/p1: request 4 has no legal choice set/);
  });
});

describe("randomLegal in A1 battles", () => {
  const engine = loadEngine();
  const sha = (record: BattleRecord) =>
    createHash("sha256").update(JSON.stringify(record.omniscient_log)).digest("hex");
  const play = (p1: () => ReturnType<typeof randomLegalActor>, p2: () => ReturnType<typeof randomLegalActor>) =>
    runBattle({
      engine,
      formatId: GOLDEN_FORMAT,
      seed: GOLDEN_SEED,
      teams: { p1: syntheticTeam(), p2: syntheticTeam() },
      actors: { p1: p1(), p2: p2() },
      maxTurns: 40,
    });

  const assertLegal = (record: BattleRecord) => {
    expect(record.turns).toBeGreaterThan(0);
    for (const player of ["p1", "p2"] as const) {
      for (const choice of record.choices.filter((c) => c.player === player)) {
        const request = record.requests[player].find((r) => r.request_seq === choice.request_seq) as RequestRecord;
        const legal = request.legal as { choices: string[]; pending: string[] };
        expect([...legal.choices, ...legal.pending]).toContain(choice.choice);
        if (legal.choices.length > 0) expect(legal.choices).toContain(choice.choice);
      }
    }
  };

  it("plays random-legal against the actor rule to the end, reproducibly", async () => {
    const a = await play(() => randomLegalActor(11), () => firstLegal);
    const b = await play(() => randomLegalActor(11), () => firstLegal);
    assertLegal(a);
    expect(a.choices.every((c) => c.refused === undefined)).toBe(true);
    expect(sha(a)).toBe(sha(b));
    expect(a.omniscient_log.at(-1)).toBeDefined();
  });

  it("plays two random-legal players to the end, reproducibly", async () => {
    const a = await play(() => randomLegalActor(21), () => randomLegalActor(22));
    const b = await play(() => randomLegalActor(21), () => randomLegalActor(22));
    assertLegal(a);
    expect(sha(a)).toBe(sha(b));
    const other = await play(() => randomLegalActor(23), () => randomLegalActor(22));
    expect(sha(other)).not.toBe(sha(a));
  });
});

describe("randomBelow (the draw the damage-aware baseline's epsilon_ppm uses)", () => {
  it("draws from the same Mulberry32 stream, uniformly in [0, n), pure and bounded", () => {
    const first = randomBelow(10, 12345);
    // the same stream as randomLegal: a 10-candidate set picks index first.value
    expect(randomLegal(set(TEN), 12345).choice).toBe(String(first.value));
    expect(randomBelow(10, 12345)).toEqual(first);
    expect(first.value).toBeGreaterThanOrEqual(0);
    expect(first.value).toBeLessThan(10);
    expect(randomBelow(1, 5).value).toBe(0);
    expect(randomBelow(2 ** 32, 5).value).toBeLessThan(2 ** 32);
    expect(() => randomBelow(0, 1)).toThrow(RangeError);
    expect(() => randomBelow(2 ** 32 + 1, 1)).toThrow(RangeError);
    expect(() => randomBelow(1.5, 1)).toThrow(RangeError);
    expect(() => randomBelow(10, -1)).toThrow(RandomStateError);
  });
});
