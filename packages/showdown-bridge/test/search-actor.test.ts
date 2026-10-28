// The search actor's pure parts (ADR-026 sections 4 and 7): the harness-derived seeds with vectors computed outside the TS code
// (Python hashlib over the canonical JSON), the A2 clock rule, the reviewed engine data of the world facts, and the privacy boundary
// of the decision as a type test.
import { describe, expect, expectTypeOf, it } from "vitest";
import { goldenWorldFactsData, required } from "../src/world-facts-data.ts";
import {
  engineSeed,
  replyIndex,
  searchSkipReason,
  type SearchActor,
  type SearchInput,
  FALLBACK_REASONS,
  DROP_REASONS,
} from "../src/search-actor.ts";
import type { BattleRecord, HiddenVerdicts, OmniscientLog, PlayerChannel } from "../src/index.ts";
import type { WorldRequest } from "../src/world-sampler.ts";

const STREAM = Uint8Array.from({ length: 32 }, (_, i) => i + 1);

describe("the world's engine seed and reply draws (section 4)", () => {
  // Computed with Python: sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"))) over the preimages below.
  it("are the SHA-256 of the JCS of {purpose, search_stream, world} and of {draw, purpose, search_stream, world}", () => {
    expect(engineSeed(STREAM, 3)).toBe("sodium,522ece2b83e9fe14df48e59e706d329f");
    expect([replyIndex(STREAM, 3, 0, 7), replyIndex(STREAM, 3, 1, 7), replyIndex(STREAM, 3, 1, 100)]).toEqual([6, 3, 51]);
  });

  it("depend on the world and the stream, stay below n and are spread over the options", () => {
    expect(engineSeed(STREAM, 4)).not.toBe(engineSeed(STREAM, 3));
    expect(engineSeed(Uint8Array.from(STREAM, (b) => b ^ 1), 3)).not.toBe(engineSeed(STREAM, 3));
    const seen = new Set<number>();
    for (let world = 0; world < 200; world += 1) {
      const index = replyIndex(STREAM, world, 0, 5);
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(5);
      seen.add(index);
    }
    expect(seen.size).toBe(5);
    expect(replyIndex(STREAM, 3, 0, 1)).toBe(0);
  });
});

describe("the A2 clock rule (section 6)", () => {
  it("skips the search below the floor and only there", () => {
    expect([searchSkipReason(29_999, 30_000), searchSkipReason(30_000, 30_000), searchSkipReason(0, 30_000)]).toEqual(["clock_floor", null, "clock_floor"]);
  });
});

describe("the world facts' engine data", () => {
  it("is read from the pinned engine's own source", () => {
    const data = goldenWorldFactsData();
    expect(data.stall_moves).toContain("protect");
    expect(data.terrains).toContain("grassyterrain");
    expect(data.toxic_stage_max).toBeGreaterThan(10);
    expect(data.swap_side_conditions).toContain("reflect");
    expect(data.action_sources).toContain("move: Round");
    expect(required(1, "x")).toBe(1);
    expect(() => required(undefined, "ceiling")).toThrow("has no ceiling");
  });
});

describe("the decision's inputs (section 7)", () => {
  it("are the player's own view, the candidates in prior rank and the harness stream, and nothing else", () => {
    expectTypeOf<keyof SearchInput>().toEqualTypeOf<"world" | "channel" | "top" | "stream">();
    expectTypeOf<keyof WorldRequest>().toEqualTypeOf<"request" | "team" | "state" | "flags" | "facts">();
  });

  it("has no field for the live battle, a BattleRecord, an OmniscientLog, hidden verdicts or the seed", () => {
    const base = {} as SearchInput;
    // @ts-expect-error the live battle is not an input
    const a: SearchInput = { ...base, battle: {} as unknown };
    // @ts-expect-error a BattleRecord is not an input
    const b: SearchInput = { ...base, record: {} as BattleRecord };
    // @ts-expect-error the omniscient log is not an input
    const c: SearchInput = { ...base, log: {} as OmniscientLog };
    // @ts-expect-error hidden verdicts are not an input
    const d: SearchInput = { ...base, verdicts: [] as HiddenVerdicts[] };
    // @ts-expect-error the battle seed is not an input
    const e: SearchInput = { ...base, seed: "sodium,00" };
    // @ts-expect-error the opponent's team is not an input
    const f: SearchInput = { ...base, opponent: [] as unknown[] };
    // The channel is the player's own lines, never a branded log of a record.
    expectTypeOf<SearchInput["channel"]>().toEqualTypeOf<readonly string[]>();
    expectTypeOf<PlayerChannel>().not.toEqualTypeOf<SearchInput["channel"]>();
    // The actor's decision takes that input and returns a choice or a fallback reason.
    expectTypeOf<SearchActor["decide"]>().parameter(0).toEqualTypeOf<SearchInput>();
    expect([a, b, c, d, e, f]).toHaveLength(6);
  });

  it("lists the fallback and drop reasons it counts", () => {
    expect(FALLBACK_REASONS).toContain("clock_floor");
    expect(DROP_REASONS).toEqual(["reply_model", "choice_refused"]);
  });
});
