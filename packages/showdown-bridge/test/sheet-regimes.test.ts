import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { jcs, type Json } from "../src/canonical-json.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import {
  checkPlanRegimes,
  closedGameCount,
  loadEngine,
  planFormats,
  readTrainingRunConfig,
  regimeCounts,
  regimeDigest,
  resolveRegimes,
  selfPlayRegimes,
  SHEET_REGIME_PURPOSE,
  SheetRegimeError,
  sortManifests,
  type TrainingRunConfig,
} from "../src/index.ts";

const CLOSED = "gen9championsvgc2026regmc";
const OPEN = "gen9championsvgc2026regmcbo3";
const sample = (name: string): Record<string, unknown> => JSON.parse(readFileSync(join(SAMPLES_DIR, `v1/training-run-config/valid/${name}.json`), "utf8")) as Record<string, unknown>;
const regimes = (change: Partial<NonNullable<TrainingRunConfig["sheet_regimes"]>> = {}) => ({ open_format: OPEN, closed_share_ppm: 500_000, open_only_generations: 1, closed_sheet_source: "revealed-only/1", ...change });
// The tiny sample: 4 self-play games, then 2 random-legal baseline games (run key "tiny-run").
const mixed = (change: Partial<NonNullable<TrainingRunConfig["sheet_regimes"]>> = {}): TrainingRunConfig => readTrainingRunConfig({ ...sample("tiny"), sheet_regimes: regimes(change) });
const sheet = (config: TrainingRunConfig) => config.sheet_regimes as NonNullable<TrainingRunConfig["sheet_regimes"]>;

describe("the sheet-regime/1 assignment (ADR-027 section 5)", () => {
  it("is a hash of the run key, generation and game, pinned against Python's hashlib", () => {
    expect(SHEET_REGIME_PURPOSE).toBe("sheet-regime/1");
    // Computed with areneon_ml.loop.sheet_regimes (hashlib over the RFC 8785 bytes).
    expect(regimeDigest("tiny-run", 2, 0)).toBe("3bd74e5c7d17e70ba2fef095a9079d48a146c125e84d47653108a04cf85847f8");
  });

  it("makes the open-only generations open and then exactly round(games * share) closed, by hash order", () => {
    const config = mixed();
    expect(selfPlayRegimes(sheet(config), "tiny-run", 1, 8)).toEqual(Array(8).fill("open"));
    // Vectors from the Python twin.
    expect(selfPlayRegimes(sheet(config), "tiny-run", 2, 8)).toEqual(["closed", "open", "closed", "open", "closed", "open", "closed", "open"]);
    expect(selfPlayRegimes(sheet(config), "tiny-run", 3, 8)).toEqual(["open", "closed", "open", "closed", "closed", "open", "open", "closed"]);
  });

  it("rounds half up and always has exactly the quota", () => {
    expect([closedGameCount(4, 500_000), closedGameCount(3, 500_000), closedGameCount(5, 500_000), closedGameCount(7, 250_000), closedGameCount(10, 1), closedGameCount(10, 999_999)]).toEqual([2, 2, 3, 2, 0, 10]);
    for (const [games, share] of [[7, 250_000], [10, 0], [10, 1_000_000], [9, 333_333]] as const) {
      const closed = selfPlayRegimes({ ...regimes(), closed_share_ppm: share } as never, "k", 5, games).filter((r) => r === "closed").length;
      expect(closed).toBe(closedGameCount(games, share));
    }
  });

  it("takes the closed games of one generation from the hash order alone: more share only adds games", () => {
    const some = selfPlayRegimes({ ...regimes(), closed_share_ppm: 250_000 } as never, "k", 3, 12);
    const more = selfPlayRegimes({ ...regimes(), closed_share_ppm: 750_000 } as never, "k", 3, 12);
    for (const [game, regime] of some.entries()) if (regime === "closed") expect(more[game]).toBe("closed");
  });
});

describe("the plan's formats", () => {
  it("name the open format for open self-play games and the closed one for the rest, baselines included", () => {
    const config = mixed();
    expect(planFormats(config, sheet(config), 1)).toEqual([OPEN, OPEN, OPEN, OPEN, CLOSED, CLOSED]);
    const second = planFormats(config, sheet(config), 2);
    expect(second.filter((f) => f === OPEN)).toHaveLength(2);
    expect(second.slice(4)).toEqual([CLOSED, CLOSED]);
  });

  const planOf = (config: TrainingRunConfig, generation: number) =>
    planFormats(config, sheet(config), generation).map((format_id, game) => ({ format_id, opponent: game < config.generation.games ? "self" : { actor: "random-legal" } }));

  it("pass the check and are counted", () => {
    const config = mixed();
    const plan = planOf(config, 2);
    expect(() => checkPlanRegimes(plan, config, 2)).not.toThrow();
    expect(regimeCounts(plan, config)).toEqual({ open: 2, closed: 4 });
  });

  it("are refused when they leave the quota, name another format, put a baseline in the open format or exist without sheet_regimes", () => {
    const config = mixed();
    const swapped = planOf(config, 2).map((e, i) => (i === 0 ? { ...e, format_id: e.format_id === OPEN ? CLOSED : OPEN } : e));
    expect(() => checkPlanRegimes(swapped, config, 2)).toThrow(/quota/);
    const foreign = planOf(config, 2).map((e, i) => (i === 1 ? { ...e, format_id: "gen9other" } : e));
    expect(() => checkPlanRegimes(foreign, config, 2)).toThrow(/neither the closed format nor the open format/);
    const openBaseline = planOf(config, 1).map((e, i) => (i === 5 ? { ...e, format_id: OPEN } : e));
    expect(() => checkPlanRegimes(openBaseline, config, 1)).toThrow(/baseline game is played in the closed format only/);
    expect(() => checkPlanRegimes(planOf(config, 2).slice(1), config, 2)).toThrow(SheetRegimeError);
    const plain = readTrainingRunConfig(sample("tiny"));
    expect(() => checkPlanRegimes([{ opponent: "self" }], plain, 1)).not.toThrow();
    expect(() => checkPlanRegimes([{ opponent: "self", format_id: CLOSED }], plain, 1)).toThrow(/no sheet_regimes/);
    expect(() => regimeCounts([], plain)).toThrow(/no sheet_regimes/);
  });
});

describe("the manifests of the two formats (ADR-027 section 5a)", () => {
  const engine = loadEngine();

  it("are resolved once, checked against the engine and sorted by their JCS bytes", () => {
    const resolved = resolveRegimes(engine, mixed());
    expect(resolved.closed.format_id).toBe(CLOSED);
    expect(resolved.open.format_id).toBe(OPEN);
    expect(resolved.closed.showdown_commit).toBe(resolved.open.showdown_commit);
    expect(resolved.sorted).toHaveLength(2);
    const keys = resolved.sorted.map((m) => jcs(m as unknown as Json));
    expect(keys).toEqual([...keys].sort());
    expect(sortManifests([resolved.open, resolved.closed, resolved.open])).toEqual(resolved.sorted);
  });

  it("are refused when the closed format is open, the open format is closed, or the pair differs beyond the sheet rules", () => {
    expect(() => resolveRegimes(engine, { ...mixed(), format_id: OPEN, sheet_regimes: regimes({ open_format: CLOSED }) } as TrainingRunConfig)).toThrow(/must be a closed-sheet format/);
    expect(() => resolveRegimes(engine, { ...mixed(), sheet_regimes: regimes({ open_format: CLOSED }) } as TrainingRunConfig)).toThrow(/must be an open-sheet format/);
    expect(() => resolveRegimes(engine, { ...mixed(), format_id: "gen9championsbssregmc" } as TrainingRunConfig)).toThrow();
    expect(() => resolveRegimes(engine, readTrainingRunConfig(sample("tiny")))).toThrow(/no sheet_regimes/);
  });
});
