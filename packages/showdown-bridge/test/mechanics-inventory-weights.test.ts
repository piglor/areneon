/* eslint-disable @typescript-eslint/no-explicit-any */
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { GOLDEN_DIR, goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { INVENTORY_FILE, type MechanicsInventory, readMechanicsInventory } from "../src/mechanics-inventory.ts";
import { INVENTORY_DOC, percent, renderInventoryDoc } from "../src/mechanics-inventory-doc.ts";
import { lineEvents, readRunEpisodes, runReport, RunReadError } from "../src/mechanics-inventory-run.ts";
import { FIXTURES_V1_DIR, InventoryTeamsError, latestUsage, readDatasetTeams } from "../src/mechanics-inventory-teams.ts";
import {
  buildIndex,
  curve,
  entitiesFor,
  entityKey,
  independenceCurve,
  teamKeys,
  teamPresence,
  usageSlots,
  WEIGHTS_FILE,
  weightsBytes,
  weightsReport,
  type WeightsReport,
} from "../src/mechanics-inventory-weights.ts";
import { readUsageWeightTable } from "../src/usage-weight-table.ts";

// #1465: weights, coverage curves, run counts, the team reader and the generated document.

let engine: Engine;
const vocabulary = goldenTeamVocabulary();
const inv: MechanicsInventory = readMechanicsInventory(JSON.parse(readFileSync(join(GOLDEN_DIR, INVENTORY_FILE), "utf8")));
beforeAll(() => {
  engine = loadEngine();
});
const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
const scratch = (): string => {
  const dir = mkdtempSync(join(tmpdir(), "inventory-weights-"));
  dirs.push(dir);
  return dir;
};
const latest = latestUsage();
const usage = readUsageWeightTable(latest.value, vocabulary);

describe("curve", () => {
  it("orders by count then key and covers a unit once its last entity is implemented", () => {
    const c = curve([["a", "b"], ["a"], ["c"], []]);
    expect(c.order).toEqual([
      { key: "a", count: 2 },
      { key: "b", count: 1 },
      { key: "c", count: 1 },
    ]);
    expect(c.units).toBe(4);
    // n = 0: the empty unit; n = 1: + ["a"]; n = 2: + ["a","b"]; n = 3: + ["c"]
    expect(c.covered_ppm).toEqual([250_000, 500_000, 750_000, 1_000_000]);
    expect(entitiesFor(c, 500_000)).toBe(1);
    expect(entitiesFor(c, 1_000_000)).toBe(3);
    expect(entitiesFor({ ...c, covered_ppm: [0, 10] }, 1_000_000)).toBeNull();
    expect(curve([]).covered_ppm).toEqual([0]);
  });

  it("the independence estimate multiplies what is left out", () => {
    const c = independenceCurve([
      { key: "x", p_ppm: 500_000 },
      { key: "y", p_ppm: 900_000 },
    ]);
    expect(c.order.map((e) => e.key)).toEqual(["y", "x"]);
    // implementing y alone leaves x out with probability 1 - 0.5; nothing implemented: 0.5 * 0.1
    expect(c.covered_ppm).toEqual([50_000, 500_000, 1_000_000]);
    expect(c.units).toBe(0);
  });

  it("team presence is one minus the chance no slot holds it", () => {
    expect(teamPresence(0, 6)).toBe(0);
    expect(teamPresence(1_000_000, 6)).toBe(1_000_000);
    expect(teamPresence(500_000, 2)).toBe(750_000);
    expect(teamPresence(2_000_000, 2)).toBe(1_000_000);
  });
});

describe("the inventory index and a team's entities", () => {
  const index = buildIndex(inv);
  it("expands a held move into the conditions it starts and counts ids the inventory lacks", () => {
    const held = teamKeys(index, [{ item: "Leftovers", ability: "Intimidate", moves: ["Tailwind", "Protect", "", "Nonesuchmove"] }]);
    for (const key of ["item:leftovers", "ability:intimidate", "move:tailwind", "move:protect", "condition:tailwind", "condition:protect"]) expect(held.keys.has(key), key).toBe(true);
    expect(held.unknown).toBe(1);
    expect(index.damage.get("move:facade")).toEqual(["onBasePower"]);
    expect(index.damage.has("condition:auroraveil")).toBe(false); // defined in the move: the move's own handlers
    expect(index.damage.has("move:auroraveil")).toBe(true);
  });
});

describe("usage slots", () => {
  const index = buildIndex(inv);
  it("scales a move's share by the moves a set holds, drops ids the inventory lacks and caps a condition at one", () => {
    const table = {
      species: [
        { usage_ppm: 500_000, items: { entries: [{ id: "leftovers", ppm: 400_000 }, { id: "nonesuch", ppm: 1 }] }, abilities: { entries: [{ id: "intimidate", ppm: 1_000_000 }] }, moves: { entries: [{ id: "tailwind", ppm: 250_000 }] } },
        { usage_ppm: 500_000, items: { entries: [] }, abilities: { entries: [] }, moves: { entries: [{ id: "tailwind", ppm: 1_000_000 }] } },
      ],
    } as any;
    const slots = usageSlots(inv, index, table, 4);
    expect(slots.get("item:leftovers")).toBe(200_000);
    expect(slots.get("item:nonesuch")).toBeUndefined();
    expect(slots.get("ability:intimidate")).toBe(500_000);
    // 500000 * 250000 * 4 / 1e6 = 500000, + 500000 * 1e6 * 4 / 1e6 = 2000000, so 2.5 million, capped
    expect(slots.get("move:tailwind")).toBe(1_000_000);
    expect(slots.get("condition:tailwind")).toBe(1_000_000);
  });
});

describe("the committed weights report", () => {
  it("equals a fresh computation from fixtures-v1 (dev split only) and the latest usage table", () => {
    const { teams, sha256, files } = readDatasetTeams(engine, FIXTURES_V1_DIR);
    expect(files).toEqual(["teams-dev.jsonl"]); // the validation split is never opened
    const report = weightsReport(inv, { teams, dataset: { id: "fixtures-v1", file_sha256: sha256 }, usage: { file: latest.file, table: usage }, team_size: vocabulary.rules.max_team_size });
    expect(readFileSync(join(GOLDEN_DIR, WEIGHTS_FILE), "utf8")).toBe(weightsBytes(report));
    expect(report.dataset.unknown_ids).toBe(0);
    expect(report.dataset.moves_per_set).toBe(4);
    expect(report.coverage.teams.covered_ppm.at(-1)).toBe(1_000_000);
    expect(report.pairs.pairs_seen).toBeGreaterThan(0);
    expect(report.pairs.top.length).toBeLessThanOrEqual(40);
    expect(report.weights.length).toBe(report.damage.length);
  });

  it("the committed document equals a fresh render (regenerate with scripts/mechanics-inventory.ts)", () => {
    const report = JSON.parse(readFileSync(join(GOLDEN_DIR, WEIGHTS_FILE), "utf8")) as WeightsReport;
    expect(readFileSync(INVENTORY_DOC, "utf8")).toBe(renderInventoryDoc(inv, report));
  });

  it("renders the edge cases: a share no order reaches, a missing weight, no start and one start unresolved", () => {
    const report = structuredClone(JSON.parse(readFileSync(join(GOLDEN_DIR, WEIGHTS_FILE), "utf8"))) as WeightsReport;
    report.coverage.teams.covered_ppm = report.coverage.teams.covered_ppm.map(() => 0);
    report.coverage.handler_names.covered_ppm = report.coverage.handler_names.covered_ppm.map(() => 0);
    report.coverage.usage_independence.covered_ppm = report.coverage.usage_independence.covered_ppm.map(() => 0);
    report.weights = [];
    const none = renderInventoryDoc({ ...inv, unresolved_starts: [] }, report);
    expect(none).toContain("never");
    expect(none).toContain("- 0 starts name a condition the engine has no condition for.");
    const one = renderInventoryDoc({ ...inv, unresolved_starts: ["move:x a:b"] }, report);
    expect(one).toContain("- 1 start name a condition the engine has no condition for: `move:x a:b`.");
    expect(renderInventoryDoc(inv, report)).toContain("start");
  });

  it("formats parts per million as percentages in integers", () => {
    expect(percent(0)).toBe("0.00%");
    expect(percent(123_456)).toBe("12.34%");
    expect(percent(1_000_000)).toBe("100.00%");
  });

  it("weightsReport is deterministic for the same input", () => {
    const { teams } = readDatasetTeams(engine, FIXTURES_V1_DIR);
    const input = { teams: teams.slice(0, 10), dataset: { id: "x", file_sha256: "0" }, usage: { file: latest.file, table: usage }, team_size: 6 };
    expect(weightsBytes(weightsReport(inv, input))).toBe(weightsBytes(weightsReport(inv, input)));
  });
});

describe("the team reader", () => {
  it("reads only the dev and training split files and refuses private-data, no files and a record with no paste", () => {
    const dir = scratch();
    expect(() => readDatasetTeams(engine, dir)).toThrow(InventoryTeamsError);
    writeFileSync(join(dir, "teams-validation.jsonl"), "{}\n");
    expect(() => readDatasetTeams(engine, dir)).toThrow(/no dev or training/);
    writeFileSync(join(dir, "records.training.jsonl"), '{"x":1}\n');
    expect(() => readDatasetTeams(engine, dir)).toThrow(/no paste/);
    const hidden = join(dir, "private-data");
    mkdirSync(hidden);
    expect(() => readDatasetTeams(engine, hidden)).toThrow(/private-data/);
    const paste = JSON.parse(readFileSync(join(FIXTURES_V1_DIR, "teams-dev.jsonl"), "utf8").split("\n")[0] as string).paste;
    writeFileSync(join(dir, "records.dev.jsonl"), `${JSON.stringify({ paste })}\n`);
    writeFileSync(join(dir, "records.training.jsonl"), `${JSON.stringify({ paste })}\n\n`);
    const read = readDatasetTeams(engine, dir);
    expect(read.files).toEqual(["records.dev.jsonl", "records.training.jsonl"]);
    expect(read.teams.length).toBe(2);
  });

  it("latestUsage names the last month and refuses an empty directory", () => {
    expect(latest.file).toMatch(/^\d{4}-\d{2}-\d+\.json$/);
    expect(() => latestUsage(scratch())).toThrow(/no usage weight table/);
  });
});

describe("run counts", () => {
  it("reads the exercised entities of a protocol line", () => {
    expect(lineEvents("|move|p1a: Zoroark|Protect|p1a: Zoroark")).toEqual([["move", "protect"]]);
    expect(lineEvents("|move|p1a: X|Mirror Move|p2a: Y|[from]move: Foo")).toEqual([]);
    expect(lineEvents("|-damage|p1a: X|50/100|[from] item: Life Orb")).toEqual([["item", "lifeorb"]]);
    expect(lineEvents("|-activate|p2a: X|ability: Intimidate")).toEqual([["ability", "intimidate"]]);
    expect(lineEvents("|-start|p1a: X|move: Taunt")).toEqual([["condition", "taunt"]]);
    expect(lineEvents("|-sidestart|p1: A|move: Tailwind")).toEqual([["condition", "tailwind"]]);
    expect(lineEvents("|-fieldstart|move: Trick Room|[of] p1a: X")).toEqual([["condition", "trickroom"]]);
    expect(lineEvents("|-weather|RainDance")).toEqual([["condition", "raindance"]]);
    expect(lineEvents("|-weather|none")).toEqual([]);
    expect(lineEvents("|-status|p1a: X|brn")).toEqual([["condition", "brn"]]);
    expect(lineEvents("|-start|p1a: X")).toEqual([]);
    expect(lineEvents("plain text")).toEqual([]);
  });

  const channel = (n: number): string[] => [
    "|move|p1a: A|Facade|p2a: B", // turn 0 (before |turn|1)
    "|turn|1",
    "|move|p1a: A|Facade|p2a: B",
    "|-activate|p2a: B|ability: Intimidate",
    "|-weather|Sandstorm",
    "|-start|p1a: A|move: Nonesuch Thing",
    "|turn|2",
    "|move|p2a: B|Round|p1a: A",
    ...(n > 1 ? ["|move|p2a: B|Facade|p1a: A"] : []),
  ];
  const report = runReport([{ data_class: "synthetic", channel: channel(1) }, { data_class: "synthetic", channel: channel(2) }], inv);

  it("counts uses, turns and Episodes of the damage-relevant entities and the unmatched events", () => {
    expect(report.episodes).toBe(2);
    expect(report.turns).toBe(4);
    expect(report.unmatched_events).toBe(2);
    const facade = report.entities.find((e) => e.key === "move:facade") as { uses: number; turns: number; episodes: number };
    expect(facade).toEqual({ key: "move:facade", uses: 5, turns: 5, episodes: 2 });
    expect(report.entities.find((e) => e.key === "move:round")).toEqual({ key: "move:round", uses: 2, turns: 2, episodes: 2 });
    expect(report.entities.find((e) => e.key === "condition:sandstorm")?.uses).toBe(2);
  });

  it("builds coverage curves and pair counts", () => {
    expect(report.coverage.episodes.units).toBe(2);
    expect(report.coverage.turns.units).toBe(6);
    expect(report.coverage.handler_names.order.length).toBeGreaterThan(0);
    // facade and round share turn 2 of the first Episode and of the second: two turns hold the pair
    expect(report.pairs).toEqual({ turns: 6, pairs_seen: expect.any(Number), pairs_ge_5_turns: 0, pairs_ge_20_turns: 0 });
    const many = runReport([{ data_class: "synthetic", channel: Array.from({ length: 25 }, () => ["|turn|1", "|move|p1a: A|Facade|x", "|move|p1a: A|Round|x"]).flat() }], inv);
    expect(many.pairs.pairs_ge_20_turns).toBe(1);
    expect(many.pairs.pairs_ge_5_turns).toBe(1);
  });

  it("refuses an Episode that is not synthetic", () => {
    expect(() => runReport([{ data_class: "private", channel: [] }], inv)).toThrow(RunReadError);
  });

  it("reads each generation's episodes and its members' episodes, one at a time, and refuses bad files", () => {
    const run = scratch();
    const g = join(run, "generations", "g000003");
    mkdirSync(join(g, "episodes"), { recursive: true });
    mkdirSync(join(g, "members", "m0", "episodes"), { recursive: true });
    mkdirSync(join(g, "members", "m1"), { recursive: true }); // a member with no episodes directory
    writeFileSync(join(g, "episodes", "a.json"), JSON.stringify({ data_class: "synthetic", channel: ["|turn|1"] }));
    writeFileSync(join(g, "episodes", "ignored.txt"), "x");
    writeFileSync(join(g, "members", "m0", "episodes", "b.json"), JSON.stringify({ data_class: "synthetic", channel: ["|turn|2", "|turn|3"] }));
    expect([...readRunEpisodes(run, [3])].map((e) => e.channel.length)).toEqual([1, 2]);
    expect([...readRunEpisodes(run, [4])]).toEqual([]); // a generation with no directory
    writeFileSync(join(g, "episodes", "c.json"), JSON.stringify({ data_class: "synthetic", channel: [1] }));
    expect(() => [...readRunEpisodes(run, [3])]).toThrow(/no data class or channel/);
    const hidden = join(run, "private-data");
    mkdirSync(hidden);
    expect(() => [...readRunEpisodes(hidden, [3])]).toThrow(/private-data/);
  });
});

describe("the entity key", () => {
  it("is kind:id", () => {
    expect(entityKey("move", "protect")).toBe("move:protect");
  });
});
