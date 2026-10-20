import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { exportEngineData } from "../src/engine-data.ts";
import { type Engine, loadEngine, type PokemonSet } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, syntheticTeam } from "../src/golden.ts";
import { SAMPLES_DIR } from "../src/samples.ts";
import { teamHash } from "../src/team-class.ts";
import {
  canonicalTeamHash,
  logicalSnapshotHash,
  readTeamDatasetManifest,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  readTeamRecord,
  readTeamSnapshotManifest,
  recordHash,
  shardLogicalHash,
  snapshotHash,
  type TeamLabelSet,
  type TeamRecord,
  TeamRecordError,
  type TeamSnapshotManifest,
} from "../src/team-records.ts";

const engine = loadEngine();
const table = exportEngineData(engine, GOLDEN_FORMAT);
const sample = <T>(path: string): T => JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", path), "utf8")) as T;
const record = sample<TeamRecord>("team-record/valid/golden.json");
const labelSet = sample<TeamLabelSet>("team-label-set/valid/golden.json");
const snapshot = sample<TeamSnapshotManifest>("team-snapshot-manifest/valid/first.json");
const edit = <T>(value: T, change: (copy: T) => void): T => {
  const copy = structuredClone(value);
  change(copy);
  return copy;
};
const permutations = <T>(items: T[]): T[][] =>
  items.length <= 1 ? [items] : items.flatMap((item, i) => permutations([...items.slice(0, i), ...items.slice(i + 1)]).map((rest) => [item, ...rest]));

describe("canonicalTeamHash (ADR-013 section 4)", () => {
  const sets = syntheticTeam();

  it("is the same for every permutation of the team", () => {
    const hashes = new Set(permutations(sets.slice(0, 5)).map((p) => canonicalTeamHash([...p, sets[5] as PokemonSet])));
    expect(hashes.size).toBe(1);
  });

  it("differs from the positional hash as soon as the order does, and equals it for the canonical order", () => {
    const sorted = [...sets].sort((a, b) => (a.species.toLowerCase() < b.species.toLowerCase() ? -1 : 1));
    const reversed = [...sorted].reverse();
    expect(teamHash(reversed)).not.toBe(teamHash(sorted));
    expect(canonicalTeamHash(reversed)).toBe(canonicalTeamHash(sorted));
    expect(canonicalTeamHash(sets)).toBe(teamHash(sorted));
  });

  it("sorts by species id, not by display name or case", () => {
    const [a, b] = [
      { ...(sets[0] as PokemonSet), species: "Zapdos" },
      { ...(sets[1] as PokemonSet), species: "Mr. Mime" },
    ] as [PokemonSet, PokemonSet];
    // "mrmime" < "zapdos" although "M" and "Z" would also order them: the id drops the dot and the space.
    expect(canonicalTeamHash([a, b])).toBe(teamHash([b, a]));
  });

  it("breaks a species tie by the set's JCS bytes, so a repeated species is still one hash", () => {
    const [x, y] = [sets[0] as PokemonSet, { ...(sets[0] as PokemonSet), moves: ["Tackle"] }] as [PokemonSet, PokemonSet];
    expect(canonicalTeamHash([x, y])).toBe(canonicalTeamHash([y, x]));
  });

  it("changes when any set changes, and ignores nothing of a set", () => {
    const changed = sets.map((s, k) => (k === 2 ? { ...s, nature: "Adamant" } : s));
    expect(canonicalTeamHash(changed)).not.toBe(canonicalTeamHash(sets));
  });
});

describe("the identities", () => {
  it("hashes a shard by its sorted record hashes, whatever their order, and refuses a repeat", () => {
    const [a, b, c] = ["a", "b", "c"].map((x) => x.repeat(64)) as [string, string, string];
    expect(shardLogicalHash([c, a, b])).toBe(shardLogicalHash([a, b, c]));
    expect(shardLogicalHash([a, b])).not.toBe(shardLogicalHash([a, b, c]));
    expect(() => shardLogicalHash([a, a])).toThrow(TeamRecordError);
  });

  it("makes the logical snapshot hash independent of the writer and the files, and the snapshot hash dependent on them", () => {
    const other = edit(snapshot, (s) => {
      s.writer = { name: "pyarrow", version: "26.0.0", binary_sha256: "e".repeat(64) };
      (s.shards[0] as { file_sha256: string; object_key: string; bytes: number }).file_sha256 = "d".repeat(64);
      (s.shards[0] as { object_key: string }).object_key = `pub/v1/shards/${"d".repeat(64)}`;
      s.parquet_schema_version = 2;
    });
    expect(logicalSnapshotHash(other)).toBe(logicalSnapshotHash(snapshot));
    expect(snapshotHash(other)).not.toBe(snapshotHash(snapshot));
    expect(readTeamSnapshotManifest(other).writer.name).toBe("pyarrow");
  });

  it("makes the logical snapshot hash independent of the shard order, and dependent on cells, datasets and the manifest", () => {
    const extended = sample<TeamSnapshotManifest>("team-snapshot-manifest/valid/extended.json");
    const swapped = edit(extended, (s) => s.shards.reverse());
    expect(logicalSnapshotHash(swapped)).toBe(logicalSnapshotHash(extended));
    const hash = logicalSnapshotHash(snapshot);
    expect(logicalSnapshotHash(edit(snapshot, (s) => (s.cells[0] as { rows: number }).rows++))).not.toBe(hash);
    expect(logicalSnapshotHash(edit(snapshot, (s) => (s.dataset_hashes = ["f".repeat(64)])))).not.toBe(hash);
    expect(logicalSnapshotHash(edit(snapshot, (s) => (s.manifest.format_id = "other")))).not.toBe(hash);
  });

  it("hashes a record by its JCS bytes: key order does not matter, a value does", () => {
    const reordered = Object.fromEntries(Object.entries(record).reverse()) as TeamRecord;
    expect(recordHash(reordered)).toBe(recordHash(record));
    expect(recordHash({ ...record, origin: { ...record.origin, index: 1 } })).not.toBe(recordHash(record));
  });
});

describe("readTeamRecord", () => {
  it("accepts the golden record, with or without its label set, and its hashes match the golden team", () => {
    expect(readTeamRecord(engine, record)).toEqual(record);
    expect(readTeamRecord(engine, record, labelSet)).toEqual(record);
    expect(record.team_hash).toBe(teamHash(syntheticTeam()));
    expect(record.canonical_team_hash).toBe(canonicalTeamHash(syntheticTeam()));
  });

  it("refuses a schema violation, and a paste the engine cannot read", () => {
    expect(() => readTeamRecord(engine, { ...record, data_class: "private" })).toThrow(/invalid team record/);
    expect(() => readTeamRecord(engine, { ...record, paste: "   " })).toThrow(/refused by the pinned engine/);
  });

  it("lets an error that is not an import refusal through", () => {
    const broken = { ...engine, sim: { ...engine.sim, Teams: { ...engine.sim.Teams, import: () => { throw new RangeError("boom"); } } } } as unknown as Engine;
    expect(() => readTeamRecord(broken, record)).toThrow(RangeError);
  });

  it("refuses labels that do not follow a given label set", () => {
    const fewer = edit(record, (r) => r.labels.pop());
    expect(() => readTeamRecord(engine, fewer, labelSet)).toThrow(/one per label/);
    expect(() => readTeamRecord(engine, edit(record, (r) => r.labels.reverse()), labelSet)).toThrow(/labels\[0\] is sun_setter/);
  });

  it("accepts a label's default value as well as its listed ones", () => {
    const checked = readTeamRecord(engine, record, labelSet);
    expect(checked.labels[0]?.value).toBe("neither");
    expect(readTeamRecord(engine, edit(record, (r) => (r.labels[0] = { label_id: "speed_control", value: "tailwind" })), labelSet).labels[0]?.value).toBe("tailwind");
  });
});

describe("readTeamLabelSet", () => {
  it("accepts the golden set with the engine's table, and a set of all four leaf kinds", () => {
    expect(readTeamLabelSet(labelSet, table)).toEqual(labelSet);
    const every = edit(labelSet, (l) => {
      l.labels = [
        {
          label_id: "everything",
          values: [
            { value: "a", when: { kind: "has_species", species: ["garchomp"] } },
            { value: "b", when: { kind: "has_item", items: ["leftovers"] } },
            { value: "c", when: { kind: "has_ability", abilities: ["roughskin"] } },
          ],
          default: "d",
        },
      ];
    });
    expect(readTeamLabelSet(every, table).labels).toHaveLength(1);
    expect(() => readTeamLabelSet(edit(every, (l) => (l.labels[0]?.values[0] as { when: { species: string[] } }).when.species.push("zzzzzz")), table)).toThrow(/zzzzzz is not in the engine-data table/);
    expect(() => readTeamLabelSet(edit(every, (l) => ((l.labels[0]?.values[1] as { when: { items: string[] } }).when.items = ["noitem"])), table)).toThrow(/noitem is not in/);
    expect(() => readTeamLabelSet(edit(every, (l) => ((l.labels[0]?.values[2] as { when: { abilities: string[] } }).when.abilities = ["noability"])), table)).toThrow(/noability is not in/);
  });

  it("refuses a schema violation, and checks the ids of an all_of's leaves", () => {
    expect(() => readTeamLabelSet({ ...labelSet, labels: [] })).toThrow(/invalid team label set/);
    const inner = edit(labelSet, (l) => ((l.labels[2]?.values[0] as { when: { of: { abilities?: string[] }[] } }).when.of[0] as { abilities: string[] }).abilities = ["zzz", "aaa"]);
    expect(() => readTeamLabelSet(inner)).toThrow(/has_ability ids: .* repeated or out of order/);
  });
});

describe("readTeamGeneratorConfig", () => {
  const config = sample<Record<string, unknown>>("team-generator-config/valid/golden.json");

  it("accepts the golden config with or without the table, and refuses a schema violation", () => {
    expect(readTeamGeneratorConfig(config).iv).toBe(31);
    expect(readTeamGeneratorConfig(config, table).item_weights).toHaveLength(2);
    expect(() => readTeamGeneratorConfig({ ...config, iv: 32 })).toThrow(/invalid team generator config/);
  });
});

describe("readTeamDatasetManifest and readTeamSnapshotManifest", () => {
  it("accept the golden samples and refuse a schema violation", () => {
    const dataset = sample<Record<string, unknown>>("team-dataset-manifest/valid/golden.json");
    expect(readTeamDatasetManifest(dataset).dataset_id).toBe("golden-1");
    expect(() => readTeamDatasetManifest({ ...dataset, splits: [] })).toThrow(/invalid team dataset manifest/);
    expect(() => readTeamSnapshotManifest({ ...snapshot, shards: [] })).toThrow(/invalid team snapshot manifest/);
    expect(readTeamSnapshotManifest(snapshot).snapshot_id).toBe("golden-snapshot-1");
  });

  it("refuses repeated splits", () => {
    const dataset = sample<{ splits: unknown[] }>("team-dataset-manifest/valid/golden.json");
    expect(() => readTeamDatasetManifest({ ...dataset, splits: [dataset.splits[0], dataset.splits[0]] })).toThrow(/splits are repeated/);
  });
});

describe("the golden synthetic team is the one the samples use", () => {
  it("stays equal to the committed fixture", () => {
    expect(JSON.parse(readFileSync(join(GOLDEN_DIR, "synthetic-team.json"), "utf8"))).toEqual(syntheticTeam());
  });
});
