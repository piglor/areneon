import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import privateSnapshotManifestSchema from "@areneon/schema/contracts/v1/private-snapshot-manifest.schema.json" with { type: "json" };
import teamSnapshotManifestSchema from "@areneon/schema/contracts/v1/team-snapshot-manifest.schema.json" with { type: "json" };
import { describe, expect, it } from "vitest";
import {
  HOLDOUT_PPM,
  HOLDOUT_RULE,
  holdoutFlag,
  PrivateSnapshotError,
  type PrivateSnapshotManifest,
  privateLogicalSnapshotHash,
  privateSnapshotHash,
  readPrivateSnapshotManifest,
} from "../src/private-snapshot.ts";
import { PRIVATE_SAMPLE_HOLDOUT_KEY } from "../src/private-samples.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const REPO = join(SAMPLES_DIR, "..", "..");
const sample = <T>(path: string): T => JSON.parse(readFileSync(join(SAMPLES_DIR, "v1", path), "utf8")) as T;
const names = (kind: string): string[] => readdirSync(join(SAMPLES_DIR, "v1", "private-snapshot-manifest", kind)).sort();
const first = sample<PrivateSnapshotManifest>("private-snapshot-manifest/valid/first.json");
const extended = sample<PrivateSnapshotManifest>("private-snapshot-manifest/valid/extended.json");
const edit = <T>(value: T, change: (copy: T) => void): T => {
  const copy = structuredClone(value);
  change(copy);
  return copy;
};

// Known answers, computed outside the TypeScript code with Python's hmac and hashlib:
//
//   import hmac, hashlib
//   key = bytes(range(32))                       # the key 000102...1e1f
//   def n(team_id):
//       tag = hmac.new(key, team_id.encode("ascii"), hashlib.sha256).digest()
//       return int.from_bytes(tag[:8], "big")
//   held_out = n(team_id) * 10**6 < ppm * 2**64   # ppm = 200000
//
//   team_id                            first 8 bytes of the tag   n                      held out
//   00112233445566778899aabbccddeeff   36ec4579078060df           3957614558671036639    no  (n / 2^64 = 0.2145427...)
//   ffeeddccbbaa99887766554433221100   391a7263632689fd           4114726980747168253    no
//   00000000000000000000000000000000   9f8f7fbc3e6c1c61           11497548820178017377   no
//   ffffffffffffffffffffffffffffffff   adc2dfda727bee91           12520816043359989393   no
//   a1b2c3d4e5f60718293a4b5c6d7e8f90   2d4161aa7fa4f691           3260994990106801809    yes
//
// The first team flips between ppm 214542 (no) and 214543 (yes); of the team
// ids format(i, "032x") for i in range(2000), 414 are held out at 200000 ppm.
//
// Manifest identities: with jcs = json.dumps(v, sort_keys=True,
// separators=(",", ":")).encode() (every value is an ASCII string or an
// integer), the logical hash of valid/first.json is
//   sha256(jcs({"shards": sorted shard logical hashes, "cells", "label_set_hash",
//              "manifest", "holdout", "first": True}))
// and its snapshot hash is sha256(jcs(the whole file)); valid/extended.json
// uses "parent_snapshot_hash" in place of "first".
const KEY = PRIVATE_SAMPLE_HOLDOUT_KEY;
const VECTORS: [string, boolean][] = [
  ["00112233445566778899aabbccddeeff", false],
  ["ffeeddccbbaa99887766554433221100", false],
  ["00000000000000000000000000000000", false],
  ["ffffffffffffffffffffffffffffffff", false],
  ["a1b2c3d4e5f60718293a4b5c6d7e8f90", true],
];
const FIRST_LOGICAL = "552472cdc8bc00163ccd52088943436e10e7654cb6b79d7bb250a50e5c8b07a0";
const FIRST_SNAPSHOT_HASH = "7eb9777b446a6bfca2ec144f8c3848d1806f20dbbbba9f61d87db43d7804c853";
const EXTENDED_LOGICAL = "ff0a4955392b0abd54675ec8ed759453eb2ddb3261f66afd221d7365b04c4088";

describe("holdoutFlag (ADR-015 section 2, hmac-sha256-team-id/1)", () => {
  it("matches the independent Python vectors", () => {
    expect(KEY).toBe("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
    for (const [id, expected] of VECTORS) expect(holdoutFlag(KEY, id), id).toBe(expected);
    expect(HOLDOUT_RULE).toBe("hmac-sha256-team-id/1");
    expect(HOLDOUT_PPM).toBe(200000);
  });

  it("decides by the threshold n / 2^64 < ppm / 10^6 in integers, exactly at the vector's boundary", () => {
    const id = "00112233445566778899aabbccddeeff";
    expect(holdoutFlag(KEY, id, 214542)).toBe(false);
    expect(holdoutFlag(KEY, id, 214543)).toBe(true);
    expect(holdoutFlag(KEY, id, 0)).toBe(false);
    expect(holdoutFlag(KEY, id, 1_000_000)).toBe(true);
  });

  it("holds about the configured share, and depends on the key", () => {
    const ids = Array.from({ length: 2000 }, (_, i) => i.toString(16).padStart(32, "0"));
    expect(ids.filter((id) => holdoutFlag(KEY, id)).length).toBe(414);
    const other = "f".repeat(64);
    expect(ids.filter((id) => holdoutFlag(other, id) !== holdoutFlag(KEY, id)).length).toBeGreaterThan(0);
  });

  it("is a function of the key and the team_id alone: the same on every call", () => {
    for (const [id, expected] of VECTORS) for (let k = 0; k < 3; k++) expect(holdoutFlag(KEY, id)).toBe(expected);
  });

  it("refuses a key, id or ppm out of shape without repeating it", () => {
    const secret = "SECRETKEY";
    for (const call of [() => holdoutFlag(secret, VECTORS[0]?.[0] as string), () => holdoutFlag("A".repeat(64), VECTORS[0]?.[0] as string), () => holdoutFlag(KEY, "abc"), () => holdoutFlag(KEY, "A".repeat(32))]) {
      expect(call).toThrow(PrivateSnapshotError);
    }
    expect(() => holdoutFlag(secret, "0".repeat(32))).not.toThrow(secret);
    for (const ppm of [-1, 1_000_001, 0.5, Number.NaN]) expect(() => holdoutFlag(KEY, "0".repeat(32), ppm)).toThrow(/ppm/);
  });
});

describe("the identities of a private snapshot", () => {
  it("recompute the independent Python digests", () => {
    expect(privateLogicalSnapshotHash(first)).toBe(FIRST_LOGICAL);
    expect(first.logical_snapshot_hash).toBe(FIRST_LOGICAL);
    expect(privateSnapshotHash(first)).toBe(FIRST_SNAPSHOT_HASH);
    expect(privateLogicalSnapshotHash(extended)).toBe(EXTENDED_LOGICAL);
  });

  it("keep the logical hash independent of the writer and file bytes, and tied to the label set, the key and the lineage", () => {
    const writer = edit(first, (m) => (m.writer.version = "9.9.9"));
    expect(privateLogicalSnapshotHash(writer)).toBe(FIRST_LOGICAL);
    expect(privateSnapshotHash(writer)).not.toBe(FIRST_SNAPSHOT_HASH);
    const bytes = edit(first, (m) => ((m.shards[0] as { file_sha256: string }).file_sha256 = "e".repeat(64)));
    expect(privateLogicalSnapshotHash(bytes)).toBe(FIRST_LOGICAL);
    expect(privateLogicalSnapshotHash(edit(first, (m) => (m.label_set_hash = "e".repeat(64))))).not.toBe(FIRST_LOGICAL);
    expect(privateLogicalSnapshotHash(edit(first, (m) => (m.holdout.key = "e".repeat(64))))).not.toBe(FIRST_LOGICAL);
    const child = edit(first, (m) => {
      delete m.first;
      m.parent_snapshot_hash = "8".repeat(64);
    });
    expect(privateLogicalSnapshotHash(child)).not.toBe(FIRST_LOGICAL);
  });
});

describe("the PrivateSnapshotManifest contract", () => {
  const valid = ajvFor("contract").compile(privateSnapshotManifestSchema);

  it("refuses each invalid sample for its own fault: with the private label put back it stays invalid", () => {
    for (const name of names("invalid")) {
      const value = sample<Record<string, unknown>>(`private-snapshot-manifest/invalid/${name}`);
      expect(valid(value), name).toBe(false);
      if (name !== "synthetic-class.json") expect(valid({ ...value, data_class: "private" }), name).toBe(false);
    }
    // The negative control: that one sample is invalid for its label alone.
    const only = sample<Record<string, unknown>>("private-snapshot-manifest/invalid/synthetic-class.json");
    expect(valid({ ...only, data_class: "private" })).toBe(true);
  });

  it("has exactly the ADR-015 section 2 fields: the public manifest's, without dataset_hashes and with the private additions", () => {
    const own = Object.keys(privateSnapshotManifestSchema.properties).sort();
    const publicKeys = Object.keys(teamSnapshotManifestSchema.properties);
    expect(own).toEqual(
      [...publicKeys.filter((k) => k !== "dataset_hashes"), "label_set_hash", "records", "holdout", "first"].sort(),
    );
    expect(privateSnapshotManifestSchema.properties.data_class.const).toBe("private");
    expect(privateSnapshotManifestSchema.required).not.toContain("parent_snapshot_hash");
    expect(privateSnapshotManifestSchema.required).not.toContain("first");
    expect(privateSnapshotManifestSchema.properties.holdout.properties.rule.const).toBe(HOLDOUT_RULE);
    expect(privateSnapshotManifestSchema.properties.holdout.properties.ppm.const).toBe(HOLDOUT_PPM);
    // The version manifest is the public one's, byte for byte (the profile check also requires it).
    expect(privateSnapshotManifestSchema.$defs.VersionManifest).toEqual(teamSnapshotManifestSchema.$defs.VersionManifest);
  });

  it("holds a hold-out key and no key commitment, and a shard names its index, never a path", () => {
    expect(Object.keys(privateSnapshotManifestSchema.properties.holdout.properties).sort()).toEqual(["key", "ppm", "rule"]);
    expect(Object.keys(privateSnapshotManifestSchema.$defs.Shard.properties)).not.toContain("object_key");
    expect(Object.keys(privateSnapshotManifestSchema.$defs.Shard.properties)).toContain("index");
  });
});

describe("readPrivateSnapshotManifest", () => {
  it("accepts every valid sample, and needs no key or engine", () => {
    for (const name of names("valid")) expect(readPrivateSnapshotManifest(sample(`private-snapshot-manifest/valid/${name}`)).data_class).toBe("private");
    expect(readPrivateSnapshotManifest(first).first).toBe(true);
    expect(readPrivateSnapshotManifest(extended).parent_snapshot_hash).toBe("8".repeat(64));
    const none = readPrivateSnapshotManifest(sample("private-snapshot-manifest/valid/all-retired.json"));
    expect(none.cells).toEqual([]);
    expect(none.records).toEqual({ active: 0, retired: 2 });
  });

  it("carries the hold-out rule of the sample key", () => {
    expect(first.holdout).toEqual({ rule: HOLDOUT_RULE, key: KEY, ppm: HOLDOUT_PPM });
  });

  it.each([
    ["both-parent-and-first", /exactly one of parent_snapshot_hash and first/],
    ["neither-parent-nor-first", /exactly one of parent_snapshot_hash and first/],
    ["shard-index-not-position", /index is not its position/],
    ["shards-out-of-order", /repeated or out of logical-hash order/],
    ["duplicate-shard", /repeated or out of logical-hash order/],
    ["ranges-do-not-tile", /cell ranges do not tile the shard/],
    ["ranges-out-of-order", /repeated or out of stratum order/],
    ["active-rows-over-rows", /more active rows than rows/],
    ["rows-not-range-sum", /rows is not the sum of its cell ranges/],
    ["cell-not-active-sum", /rows is not the shards' active sum/],
    ["cells-out-of-order", /cells are repeated or out of stratum order/],
    ["unlisted-active-cell", /cells do not list exactly the cells that hold an active record/],
    ["records-active-not-cells", /records.active is not the sum of the cells/],
    ["counts-not-shard-rows", /plus records.retired is not the shards' rows/],
    ["wrong-logical-hash", /logical_snapshot_hash is not recomputed/],
  ])("refuses private-snapshot-manifest/%s", (name, message) => {
    const value = sample(`private-snapshot-manifest/reader-invalid/${name}.json`);
    expect(() => readPrivateSnapshotManifest(value)).toThrow(PrivateSnapshotError);
    expect(() => readPrivateSnapshotManifest(value)).toThrow(message);
  });

  it("refuses a cell that lists a stratum no shard holds active, and a cell with a stale count", () => {
    expect(() => readPrivateSnapshotManifest(edit(first, (m) => (m.cells = [...m.cells, { split: "owner", stratum: 9, rows: 1 }])))).toThrow(/active sum/);
  });

  it("refuses a schema violation", () => {
    expect(() => readPrivateSnapshotManifest(edit(first, (m) => ((m as unknown as { data_class: string }).data_class = "synthetic")))).toThrow(/invalid private snapshot manifest: \/data_class const/);
    expect(() => readPrivateSnapshotManifest(null)).toThrow(/invalid private snapshot manifest: \/ type/);
  });

  it("keeps a canary out of every message: a value, a key name and a team-looking string", () => {
    const canary = "CANARYVALUE";
    const probes: unknown[] = [
      edit(first, (m) => ((m as unknown as Record<string, unknown>)[canary] = 1)),
      edit(first, (m) => (m.snapshot_id = `${canary} Garchomp @ Sitrus Berry`)),
      edit(first, (m) => (m.holdout.key = canary)),
      edit(first, (m) => ((m.holdout as unknown as Record<string, unknown>)[canary] = 1)),
      edit(first, (m) => ((m.shards[0] as unknown as Record<string, unknown>)[canary] = 1)),
      edit(first, (m) => (m.writer.name = `${canary} x`)),
      edit(first, (m) => ((m.cells[0] as unknown as { split: string }).split = canary)),
      edit(first, (m) => (m.logical_snapshot_hash = "a".repeat(64))),
      edit(first, (m) => ((m.shards[0] as { index: number }).index = 7)),
    ];
    let thrown = 0;
    for (const probe of probes) {
      try {
        readPrivateSnapshotManifest(probe);
      } catch (error) {
        thrown++;
        expect(error).toBeInstanceOf(PrivateSnapshotError);
        const text = `${(error as Error).message} ${(error as Error).stack ?? ""}`;
        expect(text).not.toContain(canary);
        expect(text).not.toContain("Garchomp");
        expect(text).not.toContain(KEY);
        expect(text).not.toContain("a".repeat(64));
      }
    }
    expect(thrown).toBe(probes.length);
    // Negative control: the canary is in the probes, so the scan would see it were it echoed.
    expect(JSON.stringify(probes)).toContain(canary);
  });
});

describe("the private manifest markers (ADR-015 sections 2 and 6)", () => {
  const source = readFileSync(join(REPO, "scripts/check_data_class.py"), "utf8");
  const markers = (/PRIVATE_MANIFEST_MARKERS = frozenset\(\s*\{([^}]*)\}/.exec(source)?.[1] ?? "")
    .split("\n")
    .map((line) => line.trim().replace(/^"|",?$/g, ""))
    .filter(Boolean);

  it("lists exactly the committed manifest samples that carry the private label", () => {
    const labelled = ["valid", "invalid", "reader-invalid"].flatMap((kind) =>
      names(kind)
        .filter((n) => sample<{ data_class: string }>(`private-snapshot-manifest/${kind}/${n}`).data_class === "private")
        .map((n) => `fixtures/contracts/v1/private-snapshot-manifest/${kind}/${n}`),
    );
    expect(markers.length).toBeGreaterThan(0);
    for (const marker of markers) expect(existsSync(join(REPO, marker)), marker).toBe(true);
    expect(markers.sort()).toEqual(labelled.sort());
  });

  it("holds no paste in any sample, and every string of a marked sample is a hash, a token or a fixed word", () => {
    for (const marker of markers) {
      const text = readFileSync(join(REPO, marker), "utf8");
      expect(text, marker).not.toMatch(/paste|Ability:|@ /);
      const strings = [...text.matchAll(/"([^"]*)"\s*[,\n\]}]/g)].map((m) => m[1] as string);
      for (const s of strings) expect(s, marker).toMatch(/^[A-Za-z0-9][A-Za-z0-9._:/+-]*$/);
    }
  });
});
