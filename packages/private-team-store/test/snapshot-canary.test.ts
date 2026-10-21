import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { inspect } from "node:util";
import {
  canonicalTeamHash,
  exportPaste,
  importPaste,
  type PrivateTeamRecord,
  privateLogicalSnapshotHash,
  teamHash,
} from "@areneon/showdown-bridge";
import { DuckdbWriter, privateColumnsFor, privateRowOf, readPrivateShard, type Runner } from "@areneon/team-dataset";
import { afterEach, describe, expect, it, vi } from "vitest";
import { parquetOf } from "../../team-dataset/test/shards/support.ts";
import { SnapshotIoError } from "../src/snapshot-io-error.ts";
import { readPrivateSnapshot } from "../src/snapshot-reader.ts";
import { writePrivateSnapshot, type WrittenSnapshot } from "../src/snapshot-writer.ts";
import { at, engine, input, labelSet, pastes, plan, recordOf, scratchRoot, standIn } from "./snapshot-support.ts";

// Paste canary tests (ADR-015 section 3, "No leaks") for the snapshot writer
// and reader: marker strings placed in a paste (a nickname), a tag, a label
// value, a column name, a file's bytes, a tool's stderr, a key and a path must
// appear in no refusal, nor may the species, items, abilities or moves of the
// teams, on any error path, in the error text, its stack, its fields or what
// the process printed. The negative controls show the scan can fail.

const NICK = "CanaryNick7";
const TAG = "canarytag7";
const BADTAG = "CanaryTag7";
const LABEL = "CanaryLabel7";
const FILE = "CanaryFile7";
const STDERR = "CanaryStderr7";
const KEY = "CanaryKey7";
const DIR = "CanaryDir7";
const SOURCE = "CanarySource7";
const MARKERS = [NICK, TAG, BADTAG, LABEL, FILE, STDERR, KEY, DIR, SOURCE];

const nickRecord = (index: number, id: number, over: Partial<PrivateTeamRecord> = {}): PrivateTeamRecord => {
  const base = recordOf(index, id, { tags: [TAG] });
  const sets = importPaste(engine, base.paste);
  at(sets, 0).name = NICK;
  const paste = exportPaste(engine, sets);
  return { ...base, paste, team_hash: teamHash(sets), canonical_team_hash: canonicalTeamHash(sets), ...over };
};

const needles = (): string[] => {
  const out = new Set<string>(MARKERS);
  for (const paste of pastes.slice(0, 20)) {
    for (const set of importPaste(engine, paste)) for (const v of [set.species, set.item, set.ability, ...set.moves]) if (v.length >= 4) out.add(v);
  }
  return [...out];
};
const NEEDLES = needles();

const surface = (error: unknown): string => {
  const e = error as SnapshotIoError;
  return [
    String(error),
    e.message,
    e.name,
    e.stack ?? "",
    JSON.stringify(error, Object.getOwnPropertyNames(error as object)),
    inspect(error, { depth: 8, showHidden: true }),
    JSON.stringify({ code: e.code, position: e.position }),
  ].join("\n");
};
const leaks = (text: string): string[] => NEEDLES.filter((n) => text.includes(n));

let printed: string[] = [];
const spies = [
  vi.spyOn(process.stdout, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true)),
  vi.spyOn(process.stderr, "write").mockImplementation((chunk: unknown) => (printed.push(String(chunk)), true)),
];
afterEach(() => {
  printed = [];
});
void spies;

const refusal = async (run: () => unknown): Promise<SnapshotIoError> => {
  try {
    await run();
  } catch (error) {
    expect(error, "must be a SnapshotIoError").toBeInstanceOf(SnapshotIoError);
    expect(leaks(surface(error)), `leaked from ${(error as SnapshotIoError).code}`).toEqual([]);
    expect(leaks(printed.join("")), "printed").toEqual([]);
    return error as SnapshotIoError;
  }
  throw new Error("expected a refusal");
};

const recordsWithNick = (n = 6): PrivateTeamRecord[] => Array.from({ length: n }, (_, k) => nickRecord(k, k + 1, { status: k % 5 === 4 ? "retired" : "active" }));

describe("the setup puts the markers where the paths read them", () => {
  it("has the nickname in the paste, the tag in the tags and real team text to look for", () => {
    const r = nickRecord(0, 1);
    expect(r.paste).toContain(NICK);
    expect(r.tags).toEqual([TAG]);
    expect(NEEDLES.length).toBeGreaterThan(30);
    expect(NEEDLES.some((n) => importPaste(engine, r.paste).some((s) => s.species === n))).toBe(true);
  });
});

describe("negative controls: the scan can fail", () => {
  it("flags an error that repeats a paste, a marker or a species", () => {
    const r = nickRecord(0, 1);
    expect(leaks(surface(new Error(`bad paste: ${r.paste}`))).length).toBeGreaterThan(0);
    expect(leaks(surface(new Error(`tag ${BADTAG}`)))).toContain(BADTAG);
    expect(leaks(surface(Object.assign(new Error("x"), { cause: new Error(NICK) })))).toContain(NICK);
    expect(leaks(surface(new SnapshotIoError("record_invalid")))).toEqual([]);
  });

  it("flags the public writer path, which forwards the tool's stderr", () => {
    const tool = standIn();
    const forwarding = DuckdbWriter.open(
      join(scratchRoot, "duckdb-stand-in"),
      ((f, a, i, e, c) => (a[0] === "--version" ? tool.runner(f, a, i, e, c) : { status: 1, stdout: "", stderr: `${STDERR} ${nickRecord(0, 1).paste}` })) as Runner,
    );
    let message = "";
    try {
      forwarding.writeShard("a", "b", privateColumnsFor(labelSet));
    } catch (error) {
      message = surface(error);
    }
    expect(leaks(message)).toContain(STDERR);
  });
});

describe("writer error paths", () => {
  it("withholds the tool's stderr, with markers and a paste in it", async () => {
    const tool = standIn();
    const failing = DuckdbWriter.open(
      join(scratchRoot, "duckdb-stand-in"),
      ((f, a, i, e, c) => (a[0] === "--version" ? tool.runner(f, a, i, e, c) : { status: 1, stdout: `${NICK}`, stderr: `${STDERR} ${nickRecord(0, 1).paste}` })) as Runner,
    );
    const error = await refusal(() => writePrivateSnapshot(input({ records: recordsWithNick(), writer: failing, pin: tool.pin })));
    expect(error.code).toBe("writer_failed");
  });

  it("refuses a record set, key, flag, labels and directory without echoing them", async () => {
    expect((await refusal(() => writePrivateSnapshot(input({ records: [nickRecord(0, 1), nickRecord(1, 1)] })))).code).toBe("records_invalid");
    expect((await refusal(() => writePrivateSnapshot(input({ records: recordsWithNick(), holdoutKey: KEY })))).code).toBe("holdout_key_invalid");
    const flipped = nickRecord(0, 1);
    expect((await refusal(() => writePrivateSnapshot(input({ records: [{ ...flipped, holdout: !flipped.holdout }] })))).code).toBe("holdout_mismatch");
    const labelled = nickRecord(0, 1, { labels: [{ label_id: LABEL, value: LABEL }] });
    expect((await refusal(() => writePrivateSnapshot(input({ records: [labelled] })))).code).toBe("labels_invalid");
    expect((await refusal(() => writePrivateSnapshot(input({ records: recordsWithNick(), workDir: `/${DIR}/private-data` })))).code).toBe("work_dir_unsafe");
    expect((await refusal(() => writePrivateSnapshot(input({ records: recordsWithNick(), shardKey: KEY })))).code).toBe("layout_failed");
  });

  it("refuses a shard file the reader cannot read back, whether it holds a marker or a bad tag", async () => {
    const tool = standIn();
    const garbage = DuckdbWriter.open(
      join(scratchRoot, "duckdb-stand-in"),
      ((f, a, i, e, c) => {
        if (a[0] === "--version") return tool.runner(f, a, i, e, c);
        writeFileSync(at(/TO '((?:[^']|'')*)'/.exec(i), 1), `${FILE} ${nickRecord(0, 1).paste}`);
        return { status: 0, stdout: "", stderr: "" };
      }) as Runner,
    );
    expect((await refusal(() => writePrivateSnapshot(input({ records: recordsWithNick(), writer: garbage, pin: tool.pin })))).code).toBe("shard_invalid");
    // A tag outside the pattern is written by the stand-in and refused on read-back.
    const badTag = nickRecord(0, 1, { tags: [BADTAG] });
    expect((await refusal(() => writePrivateSnapshot(input({ records: [badTag] })))).code).toBe("shard_invalid");
  });
});

const written = async (): Promise<WrittenSnapshot> => writePrivateSnapshot(input({ records: recordsWithNick(8), shardRows: 4 }));

function reseal(w: WrittenSnapshot, shard: number, file: Uint8Array): Record<string, unknown> {
  const shards = w.manifest.shards.map((s, i) => (i === shard ? { ...s, bytes: file.byteLength, file_sha256: createHash("sha256").update(file).digest("hex") } : s));
  const next = { ...w.manifest, shards } as Record<string, unknown>;
  next.logical_snapshot_hash = privateLogicalSnapshotHash(next as never);
  return next;
}

describe("reader error paths", () => {
  const read = (w: WrittenSnapshot, manifest: unknown, files: (i: number) => Uint8Array | Promise<Uint8Array>) =>
    readPrivateSnapshot({ engine, manifest, shard: files, labelSet, shardRows: 4 });

  /** The rows of shard 0 as the writer wrote them, changed, written as Parquet and put in the manifest. */
  async function forged(change: (rows: Record<string, unknown>[]) => void, options: (n: number) => Parameters<typeof parquetOf>[2] = () => ({})) {
    const w = await written();
    const first = await readPrivateShard(at(w.shards, 0), at(w.manifest.shards, 0), plan);
    const rows = first.records.map((r, k) => privateRowOf(r, at(first.strata, k))) as Record<string, unknown>[];
    change(rows);
    const file = parquetOf(rows, privateColumnsFor(labelSet), options(rows.length));
    return { w, manifest: reseal(w, 0, file), files: (i: number) => (i === 0 ? file : at(w.shards, i)) };
  }

  it("refuses forged shard contents with fixed messages, whatever they contain", async () => {
    const cases: [string, (rows: Record<string, unknown>[]) => void, ((n: number) => Parameters<typeof parquetOf>[2])?][] = [
      ["tag", (r) => (at(r, 0).tags = [BADTAG])],
      ["label value", (r) => (at(at(r, 0).labels as { value: string }[], 0).value = LABEL)],
      ["label column", (r) => (at(r, 0)[`label:${at(labelSet.labels, 0).label_id}`] = LABEL)],
      ["predecessor", (r) => (at(r, 0).supersedes_record_hash = FILE)],
      ["team id", (r) => (at(r, 0).team_id = NICK)],
      ["paste", (r) => (at(r, 0).paste = `${NICK} (${FILE})`)],
      ["column name", () => undefined, (n) => ({ extra: [{ name: FILE, data: Array.from({ length: n }, () => NICK) }] })],
    ];
    for (const [label, change, options] of cases) {
      const f = await forged(change, options);
      const error = await refusal(() => read(f.w, f.manifest, f.files));
      expect(error.code, label).toBe("shard_invalid");
      expect(error.position.shard, label).toBe(0);
    }
  });

  it("refuses a file of marker bytes with a matching digest, and one without", async () => {
    const w = await written();
    const bytes = new TextEncoder().encode(`${FILE} ${NICK}`);
    const manifest = reseal(w, 1, bytes);
    expect((await refusal(() => read(w, manifest, (i) => (i === 1 ? bytes : at(w.shards, i))))).code).toBe("shard_invalid");
    expect((await refusal(() => read(w, w.manifest, (i) => (i === 1 ? bytes : at(w.shards, i))))).code).toBe("shard_size_mismatch");
    const same = new Uint8Array(at(w.shards, 1).byteLength).fill(65);
    expect((await refusal(() => read(w, w.manifest, (i) => (i === 1 ? same : at(w.shards, i))))).code).toBe("shard_digest_mismatch");
  });

  it("drops the text of a failing shard source and of a manifest full of markers", async () => {
    const w = await written();
    expect((await refusal(() => read(w, w.manifest, () => Promise.reject(new Error(`${SOURCE} ${NICK}`))))).code).toBe("shard_missing");
    expect((await refusal(() => read(w, { ...w.manifest, [FILE]: NICK }, (i) => at(w.shards, i)))).code).toBe("manifest_invalid");
    expect((await refusal(() => read(w, { ...w.manifest, snapshot_id: `${NICK} ${LABEL}` }, (i) => at(w.shards, i)))).code).toBe("manifest_invalid");
    expect((await refusal(() => read(w, w.manifest, (i) => at(w.shards, i)).then(() => readPrivateSnapshot({ engine, manifest: w.manifest, shard: (i) => at(w.shards, i), labelSet, shardRows: 4, expectedSnapshotHash: KEY })))).code).toBe("snapshot_hash_mismatch");
  });

  it("refuses a record the engine refuses, a wrong flag and without echoing them", async () => {
    const bad = nickRecord(0, 1, { paste: nickRecord(0, 1).paste.replace(/Ability: .*/, `Ability: ${NICK}`) });
    const w = await writePrivateSnapshot(input({ records: [bad, nickRecord(1, 2)] }));
    const error = await refusal(() => readPrivateSnapshot({ engine, manifest: w.manifest, shard: (i) => at(w.shards, i), labelSet }));
    expect(error.code).toBe("record_invalid");
    expect(error.position).toEqual({ shard: 0, row: expect.any(Number) });

    const ok = await written();
    const other = { ...ok.manifest, holdout: { ...ok.manifest.holdout, key: "77".repeat(32) } } as Record<string, unknown>;
    other.logical_snapshot_hash = privateLogicalSnapshotHash(other as never);
    expect((await refusal(() => read(ok, other, (i) => at(ok.shards, i)))).code).toBe("holdout_mismatch");

  });
});
