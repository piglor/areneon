// What the snapshot tests share: the pinned engine, the committed label set,
// synthetic private records built from the committed CC0 fixture teams (no
// real team is ever used here), and a DuckDB stand-in that writes Parquet
// through hyparquet-writer so the writer runs where the real tool is absent.

import { createHash, randomBytes } from "node:crypto";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  canonicalTeamHash,
  type Engine,
  exportPaste,
  holdoutFlag,
  importPaste,
  loadEngine,
  type PrivateTeamRecord,
  readTeamLabelSet,
  resolveFormat,
  teamHash,
} from "@areneon/showdown-bridge";
import { type DuckdbPin, DuckdbWriter, labelTeam, locateDuckdb, readPin } from "@areneon/team-dataset";
import { standInRunner } from "../../team-dataset/test/shards/support.ts";
import { labelPlan } from "../src/snapshot-plan.ts";
import type { WriteSnapshotInput } from "../src/snapshot-writer.ts";

/** The element at `k` (negative counts from the end), or a failure: tests index without `!`. */
export function at<T>(list: readonly T[] | RegExpExecArray | null | undefined, k: number): T {
  const value = list?.at(k) as T | undefined;
  if (value === undefined) throw new Error(`no element at ${k}`);
  return value;
}

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
export const FORMAT_ID = "gen9championsvgc2026regmc";
export const engine: Engine = loadEngine();
export const format = resolveFormat(engine, FORMAT_ID);
export const labelSet = readTeamLabelSet(JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT_ID, "labels.json"), "utf8")));
export const plan = labelPlan(labelSet);
export const pastes: string[] = readFileSync(join(ROOT, "fixtures", "teams", FORMAT_ID, "datasets", "fixtures-v1", "teams-dev.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => (JSON.parse(line) as { paste: string }).paste);

export const HOLDOUT_KEY = "42".repeat(32);
export const SHARD_KEY = "ab".repeat(32);
export const hex32 = (n: number): string => n.toString(16).padStart(32, "0");

export interface RecordOptions {
  status?: "active" | "retired";
  tags?: string[];
  key?: string;
}

/** A valid private record for the paste at `index` of the fixture teams, with team_id `hex32(id)`. */
export function recordOf(index: number, id: number, options: RecordOptions = {}): PrivateTeamRecord {
  const sets = importPaste(engine, pastes[index] as string);
  const teamId = hex32(id);
  return {
    schema_version: 1,
    data_class: "private",
    team_id: teamId,
    revision: 1,
    supersedes_record_hash: null,
    team_hash: teamHash(sets),
    canonical_team_hash: canonicalTeamHash(sets),
    paste: exportPaste(engine, sets),
    status: options.status ?? "active",
    tags: options.tags ?? [],
    holdout: holdoutFlag(options.key ?? HOLDOUT_KEY, teamId),
    labels: labelTeam(labelSet, sets).map((l) => ({ label_id: l.label_id, value: l.value })),
  } as PrivateTeamRecord;
}

/** `n` records over the first `n` distinct fixture teams; every fifth is retired. */
export const records = (n: number): PrivateTeamRecord[] =>
  Array.from({ length: n }, (_, k) => recordOf(k, k + 1, { status: k % 5 === 4 ? "retired" : "active", tags: k % 2 === 0 ? ["alpha"] : [] }));

const home = realpathSync(mkdtempSync(join(tmpdir(), "private-store-")));
export const scratchRoot = home;
let counter = 0;
/** A fresh `private-data/work` directory, mode 0700. */
export function workDir(): string {
  const dir = join(home, `case-${counter++}`, "private-data", "work");
  mkdirSync(dir, { recursive: true });
  chmodSync(dir, 0o700);
  return dir;
}

/** The pin and a writer: the real tool when DUCKDB_BIN or the PATH has the pinned one, else the stand-in. */
export function standIn(): { writer: DuckdbWriter; pin: DuckdbPin; runner: ReturnType<typeof standInRunner> } {
  const file = join(home, "duckdb-stand-in");
  writeFileSync(file, `stand-in ${randomBytes(4).toString("hex")}\n`);
  const runner = standInRunner();
  const writer = DuckdbWriter.open(file, runner);
  const real = readPin();
  const sha = createHash("sha256").update(readFileSync(file)).digest("hex");
  const pin: DuckdbPin = { ...real, assets: Object.fromEntries(Object.keys(real.assets).map((k) => [k, { ...(real.assets[k] as object), binary_sha256: sha }])) as DuckdbPin["assets"] };
  return { writer, pin, runner };
}

/** The real pinned DuckDB tool, or undefined where it is absent (CI sets ARENEON_REQUIRE_DUCKDB=1 and so cannot skip). */
export function realDuckdb(): { writer: DuckdbWriter; pin: DuckdbPin } | undefined {
  const binary = locateDuckdb();
  if (binary === undefined) return undefined;
  const writer = DuckdbWriter.open(binary);
  const pin = readPin();
  try {
    writer.requirePinned(pin);
  } catch {
    return undefined;
  }
  return { writer, pin };
}

export function input(over: Partial<WriteSnapshotInput> & Pick<WriteSnapshotInput, "records">, tool = standIn()): WriteSnapshotInput {
  return {
    holdoutKey: HOLDOUT_KEY,
    parentSnapshotHash: null,
    snapshotId: "owner-test",
    labelSet,
    manifest: format.manifest,
    vocabularyHash: "ee".repeat(32),
    shardKey: SHARD_KEY,
    writer: tool.writer,
    pin: tool.pin,
    workDir: workDir(),
    ...over,
  };
}
