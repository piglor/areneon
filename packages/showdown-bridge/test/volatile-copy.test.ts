import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR, EngineError, loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import {
  exportVolatileCopy,
  readVolatileCopy,
  VOLATILE_COPY_DIR,
  VolatileCopyError,
  volatileCopyFile,
  volatileCopyHash,
} from "../src/volatile-copy.ts";

const engine = loadEngine();
const committed = JSON.parse(readFileSync(join(VOLATILE_COPY_DIR, volatileCopyFile(GOLDEN_FORMAT)), "utf8"));
const scratch = mkdtempSync(join(tmpdir(), "volatile-copy-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const seal = (body: Record<string, unknown>) => ({ ...body, volatile_copy_hash: volatileCopyHash(body as never) });
const { volatile_copy_hash: _hash, ...body } = committed;

describe("the volatile-copy table of the pinned engine", () => {
  it("equals a fresh export byte for byte", () => {
    const fresh = exportVolatileCopy(engine, GOLDEN_FORMAT);
    expect(`${JSON.stringify(fresh, null, 2)}\n`).toBe(readFileSync(join(VOLATILE_COPY_DIR, volatileCopyFile(GOLDEN_FORMAT)), "utf8"));
    expect(readVolatileCopy(committed)).toEqual(fresh);
  });

  it("lists the engine's noCopy conditions, not Substitute or Leech Seed, and Shed Tail's Substitute", () => {
    expect(committed.not_copied).toEqual(expect.arrayContaining(["encore", "trapped", "yawn"]));
    expect(committed.not_copied).not.toContain("substitute");
    expect(committed.not_copied).not.toContain("leechseed");
    expect(committed.shed_tail_copies).toEqual(["substitute"]);
  });

  it("refuses an unknown format", () => {
    expect(() => exportVolatileCopy(engine, "gen9nothing")).toThrow(EngineError);
  });

  it("refuses a checkout whose copyVolatileFrom no longer reads as the recorded rule", () => {
    const dir = join(scratch, "doctored");
    mkdirSync(join(dir, "sim"), { recursive: true });
    const source = readFileSync(join(DEFAULT_SHOWDOWN_DIR, "sim/pokemon.ts"), "utf8");
    writeFileSync(join(dir, "sim/pokemon.ts"), source.replace("i !== 'substitute'", "i !== 'other'"));
    expect(() => exportVolatileCopy(engine, GOLDEN_FORMAT, dir)).toThrow(VolatileCopyError);
  });
});

describe("readVolatileCopy", () => {
  it("refuses what is not a sealed table", () => {
    expect(() => readVolatileCopy(null)).toThrow(/not an object/);
    expect(() => readVolatileCopy([])).toThrow(/not an object/);
    expect(() => readVolatileCopy({ ...committed, extra: 1 })).toThrow(/key set/);
    expect(() => readVolatileCopy(seal({ ...body, schema_version: 2 }))).toThrow(/schema_version/);
    expect(() => readVolatileCopy({ ...committed, format_id: "" })).toThrow(/format_id/);
    expect(() => readVolatileCopy({ ...committed, not_copied: ["a", "b"].reverse() })).toThrow(/sorted/);
    expect(() => readVolatileCopy({ ...committed, not_copied: ["Bad Id"] })).toThrow(/ids/);
    expect(() => readVolatileCopy({ ...committed, not_copied: "x" })).toThrow(/ids/);
    expect(() => readVolatileCopy({ ...committed, shed_tail_copies: ["a", "a"] })).toThrow(/sorted/);
    expect(() => readVolatileCopy({ ...committed, not_copied: [...committed.not_copied, "zzz"] })).toThrow(/hash/);
  });
});
