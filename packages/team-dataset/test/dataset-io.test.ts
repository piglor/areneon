import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { eachLine, endsWithNewline, recordHashes, writeLines } from "../src/dataset-io.ts";
import { sortedUnique } from "../src/dedupe.ts";

const dir = mkdtempSync(join(tmpdir(), "dataset-io-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const lines = (path: string, chunk?: number): string[] => {
  const out: string[] = [];
  eachLine(path, (line, n) => {
    expect(n).toBe(out.length);
    out.push(line);
  }, chunk);
  return out;
};

describe("streamed JSON Lines I/O (#883)", () => {
  const text = ["alpha", "", "béta \u{1F600}", "g".repeat(100), "z"];

  it("writes the same bytes as joining the lines, at any chunk size", () => {
    const want = text.map((l) => `${l}\n`).join("");
    for (const chunk of [1, 7, 64, 1 << 20]) {
      const path = join(dir, `w${chunk}.jsonl`);
      writeLines(path, text, chunk);
      expect(readFileSync(path, "utf8")).toBe(want);
    }
    writeLines(join(dir, "empty.jsonl"), []);
    expect(readFileSync(join(dir, "empty.jsonl"), "utf8")).toBe("");
  });

  it("reads the lines back across chunk boundaries, multi-byte characters included", () => {
    const path = join(dir, "r.jsonl");
    writeLines(path, text);
    for (const chunk of [1, 2, 3, 5, 64, 1 << 20]) expect(lines(path, chunk)).toEqual(text);
  });

  it("matches split semantics: no lines for an empty file, a last line without a newline is still read", () => {
    const empty = join(dir, "e.jsonl");
    writeFileSync(empty, "");
    expect(lines(empty)).toEqual([]);
    expect(endsWithNewline(empty)).toBe(true);
    const open = join(dir, "open.jsonl");
    writeFileSync(open, "a\nb");
    expect(lines(open, 2)).toEqual(["a", "b"]);
    expect(endsWithNewline(open)).toBe(false);
    const blank = join(dir, "blank.jsonl");
    writeFileSync(blank, "\n");
    expect(lines(blank)).toEqual([""]);
    expect(endsWithNewline(blank)).toBe(true);
  });

  it("recordHashes reads every file, skips blank lines and names the line without a hash", () => {
    const a = join(dir, "a.jsonl");
    const b = join(dir, "b.jsonl");
    writeFileSync(a, '{"canonical_team_hash":"h1"}\n\n{"canonical_team_hash":"h2"}\n');
    writeFileSync(b, '{"canonical_team_hash":"h3"}');
    expect(recordHashes([a, b])).toEqual(["h1", "h2", "h3"]);
    writeFileSync(b, '{"canonical_team_hash":"h3"}\n{"x":1}\n');
    expect(() => recordHashes([b])).toThrow(/b\.jsonl line 2 has no canonical_team_hash/);
  });

  it("holds 1.2 million lines without joining them", () => {
    const path = join(dir, "big.jsonl");
    const n = 1_200_000;
    writeLines(path, (function* () { for (let k = 0; k < n; k++) yield `{"k":${k}}`; })());
    let count = 0;
    let last = "";
    eachLine(path, (line) => {
      count++;
      last = line;
    });
    expect(count).toBe(n);
    expect(last).toBe(`{"k":${n - 1}}`);
  });

  it("sortedUnique drops repeats and refuses a non-hash", () => {
    const h = (c: string): string => c.repeat(64);
    expect(sortedUnique([h("b"), h("a"), h("b"), h("a"), h("c")])).toEqual([h("a"), h("b"), h("c")]);
    expect(sortedUnique([])).toEqual([]);
    expect(() => sortedUnique(["nope"])).toThrow(/not a 64-digit/);
  });
});
