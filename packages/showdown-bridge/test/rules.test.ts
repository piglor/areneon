import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  canonicalJson,
  formatRulesRecord,
  jcs,
  type Json,
  loadRulesSnapshot,
  PINNED_SHOWDOWN_COMMIT,
  RulesError,
  rulesSnapshotHash,
  snapshotBytes,
} from "../src/index.ts";

describe("jcs (RFC 8785)", () => {
  // JSON text escapes are built at runtime so the vectors stay byte-exact.
  const u = (hex: string): string => `\\u${hex}`;
  const ch = (code: number): string => String.fromCodePoint(code);

  // RFC 8785 §3.2.2: number and string serialisation. The input is parsed
  // from its JSON text, as a published file would be.
  it("matches the RFC primitive-serialisation vector", () => {
    const input =
      `{"numbers":[333333333.33333329,1E30,4.50,2e-3,0.000000000000000000000000001],` +
      `"string":"${u("20ac")}$${u("000F")}${u("000a")}A'${u("0042")}${u("0022")}${u("005c")}\\\\\\"\\/",` +
      `"literals":[null,true,false]}`;
    expect(jcs(JSON.parse(input) as Json)).toBe(
      `{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],` +
        `"string":"${ch(0x20ac)}$${u("000f")}\\nA'B\\"\\\\\\\\\\"/"}`,
    );
  });

  // RFC 8785 §3.2.3: keys sort by UTF-16 code units, so the emoji's high
  // surrogate (0xD83D) sorts before U+FB33. Object.keys would put "1"
  // first, so the serialised text itself is compared.
  it("matches the RFC key-sorting vector", () => {
    const input =
      `{"${u("20ac")}":"Euro Sign","\\r":"Carriage Return","${u("fb33")}":"Hebrew Letter Dalet With Dagesh",` +
      `"1":"One","${u("d83d")}${u("de00")}":"Emoji: Grinning Face","${u("0080")}":"Control",` +
      `"${u("00f6")}":"Latin Small Letter O With Diaeresis"}`;
    expect(jcs(JSON.parse(input) as Json)).toBe(
      `{"\\r":"Carriage Return","1":"One","${ch(0x80)}":"Control",` +
        `"${ch(0xf6)}":"Latin Small Letter O With Diaeresis","${ch(0x20ac)}":"Euro Sign",` +
        `"${ch(0x1f600)}":"Emoji: Grinning Face","${ch(0xfb33)}":"Hebrew Letter Dalet With Dagesh"}`,
    );
  });

  it("is canonicalJson plus the I-JSON checks", () => {
    const value = { b: [1, "x"], a: { d: null, c: true } };
    expect(jcs(value)).toBe(canonicalJson(value));
  });

  it.each([
    ["a lone high surrogate in a value", { a: "\ud83d" }],
    ["a lone low surrogate in a key", { "\ude00": 1 }],
    ["a lone surrogate in an array", ["ok", "\udc00x"]],
  ])("rejects %s", (_label, value) => {
    expect(() => jcs(value as Json)).toThrow(/lone surrogate/);
  });

  it("rejects non-finite numbers", () => {
    expect(() => jcs({ a: Number.NaN })).toThrow(RangeError);
  });
});

describe("rules snapshot identity", () => {
  const snapshot = loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT);

  it("hashes the exact JCS bytes of exactly five keys", () => {
    expect(Object.keys(snapshot).sort()).toEqual(
      ["formats", "gaps", "regulations", "schema_version", "showdown_commit"],
    );
    const bytes = snapshotBytes(snapshot);
    expect(bytes.toString("utf8")).toBe(jcs(snapshot as unknown as Json));
    expect(rulesSnapshotHash(snapshot)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("changes with the engine commit", () => {
    expect(rulesSnapshotHash({ ...snapshot, showdown_commit: "0".repeat(40) })).not.toBe(
      rulesSnapshotHash(snapshot),
    );
  });

  // The historical channel check runs scripts/rules-hash.ts from bare
  // archives, so its unvalidated build must hash identically.
  it("is reproduced by the dependency-free rules-hash script", () => {
    const script = fileURLToPath(new URL("../scripts/rules-hash.ts", import.meta.url));
    const printed = JSON.parse(execFileSync("node", [script], { encoding: "utf8" })) as {
      hash: string;
      showdown_commit: string;
    };
    expect(printed).toEqual({ hash: rulesSnapshotHash(snapshot), showdown_commit: PINNED_SHOWDOWN_COMMIT });
  });

  // check-rules.ts reruns this commit's builder from a bare archive on every
  // later PR. Copy only what an archive holds (no node_modules anywhere up
  // the tree) so a non-built-in import fails here, before it reaches main.
  it("builds from a bare copy with no node_modules", () => {
    const root = mkdtempSync(join(tmpdir(), "areneon-builder-"));
    try {
      const repo = fileURLToPath(new URL("../../../", import.meta.url));
      cpSync(join(repo, "rules"), join(root, "rules"), { recursive: true });
      for (const part of ["src", "scripts"]) {
        cpSync(join(repo, "packages/showdown-bridge", part), join(root, "packages/showdown-bridge", part), {
          recursive: true,
        });
      }
      const printed = JSON.parse(
        execFileSync("node", [join(root, "packages/showdown-bridge/scripts/rules-hash.ts")], { encoding: "utf8" }),
      ) as { hash: string };
      expect(printed.hash).toBe(rulesSnapshotHash(snapshot));
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("refuses a format record that is not in the snapshot", () => {
    expect(() => formatRulesRecord(snapshot, "gen9nonsense")).toThrow(RulesError);
  });
});
