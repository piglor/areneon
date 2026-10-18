import { cpSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_RULES_DIR, loadChannel, loadRulesSnapshot, RulesError } from "../src/index.ts";

const COMMIT = "0".repeat(40);
const FORMAT = "formats/gen9championsvgc2026regmc.json";
const dirs: string[] = [];

/** A scratch copy of the committed rules, edited by `edit`. */
function rulesCopy(edit: (dir: string) => void = () => {}): string {
  const dir = mkdtempSync(join(tmpdir(), "areneon-rules-test-"));
  dirs.push(dir);
  cpSync(DEFAULT_RULES_DIR, dir, { recursive: true });
  edit(dir);
  return dir;
}

function patch(dir: string, file: string, change: (value: Record<string, unknown>) => void): void {
  const path = join(dir, file);
  const value = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  change(value);
  writeFileSync(path, JSON.stringify(value));
}

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("loadRulesSnapshot", () => {
  it("loads the committed records", () => {
    const snapshot = loadRulesSnapshot(COMMIT);
    expect(snapshot.showdown_commit).toBe(COMMIT);
    expect(snapshot.formats.gen9championsvgc2026regmc?.forbidden_rules.map((f) => f.rule)).toEqual([
      "forceopenteamsheets",
    ]);
    expect(Object.keys(snapshot.regulations)).toEqual(["M-C"]);
    expect(snapshot.gaps).toEqual({});
  });

  it.each<[string, (dir: string) => void, RegExp]>([
    ["an unknown schema_version", (d) => patch(d, FORMAT, (v) => (v.schema_version = 99)), /no format schema for schema_version 99/],
    ["a non-integer schema_version", (d) => patch(d, FORMAT, (v) => (v.schema_version = "1")), /schema_version must be an integer/],
    ["a missing schema_version", (d) => patch(d, FORMAT, (v) => delete v.schema_version), /schema_version must be an integer/],
    ["an unknown key", (d) => patch(d, FORMAT, (v) => (v.extra = true)), /must NOT have additional properties/],
    ["a game type outside the closed enum", (d) => patch(d, FORMAT, (v) => (v.game_type = "singles")), /allowed values/],
    ["a regulation record without sources", (d) => patch(d, "regulations/M-C.json", (v) => (v.sources = [])), /regulations\/M-C\.json/],
    [
      "a misnamed format file",
      (d) => renameSync(join(d, FORMAT), join(d, "formats/other.json")),
      /must be named gen9championsvgc2026regmc\.json/,
    ],
    [
      "a misnamed regulation file",
      (d) => renameSync(join(d, "regulations/M-C.json"), join(d, "regulations/MC.json")),
      /must be named M-C\.json/,
    ],
    ["an unknown regulation", (d) => patch(d, FORMAT, (v) => (v.regulation = "M-Z")), /unknown regulation M-Z/],
    [
      "unsorted forbidden rules",
      (d) => patch(d, FORMAT, (v) => (v.forbidden_rules = [{ rule: "b", reason: "x" }, { rule: "a", reason: "x" }])),
      /sorted by rule and unique/,
    ],
    [
      "duplicated forbidden rules",
      (d) => patch(d, FORMAT, (v) => (v.forbidden_rules = [{ rule: "a", reason: "x" }, { rule: "a", reason: "y" }])),
      /sorted by rule and unique/,
    ],
    ["no format records", (d) => rmSync(join(d, FORMAT)), /no format records/],
  ])("fails closed on %s", (_label, edit, message) => {
    const dir = rulesCopy(edit);
    expect(() => loadRulesSnapshot(COMMIT, dir)).toThrow(RulesError);
    expect(() => loadRulesSnapshot(COMMIT, dir)).toThrow(message);
  });

  it("fails on a malformed JSON file", () => {
    const dir = rulesCopy((d) => writeFileSync(join(d, FORMAT), "{"));
    expect(() => loadRulesSnapshot(COMMIT, dir)).toThrow(SyntaxError);
  });
});

describe("loadChannel", () => {
  it("loads the committed stable channel", () => {
    expect(loadChannel("stable")).toMatchObject({ channel: "stable", seq: expect.any(Number) });
  });

  it("rejects a channel file naming another channel", () => {
    const dir = rulesCopy((d) => {
      cpSync(join(d, "channels/stable.json"), join(d, "channels/other.json"));
    });
    expect(() => loadChannel("other", dir)).toThrow(/names channel stable/);
  });

  it("rejects an invalid channel file", () => {
    const dir = rulesCopy((d) => patch(d, "channels/stable.json", (v) => (v.seq = 0)));
    expect(() => loadChannel("stable", dir)).toThrow(/must be >= 1/);
  });
});
