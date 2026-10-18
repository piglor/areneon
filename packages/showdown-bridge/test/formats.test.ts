import { describe, expect, it } from "vitest";
import {
  formatRulesHash,
  loadEngine,
  loadRulesSnapshot,
  resolveFormat,
  type RulesSnapshot,
  UnsupportedFormatError,
} from "../src/index.ts";

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
function committedFormat(formatId: string): RulesSnapshot["formats"][string] {
  const format = snapshot.formats[formatId];
  if (!format) throw new Error(`rules/formats/${formatId}.json is missing`);
  return format;
}
const mc = committedFormat("gen9championsvgc2026regmc");

/** The committed snapshot plus `formats`, e.g. a candidate record under test. */
const withFormats = (formats: RulesSnapshot["formats"]): RulesSnapshot => ({
  ...snapshot,
  formats: { ...snapshot.formats, ...formats },
});

describe("resolveFormat", () => {
  it("resolves Reg M-C Bo1 doubles to a version manifest", () => {
    const resolved = resolveFormat(engine, "gen9championsvgc2026regmc");
    expect(resolved.rules).toEqual({
      schema_version: 1,
      format_id: "gen9championsvgc2026regmc",
      regulation: "M-C",
      game_type: "doubles",
      forbidden_rules: ["forceopenteamsheets"],
    });
    expect(resolved.manifest).toEqual({
      showdown_commit: engine.commit,
      format_id: "gen9championsvgc2026regmc",
      regulation: "M-C",
      ruleset_hash: expect.stringMatching(/^[0-9a-f]{64}$/),
      format_rules_hash: formatRulesHash(resolved.rules),
    });
    expect(resolved.rules_snapshot_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  // The engine-derived ruleset_hash is unchanged by ADR-007: same bytes as
  // the value committed with the #670 golden fixtures.
  it("keeps the engine-derived ruleset_hash byte-identical", () => {
    expect(resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash).toBe(
      "4dd1f039bbcc6156e08c434972beae2370dc5fe8c9c6a263da7383c3247b68f3",
    );
  });

  it("hashes the ruleset deterministically and per format", () => {
    const again = resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash;
    expect(again).toBe(resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash);
    const mb = resolveFormat(
      engine,
      "gen9championsvgc2026regmb",
      withFormats({ gen9championsvgc2026regmb: { ...mc, regulation: "M-B" } }),
    );
    expect(mb.manifest.ruleset_hash).not.toBe(again);
  });

  it.each([
    ["singles", "gen9championsbssregmc"],
    ["Bo3 (forced open sheets)", "gen9championsvgc2026regmcbo3"],
    ["another mod", "gen9vgc2025regi"],
    ["an unknown ID", "gen9nonsense"],
  ])("rejects %s, which has no rules record", (_label, formatId) => {
    expect(snapshot.formats[formatId]).toBeUndefined();
    expect(() => resolveFormat(engine, formatId)).toThrow(/not a supported Areneon format/);
  });

  it("rejects a record the engine defines as singles", () => {
    expect(() =>
      resolveFormat(engine, "gen9championsbssregmc", withFormats({ gen9championsbssregmc: mc })),
    ).toThrow(/not a doubles format/);
  });

  it("rejects a record the engine does not define", () => {
    expect(() =>
      resolveFormat(engine, "gen9missing", withFormats({ gen9missing: mc })),
    ).toThrow(/not a doubles format/);
  });

  it("rejects a format with a forbidden rule, giving the record's reason", () => {
    const bo3 = withFormats({ gen9championsvgc2026regmcbo3: mc });
    expect(() => resolveFormat(engine, "gen9championsvgc2026regmcbo3", bo3)).toThrow(
      UnsupportedFormatError,
    );
    expect(() => resolveFormat(engine, "gen9championsvgc2026regmcbo3", bo3)).toThrow(
      /has forceopenteamsheets: Forced Open Team Sheets/,
    );
  });

  it("rejects a snapshot built for another Showdown commit", () => {
    expect(() =>
      resolveFormat(engine, "gen9championsvgc2026regmc", { ...snapshot, showdown_commit: "0".repeat(40) }),
    ).toThrow(/built for Showdown 0{40}/);
  });

  it("binds behaviour-changing policy into format_rules_hash", () => {
    const base = resolveFormat(engine, "gen9championsvgc2026regmc").manifest.format_rules_hash;
    const extra = withFormats({
      gen9championsvgc2026regmc: {
        ...mc,
        forbidden_rules: [...mc.forbidden_rules, { rule: "zzunusedrule", reason: "test" }],
      },
    });
    expect(resolveFormat(engine, "gen9championsvgc2026regmc", extra).manifest.format_rules_hash).not.toBe(base);
  });

  it("keeps reasons and seasons out of format_rules_hash", () => {
    const base = resolveFormat(engine, "gen9championsvgc2026regmc");
    const edited = withFormats({
      gen9championsvgc2026regmc: {
        ...mc,
        season: { start: "2026-01-01T00:00Z", end: "2026-02-01T00:00Z" },
        forbidden_rules: mc.forbidden_rules.map((rule) => ({ ...rule, reason: "reworded" })),
      },
    });
    const again = resolveFormat(engine, "gen9championsvgc2026regmc", edited);
    expect(again.manifest.format_rules_hash).toBe(base.manifest.format_rules_hash);
    expect(again.rules_snapshot_hash).not.toBe(base.rules_snapshot_hash);
  });
});
