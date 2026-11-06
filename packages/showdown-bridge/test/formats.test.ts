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
    ["another mod", "gen9vgc2025regi"],
    ["an unknown ID", "gen9nonsense"],
  ])("rejects %s, which has no rules record", (_label, formatId) => {
    expect(snapshot.formats[formatId]).toBeUndefined();
    expect(() => resolveFormat(engine, formatId)).toThrow(/not a supported Areneon format/);
  });

  describe("team_sheets (ADR-027 section 4)", () => {
    const bo3 = committedFormat("gen9championsvgc2026regmcbo3");

    it("resolves the open-sheet format with team_sheets in its replay identity", () => {
      const resolved = resolveFormat(engine, "gen9championsvgc2026regmcbo3");
      expect(resolved.rules).toEqual({
        schema_version: 1,
        format_id: "gen9championsvgc2026regmcbo3",
        regulation: "M-C",
        game_type: "doubles",
        forbidden_rules: [],
        team_sheets: "open",
      });
      expect(resolved.manifest.format_rules_hash).toBe(formatRulesHash(resolved.rules));
      expect(resolved.manifest.format_rules_hash).toBe("aff5366139b1132115e3b91971c23a59fe4557309473c3232ca53c828d66d1c6");
      expect(resolved.manifest.ruleset_hash).not.toBe(resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash);
    });

    it("keeps the closed target's record bytes and format_rules_hash", () => {
      expect(mc.team_sheets).toBeUndefined();
      const rules = resolveFormat(engine, "gen9championsvgc2026regmc").rules;
      expect("team_sheets" in rules).toBe(false);
      // Pinned from before the field existed (the hash committed Episodes carry).
      expect(formatRulesHash(rules)).toBe("549cd2aa97324be093c567324676d46a5eff19d6b0c9d8491c483dab746d5dcc");
      expect(formatRulesHash(resolveFormat(engine, "gen9championsvgc2026regmc", withFormats({ gen9championsvgc2026regmc: { ...mc, team_sheets: "closed" } })).rules)).toBe(
        formatRulesHash(rules),
      );
    });

    it("fails closed when the record and the engine disagree", () => {
      expect(() =>
        resolveFormat(engine, "gen9championsvgc2026regmc", withFormats({ gen9championsvgc2026regmc: { ...mc, forbidden_rules: [], team_sheets: "open" } })),
      ).toThrow(/declared open but the engine has no forceopenteamsheets/);
      expect(() =>
        resolveFormat(engine, "gen9championsvgc2026regmcbo3", withFormats({ gen9championsvgc2026regmcbo3: { ...bo3, team_sheets: "closed" } })),
      ).toThrow(/declared closed but the engine has forceopenteamsheets/);
      expect(() =>
        resolveFormat(engine, "gen9championsvgc2026regmcbo3", withFormats({ gen9championsvgc2026regmcbo3: (({ team_sheets: _t, ...rest }) => rest)(bo3) })),
      ).toThrow(UnsupportedFormatError);
    });
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
