import { describe, expect, it } from "vitest";
import { loadEngine, resolveFormat, SUPPORTED_FORMATS, UnsupportedFormatError } from "../src/index.ts";

const engine = loadEngine();

describe("resolveFormat", () => {
  it("resolves Reg M-C Bo1 doubles to a version manifest", () => {
    const resolved = resolveFormat(engine, "gen9championsvgc2026regmc");
    expect(resolved.spec).toEqual({ regulation: "M-C", gameType: "doubles", activeSlots: 2 });
    expect(resolved.manifest).toMatchObject({
      showdown_commit: engine.commit,
      format_id: "gen9championsvgc2026regmc",
      regulation: "M-C",
    });
    expect(resolved.manifest.ruleset_hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("hashes the ruleset deterministically and per format", () => {
    const again = resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash;
    expect(again).toBe(resolveFormat(engine, "gen9championsvgc2026regmc").manifest.ruleset_hash);
    const mb = resolveFormat(engine, "gen9championsvgc2026regmb", {
      gen9championsvgc2026regmb: { regulation: "M-B", gameType: "doubles", activeSlots: 2 },
    });
    expect(mb.manifest.ruleset_hash).not.toBe(again);
  });

  it.each([
    ["singles", "gen9championsbssregmc"],
    ["Bo3 (forced open sheets)", "gen9championsvgc2026regmcbo3"],
    ["another mod", "gen9vgc2025regi"],
    ["an unknown ID", "gen9nonsense"],
  ])("rejects %s", (_label, formatId) => {
    expect(SUPPORTED_FORMATS[formatId]).toBeUndefined();
    expect(() => resolveFormat(engine, formatId)).toThrow(UnsupportedFormatError);
  });

  it("rejects a registry entry the engine defines as singles", () => {
    const registry = {
      gen9championsbssregmc: { regulation: "M-C", gameType: "doubles", activeSlots: 2 },
    } as const;
    expect(() => resolveFormat(engine, "gen9championsbssregmc", registry)).toThrow(
      /not a doubles format/,
    );
  });

  it("rejects a registry entry the engine does not define", () => {
    const registry = { gen9missing: { regulation: "X", gameType: "doubles", activeSlots: 2 } } as const;
    expect(() => resolveFormat(engine, "gen9missing", registry)).toThrow(/not a doubles format/);
  });

  it("rejects a registry entry that forces Open Team Sheets", () => {
    const registry = {
      gen9championsvgc2026regmcbo3: { regulation: "M-C", gameType: "doubles", activeSlots: 2 },
    } as const;
    expect(() => resolveFormat(engine, "gen9championsvgc2026regmcbo3", registry)).toThrow(
      /forces Open Team Sheets/,
    );
  });
});
