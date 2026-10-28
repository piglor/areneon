// The play-mode grammar of the progress evaluation (ADR-023 section 1, #1044): closed, so no mode string or match id is free text.
import { describe, expect, it } from "vitest";
import { matchSuffix, modeName, parseExtraMode, parseModes, ProgressModeError } from "../src/progress-modes.ts";

describe("parseExtraMode", () => {
  it("names the settings of an extra mode and none for the modes always played", () => {
    expect(parseExtraMode("argmax")).toBeNull();
    expect(parseExtraMode("sample")).toBeNull();
    expect(parseExtraMode("sample:t500000")).toEqual({ temperature_ppm: 500_000 });
    expect(parseExtraMode("sample:th30000")).toEqual({ truncation: { kind: "threshold", min_ppm: 30_000 } });
    expect(parseExtraMode("sample:t500000:th30000")).toEqual({ temperature_ppm: 500_000, truncation: { kind: "threshold", min_ppm: 30_000 } });
    expect(parseExtraMode("sample:t1")).toEqual({ temperature_ppm: 1 });
    expect(parseExtraMode("sample:t999999")).toEqual({ temperature_ppm: 999_999 });
  });

  it.each(["", "Sample", "argmax:t5", "sample:t0", "sample:t1000000", "sample:th1000000", "sample:t05", "sample:th0", "sample:th30000:t500000", "sample:t5:", "sample:t5.5", "sample:t-5", "sample t5", "sample:t5:th5:x", "sample:t9999999"])("refuses %j", (text) => {
    expect(() => parseExtraMode(text)).toThrow(ProgressModeError);
  });
});

describe("the names", () => {
  const both = { temperature_ppm: 500_000, truncation: { kind: "threshold" as const, min_ppm: 30_000 } };
  it("write the mode and the match-id suffix, and the mode parses back", () => {
    expect(modeName(both)).toBe("sample:t500000:th30000");
    expect(matchSuffix(both)).toBe("-t500000-th30000");
    expect(matchSuffix({ temperature_ppm: 250_000 })).toBe("-t250000");
    expect(matchSuffix({ truncation: both.truncation })).toBe("-th30000");
    expect(matchSuffix({})).toBe("");
    expect(modeName({})).toBe("sample");
    for (const text of ["sample:t500000", "sample:th30000", "sample:t500000:th30000"]) expect(modeName(parseExtraMode(text) as never)).toBe(text);
  });
});

describe("parseModes", () => {
  it("keeps the extra modes in order, each once, and ignores the defaults", () => {
    expect(parseModes(["argmax", "sample", "sample:th30000", "sample:t500000"])).toEqual([{ truncation: { kind: "threshold", min_ppm: 30_000 } }, { temperature_ppm: 500_000 }]);
    expect(parseModes([])).toEqual([]);
    expect(() => parseModes(["sample:t500000", "sample:t500000"])).toThrow(/twice/);
    expect(() => parseModes(["argmax", "argmax"])).toThrow(/twice/);
    expect(() => parseModes(["nonsense"])).toThrow(ProgressModeError);
  });
});
