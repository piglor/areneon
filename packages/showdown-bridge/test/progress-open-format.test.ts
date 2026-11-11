// The open-sheet format of `--team-sheets open` is the one the rules snapshot pairs with the run's closed format.
import { describe, expect, it } from "vitest";
import { openFormatOf } from "../src/progress-cli.ts";
import { UsageError } from "../src/generation-cli.ts";
import type { RulesSnapshot } from "../src/rules.ts";

const snapshot = (formats: Record<string, { team_sheets?: "closed" | "open"; regulation: string }>): RulesSnapshot => ({ formats }) as unknown as RulesSnapshot;

describe("openFormatOf", () => {
  it("names the open format of the closed format's regulation", () => {
    const rules = snapshot({ closed: { regulation: "M-C" }, open: { team_sheets: "open", regulation: "M-C" }, other: { regulation: "M-D" }, otherOpen: { team_sheets: "open", regulation: "M-D" } });
    expect(openFormatOf(rules, "closed")).toBe("open");
    expect(openFormatOf(rules, "other")).toBe("otherOpen");
  });

  it("refuses a closed format with no open variant and one with two", () => {
    const none = snapshot({ closed: { regulation: "M-C" } });
    expect(() => openFormatOf(none, "closed")).toThrow(UsageError);
    const two = snapshot({ closed: { regulation: "M-C" }, a: { team_sheets: "open", regulation: "M-C" }, b: { team_sheets: "open", regulation: "M-C" } });
    expect(() => openFormatOf(two, "closed")).toThrow(/exactly one open-sheet variant/);
    expect(() => openFormatOf(none, "unknown")).toThrow(/exactly one/);
  });
});
