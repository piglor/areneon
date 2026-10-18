import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { formatLabels, GAPS_DOC, renderGapsDoc } from "../src/gaps-doc.ts";
import {
  DEFAULT_RULES_DIR,
  DEFAULT_SHOWDOWN_DIR,
  formatRulesHash,
  formatRulesRecord,
  loadEngine,
  loadRulesSnapshot,
  rulesSnapshotHash,
  type RulesSnapshot,
} from "../src/index.ts";

const REPO = fileURLToPath(new URL("../../../", import.meta.url));
const FORMAT = "gen9championsvgc2026regmc";
const engine = loadEngine();
const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

/** A scratch copy of the committed rules, with `edit` applied to its gaps directory. */
function withGaps(edit: (gaps: string) => void): RulesSnapshot {
  const dir = mkdtempSync(join(tmpdir(), "areneon-gaps-test-"));
  dirs.push(dir);
  cpSync(DEFAULT_RULES_DIR, dir, { recursive: true });
  edit(join(dir, "gaps"));
  return loadRulesSnapshot(engine.commit, dir);
}

/** The same, with one gap record rewritten. */
const withGap = (file: string, change: (gap: Record<string, unknown>) => void): RulesSnapshot =>
  withGaps((gaps) => {
    const path = join(gaps, file);
    const gap = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
    change(gap);
    writeFileSync(path, JSON.stringify(gap));
  });

const committed = loadRulesSnapshot(engine.commit);

describe("gap ledger doc", () => {
  // Edit rules/gaps/ and run scripts/gaps-doc.ts; never edit the doc.
  it("matches the doc generated from the records", () => {
    expect(readFileSync(GAPS_DOC, "utf8")).toBe(renderGapsDoc(committed, formatLabels(engine, committed)));
  });

  it("names the pin and how the engine labels each format", () => {
    const doc = renderGapsDoc(committed, formatLabels(engine, committed));
    expect(doc).toContain(`\`${engine.commit}\``);
    expect(doc).toContain(`\`${FORMAT}\` (\`${engine.sim.Dex.formats.get(FORMAT).name}\`, mod \`champions\`)`);
  });

  // A pin bump that moves or renames engine code fails here.
  it("points only at engine files, symbols and fixtures that exist", () => {
    const evidence = Object.values(committed.gaps).flatMap((gap) => gap.evidence);
    expect(evidence.length).toBeGreaterThan(0);
    for (const item of evidence) {
      const path = item.kind === "engine" ? join(DEFAULT_SHOWDOWN_DIR, item.path) : join(REPO, item.path);
      expect(existsSync(path), item.path).toBe(true);
      if (item.kind === "engine") {
        const source = readFileSync(path, "utf8");
        for (const symbol of item.symbols) expect(source, `${item.path} ${symbol}`).toMatch(new RegExp(`(?<![\\w$])${symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\w$])`));
      }
    }
  });

  it("lists a gap without evidence as not yet exercised, and tabulates other statuses", () => {
    const snapshot: RulesSnapshot = {
      ...committed,
      gaps: {
        b: { area: "Seen", status: "verified", description: "Confirmed.", evidence: [{ kind: "fixture", path: "fixtures/x" }] },
        a: { area: "Dropped", status: "unsupported", description: "Not run.", evidence: [] },
        c: { area: "Open", status: "unverified", description: "Unknown.", evidence: [] },
      },
    };
    const doc = renderGapsDoc(snapshot, []);
    const table = "| Area | What Showdown does at the pin | Where |\n|---|---|---|\n";
    expect(doc).toContain(`## Verified in-game\n\n${table}| Seen | Confirmed. | fixture \`fixtures/x\` |`);
    expect(doc).toContain(`## Not supported by Areneon\n\n${table}| Dropped | Not run. |  |`);
    expect(doc).toContain("## Unknown or not yet exercised\n\n- Unknown.\n");
    expect(doc).not.toContain("unverified in-game");
  });

  it("omits empty sections", () => {
    expect(renderGapsDoc({ ...committed, gaps: {} }, [])).not.toContain("##");
  });
});

describe("gap records", () => {
  // The ledger is provenance: editing it moves the snapshot, never a
  // battle's replay identity (ADR-007).
  it("never change format_rules_hash", () => {
    const record = { schema_version: 1, gap_id: "zz-new", area: "New", status: "unverified", description: "Added.", evidence: [] };
    const edited = [
      withGap("level.json", (gap) => (gap.status = gap.status === "unsupported" ? "unverified" : "unsupported")),
      withGap("level.json", (gap) => (gap.description = "changed")),
      withGap("level.json", (gap) => (gap.evidence = [])),
      withGaps((gaps) => rmSync(join(gaps, "level.json"))),
      withGaps((gaps) => writeFileSync(join(gaps, "zz-new.json"), JSON.stringify(record))),
    ];
    const identity = (snapshot: RulesSnapshot): string => formatRulesHash(formatRulesRecord(snapshot, FORMAT));
    for (const snapshot of edited) {
      expect(identity(snapshot)).toBe(identity(committed));
      expect(rulesSnapshotHash(snapshot)).not.toBe(rulesSnapshotHash(committed));
    }
  });

  it("fail closed on an unknown status, a misnamed file or unsafe text", () => {
    const rejects = (change: (gap: Record<string, unknown>) => void): void => {
      expect(() => withGap("level.json", change)).toThrow(/gaps\/level\.json/);
    };
    rejects((gap) => (gap.status = "maybe"));
    expect(() => withGap("level.json", (gap) => (gap.gap_id = "other"))).toThrow(/must be named other\.json/);
    // Text is rendered into Markdown table cells and list items.
    rejects((gap) => (gap.description = "two\nlines"));
    rejects((gap) => (gap.description = "a | b"));
    rejects((gap) => (gap.area = " "));
    rejects((gap) => (gap.evidence = [{ kind: "engine", path: "sim/teams.ts", symbols: ["a`b"] }]));
    rejects((gap) => (gap.evidence = [{ kind: "engine", path: "sim/teams.ts", symbols: ["a", "a"] }]));
    const once = { kind: "engine", path: "sim/teams.ts", symbols: [] };
    rejects((gap) => (gap.evidence = [once, once]));
  });

  // Evidence must stay inside the submodule or fixtures/: never private-data/.
  it("reject evidence paths that climb out", () => {
    for (const evidence of [
      { kind: "engine", path: "data/../../../private-data/team.txt", symbols: [] },
      { kind: "engine", path: "./sim/teams.ts", symbols: [] },
      { kind: "fixture", path: "fixtures/../private-data/team.txt" },
      { kind: "fixture", path: "../private-data/team.txt" },
    ]) {
      expect(() => withGap("level.json", (gap) => (gap.evidence = [evidence])), evidence.path).toThrow(/gaps\/level\.json/);
    }
  });

  it("require fixture evidence for a verified gap", () => {
    expect(() => withGap("level.json", (gap) => (gap.status = "verified"))).toThrow(/gaps\/level\.json/);
    const verified = withGap("team-preview.json", (gap) => (gap.status = "verified"));
    expect(verified.gaps["team-preview"]?.status).toBe("verified");
  });
});
