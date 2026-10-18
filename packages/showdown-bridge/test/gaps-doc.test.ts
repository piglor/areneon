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

/** A scratch copy of the committed rules with one gap record rewritten. */
function withGap(file: string, change: (gap: Record<string, unknown>) => void): RulesSnapshot {
  const dir = mkdtempSync(join(tmpdir(), "areneon-gaps-test-"));
  dirs.push(dir);
  cpSync(DEFAULT_RULES_DIR, dir, { recursive: true });
  const path = join(dir, "gaps", file);
  const gap = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  change(gap);
  writeFileSync(path, JSON.stringify(gap));
  return loadRulesSnapshot(engine.commit, dir);
}

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

  it("points only at engine files and fixtures that exist", () => {
    const evidence = Object.values(committed.gaps).flatMap((gap) => gap.evidence);
    expect(evidence.length).toBeGreaterThan(0);
    for (const item of evidence) {
      const path = item.kind === "engine" ? join(DEFAULT_SHOWDOWN_DIR, item.path) : join(REPO, item.path);
      expect(existsSync(path), item.path).toBe(true);
    }
  });

  it("lists a gap without evidence as not yet exercised, and tabulates other statuses", () => {
    const snapshot: RulesSnapshot = {
      ...committed,
      gaps: {
        b: { area: "Pipes | here", status: "verified", description: "a | b", evidence: [{ kind: "fixture", path: "fixtures/x" }] },
        a: { area: "Dropped", status: "unsupported", description: "Not run.", evidence: [] },
        c: { area: "Open", status: "unverified", description: "Unknown.", evidence: [] },
      },
    };
    const doc = renderGapsDoc(snapshot, []);
    expect(doc).toContain("## Verified in-game\n\n| Area | What Showdown does at the pin | Where |\n|---|---|---|\n| Pipes \\| here | a \\| b | fixture `fixtures/x` |");
    expect(doc).toContain("## Not supported by Areneon\n\n| Area | What Showdown does at the pin | Where |\n|---|---|---|\n| Dropped | Not run. |  |");
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
    const edited = [
      withGap("level.json", (gap) => (gap.status = "verified")),
      withGap("level.json", (gap) => (gap.description = "changed")),
      withGap("level.json", (gap) => (gap.evidence = [])),
    ];
    const identity = (snapshot: RulesSnapshot): string => formatRulesHash(formatRulesRecord(snapshot, FORMAT));
    for (const snapshot of edited) {
      expect(identity(snapshot)).toBe(identity(committed));
      expect(rulesSnapshotHash(snapshot)).not.toBe(rulesSnapshotHash(committed));
    }
  });

  it("fail closed on an unknown status, a misnamed file or a bad evidence path", () => {
    expect(() => withGap("level.json", (gap) => (gap.status = "maybe"))).toThrow(/gaps\/level\.json/);
    expect(() => withGap("level.json", (gap) => (gap.gap_id = "other"))).toThrow(/must be named other\.json/);
    expect(() =>
      withGap("level.json", (gap) => (gap.evidence = [{ kind: "fixture", path: "../private-data/team.txt" }])),
    ).toThrow(/gaps\/level\.json/);
  });
});
