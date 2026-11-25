import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { PINNED_NINJASK_COMMIT } from "../../showdown-bridge/src/ninjask-pin.ts";

// Always runs (no engine, no build): the committed ledger of the native probe names the commit of the vendored fork, so a
// bump of `vendor/ninjask` without a new fresh corpus and ledger fails here instead of leaving an admission of other code.
const LEDGER = new URL("../../../config/backends/gen9championsvgc2026regmc.native-probe.admission.jsonl", import.meta.url);
const REPORT = new URL("../../../config/backends/gen9championsvgc2026regmc.native-probe.admission-report.json", import.meta.url);
const ROOT = new URL("../../../", import.meta.url).pathname;

describe("the committed native-probe ledger", () => {
  const header = JSON.parse(readFileSync(LEDGER, "utf8").split("\n")[0] as string) as { header: { backend: { backend_commit: string; kind: string } } };

  it("is for the vendored fork's commit", () => {
    const tree = execFileSync("git", ["ls-tree", "HEAD", "vendor/ninjask"], { cwd: ROOT, encoding: "utf8" });
    const gitlink = /^160000 commit ([0-9a-f]{40})\t/.exec(tree)?.[1];
    expect(gitlink).toMatch(/^[0-9a-f]{40}$/);
    expect(header.header.backend.kind).toBe("native-probe");
    expect(header.header.backend.backend_commit).toBe(gitlink);
    expect(PINNED_NINJASK_COMMIT).toBe(gitlink);
  });

  it("has a report for the same ledger, with no difference in its fresh corpus", () => {
    const report = JSON.parse(readFileSync(REPORT, "utf8")) as { fresh: { differences: number; count_mismatches: number; answered: number } };
    expect(report.fresh.differences).toBe(0);
    expect(report.fresh.count_mismatches).toBe(0);
    expect(report.fresh.answered).toBeGreaterThanOrEqual(200000);
  });
});
