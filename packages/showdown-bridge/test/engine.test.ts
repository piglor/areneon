import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_SHOWDOWN_DIR, EngineError, loadEngine, PINNED_SHOWDOWN_COMMIT } from "../src/index.ts";

describe("loadEngine", () => {
  it("loads the pinned build, matching the vendor submodule commit", () => {
    const engine = loadEngine();
    expect(engine.commit).toBe(PINNED_SHOWDOWN_COMMIT);
    const gitlink = execFileSync("git", ["ls-files", "--stage", "vendor/pokemon-showdown"], {
      cwd: join(DEFAULT_SHOWDOWN_DIR, "..", ".."),
      encoding: "utf8",
    });
    expect(gitlink).toContain(PINNED_SHOWDOWN_COMMIT);
  });

  it("refuses a checkout at another commit", () => {
    expect(() => loadEngine(DEFAULT_SHOWDOWN_DIR, "0".repeat(40))).toThrow(
      /expected pinned 0{40}/,
    );
  });

  it("refuses a directory without a Showdown build", () => {
    const empty = mkdtempSync(join(tmpdir(), "areneon-engine-"));
    expect(() => loadEngine(empty)).toThrow(EngineError);
  });
});
