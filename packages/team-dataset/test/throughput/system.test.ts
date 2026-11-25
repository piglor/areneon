// The real machine behind the runner: git, /proc, the clock and the process look-alike scan.
import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { PINNED_SHOWDOWN_COMMIT } from "@areneon/showdown-bridge";
import { UsageError } from "../../src/throughput/errors.ts";
import { defaultDeps } from "../../src/throughput/system.ts";

const REPO = new URL("../../../..", import.meta.url).pathname;

describe("defaultDeps", () => {
  const deps = defaultDeps(REPO, ["a-pattern-of-the-owners"]);

  it("names the code by git's HEAD and the engine by the pin, and says whether tracked files changed", () => {
    const code = deps.code();
    expect(code.commit).toBe(execFileSync("git", ["-C", REPO, "rev-parse", "HEAD"], { encoding: "utf8" }).trim());
    expect(code.engine_pin).toBe(PINNED_SHOWDOWN_COMMIT);
    expect(typeof code.tree_clean).toBe("boolean");
  });

  it("refuses a repository root that is not a git checkout as a usage error", () => {
    expect(() => defaultDeps("/nonexistent/checkout").code()).toThrow(UsageError);
  });

  it("describes the machine from /proc", () => {
    const machine = deps.machineInfo();
    expect(machine.logical_cpus).toBeGreaterThan(0);
    expect(machine.mem_total_kib).toBeGreaterThan(0);
    expect(machine.cpu_model).toMatch(/^[ -~]+$/);
    expect(deps.pageBytes).toBe(4096);
  });

  it("samples the machine, with no CPU busy share at the first sample", () => {
    const sample = deps.machine.sample(null);
    expect(sample.mem_available_kib).toBeGreaterThan(0);
    expect(sample.cpu_busy_ppm).toBeNull();
    expect(deps.machine.sample({ util: 3, memMib: 4 })).toMatchObject({ gpu_util_pct: 3, gpu_mem_used_mib: 4 });
  });

  it("finds a running look-alike by the busy patterns (the owner's too), and leaves out the pids it is told to", async () => {
    const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 5000)", "a-pattern-of-the-owners"], { stdio: "ignore" });
    try {
      await new Promise((resolve) => setTimeout(resolve, 200));
      const found = deps.machine.others(new Set(), null);
      expect(found.some((command) => command.includes("a-pattern-of-the-owners"))).toBe(true);
      expect(deps.machine.others(new Set([child.pid as number]), null).some((command) => command.includes("a-pattern-of-the-owners"))).toBe(false);
    } finally {
      child.kill("SIGKILL");
    }
  });

  it("has a clock and a sleep", async () => {
    const before = deps.mono();
    await deps.sleep(20);
    expect(deps.mono() - before).toBeGreaterThanOrEqual(15);
    expect(Math.abs(deps.now() - Date.now())).toBeLessThan(1000);
  });
});
