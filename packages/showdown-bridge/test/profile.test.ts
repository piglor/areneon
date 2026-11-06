import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { addSince, countDecisions, enableProfile, processFigures, profiling, readProfile, resetProfile, startTimer, timed, timedAsync, writeProcessFigures } from "../src/profile.ts";

const busy = (ms: number): void => {
  const until = performance.now() + ms;
  while (performance.now() < until) {
    // wait
  }
};

afterEach(() => resetProfile());

describe("the profile timers", () => {
  it("do nothing and read no clock while profiling is off", async () => {
    expect(profiling()).toBe(false);
    expect(startTimer()).toBe(0);
    addSince("e_sim", 0);
    expect(timed("e_sim", () => 7)).toBe(7);
    expect(await timedAsync("e_probe", () => Promise.resolve("x"))).toBe("x");
    countDecisions(5);
    expect(processFigures(null)).toMatchObject({ e_sim_ms: 0, e_probe_ms: 0, decisions: 0 });
  });

  it("add the wall time of a sync call, an async call and a mark to their phases", async () => {
    enableProfile();
    expect(profiling()).toBe(true);
    expect(timed("e_sim", () => (busy(12), "done"))).toBe("done");
    expect(() => timed("e_judge", () => (busy(6), (() => { throw new Error("boom"); })()))).toThrow("boom");
    await timedAsync("ipc_wait", async () => busy(8));
    await expect(timedAsync("e_probe", () => Promise.reject(new Error("late")))).rejects.toThrow("late");
    const mark = startTimer();
    busy(5);
    addSince("o_obs", mark);
    countDecisions(3);
    countDecisions(4);
    const figures = processFigures(1234);
    expect(figures.e_sim_ms).toBeGreaterThanOrEqual(12);
    expect(figures.e_judge_ms).toBeGreaterThanOrEqual(6); // a throw is still timed
    expect(figures.ipc_wait_ms).toBeGreaterThanOrEqual(8);
    expect(figures.o_obs_ms).toBeGreaterThanOrEqual(5);
    expect(figures.pool_load_ms).toBe(0);
    expect(figures.decisions).toBe(7);
    expect(figures.peak_rss_bytes).toBe(1234);
    expect(Number.isInteger(figures.cpu_ms) && figures.cpu_ms >= 0).toBe(true);
    expect("peak_rss_bytes" in processFigures(null)).toBe(false);
  });

  it("are summed from the files of the game processes and the workers, in name order", () => {
    const dir = mkdtempSync(join(tmpdir(), "areneon-profile-test-"));
    try {
      const game = (cpu: number, decisions: number, rss?: number) => ({ e_sim_ms: 1, e_judge_ms: 2, e_probe_ms: 3, o_obs_ms: 4, ipc_wait_ms: 5, pool_load_ms: 6, merge_ms: 7, cpu_ms: cpu, ...(rss === undefined ? {} : { peak_rss_bytes: rss }), decisions });
      writeProcessFigures(join(dir, "nested", "profile"), 0, game(1, 1)); // creates its directory
      writeProcessFigures(dir, 1, game(20, 2, 99));
      writeProcessFigures(dir, 0, game(10, 1));
      for (const [pid, cpu] of [[300, 5], [200, 7]] as const) {
        writeFileSync(join(dir, `worker-${pid}.json`), JSON.stringify({ cpu_ms: cpu, peak_rss_bytes: pid, queries: 2, w_enc_ms: 3, w_fwd_ms: 4, w_enc_cpu_ms: 1, w_fwd_cpu_ms: 2 }));
      }
      mkdirSync(join(dir, "ignored"));
      writeFileSync(join(dir, "notes.txt"), "not a figure");
      expect(readProfile(dir)).toEqual({
        decisions: 3,
        game_processes: {
          processes: 2, cpu_ms: 30, e_sim_ms: 2, e_judge_ms: 4, e_probe_ms: 6, o_obs_ms: 8, ipc_wait_ms: 10, pool_load_ms: 12, merge_ms: 14,
          per_process: [{ cpu_ms: 10, decisions: 1 }, { cpu_ms: 20, peak_rss_bytes: 99, decisions: 2 }],
        },
        policy_workers: {
          workers: 2, queries: 4, cpu_ms: 12, w_enc_ms: 6, w_fwd_ms: 8, w_enc_cpu_ms: 2, w_fwd_cpu_ms: 4,
          per_process: [{ cpu_ms: 7, peak_rss_bytes: 200 }, { cpu_ms: 5, peak_rss_bytes: 300 }],
        },
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
