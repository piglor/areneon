// A stand-in for play-generation.ts, for the throughput harness tests: it takes the same flags, burns a little CPU, writes the run
// record a real play leaves in its --out directory and prints the summary and profile lines.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    config: { type: "string" }, generation: { type: "string" }, "policy-id": { type: "string" }, preview: { type: "string" }, battle: { type: "string" },
    "run-dir": { type: "string" }, "engine-data": { type: "string" }, "encoder-version": { type: "string" }, "worker-python": { type: "string" },
    out: { type: "string" }, "policy-workers": { type: "string" }, "game-processes": { type: "string" }, profile: { type: "boolean" },
  },
});
const burn = (ms: number): void => {
  for (let usage = process.cpuUsage(); usage.user + usage.system < ms * 1000; usage = process.cpuUsage()) {
    // spin
  }
};
burn(150); // long enough for the harness to walk the tree at least once
const sample = JSON.parse(readFileSync(new URL("../../../../fixtures/contracts/v1/league-run-record/valid/tiny.json", import.meta.url), "utf8")) as Record<string, unknown>;
const out = values.out as string;
mkdirSync(join(out, "episodes"), { recursive: true });
writeFileSync(join(out, "league-run.json"), JSON.stringify({ ...sample, pair_policy_id: values["policy-id"] }));
console.log(JSON.stringify({ generation: 1, run_config_hash: sample.run_config_hash, training_set_hash: "d".repeat(64), examples: 1000 + Number(values["game-processes"]) }));
console.log(JSON.stringify({ profile: { decisions: 1500, game_processes: { processes: Number(values["game-processes"]), cpu_ms: 10, e_sim_ms: 1, e_judge_ms: 2, e_probe_ms: 3, o_obs_ms: 4, ipc_wait_ms: 5, pool_load_ms: 40, merge_ms: 6 }, policy_workers: { workers: 1, queries: 1, cpu_ms: 1, w_enc_ms: 7, w_fwd_ms: 8 } } }));
