// A stand-in arm for the throughput harness tests: a process tree with a known amount of CPU burnt and known counts, run as
//   node fake-arm.ts --children N --burn-ms M --hold-ms H --games G --decisions D [--report FILE] [--exit CODE] [--survivor]
// Every process burns M ms of CPU (it spins until process.cpuUsage() says so), the root waits H ms without CPU, then writes an ArmReport.
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    children: { type: "string", default: "0" },
    "burn-ms": { type: "string", default: "0" },
    "hold-ms": { type: "string", default: "0" },
    games: { type: "string", default: "0" },
    decisions: { type: "string", default: "0" },
    report: { type: "string" },
    exit: { type: "string", default: "0" },
    survivor: { type: "boolean", default: false },
    "survivor-file": { type: "string" },
    leaf: { type: "boolean", default: false },
    gpu: { type: "boolean", default: false },
  },
});

const burn = (ms: number): void => {
  const target = ms * 1000;
  for (let usage = process.cpuUsage(); usage.user + usage.system < target; usage = process.cpuUsage()) {
    // spin
  }
};
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const burnMs = Number(values["burn-ms"]);
if (values.leaf === true) {
  burn(burnMs);
  process.exit(0);
}

const self = process.argv[1] as string;
const children = Array.from({ length: Number(values.children) }, () => new Promise<void>((resolve) => spawn(process.execPath, [self, "--leaf", "--burn-ms", String(burnMs)], { stdio: "ignore" }).on("exit", () => resolve())));
if (values.survivor === true) {
  const stray = spawn(process.execPath, ["-e", "setTimeout(() => {}, 60000)"], { stdio: "ignore" }); // outlives the arm, in its process group
  stray.unref();
  if (values["survivor-file"] !== undefined) writeFileSync(values["survivor-file"], String(stray.pid));
}
burn(burnMs);
await Promise.all(children);
await sleep(Number(values["hold-ms"]));
if (values.report !== undefined) {
  const hash = (c: string): string => c.repeat(64);
  writeFileSync(
    values.report,
    JSON.stringify({
      schema_version: 1,
      arm: { id: "fake", label: "Fake arm", parameters: [{ name: "children", value: Number(values.children) }] },
      plan: { pool_id: "fake-pool", pool_hash: hash("1"), plan_hash: null, config_hash: null },
      counts: { games: Number(values.games), games_forfeited: 0, games_on_showdown: 0, decisions: Number(values.decisions), scripted_decisions: 0 },
      phases: [{ name: "play", kind: "play", wall_ms: Number(values["hold-ms"]), cpu_ms: null }],
      stages: [{ name: "burn", ms: burnMs }],
      gpu_used: values.gpu === true,
    }),
  );
}
process.exit(Number(values.exit));
