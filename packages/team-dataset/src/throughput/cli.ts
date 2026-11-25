// The throughput harness's command line (`scripts/throughput-bench.ts`): measure an arm, check the machine, read records, compare runs.
//
//   run        measure an arm, once per repeat (the Showdown path at one or more numbers of game processes, or any command that leaves an ArmReport)
//   guard      check the ADR-028 idle-machine conditions now
//   verify     read records with the contract reader and print their hashes
//   summarise  the median and range per arm over records or run directories
//
// Exit codes: 0 done and every run valid, 1 a run is void or a record is refused, 2 usage, 3 the machine is not idle (nothing was started).
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { readThroughputRecord, type ThroughputRecord, throughputRecordHash } from "@areneon/showdown-bridge";
import { type ArmSpec, commandArm } from "./arms.ts";
import { UsageError } from "./errors.ts";
import { checkStart, GIB_KIB, type Mode, thresholdsFor } from "./guard.ts";
import { ConditionsNotMet, type Deps, defaultRunOptions, type RunOptions, runArm } from "./runner.ts";
import { poolOf, resolveIds, showdownArm } from "./showdown-arm.ts";
import { defaultDeps } from "./system.ts";
import { renderComparison } from "./summary.ts";

/** Between two looks at a machine that is not idle yet (`--wait-idle-minutes`). */
export const RETRY_MS = 15_000;

export interface CliIo {
  out(line: string): void;
  err(line: string): void;
  /** The checkout the arm runs from by default (and git reads the commit of). */
  repoRoot: string;
  /** The machine; the real one by default. */
  deps?: (options: { repoRoot: string; busyPatterns: readonly string[] }) => Deps;
}

const FLAGS = {
  arm: { type: "string" }, out: { type: "string" }, "run-id": { type: "string" }, repeats: { type: "string" }, mode: { type: "string" },
  "min-mem-gib": { type: "string" }, "busy-pattern": { type: "string", multiple: true }, "no-abort": { type: "boolean" },
  "proc-interval-ms": { type: "string" }, "condition-interval-ms": { type: "string" }, "start-window-ms": { type: "string" },
  "wait-idle-minutes": { type: "string" }, "pool-id": { type: "string" }, "pool-hash": { type: "string" }, "keep-arm-output": { type: "boolean" },
  "repo-root": { type: "string" },
  // the command arm
  "arm-id": { type: "string" }, "arm-label": { type: "string" }, "arm-report": { type: "string" }, "arm-param": { type: "string", multiple: true }, gpu: { type: "boolean" },
  // the Showdown arm
  config: { type: "string" }, "run-dir": { type: "string" }, preview: { type: "string" }, battle: { type: "string" }, "engine-data": { type: "string" },
  "worker-python": { type: "string" }, generation: { type: "string" }, "game-processes": { type: "string" }, "policy-workers": { type: "string" },
  "policy-id": { type: "string" }, "encoder-version": { type: "string" }, "ids-python": { type: "string" }, "node-heap-mb": { type: "string" },
  nice: { type: "string" }, node: { type: "string" }, "play-script": { type: "string" },
} as const;
type Values = ReturnType<typeof parseArgs<{ options: typeof FLAGS; allowPositionals: true; strict: true }>>["values"];

const USAGE = "usage: throughput-bench.ts run|guard|verify|summarise [options] (see docs/architecture/throughput.md)";

function integer(name: string, text: string | undefined, fallback: number, min: number, max = Number.MAX_SAFE_INTEGER): number {
  if (text === undefined) return fallback;
  const value = Number(text);
  if (!/^\d+$/.test(text) || value < min || value > max) throw new UsageError(`--${name} must be an integer from ${min} to ${max}`);
  return value;
}

function need(values: Values, name: keyof Values & string): string {
  const value = values[name];
  if (typeof value !== "string" || value === "") throw new UsageError(`--${name} is required`);
  return value;
}

const modeOf = (values: Values): Mode => {
  const mode = values.mode ?? "idle";
  if (mode !== "idle" && mode !== "non-idle") throw new UsageError("--mode must be idle or non-idle");
  return mode;
};

/** The thresholds of the mode; `--min-mem-gib` changes the memory MemAvailable must have at the start. */
function thresholdsOf(values: Values, mode: Mode): RunOptions["thresholds"] {
  const gib = integer("min-mem-gib", values["min-mem-gib"], 8, 1);
  return thresholdsFor(mode, { min_mem_available_start_kib: gib * GIB_KIB });
}

interface Setup {
  /** One arm per run to make, in the order to run them (interleaved rounds), with the id of the run. */
  runs: { id: string; arm: ArmSpec }[];
}

function commandSetup(values: Values, positionals: readonly string[], runId: string, repeats: number, repoRoot: string): Setup {
  const reportPath = resolve(need(values, "arm-report"));
  const id = need(values, "arm-id");
  const label = need(values, "arm-label");
  const parameters = (values["arm-param"] ?? []).map((text) => {
    const match = /^([A-Za-z0-9][A-Za-z0-9._:-]*)=(\d+)$/.exec(text);
    if (match === null) throw new UsageError("--arm-param must be name=integer");
    return { name: match[1] as string, value: Number(match[2]) };
  });
  const [command, ...args] = positionals;
  if (command === undefined) throw new UsageError("the command to run follows --");
  const plan = { pool_id: values["pool-id"] ?? "unnamed", pool_hash: values["pool-hash"] ?? null, plan_hash: null, config_hash: null };
  const description = { arm: { id, label, parameters: parameters.sort((a, b) => (a.name < b.name ? -1 : 1)) }, plan, gpu_used: values.gpu === true };
  return { runs: Array.from({ length: repeats }, (_, i) => ({ id: `${runId}-r${i + 1}`, arm: commandArm({ command, args, cwd: repoRoot, reportPath, description }) })) };
}

function showdownSetup(values: Values, runId: string, repeats: number, repoRoot: string, outDir: string, makeIds: () => { policyId: string; encoderVersion: string }): Setup {
  const configPath = resolve(need(values, "config"));
  let configJson: { team_dataset?: { path: string; dataset_hash: string } };
  try {
    configJson = JSON.parse(readFileSync(configPath, "utf8")) as typeof configJson;
  } catch {
    throw new UsageError("cannot read the run config");
  }
  const processes = (values["game-processes"] ?? "4").split(",").map((text) => integer("game-processes", text.trim(), 4, 1, 64));
  const nice = values.nice === undefined ? null : integer("nice", values.nice, 0, 0, 19);
  const nodeHeapMb = values["node-heap-mb"] === undefined ? null : integer("node-heap-mb", values["node-heap-mb"], 0, 64);
  const ids = values["policy-id"] !== undefined && values["encoder-version"] !== undefined ? { policyId: values["policy-id"], encoderVersion: values["encoder-version"] } : undefined;
  if (ids === undefined && values["ids-python"] === undefined) throw new UsageError("give --policy-id and --encoder-version, or --ids-python");
  const pool = poolOf(configJson);
  const common = {
    repoRoot,
    node: values.node ?? process.execPath,
    nice,
    config: configPath,
    runDir: need(values, "run-dir"),
    preview: need(values, "preview"),
    battle: need(values, "battle"),
    engineData: need(values, "engine-data"),
    workerPython: need(values, "worker-python"),
    generation: integer("generation", values.generation, 1, 1),
    policyWorkers: integer("policy-workers", values["policy-workers"], 1, 1, 16),
    nodeHeapMb,
    poolId: values["pool-id"] ?? pool.poolId,
    poolHash: values["pool-hash"] ?? pool.poolHash,
    playScript: values["play-script"] ?? "packages/team-dataset/scripts/play-generation.ts",
  };
  const resolved = ids ?? makeIds();
  const runs: Setup["runs"] = [];
  for (let round = 1; round <= repeats; round++) {
    for (const gameProcesses of processes) {
      const id = `${runId}-gp${gameProcesses}-r${round}`;
      runs.push({ id, arm: showdownArm({ ...common, ...resolved, gameProcesses, outDir: join(outDir, id, "arm") }) });
    }
  }
  return { runs };
}

const seconds = (ms: number): string => `${(ms / 1000).toFixed(1)} s`;

async function run(values: Values, positionals: readonly string[], io: CliIo): Promise<number> {
  const armKind = need(values, "arm");
  if (armKind !== "showdown" && armKind !== "command") throw new UsageError("unknown arm: it is showdown or command");
  const outDir = resolve(need(values, "out"));
  const mode = modeOf(values);
  const repeats = integer("repeats", values.repeats, 1, 1, 1000);
  const runId = values["run-id"] ?? armKind;
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,100}$/.test(runId)) throw new UsageError("--run-id must be letters, digits and . _ : -");
  const repoRoot = resolve(values["repo-root"] ?? io.repoRoot);
  const busyPatterns = values["busy-pattern"] ?? [];
  const waitMinutes = integer("wait-idle-minutes", values["wait-idle-minutes"], 0, 0, 24 * 60);
  const options = {
    mode,
    thresholds: thresholdsOf(values, mode),
    procIntervalMs: integer("proc-interval-ms", values["proc-interval-ms"], 1000, 10),
    conditionIntervalMs: integer("condition-interval-ms", values["condition-interval-ms"], 60_000, 10),
    startWindowMs: integer("start-window-ms", values["start-window-ms"], 5000, 1),
    abortOnMemory: values["no-abort"] !== true,
  };
  const setup = armKind === "command"
    ? commandSetup(values, positionals, runId, repeats, repoRoot)
    : showdownSetup(values, runId, repeats, repoRoot, outDir, () => {
        try {
          return resolveIds({ python: need(values, "ids-python"), repoRoot, config: resolve(need(values, "config")), engineData: need(values, "engine-data"), preview: need(values, "preview"), battle: need(values, "battle") });
        } catch (error) {
          throw new UsageError(`cannot compute the policy ids: ${(error as Error).message.split("\n")[0]}`);
        }
      });
  if (existsSync(outDir) && !statSync(outDir).isDirectory()) throw new UsageError("--out is not a directory");
  const deps = (io.deps ?? ((o) => defaultDeps(o.repoRoot, o.busyPatterns)))({ repoRoot, busyPatterns });
  const records: ThroughputRecord[] = [];
  let code = 0;
  for (const { id, arm } of setup.runs) {
    const runOptions = defaultRunOptions({ ...options, runId: id, outDir: join(outDir, id) });
    let result: Awaited<ReturnType<typeof runArm>> | undefined;
    for (let attempt = 1; result === undefined; attempt++) {
      try {
        result = await runArm(arm, runOptions, deps);
      } catch (error) {
        if (!(error instanceof ConditionsNotMet)) throw error;
        io.err(`${id}: attempt ${attempt}: ${error.message}`);
        if (attempt > (waitMinutes * 60_000) / RETRY_MS) {
          io.out(renderComparison(records));
          return 3;
        }
        await deps.sleep(RETRY_MS);
      }
    }
    const { record } = result;
    records.push(record);
    if (!record.valid) code = 1;
    if (armKind === "showdown" && values["keep-arm-output"] !== true) rmSync(join(outDir, id, "arm"), { recursive: true, force: true });
    const rate = record.rates.decisions_per_core_hour.toLocaleString("en-US");
    io.out(`${id}: ${record.valid ? "VALID" : `VOID (${record.void_reasons.join(", ")})`}, ${rate} decisions per host-core-hour (${record.report.counts.decisions} decisions, ${seconds(record.cpu.total_ms)} host CPU, ${seconds(record.window.wall_ms)} wall); record ${join(outDir, id, "record.json")} sha256 ${result.hash}`);
    if (result.collectError !== null) io.err(`${id}: the arm's report could not be read: ${result.collectError}`);
  }
  io.out(renderComparison(records));
  return code;
}

async function guard(values: Values, io: CliIo): Promise<number> {
  const mode = modeOf(values);
  const thresholds = thresholdsOf(values, mode);
  const repoRoot = resolve(values["repo-root"] ?? io.repoRoot);
  const busyPatterns = values["busy-pattern"] ?? [];
  const deps = (io.deps ?? ((o) => defaultDeps(o.repoRoot, o.busyPatterns)))({ repoRoot, busyPatterns });
  const sample = await deps.machine.sampleOverWindow(await deps.gpu.once(), integer("start-window-ms", values["start-window-ms"], 5000, 1), deps.sleep);
  const failures = checkStart(sample, deps.machine.others(new Set([process.pid]), null), thresholds);
  io.out(`load average ${sample.load1_centi / 100}, MemAvailable ${(sample.mem_available_kib / GIB_KIB).toFixed(1)} GiB, swap in use ${sample.swap_used_kib} KiB, cpu busy ${sample.cpu_busy_ppm === null ? "n/a" : `${sample.cpu_busy_ppm / 10_000}%`}`);
  if (failures.length === 0) {
    io.out(`${mode}: every condition holds`);
    return 0;
  }
  for (const failure of failures) io.err(`not idle: ${failure.detail}`);
  return 3;
}

function recordFiles(paths: readonly string[]): string[] {
  return paths.flatMap((path) => {
    if (!statSync(path).isDirectory()) return [path];
    const here = join(path, "record.json");
    return existsSync(here) ? [here] : readdirSync(path).sort().map((name) => join(path, name, "record.json")).filter((file) => existsSync(file));
  });
}

function verify(paths: readonly string[], io: CliIo): number {
  if (paths.length === 0) throw new UsageError("verify takes record files");
  let code = 0;
  for (const path of paths) {
    try {
      io.out(`${throughputRecordHash(readThroughputRecord(JSON.parse(readFileSync(path, "utf8")) as unknown))} ${path}`);
    } catch (error) {
      io.err(`${path}: ${(error as Error).message}`);
      code = 1;
    }
  }
  return code;
}

function summarise(paths: readonly string[], io: CliIo): number {
  if (paths.length === 0) throw new UsageError("summarise takes record files or run directories");
  try {
    io.out(renderComparison(recordFiles(paths).map((file) => readThroughputRecord(JSON.parse(readFileSync(file, "utf8")) as unknown))));
    return 0;
  } catch (error) {
    io.err((error as Error).message);
    return 1;
  }
}

export async function runThroughputCli(argv: readonly string[], io: CliIo): Promise<number> {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: FLAGS, allowPositionals: true, strict: true });
    const [command, ...rest] = positionals;
    switch (command) {
      case "run":
        return await run(values, rest, io);
      case "guard":
        return await guard(values, io);
      case "verify":
        return verify(rest, io);
      case "summarise":
        return summarise(rest, io);
      case undefined:
        throw new UsageError(USAGE);
      default:
        throw new UsageError(`unknown command ${command}; ${USAGE}`);
    }
  } catch (error) {
    if (error instanceof UsageError || (error instanceof TypeError && /option|argument/i.test(error.message))) {
      io.err(`throughput-bench: ${(error as Error).message}`);
      return 2;
    }
    throw error;
  }
}
