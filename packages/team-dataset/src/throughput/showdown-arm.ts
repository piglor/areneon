// The Showdown arm of the throughput comparison: the current path (Node game processes, the pinned engine, the feature builder, the
// policy workers) run exactly as the generation loop runs it, as `play-generation.ts` with the loop's own flags, on a frozen pair and
// a plan the run config names. Nothing in it is modified; it only reads what the script prints and the run record it leaves.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { type ArmReport, readLeagueRunRecord } from "@areneon/showdown-bridge";
import { type ArmDescription, type ArmSpec, type CollectContext, defaultClassifier } from "./arms.ts";

export interface ShowdownArmOptions {
  /** The checkout the play script runs from (the working directory of the arm). */
  repoRoot: string;
  node: string;
  /** Run the arm under `nice -n N`; null leaves its priority alone (a gate measurement does). */
  nice: number | null;
  /** The run config (a TrainingRunConfig file) and the run directory holding `checkpoints/<id>.safetensors`, read only. */
  config: string;
  runDir: string;
  preview: string;
  battle: string;
  policyId: string;
  encoderVersion: string;
  engineData: string;
  workerPython: string;
  generation: number;
  gameProcesses: number;
  policyWorkers: number;
  nodeHeapMb: number | null;
  poolId: string;
  poolHash: string | null;
  /** The play script's `--out`: the generation it writes (episodes, stream, run record) goes here. */
  outDir: string;
  /** The play script, relative to the repository root (or absolute). */
  playScript: string;
}

/** The pool a run config names: its team dataset's directory name and hash, or the interim pool. */
export function poolOf(config: { team_dataset?: { path: string; dataset_hash: string } }): { poolId: string; poolHash: string | null } {
  if (config.team_dataset === undefined) return { poolId: "interim", poolHash: null };
  const name = basename(config.team_dataset.path.replace(/\/+$/, "")).replace(/[^A-Za-z0-9._:-]/g, "-");
  return { poolId: name, poolHash: config.team_dataset.dataset_hash };
}

/** The roles of the Showdown path's processes. */
export function classifyShowdown(cmdline: readonly string[], comm: string): string {
  const text = cmdline.join(" ");
  if (text.includes("areneon_ml.worker")) return "policy_worker";
  if (text.includes("play-generation.ts")) return "game_process";
  return defaultClassifier(cmdline, comm);
}

type Figures = Record<string, number>;
export interface PlayProfile {
  decisions: number;
  game_processes: Figures & { processes: number };
  policy_workers: Figures;
}
export interface PlayOutput {
  summary: { run_config_hash: string; examples: number };
  profile: PlayProfile | undefined;
}

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const integers = (value: unknown, keys: readonly string[]): boolean => isObject(value) && keys.every((key) => Number.isSafeInteger(value[key]) && (value[key] as number) >= 0);

/** What the play script printed: the summary line (`examples` is the training set's size) and, with `--profile`, the profile line. */
export function parsePlayOutput(stdout: string): PlayOutput {
  const objects: Record<string, unknown>[] = [];
  for (const line of stdout.split("\n")) {
    try {
      const value: unknown = JSON.parse(line);
      if (isObject(value)) objects.push(value);
    } catch {
      // a line that is not JSON is not one of the script's records
    }
  }
  const summary = objects.find((object) => "examples" in object && "run_config_hash" in object);
  if (summary === undefined || !Number.isSafeInteger(summary.examples) || typeof summary.run_config_hash !== "string") throw new Error("the play script printed no summary line");
  const found = objects.find((object) => "profile" in object)?.profile;
  if (found === undefined) return { summary: summary as PlayOutput["summary"], profile: undefined };
  const profile = found as Record<string, unknown>;
  const ok = integers(profile, ["decisions"]) && integers(profile.game_processes, ["processes", "e_sim_ms", "e_judge_ms", "e_probe_ms", "o_obs_ms", "pool_load_ms", "merge_ms"]) && integers(profile.policy_workers, ["w_enc_ms", "w_fwd_ms"]);
  if (!ok) throw new Error("the play script's profile line is not in the shape the harness reads");
  return { summary: summary as PlayOutput["summary"], profile: profile as unknown as PlayProfile };
}

/** The arm: `play-generation.ts` run as the loop runs it. */
export function showdownArm(options: ShowdownArmOptions): ArmSpec & { description: ArmDescription } {
  const script = resolve(options.repoRoot, options.playScript);
  const play = [
    ...(options.nodeHeapMb === null ? [] : [`--max-old-space-size=${options.nodeHeapMb}`]),
    script,
    "--config", options.config,
    "--generation", String(options.generation),
    "--policy-id", options.policyId,
    "--preview", options.preview,
    "--battle", options.battle,
    "--run-dir", options.runDir,
    "--engine-data", options.engineData,
    "--encoder-version", options.encoderVersion,
    "--worker-python", options.workerPython,
    "--out", options.outDir,
    "--policy-workers", String(options.policyWorkers),
    "--game-processes", String(options.gameProcesses),
    "--profile",
  ];
  const parameters = [
    { name: "game_processes", value: options.gameProcesses },
    { name: "generation", value: options.generation },
    ...(options.nodeHeapMb === null ? [] : [{ name: "node_heap_mb", value: options.nodeHeapMb }]),
    { name: "policy_workers", value: options.policyWorkers },
  ];
  const description: ArmDescription = {
    arm: { id: "showdown", label: `Showdown path: play-generation.ts, ${options.gameProcesses} game processes x ${options.policyWorkers} policy worker${options.policyWorkers === 1 ? "" : "s"}`, parameters },
    plan: { pool_id: options.poolId, pool_hash: options.poolHash, plan_hash: null, config_hash: null },
    gpu_used: false,
  };
  return {
    command: options.nice === null ? options.node : "nice",
    args: options.nice === null ? play : ["-n", String(options.nice), options.node, ...play],
    cwd: options.repoRoot,
    env: {},
    description,
    classify: classifyShowdown,
    collect: (context: CollectContext) => Promise.resolve().then(() => collectShowdown(options, description, context)),
  };
}

function collectShowdown(options: ShowdownArmOptions, description: ArmDescription, context: CollectContext): ArmReport {
  const { summary, profile } = parsePlayOutput(readFileSync(context.stdoutPath, "utf8"));
  let text: string;
  try {
    text = readFileSync(join(options.outDir, "league-run.json"), "utf8");
  } catch {
    throw new Error("the play left no league-run.json in its output directory");
  }
  const run = readLeagueRunRecord(JSON.parse(text) as unknown);
  if (run.run_config_hash !== summary.run_config_hash) throw new Error("the play's run config hash differs between its summary line and its run record");
  if (run.pair_policy_id !== options.policyId) throw new Error("the play's pair policy id differs from the policy id the arm was given");
  const scripted = profile === undefined ? 0 : Math.max(0, profile.decisions - summary.examples);
  const stages =
    profile === undefined
      ? []
      : [
          { name: "engine_step", ms: profile.game_processes.e_sim_ms as number },
          { name: "features", ms: profile.game_processes.e_probe_ms as number },
          { name: "legal_mask", ms: profile.game_processes.e_judge_ms as number },
          { name: "observation", ms: profile.game_processes.o_obs_ms as number },
          { name: "policy_encode", ms: profile.policy_workers.w_enc_ms as number },
          { name: "policy_forward", ms: profile.policy_workers.w_fwd_ms as number },
        ];
  let phases: { name: string; kind: "warmup" | "play" | "teardown"; wall_ms: number; cpu_ms: number | null }[] = [];
  if (profile !== undefined) {
    const pool = Math.floor((profile.game_processes.pool_load_ms as number) / Math.max(1, profile.game_processes.processes));
    const merge = profile.game_processes.merge_ms as number;
    phases = [
      { name: "pool_load", kind: "warmup", wall_ms: pool, cpu_ms: null },
      { name: "play", kind: "play", wall_ms: Math.max(0, context.wallMs - pool - merge), cpu_ms: null },
      { name: "merge", kind: "teardown", wall_ms: merge, cpu_ms: null },
    ];
  }
  return {
    schema_version: 1,
    arm: description.arm,
    plan: { ...description.plan, plan_hash: run.plan_hash, config_hash: run.run_config_hash },
    counts: { games: run.games, games_forfeited: run.forfeits, games_on_showdown: run.games, decisions: summary.examples, scripted_decisions: scripted },
    phases,
    stages,
    gpu_used: false,
  };
}

export interface IdsRequest {
  python: string;
  repoRoot: string;
  config: string;
  engineData: string;
  preview: string;
  battle: string;
}

export type RunCommand = (command: string, args: readonly string[], env: Record<string, string>) => string;
const spawnCommand: RunCommand = (command, args, env) => execFileSync(command, [...args], { env: { ...process.env, ...env }, encoding: "utf8" });

/**
 * The policy id and encoder version the play script needs, from the Python package that defines them (the loop computes them the
 * same way). Run it before the harness starts the arm: it spawns a process, which must not be reaped while the arm's CPU accumulates.
 */
export function resolveIds(request: IdsRequest, run: RunCommand = spawnCommand): { policyId: string; encoderVersion: string } {
  const root = request.repoRoot;
  const out = run(request.python, ["-s", "-P", "-m", "areneon_ml.loop.play_ids", "--config", request.config, "--engine-data", request.engineData, "--preview", request.preview, "--battle", request.battle], { PYTHONPATH: `${root}/ml:${root}/packages/schema/python:${root}` });
  let found: unknown;
  try {
    found = JSON.parse(out);
  } catch {
    found = undefined;
  }
  if (!isObject(found) || typeof found.policy_id !== "string" || typeof found.encoder_version !== "string") throw new Error("the Python package did not print the policy ids as JSON");
  return { policyId: found.policy_id, encoderVersion: found.encoder_version };
}
