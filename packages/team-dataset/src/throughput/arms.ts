// An arm of the throughput comparison: any command that plays a plan and leaves an ArmReport. The harness measures everything else
// (wall clock, host CPU of the process tree, the machine's conditions), so every arm, whatever it computes on, is measured alike.
import { readFileSync } from "node:fs";
import { type ArmReport, readArmReport } from "@areneon/showdown-bridge";
import type { Classifier } from "./monitor.ts";

/** What is known of an arm before it runs: it is also what a run that left no report is recorded with. */
export type ArmDescription = Pick<ArmReport, "arm" | "plan" | "gpu_used">;

/** What the harness hands the arm's collector once the arm has ended. */
export interface CollectContext {
  /** The run directory: the arm's stdout and stderr are in it. */
  runDir: string;
  stdoutPath: string;
  stderrPath: string;
  exitCode: number;
  wallMs: number;
}

export interface ArmSpec {
  command: string;
  args: readonly string[];
  cwd: string;
  env: Record<string, string>;
  description: ArmDescription;
  /** Names a process of the arm by its command line (role in the record: game process, policy worker, ...). */
  classify: Classifier;
  /** The arm's own report of what it played; throws when it left none. */
  collect(context: CollectContext): ArmReport | Promise<ArmReport>;
}

/** A report of a run that counted nothing: games and decisions 0, no phases. */
export const emptyReport = (description: ArmDescription): ArmReport => ({
  schema_version: 1,
  ...description,
  counts: { games: 0, games_forfeited: 0, games_on_showdown: 0, decisions: 0, scripted_decisions: 0 },
  phases: [],
  stages: [],
});

/** The role of a process when the arm names none: the program it runs (the last part of argv[0]; Node's kernel name is not that), else its kernel name, as a token. */
export const defaultClassifier: Classifier = (cmdline, comm) => {
  const program = (cmdline[0] ?? "").split("/").pop() as string;
  const token = (program === "" ? comm : program).replace(/[^A-Za-z0-9._:-]/g, "-").replace(/^[^A-Za-z0-9]/, "p").slice(0, 64);
  return token === "" ? "process" : token;
};

export interface CommandArmOptions {
  command: string;
  args: readonly string[];
  cwd: string;
  env?: Record<string, string>;
  /** The file the command writes its ArmReport to. */
  reportPath: string;
  description: ArmDescription;
  classify?: Classifier;
}

/** Any command that writes an ArmReport to a file: what a native or GPU arm's launcher is. */
export function commandArm(options: CommandArmOptions): ArmSpec {
  return {
    command: options.command,
    args: options.args,
    cwd: options.cwd,
    env: options.env ?? {},
    description: options.description,
    classify: options.classify ?? defaultClassifier,
    collect: () => {
      let text: string;
      try {
        text = readFileSync(options.reportPath, "utf8");
      } catch {
        throw new Error("the arm left no arm report at the path it was given");
      }
      return readArmReport(JSON.parse(text) as unknown);
    },
  };
}
