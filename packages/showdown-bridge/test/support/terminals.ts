// Run a script with a chosen mix of terminal and pipe stdio, with or without a
// controlling terminal, for the guard tests. python3's pty module needs no
// dependency; every CI runner has python3 (callers skip where it is missing).
import { spawnSync } from "node:child_process";

export const hasPython = spawnSync("python3", ["-V"]).status === 0;

const RUNNER = `
import json, os, subprocess, sys, fcntl, termios
mask, ctty, env_json = sys.argv[1], sys.argv[2] == "ctty", sys.argv[3]
argv = sys.argv[4:]
master, slave = os.openpty()
def setctty():
    fcntl.ioctl(0, termios.TIOCSCTTY, 0) if ctty else None
std = [slave if flag == "t" else subprocess.PIPE for flag in mask]
if ctty:
    # the controlling terminal must be the child's stdin for TIOCSCTTY
    std[0] = slave
proc = subprocess.Popen(argv, stdin=std[0], stdout=std[1], stderr=std[2], env=json.loads(env_json),
                        start_new_session=True, preexec_fn=setctty)
proc.communicate(timeout=60)
print(proc.returncode)
`;

/** "t" for a terminal, "p" for a pipe, for stdin, stdout and stderr in turn. */
export type Mask = `${"t" | "p"}${"t" | "p"}${"t" | "p"}`;

export function runWithStdio(
  mask: Mask,
  ctty: boolean,
  env: Record<string, string>,
  argv: string[],
): number {
  if (ctty && mask[0] !== "t") throw new Error("a controlling terminal needs a terminal stdin here");
  const result = spawnSync("python3", ["-I", "-c", RUNNER, mask, ctty ? "ctty" : "no-ctty", JSON.stringify(env), ...argv], {
    encoding: "utf8",
    timeout: 90_000,
  });
  if (result.status !== 0) throw new Error(`runner failed: ${String(result.stderr)}`);
  return Number(String(result.stdout).trim());
}

/** The environment without the agent variables, plus `extra`, so a run behaves the same in any session. */
export const cleanEnv = (extra: Record<string, string> = {}): Record<string, string> =>
  Object.fromEntries(
    Object.entries({ ...process.env, ...extra }).filter(
      ([key, value]) => value !== undefined && (key in extra || (key !== "CLAUDECODE" && key !== "AI_AGENT")),
    ),
  ) as Record<string, string>;
