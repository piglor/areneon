// The owner-session guard (ADR-015 erratum E1, items 7-9). Every command that
// reads private teams calls it before it reads any team: it refuses in an
// agent session (`CLAUDECODE` or `AI_AGENT` set) and, for the top-level
// command, unless stdin, stdout and stderr are terminals and `/dev/tty`
// opens. There is no override. A child process a guarded driver starts has
// pipes for stdio, so it checks the environment only (`assertGuardedChild`).
// The Python twin is `ml/areneon_ml/private/owner_session.py`; both say the
// same fixed words, which `fixtures/private-guard/refusals.json` pins in a
// test on each side. A message names no path, team or variable value.
// Node built-ins only.

import { closeSync, openSync, writeSync } from "node:fs";

export const OWNER_SESSION_REFUSALS = {
  agent_session: "refusing: an agent session is running this; use your own terminal",
  not_a_terminal: "refusing: stdin, stdout and stderr must all be a terminal; run it in an interactive terminal",
  no_controlling_terminal: "refusing: no controlling terminal to show the detail on",
} as const;

export type OwnerSessionCode = keyof typeof OWNER_SESSION_REFUSALS;

export class OwnerSessionError extends Error {
  readonly code: OwnerSessionCode;
  constructor(code: OwnerSessionCode) {
    super(OWNER_SESSION_REFUSALS[code]);
    this.name = "OwnerSessionError";
    this.code = code;
  }
}

export interface TerminalFlags {
  stdin: boolean;
  stdout: boolean;
  stderr: boolean;
}

/** A handle on the terminal device. */
export interface TerminalHandle {
  write(text: string): void;
  close(): void;
}

/** What the guard looks at: a seam for tests; the default reads the process. */
export interface OwnerSession {
  env: Readonly<Record<string, string | undefined>>;
  isTerminal: TerminalFlags;
  openTerminal: () => TerminalHandle;
}

/** Open `/dev/tty` for writing; the fixed refusal if the process has no controlling terminal. `device` is a seam for tests. */
export function openControllingTerminal(device = "/dev/tty"): TerminalHandle {
  let fd: number;
  try {
    fd = openSync(device, "w");
  } catch {
    throw new OwnerSessionError("no_controlling_terminal");
  }
  return { write: (text) => void writeSync(fd, text), close: () => closeSync(fd) };
}

export const currentSession = (): OwnerSession => ({
  env: process.env,
  isTerminal: {
    stdin: Boolean(process.stdin.isTTY),
    stdout: Boolean(process.stdout.isTTY),
    stderr: Boolean(process.stderr.isTTY),
  },
  openTerminal: openControllingTerminal,
});

/** The fixed code for an agent environment, else null. Any non-empty value counts. */
export const agentRefusalCode = (env: Readonly<Record<string, string | undefined>>): OwnerSessionCode | null =>
  env["CLAUDECODE"] || env["AI_AGENT"] ? "agent_session" : null;

/** The refusal message for the environment and the stdio, or null. Opens nothing. */
export function terminalRefusal(env: Readonly<Record<string, string | undefined>>, isTerminal: TerminalFlags): string | null {
  const code: OwnerSessionCode | null =
    agentRefusalCode(env) ?? (isTerminal.stdin && isTerminal.stdout && isTerminal.stderr ? null : "not_a_terminal");
  return code === null ? null : OWNER_SESSION_REFUSALS[code];
}

/**
 * The top-level check: environment, then the three stdio, then `/dev/tty`
 * (opened and closed at once). Throws an `OwnerSessionError`; call it before
 * reading any team.
 */
export function assertOwnerSession(session: OwnerSession = currentSession()): void {
  const code = agentRefusalCode(session.env);
  if (code !== null) throw new OwnerSessionError(code);
  const { stdin, stdout, stderr } = session.isTerminal;
  if (!stdin || !stdout || !stderr) throw new OwnerSessionError("not_a_terminal");
  session.openTerminal().close();
}

/** A child of a guarded driver: its stdio are pipes, so only the environment is checked. */
export function assertGuardedChild(env: Readonly<Record<string, string | undefined>> = process.env): void {
  const code = agentRefusalCode(env);
  if (code !== null) throw new OwnerSessionError(code);
}
