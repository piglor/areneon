import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import {
  agentRefusalCode,
  assertGuardedChild,
  assertOwnerSession,
  currentSession,
  openControllingTerminal,
  OWNER_SESSION_REFUSALS,
  OwnerSessionError,
  type OwnerSession,
  terminalRefusal,
} from "../src/owner-session.ts";
import { cleanEnv, hasPython, type Mask, runWithStdio } from "./support/terminals.ts";

const probe = fileURLToPath(new URL("./support/owner-session-probe.ts", import.meta.url));
const SHARED = JSON.parse(readFileSync(fileURLToPath(new URL("../../../fixtures/private-guard/refusals.json", import.meta.url)), "utf8")) as {
  owner_session: Record<string, string>;
};
const scratch = mkdtempSync(join(tmpdir(), "owner-session-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const ALL = { stdin: true, stdout: true, stderr: true };
const session = (over: Partial<OwnerSession> = {}, opened: { n: number } = { n: 0 }): OwnerSession => ({
  env: {},
  isTerminal: ALL,
  openTerminal: () => {
    opened.n++;
    return { write: () => undefined, close: () => undefined };
  },
  ...over,
});
const codeOf = (run: () => void): string => {
  try {
    run();
  } catch (error) {
    if (error instanceof OwnerSessionError) return error.code;
    throw error;
  }
  return "ok";
};

describe("shared words", () => {
  it("equal the fixed words the Python twin is pinned to", () => {
    expect(OWNER_SESSION_REFUSALS).toEqual(SHARED.owner_session);
  });
});

describe("assertOwnerSession (unit)", () => {
  it("passes a terminal session with neither variable, opening the terminal once", () => {
    const opened = { n: 0 };
    expect(codeOf(() => assertOwnerSession(session({}, opened)))).toBe("ok");
    expect(opened.n).toBe(1);
  });

  it("refuses under CLAUDECODE or AI_AGENT before looking at the stdio or opening anything", () => {
    for (const env of [{ CLAUDECODE: "1" }, { AI_AGENT: "x" }]) {
      const opened = { n: 0 };
      expect(codeOf(() => assertOwnerSession(session({ env, isTerminal: { ...ALL, stdin: false } }, opened)))).toBe("agent_session");
      expect(opened.n).toBe(0);
    }
    expect(agentRefusalCode({ CLAUDECODE: "" })).toBeNull();
  });

  it("refuses each piped stdio, without opening the terminal", () => {
    for (const off of ["stdin", "stdout", "stderr"] as const) {
      const opened = { n: 0 };
      expect(codeOf(() => assertOwnerSession(session({ isTerminal: { ...ALL, [off]: false } }, opened)))).toBe("not_a_terminal");
      expect(opened.n).toBe(0);
    }
  });

  it("refuses when /dev/tty does not open", () => {
    const refuse = session({
      openTerminal: () => {
        throw new OwnerSessionError("no_controlling_terminal");
      },
    });
    expect(codeOf(() => assertOwnerSession(refuse))).toBe("no_controlling_terminal");
  });

  it("carries the fixed words and nothing else", () => {
    const error = new OwnerSessionError("agent_session");
    expect([error.message, error.name]).toEqual([OWNER_SESSION_REFUSALS.agent_session, "OwnerSessionError"]);
  });

  it("terminalRefusal gives the same words without opening anything", () => {
    expect(terminalRefusal({}, ALL)).toBeNull();
    expect(terminalRefusal({ AI_AGENT: "1" }, ALL)).toBe(OWNER_SESSION_REFUSALS.agent_session);
    expect(terminalRefusal({}, { ...ALL, stderr: false })).toBe(OWNER_SESSION_REFUSALS.not_a_terminal);
  });
});

describe("assertGuardedChild (unit)", () => {
  it("checks the environment only", () => {
    expect(codeOf(() => assertGuardedChild({}))).toBe("ok");
    expect(codeOf(() => assertGuardedChild({ CLAUDECODE: "1" }))).toBe("agent_session");
    expect(codeOf(() => assertGuardedChild({ AI_AGENT: "1" }))).toBe("agent_session");
  });
});

describe("the real session", () => {
  it("reads the process: stdio flags are booleans and the default is the process environment", () => {
    const real = currentSession();
    expect(real.env).toBe(process.env);
    expect(Object.values(real.isTerminal).every((v) => typeof v === "boolean")).toBe(true);
  });

  it("writes to and closes the device it opened, and refuses a device that does not open", () => {
    const device = join(scratch, "device.txt");
    const handle = openControllingTerminal(device);
    handle.write("detail");
    handle.close();
    expect(readFileSync(device, "utf8")).toBe("detail");
    expect(() => openControllingTerminal(join(scratch, "no", "such"))).toThrow(OwnerSessionError);
  });

  it("opens /dev/tty or says the fixed refusal", () => {
    try {
      openControllingTerminal().close();
    } catch (error) {
      expect(error).toBeInstanceOf(OwnerSessionError);
      expect((error as OwnerSessionError).code).toBe("no_controlling_terminal");
    }
  });

  it("the default session of the guard is the real one: it refuses under vitest's piped stdio or an agent", () => {
    // Under a test runner stdio are pipes; on a developer's terminal vitest may still pipe them.
    const code = codeOf(() => assertOwnerSession());
    expect(["ok", "agent_session", "not_a_terminal", "no_controlling_terminal"]).toContain(code);
    expect(codeOf(() => assertGuardedChild())).toMatch(/^(ok|agent_session)$/);
  });
});

describe.skipIf(!hasPython)("the guard in a real child process (needs python3)", () => {
  let n = 0;
  const run = (mode: "top" | "child", mask: Mask, ctty: boolean, env: Record<string, string> = {}): { exit: number; result: { ok: boolean; code?: string; message?: string } } => {
    const out = join(scratch, `out-${n++}.json`);
    const exit = runWithStdio(mask, ctty, cleanEnv(env), [process.execPath, probe, mode, out]);
    return { exit, result: JSON.parse(readFileSync(out, "utf8")) as { ok: boolean; code?: string; message?: string } };
  };

  it("passes in a pseudo-terminal with a controlling terminal and neither variable", () => {
    expect(run("top", "ttt", true).result).toEqual({ ok: true });
  });

  it("refuses under CLAUDECODE and under AI_AGENT even on a terminal", () => {
    for (const env of [{ CLAUDECODE: "1" }, { AI_AGENT: "1" }]) {
      expect(run("top", "ttt", true, env).result).toEqual({ ok: false, code: "agent_session", message: OWNER_SESSION_REFUSALS.agent_session });
    }
  });

  it("refuses a piped stdin, a piped stdout and a piped stderr", () => {
    for (const mask of ["ptt", "tpt", "ttp", "ppp"] as const) {
      expect(run("top", mask, false).result).toMatchObject({ ok: false, code: "not_a_terminal" });
    }
  });

  it("refuses on three terminals without a controlling terminal", () => {
    expect(run("top", "ttt", false).result).toMatchObject({ ok: false, code: "no_controlling_terminal" });
  });

  it("a child with pipes passes unless an agent variable is set", () => {
    expect(run("child", "ppp", false).result).toEqual({ ok: true });
    expect(run("child", "ppp", false, { CLAUDECODE: "1" }).result).toMatchObject({ ok: false, code: "agent_session" });
  });
});
