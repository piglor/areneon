// SPIKE (#677): throwaway. Starts the pinned, unmodified Showdown server on
// localhost only. The only thing written into the vendor checkout is
// config/config.js, which upstream gitignores and generates itself.
import { type ChildProcess, spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { DEFAULT_SHOWDOWN_DIR, loadEngine } from "../../../packages/showdown-bridge/src/index.ts";

export const HOST = "127.0.0.1";
export const DEFAULT_PORT = 8000;

/**
 * Upstream defaults from config-example.js with the overrides a private
 * sparring server needs. `--no-security` (passed at start) adds
 * nothrottle, noguestsecurity and noipchecks, so names need no login-server
 * assertion and nothing ever talks to the public login server.
 */
const CONFIG = `'use strict';
// Written by spikes/677-bot-spike/src/server.ts. Local sparring only.
Object.assign(exports, require('./config-example.js'));
exports.bindaddress = '${HOST}';   // never reachable from another machine
exports.backdoor = false;          // no remote sysop console access
exports.consoleips = [];           // no dev console
exports.repl = false;
exports.reportjoins = false;
exports.forcetimer = true;         // VGC Timer runs in every battle
`;

export function writeLocalConfig(dir: string = DEFAULT_SHOWDOWN_DIR): void {
  writeFileSync(join(dir, "config", "config.js"), CONFIG);
}

/** Start the server and resolve once it accepts connections. */
export async function startServer(port: number = DEFAULT_PORT): Promise<ChildProcess> {
  loadEngine(); // refuses an unbuilt or unpinned checkout
  writeLocalConfig();
  const child = spawn(
    process.execPath,
    ["pokemon-showdown", "start", "--skip-build", "--no-security", String(port)],
    { cwd: DEFAULT_SHOWDOWN_DIR, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stderr?.on("data", (d: Buffer) => process.stderr.write(`[server] ${d.toString()}`));
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("server did not start in 60 s")), 60_000);
    child.stdout?.on("data", (d: Buffer) => {
      const text = d.toString();
      process.stderr.write(`[server] ${text}`);
      if (text.includes("Test your server at")) {
        clearTimeout(timeout);
        resolve();
      }
    });
    child.on("exit", (code) => reject(new Error(`server exited early with code ${code}`)));
  });
  return child;
}

export async function stopServer(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return;
  const exited = new Promise<void>((resolve) => child.on("exit", () => resolve()));
  child.kill("SIGTERM");
  await exited;
}

// `node spikes/677-bot-spike/src/server.ts [port]` runs the server for a human match.
if (import.meta.main) {
  const port = Number(process.argv[2] ?? DEFAULT_PORT);
  const child = await startServer(port);
  process.stderr.write(`Local server on ws://${HOST}:${port}. Ctrl-C to stop.\n`);
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.on(signal, () => void stopServer(child).then(() => process.exit(0)));
  }
}
