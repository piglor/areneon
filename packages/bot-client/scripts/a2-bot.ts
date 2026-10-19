// The A2 sparring bot on the pinned local server (ADR-005): see `runCli` in src/cli.ts.
//   node packages/bot-client/scripts/a2-bot.ts --challenger NAME --team PASTE
// Pure wiring: the real engine, launcher, bot, signals and files.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRuntimeCopy, launch } from "@areneon/local-server";
import { loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { runCli } from "../src/cli.ts";
import { runA2Bot } from "../src/session.ts";

const REPO = fileURLToPath(new URL("../../../", import.meta.url));
const engine = loadEngine();
const code = await runCli(process.argv.slice(2), {
  engine,
  snapshot: loadRulesSnapshot(engine.commit),
  readFile: (path) => readFileSync(path, "utf8"),
  createRuntimeCopy,
  launch,
  runBot: runA2Bot,
  onSignal(handler) {
    const signals = ["SIGINT", "SIGTERM"] as const;
    for (const name of signals) process.on(name, handler);
    return () => {
      for (const name of signals) process.off(name, handler);
    };
  },
  makeDir: (dir, mode) => void mkdirSync(dir, { recursive: true, mode }),
  writeFile: (path, data, mode) => writeFileSync(path, data, { mode, flag: "wx" }),
  repoRoot: REPO,
  privateDataDir: join(REPO, "private-data"),
  defaultOutDir: join(REPO, "private-data", "a2"),
  now: () => Date.now(),
  stdout: (line) => process.stdout.write(`${line}\n`),
  stderr: (line) => process.stderr.write(`${line}\n`),
});
process.exit(code);
