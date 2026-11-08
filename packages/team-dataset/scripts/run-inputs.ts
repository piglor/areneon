// Publish or fetch the public inputs of a run (ADR-024 section 3.5). See
// src/run-inputs-cli.ts for the commands; the credentials come from the
// environment only.
import { execFileSync } from "node:child_process";
import { R2ObjectStore } from "@areneon/object-store";
import { loadEngine } from "@areneon/showdown-bridge";
import { runInputsCli } from "../src/run-inputs-cli.ts";

process.exitCode = await runInputsCli(process.argv.slice(2), {
  env: process.env,
  engine: () => loadEngine(process.env.SHOWDOWN_DIR || undefined),
  openStore: (settings, layout) => new R2ObjectStore(settings, layout),
  repoState: (dir) => {
    const git = (...args: string[]): string => execFileSync("git", ["-C", dir, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return { commit: git("rev-parse", "HEAD").trim(), dirty: git("status", "--porcelain", "--untracked-files=no").trim() !== "" };
  },
  out: (line) => process.stdout.write(`${line}\n`),
  err: (line) => process.stderr.write(`${line}\n`),
});
