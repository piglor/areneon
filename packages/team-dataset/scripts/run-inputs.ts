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
  headCommit: (dir) => execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(),
  out: (line) => process.stdout.write(`${line}\n`),
  err: (line) => process.stderr.write(`${line}\n`),
});
