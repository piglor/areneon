// CI gate for the rules records (ADR-007), offline once the repo is checked
// out with full history:
//   node packages/showdown-bridge/scripts/check-rules.ts
// 1. Every record passes its schema and every format resolves against the
//    pinned engine.
// 2. rules/channels/stable.json names an allowed snapshot built for the
//    current pin, and its seq rises by exactly 1 against the base commit.
// RULES_BASE_REF overrides the base (default: merge-base with origin/main,
// or the first parent when HEAD is origin/main).
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { checkChannel } from "../src/rules-channel.ts";
import { loadChannel, loadRulesSnapshot } from "../src/rules-load.ts";
import { type ChannelFile, rulesSnapshotHash } from "../src/rules.ts";

const CHANNEL = "stable";
const git = (...args: string[]): string => execFileSync("git", args, { encoding: "utf8" }).trim();
const tryGit = (...args: string[]): string | undefined => {
  try {
    return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return undefined;
  }
};

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
for (const formatId of Object.keys(snapshot.formats)) resolveFormat(engine, formatId, snapshot);
const head = loadChannel(CHANNEL);

const base =
  process.env.RULES_BASE_REF ??
  (git("rev-parse", "HEAD") === tryGit("rev-parse", "origin/main")
    ? tryGit("rev-parse", "HEAD^1")
    : tryGit("merge-base", "HEAD", "origin/main"));

const allowed = new Map([[rulesSnapshotHash(snapshot), engine.commit]]);
let baseChannel: ChannelFile | undefined;
if (base) {
  const text = tryGit("show", `${base}:rules/channels/${CHANNEL}.json`);
  baseChannel = text === undefined ? undefined : (JSON.parse(text) as ChannelFile);
  const builder = "packages/showdown-bridge/scripts/rules-hash.ts";
  const commits = git(
    "rev-list", "--first-parent", base, "--",
    "rules", "packages/showdown-bridge/src/rules.ts", "packages/showdown-bridge/src/engine.ts",
  ).split("\n").filter(Boolean);
  for (const commit of commits) {
    if (tryGit("cat-file", "-e", `${commit}:${builder}`) === undefined) continue;
    const dir = mkdtempSync(join(tmpdir(), "areneon-rules-"));
    try {
      execFileSync("sh", ["-c", `git archive ${commit} rules packages/showdown-bridge | tar -x -C "${dir}"`]);
      const built = JSON.parse(
        execFileSync("node", [join(dir, builder)], { encoding: "utf8" }),
      ) as { hash: string; showdown_commit: string };
      allowed.set(built.hash, built.showdown_commit);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
}

const problems = checkChannel({ head, base: baseChannel, allowed, pin: engine.commit });
if (problems.length > 0) {
  console.error(`rules/channels/${CHANNEL}.json:\n- ${problems.join("\n- ")}`);
  process.exit(1);
}
console.log(`rules ok: snapshot ${rulesSnapshotHash(snapshot)}, ${CHANNEL} seq ${head.seq} -> ${head.hash}`);
