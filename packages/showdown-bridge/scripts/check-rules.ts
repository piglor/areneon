// CI gate for the rules records (ADR-007), offline once the repo is checked
// out with full history:
//   node packages/showdown-bridge/scripts/check-rules.ts
// 1. Every record passes its schema and every format resolves against the
//    pinned engine.
// 2. rules/channels/stable.json names an allowed snapshot built for the
//    current pin, and its seq rises by exactly 1 against the base commit.
// CI sets RULES_BASE_REF=HEAD^1; locally the base is the merge-base with
// origin/main. The policy lives in src/rules-channel.ts and
// src/rules-history.ts; this file only wires git and node to it.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { checkChannel } from "../src/rules-channel.ts";
import {
  type BuiltSnapshot,
  chooseBase,
  type GitReader,
  historicalSnapshots,
  RULES_BUILDER,
} from "../src/rules-history.ts";
import { loadChannel, loadRulesSnapshot, parseChannel } from "../src/rules-load.ts";
import { type ChannelFile, rulesSnapshotHash } from "../src/rules.ts";

const CHANNEL = "stable";
const git: GitReader = (...args) => {
  try {
    return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return undefined;
  }
};

/** Run `commit`'s own builder from a bare archive: no node_modules, no shell. */
function buildAt(commit: string): BuiltSnapshot {
  const dir = mkdtempSync(join(tmpdir(), "areneon-rules-"));
  try {
    const archive = join(dir, "tree.tar");
    execFileSync("git", ["archive", "-o", archive, commit, "rules", "packages/showdown-bridge"]);
    execFileSync("tar", ["-xf", archive, "-C", dir]);
    return JSON.parse(
      execFileSync("node", [join(dir, RULES_BUILDER)], { encoding: "utf8" }),
    ) as BuiltSnapshot;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
for (const formatId of Object.keys(snapshot.formats)) resolveFormat(engine, formatId, snapshot);
const head = loadChannel(CHANNEL);

const allowed = new Map([[rulesSnapshotHash(snapshot), engine.commit]]);
let base: ChannelFile | undefined;
const baseCommit = chooseBase(process.env, git);
if (baseCommit) {
  const where = `${baseCommit}:rules/channels/${CHANNEL}.json`;
  const text = git("show", where);
  base = text === undefined ? undefined : parseChannel(CHANNEL, JSON.parse(text), where);
  for (const built of historicalSnapshots(baseCommit, git, buildAt, console.warn)) {
    if (!allowed.has(built.hash)) allowed.set(built.hash, built.showdown_commit);
  }
} else if (process.env.RULES_BASE_REF) {
  console.error(`RULES_BASE_REF=${process.env.RULES_BASE_REF} does not resolve to a commit`);
  process.exit(1);
}

const problems = checkChannel({ head, base, allowed, pin: engine.commit });
if (problems.length > 0) {
  console.error(`rules/channels/${CHANNEL}.json:\n- ${problems.join("\n- ")}`);
  process.exit(1);
}
console.log(`rules ok: snapshot ${rulesSnapshotHash(snapshot)}, ${CHANNEL} seq ${head.seq} -> ${head.hash}`);
