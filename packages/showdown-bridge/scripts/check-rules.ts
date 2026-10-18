// CI gate for the rules records (ADR-007), offline once the repo is checked
// out with full history:
//   node packages/showdown-bridge/scripts/check-rules.ts
// 1. Every record passes its schema and every format resolves against the
//    pinned engine.
// 2. rules/channels/ holds exactly one file per channel the schema allows;
//    each names an allowed snapshot built for the current pin, and its seq
//    rises by exactly 1 against the base commit.
// CI sets RULES_BASE_REF=HEAD^1; locally the base is the merge-base with
// origin/main. The policy lives in src/rules-channel.ts and
// src/rules-history.ts; this file only wires git and node to it.
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { checkChannel } from "../src/rules-channel.ts";
import { baseChannel, buildAt, gitAt, historicalSnapshots, REPO_ROOT, resolveBase } from "../src/rules-history.ts";
import { loadChannels, loadRulesSnapshot } from "../src/rules-load.ts";
import { RulesError, rulesSnapshotHash } from "../src/rules.ts";

const git = gitAt(REPO_ROOT);

try {
  const engine = loadEngine();
  const snapshot = loadRulesSnapshot(engine.commit);
  for (const formatId of Object.keys(snapshot.formats)) resolveFormat(engine, formatId, snapshot);
  const channels = loadChannels();

  const allowed = new Map([[rulesSnapshotHash(snapshot), engine.commit]]);
  const base = resolveBase(process.env, git, console.warn);
  if (base) {
    for (const built of historicalSnapshots(base, git, buildAt, console.warn)) {
      if (!allowed.has(built.hash)) allowed.set(built.hash, built.showdown_commit);
    }
  }

  let failed = false;
  for (const head of channels) {
    const problems = checkChannel({
      head,
      base: base ? baseChannel(base, head.channel, git) : undefined,
      allowed,
      pin: engine.commit,
    });
    if (problems.length > 0) {
      console.error(`rules/channels/${head.channel}.json:\n- ${problems.join("\n- ")}`);
      failed = true;
    } else {
      console.log(`rules ok: ${head.channel} seq ${head.seq} -> ${head.hash}`);
    }
  }
  console.log(`rules snapshot ${rulesSnapshotHash(snapshot)}`);
  if (failed) process.exit(1);
} catch (error) {
  if (!(error instanceof RulesError)) throw error;
  console.error(error.message);
  process.exit(1);
}
