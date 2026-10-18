// rules-publish (ADR-007 §6), run by .github/workflows/rules-publish.yml on
// the merged main commit, after check-rules.ts:
//   node packages/showdown-bridge/scripts/rules-publish.ts build <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts release <dir>
// `build` writes the commit's artifacts into <dir> (the workflow attests them
// next); `release` creates their Releases, skipping any tag that exists.
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { baseChannel, gitAt, REPO_ROOT } from "../src/rules-history.ts";
import { loadChannels, loadRulesSnapshot } from "../src/rules-load.ts";
import { type Artifact, type GitHubCli, plan, publish, snapshotTag } from "../src/rules-publish.ts";
import { RulesError } from "../src/rules.ts";

const [mode, dir] = process.argv.slice(2);
const git = gitAt(REPO_ROOT);
const gh: GitHubCli = (args) => {
  try {
    execFileSync("gh", args, { stdio: ["ignore", "ignore", "inherit"] });
    return true;
  } catch {
    return false;
  }
};

try {
  if (!dir || (mode !== "build" && mode !== "release")) throw new RulesError("usage: rules-publish.ts build|release <dir>");
  const sourceCommit = git("rev-parse", "HEAD");
  if (!sourceCommit) throw new RulesError("git could not name HEAD");
  if (mode === "build") {
    const parent = git("rev-parse", "HEAD^1");
    const artifacts = plan({
      snapshot: loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT),
      channels: loadChannels(),
      base: (channel) => (parent ? baseChannel(parent, channel, git) : undefined),
      sourceCommit,
      published: (hash) => gh(["release", "view", snapshotTag(hash)]),
    });
    mkdirSync(dir, { recursive: true });
    for (const a of artifacts) writeFileSync(join(dir, `${a.tag}.json`), a.bytes);
    console.log(artifacts.map((a) => `built ${a.tag}`).join("\n"));
  } else {
    const artifacts: Artifact[] = readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .sort()
      .map((f) => {
        const tag = f.slice(0, -".json".length);
        return { kind: tag.startsWith("rules-snapshot-") ? "snapshot" : "channel", tag, bytes: readFileSync(join(dir, f)) };
      });
    const created = publish(artifacts, sourceCommit, (a) => join(dir, `${a.tag}.json`), gh);
    console.log(created.length > 0 ? created.map((t) => `released ${t}`).join("\n") : "nothing new to release");
  }
} catch (error) {
  if (!(error instanceof RulesError)) throw error;
  console.error(error.message);
  process.exit(1);
}
