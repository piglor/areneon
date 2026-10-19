// rules-publish (ADR-007 §6), run by .github/workflows/rules-publish.yml on
// the merged main commit, after check-rules.ts:
//   node packages/showdown-bridge/scripts/rules-publish.ts plan <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts verify <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts release <dir>
// `plan` (the plan job, no environment) writes the artifacts still to release
// and their sha256 manifest into <dir>, and sets the step output
// `publish=true|false`. `verify` (the publish job, before attesting) and
// `release` (after) check <dir> against its manifest and against the plan
// made again now, then `release` creates the Releases, snapshots first.
import { execFileSync } from "node:child_process";
import { appendFileSync } from "node:fs";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { gitAt, REPO_ROOT } from "../src/rules-history.ts";
import { loadChannels, loadRulesSnapshot } from "../src/rules-load.ts";
import {
  type Artifact,
  artifactPath,
  type GitHubCli,
  isReleased,
  plan,
  publish,
  publishOutput,
  readPlan,
  verifyPlan,
  writePlan,
} from "../src/rules-publish.ts";
import { RulesError } from "../src/rules.ts";

const [mode, dir] = process.argv.slice(2);
const git = gitAt(REPO_ROOT);
const gh: GitHubCli = (args) => {
  try {
    return execFileSync("gh", args, { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] });
  } catch {
    return undefined;
  }
};

try {
  if (!dir || (mode !== "plan" && mode !== "verify" && mode !== "release")) {
    throw new RulesError("usage: rules-publish.ts plan|verify|release <dir>");
  }
  const commit = git("rev-parse", "HEAD");
  if (!commit) throw new RulesError("git could not name HEAD");
  const artifacts: Artifact[] = plan({
    snapshot: loadRulesSnapshot(PINNED_SHOWDOWN_COMMIT),
    commit,
    channels: loadChannels().map((file) => {
      const changedIn = git("log", "-1", "--first-parent", "--format=%H", "--", `:(top)rules/channels/${file.channel}.json`);
      if (!changedIn) throw new RulesError(`git could not find the commit that last changed rules/channels/${file.channel}.json`);
      return { file, changedIn };
    }),
    released: (tag) => isReleased(tag, gh),
  });
  if (mode === "plan") {
    const output = publishOutput(writePlan(dir, commit, artifacts));
    console.log(artifacts.length > 0 ? artifacts.map((a) => `planned ${a.tag}`).join("\n") : "nothing to release");
    console.log(output);
    if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `${output}\n`);
  } else {
    const { manifest, files } = readPlan(dir);
    const { release, skipped } = verifyPlan(manifest, commit, files, artifacts);
    for (const tag of skipped) console.log(`skipped ${tag}: released meanwhile`);
    if (mode === "verify") {
      console.log(release.length > 0 ? release.map((a) => `verified ${a.tag}`).join("\n") : "nothing left to release");
    } else {
      const created = publish(release, (a) => artifactPath(dir, a), gh);
      console.log(created.length > 0 ? created.map((t) => `released ${t}`).join("\n") : "nothing new to release");
    }
  }
} catch (error) {
  if (!(error instanceof RulesError)) throw error;
  console.error(error.message);
  process.exit(1);
}
