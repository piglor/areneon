// rules-publish (ADR-007 §6), run by .github/workflows/rules-publish.yml on
// the merged main commit, after check-rules.ts:
//   node packages/showdown-bridge/scripts/rules-publish.ts plan <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts verify <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts release <dir>
// `plan` (the plan job, no environment) writes the artifacts still to release
// and their sha256 manifest into <dir>, and sets the step output
// `publish=true|false`. `verify` (the publish job, before attesting) checks
// <dir> against its manifest and against the plan made again now, writes the
// release set to <dir>/verified/ and sets `attest=true|false`. `release`
// (after attesting) checks again, requires <dir>/verified/ to be exactly
// that set, and creates its Releases, snapshots first.
import { execFileSync } from "node:child_process";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { gitAt, REPO_ROOT } from "../src/rules-history.ts";
import { loadChannels, loadRulesSnapshot } from "../src/rules-load.ts";
import {
  type Artifact,
  checkVerified,
  emitOutput,
  type GitHubCli,
  isReleased,
  plan,
  publish,
  publishOutput,
  readPlan,
  verifiedPath,
  verifyPlan,
  writePlan,
  writeVerified,
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
    emitOutput(output, process.env.GITHUB_OUTPUT);
  } else {
    const { manifest, files } = readPlan(dir);
    const { release, skipped } = verifyPlan(manifest, commit, files, artifacts);
    for (const tag of skipped) console.log(`skipped ${tag}: released meanwhile`);
    if (mode === "verify") {
      const output = writeVerified(dir, release);
      console.log(release.length > 0 ? release.map((a) => `verified ${a.tag}`).join("\n") : "nothing left to release");
      console.log(output);
      emitOutput(output, process.env.GITHUB_OUTPUT);
    } else {
      checkVerified(dir, release);
      const created = publish(release, (a) => verifiedPath(dir, a), gh);
      console.log(created.length > 0 ? created.map((t) => `released ${t}`).join("\n") : "nothing new to release");
    }
  }
} catch (error) {
  if (!(error instanceof RulesError)) throw error;
  console.error(error.message);
  process.exit(1);
}
