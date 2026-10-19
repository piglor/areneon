// rules-publish (ADR-007 §6), run by .github/workflows/rules-publish.yml on
// the merged main commit, after check-rules.ts:
//   node packages/showdown-bridge/scripts/rules-publish.ts build <dir>
//   node packages/showdown-bridge/scripts/rules-publish.ts release <dir>
// `build` writes the artifacts still to release into <dir> (the workflow
// attests them next); `release` plans again, refuses unless <dir> holds
// exactly those files, and creates their Releases, snapshots first.
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { PINNED_SHOWDOWN_COMMIT } from "../src/engine.ts";
import { gitAt, REPO_ROOT } from "../src/rules-history.ts";
import { loadChannels, loadRulesSnapshot } from "../src/rules-load.ts";
import { type Artifact, checkBuilt, type GitHubCli, isReleased, plan, publish } from "../src/rules-publish.ts";
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
  if (!dir || (mode !== "build" && mode !== "release")) throw new RulesError("usage: rules-publish.ts build|release <dir>");
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
  if (mode === "build") {
    mkdirSync(dir, { recursive: true });
    for (const a of artifacts) writeFileSync(join(dir, `${a.tag}.json`), a.bytes);
    console.log(artifacts.length > 0 ? artifacts.map((a) => `built ${a.tag}`).join("\n") : "nothing to release");
  } else {
    const names = (() => {
      try {
        return readdirSync(dir);
      } catch {
        return [];
      }
    })();
    const files = new Map(names.map((name) => [name, readFileSync(join(dir, name))]));
    const problems = checkBuilt(artifacts, files);
    if (problems.length > 0) throw new RulesError(`${dir} is not what this commit plans:\n- ${problems.join("\n- ")}`);
    const created = publish(artifacts, (a) => join(dir, `${a.tag}.json`), gh);
    console.log(created.length > 0 ? created.map((t) => `released ${t}`).join("\n") : "nothing new to release");
  }
} catch (error) {
  if (!(error instanceof RulesError)) throw error;
  console.error(error.message);
  process.exit(1);
}
