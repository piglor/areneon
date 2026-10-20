// The committed dev and validation fixtures (src/fixtures.ts).
//   node packages/team-dataset/scripts/fixtures.ts write [--full]   (re)make the files
//   node packages/team-dataset/scripts/fixtures.ts check [--full]   fail if a committed file differs
// Without --full the recorded pilot.json is reused (about 15 s); --full runs the config's
// pilot again too (about 1 minute) and so also checks pilot.json.
import { join } from "node:path";
import { loadEngine } from "@areneon/showdown-bridge";
import { DATASETS_DIR, differing, loadFixtureInputs, readFixtureRun, recordedPilot, regenerate, writeFixtures } from "../src/fixtures.ts";

const [command, ...flags] = process.argv.slice(2);
const full = flags.includes("--full");
const formatId = "gen9championsvgc2026regmc";
const dir = join(DATASETS_DIR(formatId), "fixtures-v1");
if (command !== "write" && command !== "check") {
  console.error("usage: fixtures.ts <write|check> [--full]");
  process.exit(2);
}
const inputs = loadFixtureInputs(loadEngine(), formatId);
const run = readFixtureRun(dir);
const files = regenerate(inputs, run, full ? undefined : recordedPilot(inputs, dir));
if (command === "write") {
  console.log(writeFixtures(dir, files).join("\n"));
} else {
  const changed = differing(dir, files);
  if (changed.length > 0) {
    console.error(`differ from a fresh run: ${changed.join(", ")}`);
    process.exit(1);
  }
  console.log("the committed fixtures are what the generator makes");
}
