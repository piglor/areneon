// Record the synthetic bot-vs-bot fixtures of fixtures/a2/ (ADR-005 §5):
//   node packages/local-server/scripts/record-a2.ts [--out DIR] [--runtime-root DIR]
// Pure wiring: the real engine, launcher and files; see fixtures/a2/README.md.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { importPaste, loadEngine, loadRulesSnapshot } from "@areneon/showdown-bridge";
import { createRuntimeCopy } from "../src/index.ts";
import { launchHarness, recordHarnessBattle } from "../src/harness.ts";
import { FIXTURE_FORMAT, RECORDINGS, recordingProvenance } from "../src/recordings.ts";

const REPO = fileURLToPath(new URL("../../../", import.meta.url));
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const at = args.indexOf(name);
  return at < 0 ? undefined : args[at + 1];
};
const out = flag("--out") ?? join(REPO, "fixtures", "a2", FIXTURE_FORMAT);
const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
const team = (file: string) => importPaste(engine, readFileSync(join(REPO, "fixtures", "teams", FIXTURE_FORMAT, file), "utf8"));

const root = flag("--runtime-root");
const copy = createRuntimeCopy(root === undefined ? {} : { runtimeRoot: root });
const server = await launchHarness({ dir: copy.dir, port: Number(flag("--port") ?? 8000) });
try {
  mkdirSync(out, { recursive: true });
  for (const recording of RECORDINGS) {
    const fixture = await recordHarnessBattle({
      engine,
      snapshot,
      server,
      dir: copy.dir,
      format: FIXTURE_FORMAT,
      names: recording.names,
      teams: { sparring: team(recording.teams.sparring), challenger: team(recording.teams.challenger) },
      teamLabels: recording.teams,
      provenance: recordingProvenance(engine.commit),
    });
    const path = join(out, recording.file);
    writeFileSync(path, `${JSON.stringify(fixture, null, 2)}\n`);
    console.log(`wrote ${path} (seed ${fixture.seed}, winner ${String(fixture.winner)}, ${fixture.turns} turns)`);
  }
} finally {
  await server.stop();
}
