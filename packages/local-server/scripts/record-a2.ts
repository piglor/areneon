// Record the synthetic bot-vs-bot fixtures of fixtures/a2/ (ADR-005 §5):
//   node packages/local-server/scripts/record-a2.ts [--out DIR] [--runtime-root DIR] [--port N] [--only FILE]
// Pure wiring: the real engine, launcher and files; see fixtures/a2/README.md.
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { importPaste, loadEngine, loadRulesSnapshot, teamHash } from "@areneon/showdown-bridge";
import { createRuntimeCopy } from "../src/index.ts";
import { launchHarness, ReconnectNotPlannedError, recordHarnessBattle, withPlayedForms } from "../src/harness.ts";
import { FIXTURE_FORMAT, RECORDINGS, recordingProvenance } from "../src/recordings.ts";

const REPO = fileURLToPath(new URL("../../../", import.meta.url));
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const at = args.indexOf(name);
  return at < 0 ? undefined : args[at + 1];
};
const only = flag("--only");
const out = flag("--out") ?? join(REPO, "fixtures", "a2", FIXTURE_FORMAT);
const engine = loadEngine();
const snapshot = loadRulesSnapshot(engine.commit);
// The synthetic teams there are: every committed paste fixture. A recording of any other team is refused.
const TEAM_DIR = join(REPO, "fixtures", "teams", FIXTURE_FORMAT);
const pastes = readdirSync(TEAM_DIR)
  .filter((file) => file.endsWith(".txt"))
  .map((file) => importPaste(engine, readFileSync(join(TEAM_DIR, file), "utf8")));
const known_synthetic = withPlayedForms(engine, FIXTURE_FORMAT, new Set(pastes.map(teamHash)), pastes);
const team = (file: string) => importPaste(engine, readFileSync(join(REPO, "fixtures", "teams", FIXTURE_FORMAT, file), "utf8"));

const root = flag("--runtime-root");
const copy = createRuntimeCopy(root === undefined ? {} : { runtimeRoot: root });
const port = Number(flag("--port") ?? 8000);
/** A battle with a planned reconnect is random: how often a refused one (`ReconnectNotPlannedError`) is played again. */
const ATTEMPTS = 4;

/** One recording on a fresh server (so its room is battle-<format>-1 as before) and a fresh log directory. */
async function play(recording: (typeof RECORDINGS)[number]) {
  rmSync(join(copy.dir, "logs"), { recursive: true, force: true });
  const server = await launchHarness({ dir: copy.dir, port });
  try {
    return await recordHarnessBattle({
      engine,
      snapshot,
      server,
      dir: copy.dir,
      format: FIXTURE_FORMAT,
      names: recording.names,
      teams: { sparring: team(recording.teams.sparring), challenger: team(recording.teams.challenger) },
      teamLabels: recording.teams,
      known_synthetic,
      provenance: recordingProvenance(engine.commit, recording.drop),
      ...(recording.drop === undefined ? {} : { drop: recording.drop }),
    });
  } finally {
    await server.stop();
  }
}

mkdirSync(out, { recursive: true });
for (const recording of RECORDINGS.filter((r) => only === undefined || r.file === only)) {
  let fixture;
  for (let attempt = 1; fixture === undefined; attempt++) {
    try {
      fixture = await play(recording);
    } catch (error) {
      if (!(error instanceof ReconnectNotPlannedError) || attempt >= ATTEMPTS) throw error;
      console.log(`${recording.file}: attempt ${attempt} refused (${error.message}); playing again`);
    }
  }
  const path = join(out, recording.file);
  writeFileSync(path, `${JSON.stringify(fixture, null, 2)}\n`);
  console.log(`wrote ${path} (seed ${fixture.seed}, winner ${String(fixture.winner)}, ${fixture.turns} turns)`);
}
