// Re-capture the `[Unavailable choice]` channel corpus (src/random-corpus.ts)
// after a Showdown pin bump or an engine-facing change:
//   node packages/showdown-bridge/scripts/capture-unavailable-corpus.ts
// Seeded battles on the teams of the committed fixture dataset, both seats
// playing `pendingFirstActor`; the first few whose channels hold an `|error|`
// line are kept. The found specs are committed; CI replays them and never
// re-runs this search.
// SHOWDOWN_DIR points at another built checkout (a worktree without its own).
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import {
  pendingFirstActor,
  playRandomBattle,
  RANDOM_MANIFEST,
  type RandomBattleSpec,
  randomBattleFile,
  UNAVAILABLE_DIR,
} from "../src/random-corpus.ts";
import { importPaste, validateTeam } from "../src/teams.ts";

const KEEP = 2;
const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const format = resolveFormat(engine, GOLDEN_FORMAT);
const source = join(
  import.meta.dirname,
  "../../../fixtures/teams",
  GOLDEN_FORMAT,
  "datasets/fixtures-v1/teams-dev.jsonl",
);
// The team a battle starts with is the validator's normalised form (as the training loop plays it).
const teams = readFileSync(source, "utf8")
  .split("\n")
  .filter((line) => line !== "")
  .map((line) => {
    const team = importPaste(engine, (JSON.parse(line) as { paste: string }).paste);
    validateTeam(engine, format, team);
    return team;
  });

rmSync(UNAVAILABLE_DIR, { recursive: true, force: true });
mkdirSync(UNAVAILABLE_DIR, { recursive: true });
const manifest: { id: string; seed: string; turns: number; winner: string | null }[] = [];
for (let i = 0; manifest.length < KEEP; i++) {
  if (i > 2000) throw new Error("no battle with an |error| line found");
  const spec: RandomBattleSpec = {
    id: `b${String(manifest.length).padStart(2, "0")}`,
    seed: `sodium,${(i + 0x5000).toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: (2 * i + 1) * 2654435761 >>> 0, p2: (2 * i + 2) * 2654435761 >>> 0 },
    teams: { p1: teams[(2 * i) % teams.length] ?? [], p2: teams[(2 * i + 1) % teams.length] ?? [] },
  };
  let file: ReturnType<typeof randomBattleFile>;
  try {
    file = randomBattleFile(spec, await playRandomBattle(engine, spec, pendingFirstActor));
  } catch (error) {
    // e.g. Revival Blessing, which the enumerator refuses (a separate matter): not this corpus.
    console.log(`skipped battle ${i}: ${(error as Error).message}`);
    continue;
  }
  const errors = [...file.channels.p1.lines, ...file.channels.p2.lines].filter((l) => l.startsWith("|error|"));
  if (errors.length === 0) continue;
  writeFileSync(join(UNAVAILABLE_DIR, `${spec.id}.json`), `${JSON.stringify(file)}\n`);
  manifest.push({ id: spec.id, seed: spec.seed, turns: file.turns, winner: file.winner });
  console.log(`wrote ${spec.id}.json (battle ${i}, ${errors.length} error lines)`);
}
writeFileSync(
  join(UNAVAILABLE_DIR, RANDOM_MANIFEST),
  `${JSON.stringify(
    {
      provenance: "synthetic: see b*.json; teams of the committed fixtures-v1 dataset, found by capture-unavailable-corpus.ts",
      format: GOLDEN_FORMAT,
      battles: manifest,
    },
    null,
    2,
  )}\n`,
);
