// Re-capture the targeted fidelity corpus (src/search-fidelity-corpus.ts, ADR-026
// Evidence 1, slice T) after a Showdown pin bump, an engine-facing change or an
// edit of `targets.json`:
//   node packages/showdown-bridge/scripts/capture-search-fidelity-corpus.ts [--dataset DIR]...
// Teams come from the committed fixtures-v1 dev split and from every `--dataset DIR`
// (a `generate` output; its records.dev.jsonl is read), because the committed split
// holds no Last Respects or Mirror Coat team. Only the battles kept, with their teams,
// are committed; CI replays them and never re-runs this search.
// SHOWDOWN_DIR points at another built checkout (a worktree without its own).
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import { resolveFormat } from "../src/formats.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadRandomCorpus, RANDOM_MANIFEST } from "../src/random-corpus.ts";
import {
  captureFidelityCorpus,
  checkUnreachable,
  FIDELITY_DIR,
  FIDELITY_SUMMARY,
  fidelitySummary,
  loadFidelityTargets,
  type FidelityBattleFile,
  type PoolTeam,
} from "../src/search-fidelity-corpus.ts";
import { recordHash } from "../src/team-records.ts";
import { importPaste, validateTeam } from "../src/teams.ts";

const extra: string[] = [];
const args = process.argv.slice(2);
const extend = args.includes("--extend");
if (extend) args.splice(args.indexOf("--extend"), 1);
for (let i = 0; i < args.length; i += 2) {
  if (args[i] !== "--dataset" || args[i + 1] === undefined) throw new Error("usage: [--extend] [--dataset DIR]...");
  extra.push(args[i + 1] as string);
}

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const format = resolveFormat(engine, GOLDEN_FORMAT);
const targets = loadFidelityTargets();
checkUnreachable(targets.unreachable, goldenTeamVocabulary().roster);

interface DatasetRecord {
  paste: string;
  canonical_team_hash: string;
  origin?: { dataset_id: string };
}
const sources = [
  { file: join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "datasets/fixtures-v1/teams-dev.jsonl"), manifest: undefined },
  ...extra.map((dir) => ({ file: join(dir, "records.dev.jsonl"), manifest: join(dir, "manifest.json") })),
];
const pool: PoolTeam[] = [];
const datasets: Record<string, unknown>[] = [];
for (const source of sources) {
  const before = pool.length;
  let dataset_id = "fixtures-v1";
  for (const line of readFileSync(source.file, "utf8").split("\n").filter((l) => l !== "")) {
    const record = JSON.parse(line) as DatasetRecord;
    dataset_id = record.origin?.dataset_id ?? dataset_id;
    // The team a battle starts with is the validator's normalised form (as the training loop plays it).
    const team = importPaste(engine, record.paste);
    validateTeam(engine, format, team);
    pool.push({ team, origin: { dataset_id, canonical_team_hash: record.canonical_team_hash } });
  }
  const manifest =
    source.manifest === undefined ? undefined : (JSON.parse(readFileSync(source.manifest, "utf8")) as Record<string, unknown>);
  datasets.push({
    dataset_id,
    teams: pool.length - before,
    ...(manifest === undefined
      ? { note: "committed: fixtures/teams/<format>/datasets/fixtures-v1 (dev split)" }
      : {
          manifest_hash: recordHash(manifest),
          generator: manifest.generator,
          note: "generated with `generate` (split dev; public keys in the README), not committed; only the teams of the kept battles are",
        }),
  });
}

// --extend keeps the committed battles as they are and adds only what a target still lacks. The seeds start at 0x7000 plus
// 0x1000 per committed battle, so an extension never repeats the seeds of an earlier capture or extension.
const existing = extend ? (loadRandomCorpus(FIDELITY_DIR) as unknown as FidelityBattleFile[]) : [];
const files = await captureFidelityCorpus(engine, pool, targets, {
  seed_base: extend ? 0x7000 + existing.length * 0x1000 : 0x6000,
  log: console.log,
  existing,
});
// Only the generated files are replaced: targets.json (the reviewed input) and README.md stay as they are.
mkdirSync(FIDELITY_DIR, { recursive: true });
for (const name of readdirSync(FIDELITY_DIR)) {
  if (/^b\d+\.json$/.test(name)) rmSync(join(FIDELITY_DIR, name));
}
for (const file of files) writeFileSync(join(FIDELITY_DIR, `${file.id}.json`), `${JSON.stringify(file)}\n`);
writeFileSync(
  join(FIDELITY_DIR, RANDOM_MANIFEST),
  `${JSON.stringify(
    {
      provenance: "synthetic: see b*.json; found by capture-search-fidelity-corpus.ts from the sources below",
      format: GOLDEN_FORMAT,
      sources: datasets,
      battles: files.map((b) => ({ id: b.id, seed: b.seed, turns: b.turns, winner: b.winner, actors: b.actors })),
    },
    null,
    2,
  )}\n`,
);
const summary = fidelitySummary(files, targets);
writeFileSync(join(FIDELITY_DIR, FIDELITY_SUMMARY), `${JSON.stringify(summary, null, 2)}\n`);
for (const item of summary.items) console.log(`${item.id}: ${item.turns}/${item.min_turns} turns in ${item.battles} battles`);
console.log(`${files.length} battles, ${summary.battle_turns} turns`);
