// Re-capture the open-regime channel corpus (src/open-sheet-corpus.ts) after a
// Showdown pin bump or an engine-facing change:
//   node packages/showdown-bridge/scripts/capture-open-regime-corpus.ts
// Seeded A1 battles of the open-sheet format, one game each, on the random
// corpus's teams. SHOWDOWN_DIR points at another built checkout.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine } from "../src/engine.ts";
import {
  OPEN_SHEET_DIR,
  OPEN_SHEET_FORMAT,
  OPEN_SHEET_PROVENANCE,
  openSheetSpecs,
} from "../src/open-sheet-corpus.ts";
import { playRandomBattle, RANDOM_MANIFEST, randomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
rmSync(OPEN_SHEET_DIR, { recursive: true, force: true });
mkdirSync(OPEN_SHEET_DIR, { recursive: true });
const battles: { id: string; seed: string; turns: number; winner: string | null }[] = [];
for (const spec of openSheetSpecs()) {
  const record = await playRandomBattle(engine, spec, randomLegalActor, OPEN_SHEET_FORMAT);
  if (record.forfeit !== undefined) throw new Error(`${spec.id} forfeited`);
  writeFileSync(
    join(OPEN_SHEET_DIR, `${spec.id}.json`),
    `${JSON.stringify(randomBattleFile(spec, record, OPEN_SHEET_PROVENANCE), null, 2)}\n`,
  );
  battles.push({ id: spec.id, seed: spec.seed, turns: record.turns, winner: record.winner });
}
writeFileSync(
  join(OPEN_SHEET_DIR, RANDOM_MANIFEST),
  `${JSON.stringify({ format: OPEN_SHEET_FORMAT, battles }, null, 2)}\n`,
);
console.log(`wrote ${battles.length} open-regime battles to ${OPEN_SHEET_DIR}`);
