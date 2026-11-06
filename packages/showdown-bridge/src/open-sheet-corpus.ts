// The open-regime channel corpus (ADR-027 section 4, section 10 slice 1):
// seeded A1 battles of the engine's open-sheet format, ONE game each (no
// series), with each player's own channel kept as a PlayerChannelLog. The
// channels hold the engine's `|showteam|` lines; reading them is slice 2.
// Teams are the random-play corpus's synthetic teams, which the open-sheet
// format accepts (it differs from the target in the sheet and series rules
// only, `checkSheetVariants`). CI replays each committed spec and compares.
import { join } from "node:path";
import { GOLDEN_DIR } from "./golden.ts";
import { loadRandomCorpus, type RandomBattleFile, type RandomBattleSpec } from "./random-corpus.ts";

/** The engine's open-sheet variant of the target (`team_sheets: open`). */
export const OPEN_SHEET_FORMAT = "gen9championsvgc2026regmcbo3";
export const OPEN_SHEET_DIR = join(GOLDEN_DIR, "..", OPEN_SHEET_FORMAT, "open-regime");
export const OPEN_SHEET_BATTLES = 8;
/** Seeds sit apart from the random corpus's (`i` from 0), so no battle repeats one. */
export const OPEN_SHEET_SEED_BASE = 0x0e000;

export const OPEN_SHEET_PROVENANCE =
  "synthetic: random-legal vs random-legal on the random corpus's teams, one game of the open-sheet format (no series)";

/** The specs to play: the first battles of the random corpus lend their teams. */
export function openSheetSpecs(): RandomBattleSpec[] {
  return loadRandomCorpus()
    .slice(0, OPEN_SHEET_BATTLES)
    .map((source, i) => ({
      id: `o${String(i).padStart(2, "0")}`,
      seed: `sodium,${(OPEN_SHEET_SEED_BASE + i).toString(16).padStart(32, "0")}`,
      actor_seeds: { p1: OPEN_SHEET_SEED_BASE + 2 * i, p2: OPEN_SHEET_SEED_BASE + 2 * i + 1 },
      teams: source.teams,
    }));
}

export const loadOpenSheetCorpus = (): RandomBattleFile[] => loadRandomCorpus(OPEN_SHEET_DIR, /^o\d+\.json$/);
