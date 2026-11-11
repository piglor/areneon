// Sheet regimes of a mixed-format run (ADR-027 section 5 and 5a, ADR-010 erratum A8, #1153).
// With `TrainingRunConfig.sheet_regimes` a run trains on games of two formats: the closed (target)
// format `format_id` and the engine's open-sheet variant `open_format`. Which regime a game has is
// a pure function of the config, the generation and the game (purpose `sheet-regime/1`), so a
// killed generation replays the same plan. A config without the field never reaches this module's
// plan functions and keeps every byte.

import { createHash } from "node:crypto";
import type { Engine } from "./engine.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import { loadRulesSnapshot } from "./rules-load.ts";
import { checkSheetVariants } from "./sheet-variants.ts";
import { sheetRegimeIn } from "./team-sheet-binding.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";

/** Names the derivation of the regime assignment; a new byte layout gets a new number. */
export const SHEET_REGIME_PURPOSE = "sheet-regime/1";

export type SheetRegime = "open" | "closed";

export type SheetRegimes = NonNullable<TrainingRunConfig["sheet_regimes"]>;

export class SheetRegimeError extends Error {
  override name = "SheetRegimeError";
}

/** SHA-256 over the JCS of `{run_key, generation, game, purpose: "sheet-regime/1"}`: the game's place in the hash order. */
export const regimeDigest = (run_key: string, generation: number, game: number): string =>
  createHash("sha256").update(jcs({ run_key, generation, game, purpose: SHEET_REGIME_PURPOSE } as Json)).digest("hex");

/** `round(games * share / 10^6)`, half up, in integers: how many self-play games of a generation are closed. */
export const closedGameCount = (games: number, closed_share_ppm: number): number => Math.floor((games * closed_share_ppm + 500_000) / 1_000_000);

/**
 * The regime of each self-play game of `generation` (counted from 1), by game index: all open in the
 * open-only generations, then exactly `closedGameCount` closed games, the ones whose `regimeDigest`
 * is lowest (ties, which SHA-256 does not give in practice, by game index).
 */
export function selfPlayRegimes(sheet: SheetRegimes, run_key: string, generation: number, games: number): SheetRegime[] {
  const regimes: SheetRegime[] = Array.from({ length: games }, () => "open");
  if (generation <= sheet.open_only_generations) return regimes;
  const order = regimes
    .map((_, game) => ({ game, digest: regimeDigest(run_key, generation, game) }))
    .sort((a, b) => (a.digest < b.digest ? -1 : a.digest > b.digest ? 1 : a.game - b.game));
  for (const { game } of order.slice(0, closedGameCount(games, sheet.closed_share_ppm))) regimes[game] = "closed";
  return regimes;
}

/** The `format_id` of each plan entry of a mixed-format generation: the self-play games by regime, every baseline game closed. */
export function planFormats(config: TrainingRunConfig, sheet: SheetRegimes, generation: number): string[] {
  const { games, baseline_games } = config.generation;
  const self = selfPlayRegimes(sheet, config.seeds.run_key, generation, games);
  const baselines = baseline_games.reduce((sum, baseline) => sum + baseline.games, 0);
  return [...self.map((regime) => (regime === "open" ? sheet.open_format : config.format_id)), ...Array.from({ length: baselines }, () => config.format_id)];
}

/** The regime counts of a plan whose entries name a `format_id` (what `LeagueRunRecord.regimes` holds). */
export function regimeCounts(plan: readonly { format_id?: string }[], config: TrainingRunConfig): { open: number; closed: number } {
  const sheet = config.sheet_regimes;
  if (sheet === undefined) throw new SheetRegimeError("the run has no sheet_regimes");
  let open = 0;
  for (const entry of plan) if (entry.format_id === sheet.open_format) open += 1;
  return { open, closed: plan.length - open };
}

/**
 * Refuse a plan that does not follow the config's regimes: every entry names one of the two formats, a baseline
 * game is closed (the damage-aware mechanics config exists for the closed format only, ADR-027 5a), and the formats
 * are the `sheet-regime/1` quota. A plan of a config without `sheet_regimes` names no format at all.
 */
export function checkPlanRegimes(plan: readonly { format_id?: string; opponent: unknown }[], config: TrainingRunConfig, generation: number): void {
  const sheet = config.sheet_regimes;
  if (sheet === undefined) {
    if (plan.some((entry) => entry.format_id !== undefined)) throw new SheetRegimeError("a plan entry names a format, but the config has no sheet_regimes");
    return;
  }
  for (const entry of plan) {
    if (entry.format_id !== config.format_id && entry.format_id !== sheet.open_format) throw new SheetRegimeError("a plan entry's format is neither the closed format nor the open format");
    if (entry.opponent !== "self" && entry.format_id !== config.format_id) throw new SheetRegimeError("a baseline game is played in the closed format only");
  }
  const expected = planFormats(config, sheet, generation);
  if (plan.length !== expected.length || plan.some((entry, game) => entry.format_id !== expected[game])) {
    throw new SheetRegimeError("the plan's formats are not the sheet-regime/1 quota of the config");
  }
}

export interface RegimeManifests {
  /** The closed (target) format's manifest: `CheckpointRecord.manifest`. */
  closed: VersionManifest;
  open: VersionManifest;
  /** Both, sorted by JCS bytes, each once: `game_manifests`. */
  sorted: VersionManifest[];
}

/**
 * Resolve both formats once (ADR-027 5a) and check them against the engine: `format_id`'s rules record is
 * `closed`, `open_format`'s is `open`, and the sheet-variant check passes (the two differ in the sheet and series
 * rules only). Throws `SheetRegimeError` (or the variant check's own error) otherwise.
 */
export function resolveRegimes(engine: Engine, config: TrainingRunConfig): RegimeManifests {
  const sheet = config.sheet_regimes;
  if (sheet === undefined) throw new SheetRegimeError("the run has no sheet_regimes");
  const snapshot = loadRulesSnapshot(engine.commit);
  if (sheetRegimeIn(snapshot, config.format_id) !== "closed") throw new SheetRegimeError("sheet_regimes: format_id's rules record must be a closed-sheet format");
  if (sheetRegimeIn(snapshot, sheet.open_format) !== "open") throw new SheetRegimeError("sheet_regimes: open_format's rules record must be an open-sheet format");
  checkSheetVariants(engine, config.format_id, sheet.open_format);
  const closed = resolveFormat(engine, config.format_id, snapshot).manifest;
  const open = resolveFormat(engine, sheet.open_format, snapshot).manifest;
  return { closed, open, sorted: sortManifests([closed, open]) };
}

/** Version manifests sorted by their JCS bytes, each once. */
export const sortManifests = (manifests: readonly VersionManifest[]): VersionManifest[] => {
  const keyed = manifests.map((manifest) => ({ manifest, key: jcs(manifest as unknown as Json) }));
  return [...new Map(keyed.map((entry) => [entry.key, entry])).values()].sort((a, b) => (a.key < b.key ? -1 : 1)).map((entry) => entry.manifest);
};
