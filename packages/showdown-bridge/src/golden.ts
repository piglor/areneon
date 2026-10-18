import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Engine, PokemonSet } from "./engine.ts";
import { resolveFormat, type VersionManifest } from "./formats.ts";
import type { FormatRulesRecord } from "./rules.ts";
import type { JsonObject, RequestPhase } from "./request.ts";
import { type Actor, type BattleRecord, type Player, runBattle } from "./runner.ts";
import { importPaste, type TeamVerdict, validateTeam } from "./teams.ts";

/** Sanitized request fixtures for the pinned engine (CC0, see fixtures/LICENSE). */
export const GOLDEN_DIR = fileURLToPath(
  new URL("../../../fixtures/showdown/gen9championsvgc2026regmc", import.meta.url),
);
export const GOLDEN_FORMAT = "gen9championsvgc2026regmc";
export const GOLDEN_SEED = "sodium,000102030405060708090a0b0c0d0e0f";

export interface RequestFixture {
  provenance: string;
  manifest: VersionManifest;
  /** The embedded format rules record (ADR-007), so each fixture replays alone. */
  format_rules: FormatRulesRecord;
  seed: string;
  player: string;
  request_seq: number;
  request_hash: string;
  phase: RequestPhase;
  request: JsonObject;
}

export const GOLDEN_PHASES: Readonly<Record<string, RequestPhase>> = {
  "preview.request.json": "preview",
  "turn.request.json": "move",
  "force-switch.request.json": "forceSwitch",
};

/** The synthetic team both sides bring. It is not a real competitive team. */
export function syntheticTeam(): PokemonSet[] {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, "synthetic-team.json"), "utf8")) as PokemonSet[];
}

/** Let the engine pick: Showdown's own `default` choice is always legal. */
export const engineDefault: Actor = () => "default";

/** Replay the golden battle on `engine`. */
export function goldenBattle(engine: Engine): Promise<BattleRecord> {
  const team = syntheticTeam();
  return runBattle({
    engine,
    formatId: GOLDEN_FORMAT,
    seed: GOLDEN_SEED,
    teams: { p1: team, p2: team },
    actors: { p1: engineDefault, p2: engineDefault },
  });
}

export const GOLDEN_SUMMARY = "battle.summary.json";

/**
 * Capture-time provenance (ADR-007: fixtures record `rules_snapshot_hash`).
 * It sits apart from the compared fixtures because documentation-only rules
 * edits change it, and those must never force a golden recapture.
 */
export const GOLDEN_PROVENANCE = "capture.provenance.json";

export interface CaptureProvenance {
  rules_snapshot_hash: string;
}

export const captureProvenance = (record: BattleRecord): CaptureProvenance => ({
  rules_snapshot_hash: record.rules_snapshot_hash,
});

/**
 * Committed identity of the whole golden battle. A fresh process must
 * reproduce `omniscient_log_sha256` exactly; the log itself is not
 * committed because it carries both sides' hidden information.
 */
export interface BattleSummary {
  provenance: string;
  manifest: VersionManifest;
  /** The embedded format rules record (ADR-007). */
  format_rules: FormatRulesRecord;
  seed: string;
  winner: Player | null;
  turns: number;
  omniscient_log_sha256: string;
  choices: BattleRecord["choices"];
}

export function battleSummary(record: BattleRecord): BattleSummary {
  return {
    provenance: "synthetic: Areneon golden battle, engine default choices",
    manifest: record.manifest,
    format_rules: record.format_rules,
    seed: record.seed,
    winner: record.winner,
    turns: record.turns,
    omniscient_log_sha256: createHash("sha256")
      .update(record.omniscient_log.join("\n"))
      .digest("hex"),
    choices: record.choices,
  };
}

/** The first p1 request of each golden phase, keyed by fixture file name. */
export function goldenFixtures(record: BattleRecord): Record<string, RequestFixture> {
  const fixtures: Record<string, RequestFixture> = {};
  for (const [file, phase] of Object.entries(GOLDEN_PHASES)) {
    const found = record.requests.p1.find((request) => request.phase === phase);
    if (!found) throw new Error(`golden battle produced no p1 ${phase} request`);
    fixtures[file] = {
      provenance: "synthetic: Areneon golden battle, engine default choices",
      manifest: record.manifest,
      format_rules: record.format_rules,
      seed: record.seed,
      player: found.player,
      request_seq: found.request_seq,
      request_hash: found.request_hash,
      phase: found.phase,
      request: found.request,
    };
  }
  return fixtures;
}

/** Sanitized PokePaste team fixtures and their committed validator log. */
export const TEAMS_DIR = fileURLToPath(
  new URL("../../../fixtures/teams/gen9championsvgc2026regmc", import.meta.url),
);
export const TEAM_VERDICTS = "verdicts.json";

export interface TeamFixtureManifest {
  provenance: string;
  license: string;
  format_id: string;
  /** `reason`: text the validator's first problem must contain (illegal only). */
  fixtures: Record<string, { expect: "legal" | "illegal"; purpose: string; reason?: string }>;
}

export function teamFixtureManifest(): TeamFixtureManifest {
  return JSON.parse(readFileSync(join(TEAMS_DIR, "manifest.json"), "utf8")) as TeamFixtureManifest;
}

/** Paste files present on disk, whether or not the manifest lists them. */
export function teamFixtureFiles(): string[] {
  return readdirSync(TEAMS_DIR)
    .filter((file) => file.endsWith(".txt"))
    .sort();
}

/** The committed validator log: every team fixture's verdict under one manifest. */
export interface TeamVerdictLog {
  manifest: VersionManifest;
  format_rules: FormatRulesRecord;
  verdicts: Record<string, TeamVerdict>;
}

/** The pinned validator's verdict on every team fixture, keyed by file. */
export function teamVerdicts(engine: Engine): TeamVerdictLog {
  const format = resolveFormat(engine, teamFixtureManifest().format_id);
  const verdicts: Record<string, TeamVerdict> = {};
  for (const file of teamFixtureFiles()) {
    const team = importPaste(engine, readFileSync(join(TEAMS_DIR, file), "utf8"));
    verdicts[file] = validateTeam(engine, format, team);
  }
  return { manifest: format.manifest, format_rules: format.rules, verdicts };
}
