import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Engine, PokemonSet } from "./engine.ts";
import type { VersionManifest } from "./formats.ts";
import type { FormatRulesRecord } from "./rules.ts";
import type { JsonObject, RequestPhase } from "./request.ts";
import { type Actor, type BattleRecord, type Player, runBattle } from "./runner.ts";

/** Sanitized request fixtures for the pinned engine (CC0, see fixtures/LICENSE). */
export const GOLDEN_DIR = fileURLToPath(
  new URL("../../../fixtures/showdown/gen9championsvgc2026regmc", import.meta.url),
);
export const GOLDEN_FORMAT = "gen9championsvgc2026regmc";
export const GOLDEN_SEED = "sodium,000102030405060708090a0b0c0d0e0f";

export interface RequestFixture {
  provenance: string;
  manifest: VersionManifest;
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
 * Committed identity of the whole golden battle. A fresh process must
 * reproduce `omniscient_log_sha256` exactly; the log itself is not
 * committed because it carries both sides' hidden information.
 */
export interface BattleSummary {
  provenance: string;
  manifest: VersionManifest;
  /**
   * The embedded format rules record (ADR-007). `rules_snapshot_hash` is
   * left out on purpose: it changes with documentation-only edits, and this
   * file is compared byte for byte.
   */
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
