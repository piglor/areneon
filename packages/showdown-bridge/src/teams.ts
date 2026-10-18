import type { Engine, PokemonSet } from "./engine.ts";
import type { ResolvedFormat } from "./formats.ts";

export class TeamImportError extends Error {
  override name = "TeamImportError";
}

/** The pinned validator's judgement of one team. */
export interface TeamVerdict {
  valid: boolean;
  /** The validator's own messages, verbatim; empty when valid. */
  problems: string[];
}

const normaliseLine = (line: string): string => line.trim().replace(/\s+/g, " ").toLowerCase();

/** Write `team` as PokePaste text with the pinned engine's `Teams.export`. */
export function exportPaste(engine: Engine, team: PokemonSet[]): string {
  return engine.sim.Teams.export(team);
}

/**
 * Read PokePaste text with the pinned engine's `Teams.import`.
 *
 * Showdown skips lines it does not understand, which would silently lose
 * data (an `SPs:` line leaves every Stat Point at 0). So every non-empty
 * input line must survive into the re-exported paste, or the import fails.
 * Pastes exported by Showdown always pass.
 */
export function importPaste(engine: Engine, paste: string): PokemonSet[] {
  const team = engine.sim.Teams.import(paste);
  if (!team || team.length === 0) {
    throw new TeamImportError("no Pokémon found in the paste");
  }
  const kept = new Set(exportPaste(engine, team).split("\n").map(normaliseLine));
  const dropped = paste
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "" && !kept.has(normaliseLine(line)));
  if (dropped.length > 0) {
    throw new TeamImportError(`the pinned engine ignored: ${dropped.join(" | ")}`);
  }
  return team;
}

/**
 * Judge `team` with the pinned validator for a format `resolveFormat`
 * accepted. Areneon adds no rules of its own (ADR-003): the verdict is
 * exactly the engine's.
 */
export function validateTeam(engine: Engine, format: ResolvedFormat, team: PokemonSet[]): TeamVerdict {
  const problems = engine.sim.TeamValidator.get(format.formatId).validateTeam(team) ?? [];
  return { valid: problems.length === 0, problems };
}
