import type { Engine, PokemonSet } from "./engine.ts";
import type { ResolvedFormat } from "./formats.ts";

export class TeamImportError extends Error {
  override name = "TeamImportError";
  /** The input lines the pinned engine's export did not reproduce. */
  readonly lines: string[];

  constructor(message: string, lines: string[] = []) {
    super(message);
    this.lines = lines;
  }
}

/** The pinned validator's judgement of one team. */
export interface TeamVerdict {
  valid: boolean;
  /** The validator's own messages, verbatim; empty when valid. */
  problems: string[];
}

const normaliseLine = (line: string): string => line.trim().replace(/\s+/g, " ");

/** One Pokémon per blank-line-separated block, as non-empty normalised lines. */
const blocks = (paste: string): string[][] =>
  paste
    .split(/\n\s*\n/)
    .map((block) => block.split("\n").map(normaliseLine).filter((line) => line !== ""))
    .filter((block) => block.length > 0);

/** Write `team` as PokePaste text with the pinned engine's `Teams.export`. */
export function exportPaste(engine: Engine, team: PokemonSet[]): string {
  return engine.sim.Teams.export(team);
}

/**
 * Read PokePaste text with the pinned engine's `Teams.import`.
 *
 * Showdown skips lines it does not understand and lets a repeated line
 * overwrite an earlier one, which would silently lose data (an `SPs:` line
 * leaves every Stat Point at 0). So each Pokémon's input lines must all
 * reappear, as many times, in that Pokémon's re-export, or the import fails.
 * This accepts only the canonical form the pinned engine exports: a line it
 * would write differently (reordered stats, `Level: 100`, `IVs: 31 HP`, a
 * `=== [format] ===` header) is rejected even though it parsed.
 */
export function importPaste(engine: Engine, paste: string): PokemonSet[] {
  const team = engine.sim.Teams.import(paste);
  if (!team || team.length === 0) {
    throw new TeamImportError("no Pokémon found in the paste");
  }
  const input = blocks(paste);
  if (input.length !== team.length) {
    throw new TeamImportError(
      `the paste has ${input.length} blocks but the pinned engine read ${team.length} Pokémon`,
    );
  }
  const unreproduced = input.flatMap((lines, i) => {
    const exported = blocks(exportPaste(engine, team.slice(i, i + 1))).flat();
    return lines.filter((line) => {
      const at = exported.indexOf(line);
      if (at === -1) return true;
      exported.splice(at, 1);
      return false;
    });
  });
  if (unreproduced.length > 0) {
    throw new TeamImportError(
      `not reproduced by the pinned engine's export: ${unreproduced.join(" | ")}`,
      unreproduced,
    );
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
