import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * The smogon/pokemon-showdown commit Areneon treats as the reference engine
 * (ADR-003). Bumping it is a deliberate PR that also moves the
 * `vendor/pokemon-showdown` submodule and re-captures the golden fixtures.
 */
export const PINNED_SHOWDOWN_COMMIT = "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b";

export const DEFAULT_SHOWDOWN_DIR = fileURLToPath(
  new URL("../../../vendor/pokemon-showdown", import.meta.url),
);

export type StatTable = Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>;

/** One set as Showdown's team validator and `Teams.pack` accept it. */
export interface PokemonSet {
  name: string;
  species: string;
  item: string;
  ability: string;
  moves: string[];
  nature: string;
  gender: string;
  /** In the `champions` mod these hold Stat Points, not EVs. */
  evs: StatTable;
  ivs: StatTable;
  level: number;
}

/** A Showdown object stream, narrowed to what Areneon calls. */
export interface SimStream extends AsyncIterable<string> {
  write(chunk: string): void;
  writeEnd(): void;
}

export interface SimFormat {
  exists: boolean;
  id: string;
  mod: string;
  gameType: string;
}

/** A resolved format rule table: rule ID -> where it came from. */
export interface SimRuleTable extends Map<string, string> {
  /** Parameterised rules, e.g. `timerstarting` -> `420`. */
  valueRules: Map<string, string>;
  minTeamSize: number;
  maxTeamSize: number;
  pickedTeamSize: number | null;
  adjustLevel: number | null;
  evLimit: number | null;
}

/** The slice of Showdown's `sim` module Areneon depends on. */
export interface ShowdownSim {
  BattleStream: new () => object;
  getPlayerStreams(stream: object): Record<"omniscient" | "p1" | "p2", SimStream>;
  Dex: {
    formats: {
      get(id: string): SimFormat;
      getRuleTable(format: SimFormat): SimRuleTable;
    };
  };
  TeamValidator: { get(formatId: string): { validateTeam(team: PokemonSet[]): string[] | null } };
  Teams: { pack(team: PokemonSet[]): string };
}

export interface Engine {
  commit: string;
  sim: ShowdownSim;
}

export class EngineError extends Error {
  override name = "EngineError";
}

/**
 * Load the pinned, built Showdown checkout. Refuses a missing build or a
 * checkout at any other commit, so no result is ever attributed to the
 * wrong engine.
 */
export function loadEngine(
  dir: string = DEFAULT_SHOWDOWN_DIR,
  expectedCommit: string = PINNED_SHOWDOWN_COMMIT,
): Engine {
  const entry = join(dir, "dist", "sim", "index.js");
  if (!existsSync(entry)) {
    throw new EngineError(`Showdown build not found at ${entry}; run \`make showdown\``);
  }
  const commit = execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
  if (commit !== expectedCommit) {
    throw new EngineError(`Showdown checkout is at ${commit}, expected pinned ${expectedCommit}`);
  }
  const sim = createRequire(import.meta.url)(entry) as ShowdownSim;
  return { commit, sim };
}
