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
  name: string;
  mod: string;
  gameType: string;
  /** The format's `battle` options (for example `trunc`), when it sets any. */
  battle?: Record<string, unknown>;
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
  /** The level and move-count rules the format sets, read by the mechanics scenario check. */
  adjustLevelDown?: number | null;
  minLevel?: number;
  maxLevel?: number;
  defaultLevel?: number;
  maxMoveCount?: number;
  minSourceGen?: number;
  maxTotalLevel?: number | null;
}

/** One Pokémon of a battle, narrowed to the stats the mechanics probe reads. */
export interface SimPokemon {
  maxhp: number;
  /** The five stats other than HP, as the engine computed them at the format's level. */
  storedStats: Record<"atk" | "def" | "spa" | "spd" | "spe", number>;
}

/** One parsed action of a side's choice (`side.choice.actions`). */
export interface SimAction {
  choice: string;
  /** The Pokémon acting (move, team) or switching out. */
  pokemon?: object;
  /** The Pokémon switched in. */
  target?: object;
  moveid?: string;
  targetLoc?: number;
  mega?: boolean;
  megax?: boolean;
  megay?: boolean;
  terastallize?: string;
  /** Team Preview: the position this Pokémon was picked for. */
  index?: number;
}

/** One side of an in-process battle, narrowed to checking a choice. */
export interface SimSide {
  id: string;
  /** The side's Pokémon, in the order its requests list them. */
  pokemon: object[];
  /** `teampreview`, `move`, `switch`, or `` when nothing is asked. */
  requestState: string;
  /** The request last sent to this side, as JSON-safe data. */
  activeRequest: object | null;
  choice: { error: string; actions: SimAction[] };
  /** Parse and check a whole choice string without committing the turn. */
  choose(input: string): boolean;
  isChoiceDone(): boolean;
  clearChoice(): void;
}

/**
 * An in-process battle (ADR-006). `CloneJudge` reads it; the choice corpus
 * drives one directly to reach each case's request.
 */
export interface SimBattle {
  sides: SimSide[];
  ended: boolean;
  turn: number;
  /** When set, a refused choice throws instead of returning false. */
  strictChoices?: boolean;
  setPlayer(slot: string, options: { name: string; team: string }): void;
  /** Commits the turn once both sides are done. Never called by a judge. */
  choose(slot: string, input: string): boolean;
  toJSON(): object;
}

/** One mod's data tables, narrowed to name lookups. */
export interface SimModdedDex {
  items: { get(name: string): { exists: boolean; name: string; megaStone?: Record<string, string> } };
  species: { get(name: string): { exists: boolean; name: string; baseSpecies: string } };
  abilities: { get(name: string): { exists: boolean; name: string } };
}

/** The slice of Showdown's `sim` module Areneon depends on. */
export interface ShowdownSim {
  Battle: {
    new (options: { formatid: string; seed: string }): SimBattle;
    /** Rebuild a battle from `toJSON()`; without `send` it emits nothing. */
    fromJSON(state: object): SimBattle;
  };
  /** `battle` is the in-process battle once `>start` has been written. */
  BattleStream: new () => { battle: SimBattle | null };
  getPlayerStreams(stream: object): Record<"omniscient" | "p1" | "p2", SimStream>;
  /** The engine's seeded generator (`sodium,<hex>` seeds): `random()` is a 32-bit word divided by 2^32. */
  PRNG: new (seed: string) => { random(): number };
  Dex: {
    formats: {
      get(id: string): SimFormat;
      getRuleTable(format: SimFormat): SimRuleTable;
    };
    /** The data of one mod (a format's `mod`), narrowed to what the team inventory reads. */
    mod(mod: string): SimModdedDex;
  };
  TeamValidator: {
    get(formatId: string): {
      validateTeam(team: PokemonSet[]): string[] | null;
      /** One set against the format's rules; it normalises `set` in place, so pass a copy. */
      validateSet(set: PokemonSet, teamHas: Record<string, number>): string[] | null;
    };
  };
  Teams: {
    pack(team: PokemonSet[]): string;
    unpack(buf: string): PokemonSet[] | null;
    export(team: PokemonSet[]): string;
    import(buffer: string): PokemonSet[] | null;
  };
}

export interface Engine {
  commit: string;
  sim: ShowdownSim;
}

export class EngineError extends Error {
  override name = "EngineError";
}

/**
 * Check that `dir` is a built Showdown checkout at `expectedCommit`, without
 * loading it. Refuses a missing build or a checkout at any other commit.
 * `loadEngine` and the local-server runtime copy (ADR-005 §4) share it.
 * Returns the commit and the sim entry point (`dist/sim/index.js`).
 */
export function verifyShowdownCheckout(
  dir: string = DEFAULT_SHOWDOWN_DIR,
  expectedCommit: string = PINNED_SHOWDOWN_COMMIT,
): { commit: string; entry: string } {
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
  return { commit, entry };
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
  const { commit, entry } = verifyShowdownCheckout(dir, expectedCommit);
  const sim = createRequire(import.meta.url)(entry) as ShowdownSim;
  return { commit, sim };
}
