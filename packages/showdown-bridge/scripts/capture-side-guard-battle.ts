// Re-capture the side-guard battle (fixtures/showdown/<format>/side-guard/):
//   node packages/showdown-bridge/scripts/capture-side-guard-battle.ts
// A seeded random-legal vs random-legal battle (the random corpus's machinery) whose side p1 leads
// two Pokemon that both know Wide Guard, found by searching actor seeds until both use it in the
// same turn. The second use prints neither `-singleturn` nor `-fail`: `Side.addSideCondition`
// returns false without a message because the condition is already up, while the move's
// `onHitSide` still adds the user's `stall` volatile and the move does not fail (#1281). The found
// spec is committed; CI replays it and never re-runs this search.
// SHOWDOWN_DIR points at another built checkout (a worktree without its own).
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, type PokemonSet } from "../src/engine.ts";
import { syntheticTeam } from "../src/golden.ts";
import {
  playRandomBattle,
  RANDOM_DIR,
  RANDOM_MAX_TURNS,
  type RandomBattleSpec,
  randomBattleFile,
} from "../src/random-corpus.ts";

const GUARD_DIR = join(RANDOM_DIR, "..", "side-guard");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const spread = { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 };
const set = (species: string, item: string, ability: string, moves: string[]): PokemonSet => ({
  name: "",
  species,
  item,
  ability,
  moves,
  nature: "Serious",
  gender: "",
  evs: spread,
  ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
  level: 50,
});
const golden = syntheticTeam();
const aegislash = set("Aegislash", "", "Stance Change", ["Wide Guard", "King's Shield", "Shadow Ball", "Iron Head"]);
const aerodactyl = set("Aerodactyl", "", "Pressure", ["Wide Guard", "Rock Slide", "Earthquake", "Protect"]);
const teams = { p1: [aegislash, aerodactyl, ...golden.slice(1, 5)], p2: golden.slice(0, 6) };

/** A `|move|` line of a stalling side-condition move followed by a line that is neither outcome. */
const SILENT = /^\|move\|p1[ab]: [^|]+\|Wide Guard\|/;

for (let i = 0; i < 4000; i++) {
  const spec: RandomBattleSpec = {
    id: "side-guard",
    seed: `sodium,${i.toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: i, p2: i + 1 },
    teams,
  };
  const record = await playRandomBattle(engine, spec);
  if (record.forfeit !== undefined || record.turns >= RANDOM_MAX_TURNS) continue;
  const lines = record.channels.p1 as string[];
  const silent = lines.some((l, at) => SILENT.test(l) && !/^\|-(singleturn|fail)\|/.test(lines[at + 1] ?? ""));
  if (!silent) continue;
  mkdirSync(GUARD_DIR, { recursive: true });
  writeFileSync(join(GUARD_DIR, "battle.json"), `${JSON.stringify(randomBattleFile(spec, record), null, 2)}\n`);
  console.log(`wrote side-guard/battle.json (${spec.seed}, actor seeds ${i}, ${i + 1}, ${record.turns} turns)`);
  process.exit(0);
}
throw new Error("no battle with a silent Wide Guard in 4000 seeds");
