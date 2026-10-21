// Re-capture the Illusion battle (fixtures/showdown/<format>/illusion/):
//   node packages/showdown-bridge/scripts/capture-illusion-battle.ts
// A seeded random-legal vs random-legal battle (the random corpus's machinery) on
// the golden team with its last Pokémon replaced by a Zoroark (Illusion) on both
// sides, found by searching actor seeds until each side's disguise breaks
// (`|replace|`) before the battle ends on its own. The found spec is committed; CI
// replays it and never re-runs this search.
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

const ILLUSION_DIR = join(RANDOM_DIR, "..", "illusion");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const golden = syntheticTeam();
const zoroark: PokemonSet = {
  name: "",
  species: "Zoroark",
  item: "Big Root",
  ability: "Illusion",
  moves: ["Shadow Claw", "Calm Mind", "Spite", "Rain Dance"],
  nature: "Serious",
  gender: "",
  evs: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 },
  ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
  level: 50,
};
const team = [...golden.slice(0, -1), zoroark];

for (let i = 0; i < 3000; i++) {
  const spec: RandomBattleSpec = {
    id: "illusion",
    seed: `sodium,${i.toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: i, p2: i + 1 },
    teams: { p1: team, p2: team },
  };
  const record = await playRandomBattle(engine, spec);
  const lines = record.channels.p1;
  const broken = (player: string) => lines.some((l) => l.startsWith(`|replace|${player}`));
  if (record.forfeit !== undefined || record.turns >= RANDOM_MAX_TURNS || !broken("p1") || !broken("p2")) continue;
  mkdirSync(ILLUSION_DIR, { recursive: true });
  writeFileSync(join(ILLUSION_DIR, "battle.json"), `${JSON.stringify(randomBattleFile(spec, record), null, 2)}\n`);
  console.log(`wrote illusion/battle.json (${spec.seed}, actor seeds ${i}, ${i + 1}, ${record.turns} turns)`);
  process.exit(0);
}
throw new Error("no battle with both disguises broken in 3000 seeds");
