// Re-capture the Fling battle (fixtures/showdown/<format>/fling/):
//   node packages/showdown-bridge/scripts/capture-fling-battle.ts
// A seeded random-legal vs random-legal battle (the random corpus's machinery) with a Dedenne
// that holds a Grassy Seed and knows Fling against an Arboliva (Seed Sower), found by searching
// actor seeds until the Fling's hit starts Grassy Terrain, the seed is eaten on the spot, and the
// move's own `onUpdate` then ends an item that is already gone: `|-enditem|POKEMON||[from] move:
// Fling` with no item name (#1162). The found spec is committed; CI replays it and never re-runs
// this search.
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

const FLING_DIR = join(RANDOM_DIR, "..", "fling");

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
const dedenne = set("Dedenne", "Grassy Seed", "Plus", ["Fling", "Light Screen", "Charge", "Dazzling Gleam"]);
const arboliva = set("Arboliva", "Lum Berry", "Seed Sower", ["Grass Knot", "Protect", "Earth Power", "Leech Seed"]);
const teams = { p1: [dedenne, ...golden.slice(0, 5)], p2: [arboliva, ...golden.slice(0, 5)] };

for (let i = 0; i < 4000; i++) {
  const spec: RandomBattleSpec = {
    id: "fling",
    seed: `sodium,${i.toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: i, p2: i + 1 },
    teams,
  };
  const record = await playRandomBattle(engine, spec);
  if (record.forfeit !== undefined || record.turns >= RANDOM_MAX_TURNS) continue;
  const lines = record.channels.p1 as string[];
  if (!lines.some((l) => /^\|-enditem\|[^|]+\|\|\[from\] move: Fling$/.test(l))) continue;
  mkdirSync(FLING_DIR, { recursive: true });
  writeFileSync(join(FLING_DIR, "battle.json"), `${JSON.stringify(randomBattleFile(spec, record), null, 2)}\n`);
  console.log(`wrote fling/battle.json (${spec.seed}, actor seeds ${i}, ${i + 1}, ${record.turns} turns)`);
  process.exit(0);
}
throw new Error("no battle with an empty Fling item end in 4000 seeds");
