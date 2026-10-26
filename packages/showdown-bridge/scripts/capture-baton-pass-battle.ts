// Re-capture the Baton Pass battle (fixtures/showdown/<format>/baton-pass/):
//   node packages/showdown-bridge/scripts/capture-baton-pass-battle.ts
// A seeded random-legal vs random-legal battle (the random corpus's machinery) on a team of
// three golden Pokémon and three stat-boosters that know Swords Dance and Baton Pass, found by
// searching actor seeds until a Baton Pass hands raised stages to the incoming Pokémon and that
// Pokémon then raises the same stat again (the engine prints no line for the stages it copies,
// only `|switch|…|[from] Baton Pass`, so a reducer that resets them at the switch reaches an
// impossible stage count later; #1099). The found spec is committed; CI replays it and never
// re-runs this search.
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

const BATON_DIR = join(RANDOM_DIR, "..", "baton-pass");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const spread = { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 };
const booster = (species: string, item: string, ability: string, attack: string): PokemonSet => ({
  name: "",
  species,
  item,
  ability,
  moves: ["Swords Dance", "Baton Pass", "Protect", attack],
  nature: "Serious",
  gender: "",
  evs: spread,
  ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
  level: 50,
});
const team = [
  ...syntheticTeam().slice(0, 3),
  booster("Leafeon", "Sitrus Berry", "Leaf Guard", "Leaf Blade"),
  booster("Gliscor", "Focus Sash", "Hyper Cutter", "Earthquake"),
  booster("Hawlucha", "Lum Berry", "Limber", "Close Combat"),
];

/** A side's Baton Pass switches: the stages the passer had, and whether the receiver raised a passed stat again. */
function passes(lines: string[], side: "p1" | "p2"): { carried: number; raisedAgain: boolean } {
  const stages = new Map<string, number>(); // "p1a:atk" -> stage of the Pokémon in that position
  const passedStats = new Map<string, Set<string>>();
  let carried = 0;
  let raisedAgain = false;
  for (const line of lines) {
    const part = line.split("|");
    const slot = /^(p[12][ab]): /.exec(part[2] ?? "")?.[1];
    if (slot === undefined || !slot.startsWith(side)) continue;
    if (part[1] === "-boost" || part[1] === "-unboost") {
      const key = `${slot}:${part[3]}`;
      stages.set(key, (stages.get(key) ?? 0) + (part[1] === "-boost" ? 1 : -1) * Number(part[4]));
      if (part[1] === "-boost" && passedStats.get(slot)?.has(part[3] as string)) raisedAgain = true;
    } else if (part[1] === "switch" || part[1] === "drag") {
      const keep = [...stages].filter(([key]) => key.startsWith(`${slot}:`));
      for (const [key] of keep) stages.delete(key);
      passedStats.delete(slot);
      if (!line.includes("[from] Baton Pass")) continue;
      for (const [key, value] of keep) {
        if (value <= 0) continue;
        stages.set(key, value);
        carried += value;
        passedStats.set(slot, (passedStats.get(slot) ?? new Set()).add(key.split(":")[1] as string));
      }
    }
  }
  return { carried, raisedAgain };
}

for (let i = 0; i < 3000; i++) {
  const spec: RandomBattleSpec = {
    id: "baton-pass",
    seed: `sodium,${i.toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: i, p2: i + 1 },
    teams: { p1: team, p2: team },
  };
  const record = await playRandomBattle(engine, spec);
  if (record.forfeit !== undefined || record.turns >= RANDOM_MAX_TURNS) continue;
  const seen = passes(record.channels.p1 as string[], "p1");
  if (seen.carried === 0 || !seen.raisedAgain) continue;
  mkdirSync(BATON_DIR, { recursive: true });
  writeFileSync(join(BATON_DIR, "battle.json"), `${JSON.stringify(randomBattleFile(spec, record), null, 2)}\n`);
  console.log(`wrote baton-pass/battle.json (${spec.seed}, actor seeds ${i}, ${i + 1}, ${record.turns} turns)`);
  process.exit(0);
}
throw new Error("no battle with a Baton Pass of raised stages in 3000 seeds");
