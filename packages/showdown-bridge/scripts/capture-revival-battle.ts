// Re-capture the Revival Blessing battle (fixtures/showdown/<format>/revival/):
//   node packages/showdown-bridge/scripts/capture-revival-battle.ts
// A seeded random-legal vs random-legal battle (the random corpus's
// machinery) on synthetic teams, one of which brings a Pawmot with Revival
// Blessing, found by searching actor seeds until p1's own channel holds a
// `reviving` request. The found spec is committed; CI replays it and never
// re-runs this search.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, type PokemonSet } from "../src/engine.ts";
import { syntheticTeam } from "../src/golden.ts";
import {
  playRandomBattle,
  RANDOM_DIR,
  type RandomBattleSpec,
  randomBattleFile,
} from "../src/random-corpus.ts";

const REVIVAL_DIR = join(RANDOM_DIR, "..", "revival");

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const golden = syntheticTeam();
const pawmot: PokemonSet = {
  name: "",
  species: "Pawmot",
  item: "Lum Berry",
  ability: "Iron Fist",
  moves: ["Revival Blessing", "Protect", "Close Combat", "Thunder Punch"],
  nature: "Serious",
  gender: "",
  evs: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 },
  ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
  level: 50,
};

for (let i = 0; i < 5000; i++) {
  const spec: RandomBattleSpec = {
    id: "revival",
    seed: `sodium,${i.toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: i, p2: i + 1 },
    teams: { p1: [pawmot, ...golden.slice(1)], p2: golden },
  };
  const record = await playRandomBattle(engine, spec);
  if (!record.channels.p1.some((l) => l.startsWith("|request|") && l.includes('"reviving":true'))) {
    continue;
  }
  mkdirSync(REVIVAL_DIR, { recursive: true });
  writeFileSync(
    join(REVIVAL_DIR, "battle.json"),
    `${JSON.stringify(randomBattleFile(spec, record), null, 2)}\n`,
  );
  // The engine-judged legal set of every reviving request, for the Python side.
  const legal = record.requests.p1
    .filter((r) => r.phase === "forceSwitch" && JSON.stringify(r.request).includes('"reviving":true'))
    .map((r) => r.legal);
  writeFileSync(join(REVIVAL_DIR, "legal.json"), `${JSON.stringify(legal, null, 2)}\n`);
  console.log(`wrote revival/battle.json (${spec.seed}, actor seeds ${i}, ${i + 1}, ${record.turns} turns)`);
  process.exit(0);
}
throw new Error("no reviving request in 5000 battles");
