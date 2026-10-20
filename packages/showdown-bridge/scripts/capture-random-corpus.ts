// Re-capture the random-play channel corpus (src/random-corpus.ts) after a
// Showdown pin bump or an engine-facing change:
//   node packages/showdown-bridge/scripts/capture-random-corpus.ts
// Teams are drawn here, from a fixed seed, out of the pinned engine's
// learnsets and kept only if the pinned TeamValidator accepts them. About
// half of the Pokémon hold their Mega Stone when they have one. The found
// specs are committed; CI replays them and never re-runs this search.
// SHOWDOWN_DIR points at another built checkout (a worktree without its own).
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEngine, type PokemonSet } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import {
  playRandomBattle,
  RANDOM_DIR,
  RANDOM_MANIFEST,
  type RandomBattleSpec,
  randomBattleFile,
} from "../src/random-corpus.ts";

const BATTLES = 40;
const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);

let state = 0x5eed1234;
const next = (): number => {
  state = (state + 0x6d2b79f5) >>> 0;
  let t = Math.imul(state ^ (state >>> 15), state | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const pick = <T>(items: T[]): T => items[Math.floor(next() * items.length)] as T;

interface Named {
  id: string;
  name: string;
}
const data = JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as {
  species: Named[];
  moves: Named[];
  items: Named[];
};
const moveIds = new Set(data.moves.map((m) => m.id));
const format = engine.sim.Dex.formats.get(GOLDEN_FORMAT);
// The narrow interfaces have no learnsets; this script reads them untyped.
interface RawDex {
  species: {
    get(name: string): {
      exists: boolean;
      id: string;
      name: string;
      abilities: Record<string, string>;
    };
    getLearnsetData?(id: string): { learnset?: Record<string, unknown> };
  };
  moves: { get(name: string): { name: string } };
  items: { get(name: string): { megaStone?: Record<string, string> } };
}
const dex = engine.sim.Dex.mod(format.mod) as unknown as RawDex;
const validator = engine.sim.TeamValidator.get(GOLDEN_FORMAT);

const stones = new Map<string, string[]>();
for (const item of data.items) {
  for (const base of Object.keys(dex.items.get(item.name).megaStone ?? {})) {
    stones.set(base, [...(stones.get(base) ?? []), item.name]);
  }
}

function randomSet(): PokemonSet {
  for (;;) {
    const species = dex.species.get(pick(data.species).name);
    if (!species.exists) continue;
    const learnt = Object.keys(dex.species.getLearnsetData?.(species.id)?.learnset ?? {}).filter((m) =>
      moveIds.has(m),
    );
    if (learnt.length < 4) continue;
    const moves: string[] = [];
    while (moves.length < 4) {
      const move = dex.moves.get(pick(learnt)).name;
      if (!moves.includes(move)) moves.push(move);
    }
    const stone = stones.get(species.name);
    const item = stone && next() < 0.5 ? pick(stone) : pick(data.items).name;
    return {
      name: "",
      species: species.name,
      item,
      ability: pick(Object.values(species.abilities)),
      moves,
      nature: "Serious",
      gender: "",
      evs: { hp: 4, atk: 0, def: 0, spa: 0, spd: 0, spe: 4 },
      ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
      level: 50,
    };
  }
}

function randomTeam(): PokemonSet[] {
  for (let attempt = 0; attempt < 2000; attempt++) {
    const team = Array.from({ length: 6 }, randomSet);
    for (let repair = 0; repair < 40; repair++) {
      const problems = validator.validateTeam(team);
      if (!problems || problems.length === 0) return team;
      // Re-roll every Pokémon a problem names.
      let changed = false;
      team.forEach((set, i) => {
        if (problems.some((p) => p.includes(set.species))) {
          team[i] = randomSet();
          changed = true;
        }
      });
      if (!changed) break;
    }
  }
  throw new Error("no valid random team found");
}

rmSync(RANDOM_DIR, { recursive: true, force: true });
mkdirSync(RANDOM_DIR, { recursive: true });
const manifest: { id: string; seed: string; turns: number; winner: string | null }[] = [];
for (let i = 0; i < BATTLES; i++) {
  const id = `b${String(i).padStart(2, "0")}`;
  const spec: RandomBattleSpec = {
    id,
    seed: `sodium,${(i + 1).toString(16).padStart(32, "0")}`,
    actor_seeds: { p1: (2 * i + 1) * 2654435761 >>> 0, p2: (2 * i + 2) * 2654435761 >>> 0 },
    teams: { p1: randomTeam(), p2: randomTeam() },
  };
  const file = randomBattleFile(spec, await playRandomBattle(engine, spec));
  writeFileSync(join(RANDOM_DIR, `${id}.json`), `${JSON.stringify(file)}\n`);
  manifest.push({ id, seed: spec.seed, turns: file.turns, winner: file.winner });
  console.log(`wrote ${id}.json (${file.turns} turns, ${file.channels.p1.lines.length} lines)`);
}
writeFileSync(
  join(RANDOM_DIR, RANDOM_MANIFEST),
  `${JSON.stringify(
    { provenance: "synthetic: see b*.json; teams drawn by capture-random-corpus.ts", format: GOLDEN_FORMAT, battles: manifest },
    null,
    2,
  )}\n`,
);
