import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, type PokemonSet, type StatTable } from "../src/index.ts";

// The champions mod's stat formula against stats the game itself displays.
// Each case is read from an official screenshot of the in-game Stat Points
// screen (species, Stat Points, stat alignment and the six resulting stats);
// sources and access dates are in docs/research/stat-formula-and-turn-order-evidence.md
// (S2-S5). The screenshots are promotional material, not a captured fixture,
// so this pins the engine to the published values: a pin bump that changes
// `statModify` in data/mods/champions/scripts.ts fails here.

interface Shown {
  /** Where the screenshot is published. */
  source: string;
  species: string;
  nature: string;
  sp: StatTable;
  stats: StatTable;
}

const NONE: StatTable = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

const SHOWN: readonly Shown[] = [
  {
    // S2 champions.pokemon.com/en-us/pokemon/ "Train Your Pokémon" (p04_01.jpg); S3 pokemonchampions.jp/ja/training/
    source: "official site, training screen",
    species: "Empoleon",
    nature: "Modest",
    sp: { ...NONE, hp: 32, spa: 32, spe: 2 },
    stats: { hp: 191, atk: 95, def: 108, spa: 179, spd: 121, spe: 82 },
  },
  {
    // S2 champions.pokemon.com/en-us/pokemon/ "Recruiting" (p02_02.jpg); S3 pokemonchampions.jp/ja/pokemon/
    source: "official site, recruit screen",
    species: "Whimsicott",
    nature: "Bold",
    sp: { ...NONE, hp: 32, def: 32, spe: 2 },
    stats: { hp: 167, atk: 78, def: 150, spa: 97, spd: 95, spe: 138 },
  },
  {
    // S4 App Store listing, screenshot ENG_04 (iPhone and iPad)
    source: "App Store listing, recruit screen",
    species: "Pikachu",
    nature: "Jolly",
    sp: { ...NONE, atk: 32, def: 2, spe: 32 },
    stats: { hp: 110, atk: 107, def: 62, spa: 63, spd: 70, spe: 156 },
  },
  {
    // S5 Nintendo eShop product page, "train" section image
    source: "Nintendo eShop page, training screen",
    species: "Gardevoir",
    nature: "Timid",
    sp: { ...NONE, spa: 32, spd: 2, spe: 32 },
    stats: { hp: 143, atk: 76, def: 85, spa: 177, spd: 137, spe: 145 },
  },
  {
    // S5 Nintendo eShop product page, store screenshot f8c54a3f…
    source: "Nintendo eShop page, recruit screen",
    species: "Arcanine",
    nature: "Relaxed",
    sp: { ...NONE, atk: 2, def: 32, spa: 32 },
    stats: { hp: 165, atk: 132, def: 145, spa: 152, spd: 100, spe: 103 },
  },
];

const engine = loadEngine();
const IVS: StatTable = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

const set = (shown: Shown): PokemonSet => ({
  name: shown.species,
  species: shown.species,
  item: "",
  ability: "",
  moves: ["protect"],
  nature: shown.nature,
  gender: "",
  evs: shown.sp,
  ivs: IVS,
  level: 50,
});

/** The six stats the engine computes for each set, read from p1's Team Preview request. */
function engineStats(sets: PokemonSet[]): StatTable[] {
  const battle = new engine.sim.Battle({ formatid: GOLDEN_FORMAT, seed: "sodium,00000000000000000000000000000000" });
  const team = engine.sim.Teams.pack(sets);
  battle.setPlayer("p1", { name: "p1", team });
  battle.setPlayer("p2", { name: "p2", team });
  const side = battle.sides.find((s) => s.id === "p1");
  const request = side?.activeRequest as EngineRequest;
  return request.side.pokemon.map((pokemon) => ({
    hp: Number(pokemon.condition.split("/")[1]),
    ...pokemon.stats,
  }));
}

describe("champions stat formula against in-game screenshots", () => {
  const computed = engineStats(SHOWN.map(set));
  it.each(SHOWN.map((shown, i) => [shown.species, shown, i] as const))(
    "%s: the engine's stats equal the displayed ones",
    (_species, shown, i) => {
      expect(computed[i], shown.source).toEqual(shown.stats);
    },
  );
});
