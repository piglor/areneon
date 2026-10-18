// SPIKE (#677): throwaway. The bot's team is the repo's synthetic, CC0
// golden team. It is not a real competitive team. Never point this at
// private-data/: the evidence files would then carry a private team.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { PokemonSet } from "../../../packages/showdown-bridge/src/index.ts";
import { FORMAT_ID } from "./bot.ts";

export function syntheticTeam(): PokemonSet[] {
  const path = join(
    import.meta.dirname,
    "../../../fixtures/showdown",
    FORMAT_ID,
    "synthetic-team.json",
  );
  return JSON.parse(readFileSync(path, "utf8")) as PokemonSet[];
}
