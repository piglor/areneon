// Spike #682 (throwaway): list species in the pinned `champions` mod, not
// marked nonstandard, with an ability that sets maybeTrapped (ADR-006 S10).
//
//   make showdown && node spikes/682-legal-choices/roster.ts
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const showdownDir = process.env.ARENEON_SHOWDOWN_DIR ?? `${root}vendor/pokemon-showdown`;
interface Species {
  name: string;
  isNonstandard: string | null;
  abilities: Record<string, string>;
}
const { Dex } = createRequire(import.meta.url)(`${showdownDir}/dist/sim/index.js`) as {
  Dex: { mod(id: string): { species: { all(): Species[] } } };
};

const TRAPPING = new Set(["Shadow Tag", "Arena Trap", "Magnet Pull"]);
const hits = Dex.mod("champions")
  .species.all()
  .filter((s) => !s.isNonstandard && Object.values(s.abilities).some((a) => TRAPPING.has(a)))
  .map((s) => `${s.name}: ${Object.values(s.abilities).filter((a) => TRAPPING.has(a)).join(", ")}`);
console.log(hits.join("\n"));
