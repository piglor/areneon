// The search's leaf values, in integer ppm of the game score (ADR-026 section 2). The primary leaf is the policy's value head on the
// hypothetical channel (the worker's `evaluate`, `search-actor.ts`); this module holds the game-end score both leaves share and the
// non-learned secondary leaf, the HP and knock-out differential.
//
// The differential reads the simulated world after the turn: the opponent's real values in the world, which is ours (a sampled
// opponent), not the player's information. `LEAF_FORMAT` (`search-config.ts`) is bumped when it changes.

import type { SimBattle } from "./engine.ts";
import type { Outcome } from "./search-turn.ts";

/** The game score of an ended game for the seat, in ppm (win 1, tie 1/2, loss 0: `ppo.GAME_SCORE`). */
export const outcomeScore = (outcome: Outcome): number => (outcome === "own" ? 1_000_000 : outcome === "tie" ? 500_000 : 0);

interface HpSide {
  id: string;
  pokemon: { hp: number; maxhp: number }[];
}

/** A side's share of its hit points (the mean over its Pokemon of floor(hp / maxhp), ppm) and its share still standing (ppm). */
function share(side: HpSide): { hp: number; alive: number } {
  const n = side.pokemon.length;
  const hp = side.pokemon.reduce((total, mon) => total + Math.floor((mon.hp * 1_000_000) / mon.maxhp), 0);
  const alive = side.pokemon.filter((mon) => mon.hp > 0).length;
  return { hp: Math.floor(hp / n), alive: Math.floor((alive * 1_000_000) / n) };
}

/**
 * The non-learned leaf: 1/2 plus a quarter of the sum of the HP-share difference and the standing-share difference (own minus
 * opponent), in ppm, so it lies in 0..1,000,000 and 500,000 is level. Integers only.
 */
export function hpKoDifferential(battle: SimBattle, seat: "p1" | "p2"): number {
  const sides = battle.sides as unknown as HpSide[];
  const own = share(sides.find((side) => side.id === seat) as HpSide);
  const foe = share(sides.find((side) => side.id !== seat) as HpSide);
  return 500_000 + Math.floor((own.hp - foe.hp + (own.alive - foe.alive)) / 4);
}
