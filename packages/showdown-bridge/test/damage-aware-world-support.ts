// Shared by the tests of `damage-aware@v2`: a stand-in for the belief port (the real one is joined in
// packages/team-dataset), v2's committed record, and worlds built only from what a player's own view shows.
import type { PublicMon, PublicSide } from "@areneon/public-state";
import { type DamageAwareConfigV2, loadDamageAwareConfigV2 } from "../src/damage-aware-config.ts";
import { type BeliefWorldsPort, type WorldLike, type WorldSlotLike } from "../src/damage-aware-worlds.ts";
import type { WorldSetLike } from "../src/damage-aware-assumed-set.ts";
import type { DamageAwareOptions } from "../src/damage-aware-actor.ts";
import { damageAwareActor } from "../src/damage-aware-actor.ts";
import { engine, mechanics, options, snapshot, vocabulary } from "./damage-aware-support.ts";

export const configV2: DamageAwareConfigV2 = loadDamageAwareConfigV2(vocabulary);

/** Spread variants (each sums to the v1 record's 66 points): world `variant` moves `variant` points from hp to atk. */
const spread = (variant: number): WorldSetLike["stat_points"] => ({ hp: 11 - variant, atk: 11 + variant, def: 11, spa: 11, spd: 11, spe: 11 });

export interface WorldOptions {
  /** The nature of the world's sets (default hardy). */
  nature?: string;
  /** The variant: a spread shift and, for a Pokemon with no revealed move, which pool move fills its set. */
  variant?: number;
}

/** One preview slot of a world made from the public facts of a roster entry only (never a real team). */
export function slotOf(mon: PublicMon, index: number, change: WorldOptions = {}): WorldSlotLike {
  const variant = change.variant ?? 0;
  const entry = vocabulary.roster.find((r) => r.id === mon.base_species);
  if (entry === undefined) throw new Error(`no roster entry for ${mon.base_species}`);
  const revealed = mon.moves_revealed.filter((move) => entry.moves.includes(move));
  const megaEntry = mon.mega ? vocabulary.roster.find((r) => r.required_item === mon.mega_stone && r.id !== entry.id) : undefined;
  const set: WorldSetLike = {
    species: entry.id,
    forme: megaEntry?.id ?? entry.id,
    item: mon.mega ? (mon.mega_stone ?? null) : null,
    ability: entry.abilities[0] as string,
    nature: change.nature ?? "hardy",
    stat_points: spread(variant),
    moves: revealed.length > 0 ? [...revealed].sort() : [entry.moves[variant % entry.moves.length] as string],
    level: 50,
    gender: null,
  };
  return { index, brought: true, set };
}

export const worldOf = (opponent: PublicSide, change: WorldOptions = {}): WorldLike => ({ slots: opponent.roster.map((mon, index) => slotOf(mon, index, change)) });

/** A port whose worlds come from `make(world index, opponent)`; every call is recorded. */
export function fakePort(
  make: (index: number, opponent: PublicSide) => WorldLike = (index, opponent) => worldOf(opponent, { variant: index % 3 }),
  change: Partial<BeliefWorldsPort> = {},
): BeliefWorldsPort & { calls: { stream: Uint8Array; flags: number; n: number }[] } {
  const calls: { stream: Uint8Array; flags: number; n: number }[] = [];
  return {
    identity_hash: "b".repeat(64),
    prior_hash: configV2.assumed_set.prior_hash,
    belief_config_hash: configV2.assumed_set.belief_config_hash,
    prior_format_id: configV2.format_id,
    sample: ({ opponent, flags, stream, n }) => {
      calls.push({ stream, flags: flags.length, n });
      return { worlds: Array.from({ length: n }, (_, index) => ({ world: make(index, opponent) })), redraws: 0 };
    },
    calls,
    ...change,
  };
}

export const optionsV2 = (change: Partial<DamageAwareOptions> = {}): DamageAwareOptions => ({ ...options(), config: configV2, worlds: fakePort(), ...change });
export const newActorV2 = (change: Partial<DamageAwareOptions> = {}) => damageAwareActor(optionsV2(change));
export { engine, mechanics, snapshot, vocabulary };
