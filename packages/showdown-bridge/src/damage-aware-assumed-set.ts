// Assumed opponent sets for the damage-aware baseline (ADR-018 section 3, ADR-016 section 5).
//
// The mechanics probe needs whole, legal sets. The baseline only ever sees the opponent through its
// own channel, so an opponent's set is built from what that channel REVEALED (species, forme, level,
// gender, moves used, item, ability, Mega, shown hit points) plus the reviewed rules of the
// `DamageAwareConfig` for every other field. The input is one `PublicMon` of the shared player-view
// state reader: the opponent's real team, the omniscient log and the seed are not parameters of
// anything here, so they cannot be read by accident.
//
// Nothing is a game rule: which species, abilities and moves exist comes from the team vocabulary,
// and whether the result is legal is the pinned validator's verdict (`validateSet`), checked on every
// set. An unrevealed move is never invented. A move is added only when the set would otherwise have
// none (a set needs one to be legal); it is listed in `filler_moves` and is never a threat.

import type { PublicMon } from "@areneon/public-state";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import { type Engine, type PokemonSet, type StatTable, defaultIvs } from "./engine.ts";
import { exportPaste, importPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

export class AssumedSetError extends Error {
  override name = "AssumedSetError";
}

/** An assumed opponent Pokemon: the legal set and which of its parts were revealed. */
export interface AssumedSet {
  /** A legal set (the target format's validator took it), in the engine's canonical paste form below. */
  set: PokemonSet;
  paste: string;
  /** Moves the opponent has used and the set holds, in the order they were revealed: the only threats. */
  revealed_moves: string[];
  /** Moves added only to make the set legal. Never scored as a threat. */
  filler_moves: string[];
  /** Revealed moves no legal set of this species can hold (Transform copies, Struggle): known threats the probe cannot be asked about. */
  unplaceable_moves: string[];
  /** The shown hit-point percentage, 0 to 100, or null before the Pokemon was seen. */
  shown_hp_percent: number | null;
  /** The Pokemon has Mega Evolved: the probe query declares it so. */
  mega: boolean;
}

export interface AssumedSetContext {
  engine: Engine;
  vocabulary: TeamVocabulary;
  config: DamageAwareConfig;
}

const pick = <T>(list: readonly T[], rule: "first_in_vocabulary_order" | "last_in_vocabulary_order"): T | undefined =>
  rule === "first_in_vocabulary_order" ? list[0] : list[list.length - 1];

/** The hit points of a shown percentage of `maxHp`, rounded as the record states; a shown percentage above 0 never reaches 0. */
export function assumedHp(percent: number, maxHp: number, rounding: DamageAwareConfig["hp_rounding"]): number {
  if (!Number.isInteger(percent) || percent < 0 || percent > 100) throw new AssumedSetError("a shown hit-point percentage is an integer from 0 to 100");
  if (!Number.isInteger(maxHp) || maxHp < 1) throw new AssumedSetError("maximum hit points must be a positive integer");
  if (percent === 0) return 0;
  const scaled = maxHp * percent;
  const raw = rounding === "floor" ? Math.floor(scaled / 100) : rounding === "ceil" ? Math.ceil(scaled / 100) : Math.floor((scaled * 2 + 100) / 200);
  return Math.min(maxHp, Math.max(1, raw));
}

function shownPercent(mon: PublicMon): number | null {
  if (mon.fainted) return 0;
  if (mon.hp_num === null) return null;
  if (mon.hp_den !== 100) throw new AssumedSetError("an opponent's hit points are shown out of 100");
  return mon.hp_num;
}

/**
 * Build the assumed set of one opponent Pokemon of the player's own view. Throws `AssumedSetError`
 * when no legal set holds what was revealed (the validator's words are in the message).
 */
export function buildAssumedSet(context: AssumedSetContext, mon: PublicMon, skip_abilities: readonly string[] = []): AssumedSet {
  const { engine, vocabulary, config } = context;
  if (mon.own) throw new AssumedSetError("an assumed set is for an opponent's Pokemon");
  const rule = config.assumed_set;
  const candidates = mon.mega ? [mon.base_species] : [...(mon.species === null ? [] : [mon.species]), mon.base_species];
  // The forme shown comes first; a forme the roster does not list (a battle-only forme) falls back to its base species.
  const entry = candidates.map((id) => vocabulary.roster.find((r) => r.id === id)).find((r) => r !== undefined);
  if (entry === undefined) throw new AssumedSetError("the species is not in the team vocabulary's roster");
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(config.format_id).mod);

  // An ability the caller has ruled out (public facts contradict it: the Pokemon holds a status that ability blocks) is never assumed.
  const allowed = entry.abilities.filter((id) => !skip_abilities.includes(dex.abilities.get(id).name));
  const ability = mon.ability !== null && entry.abilities.includes(mon.ability) ? mon.ability : pick(allowed, rule.ability);
  if (ability === undefined) throw new AssumedSetError("the species has no ability in the team vocabulary");

  let item: string;
  // A Mega Evolved Pokemon is declared as its base forme holding the revealed stone; the probe query sets `mega`.
  if (mon.mega) item = mon.mega_stone ?? "";
  else if (mon.item_known) item = mon.item ?? "";
  else item = rule.item.kind === "fixed" ? rule.item.item : "";

  const pool = new Set(entry.moves);
  // Every revealed move the species' pool holds goes on the set; the validator, not a count written here, refuses too many.
  const revealed = mon.moves_revealed.filter((move) => pool.has(move));
  const unplaceable = mon.moves_revealed.filter((move) => !pool.has(move));
  const filler = revealed.length === 0 ? pick(entry.moves, rule.filler_move) : undefined;
  if (revealed.length === 0 && filler === undefined) throw new AssumedSetError("the species has no move in the team vocabulary");

  const natureName = vocabulary.natures.find((n) => n.id === rule.nature)?.name;
  if (natureName === undefined) throw new AssumedSetError("the record's nature is not in the team vocabulary");
  const evs = { ...rule.stat_points } as StatTable;
  const ivs: StatTable = defaultIvs();
  const set: PokemonSet = {
    name: "",
    species: dex.species.get(entry.id).name,
    item: item === "" ? "" : dex.items.get(item).name,
    ability: dex.abilities.get(ability).name,
    moves: (revealed.length > 0 ? revealed : [filler as string]).map((move) => dex.moves.get(move).name),
    nature: natureName,
    gender: mon.gender ?? "",
    evs,
    ivs,
    level: mon.level ?? vocabulary.rules.adjust_level,
  };
  const problems = engine.sim.TeamValidator.get(config.format_id).validateSet(structuredClone(set), {}) ?? [];
  if (problems.length > 0) throw new AssumedSetError(`no legal set holds what was revealed: ${problems.join("; ")}`);
  const paste = exportPaste(engine, [set]);
  // The probe takes only the engine's canonical paste: the set must survive the import round trip unchanged.
  const back = importPaste(engine, paste);
  if (back.length !== 1 || exportPaste(engine, back) !== paste) throw new AssumedSetError("the assumed set is not in the engine's canonical paste form");
  return {
    set,
    paste,
    revealed_moves: revealed,
    filler_moves: filler === undefined ? [] : [filler],
    unplaceable_moves: unplaceable,
    shown_hp_percent: shownPercent(mon),
    mega: mon.mega,
  };
}

/** A sampled world's set contradicts a public fact of the Pokemon it is meant for. The world is dropped, never repaired. */
export class WorldMismatchError extends Error {
  override name = "WorldMismatchError";
}

/** What the baseline reads of a sampled opponent set: the structural shape of the opponent model's `WorldSet`. */
export interface WorldSetLike {
  /** The preview species id. */
  readonly species: string;
  /** The Mega forme id for a Mega slot (its stone is the item); else the species or one of its formes. */
  readonly forme: string;
  readonly item: string | null;
  readonly ability: string;
  readonly nature: string;
  readonly stat_points: Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;
  readonly moves: readonly string[];
  readonly level: number;
  readonly gender: string | null;
}

/**
 * The assumed set of one opponent Pokemon from a sampled world's set (ADR-025 section 6): the same
 * `AssumedSet` `buildAssumedSet` makes, with the fields the channel does not show taken from the world
 * instead of the record's field rules. The Pokemon's public facts win over the world, as in v1: a shown
 * item (or a lost one), a revealed ability, the Mega Evolution and its stone, the shown hit points. A world
 * that cannot be that Pokemon (another species, a Mega Evolution its forme does not explain, a revealed move
 * its set lacks) is a `WorldMismatchError`. A revealed move stays the only threat: `revealed_moves` is what
 * the channel showed, never a move the world added. The target format's validator judges the result.
 */
export function buildWorldAssumedSet(context: Pick<AssumedSetContext, "engine" | "vocabulary"> & { format_id: string }, mon: PublicMon, world: WorldSetLike): AssumedSet {
  const { engine, vocabulary, format_id } = context;
  if (mon.own) throw new AssumedSetError("an assumed set is for an opponent's Pokemon");
  if (world.species !== mon.base_species && world.species !== mon.species) throw new WorldMismatchError("a world's set is for another species than the Pokemon on the field");
  const entry = vocabulary.roster.find((r) => r.id === world.species);
  if (entry === undefined) throw new AssumedSetError("the species is not in the team vocabulary's roster");
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(format_id).mod);
  const formeEntry = vocabulary.roster.find((r) => r.id === world.forme);
  const worldMega = formeEntry?.required_item !== undefined && world.forme !== world.species;
  if (mon.mega && !worldMega) throw new WorldMismatchError("the Pokemon has Mega Evolved and the world's forme is not a Mega forme");

  const pool = new Set(entry.moves);
  const revealed = mon.moves_revealed.filter((move) => pool.has(move));
  const unplaceable = mon.moves_revealed.filter((move) => !pool.has(move));
  if (!revealed.every((move) => world.moves.includes(move))) throw new WorldMismatchError("a revealed move is not in the world's set");

  const worldAbility = entry.abilities.includes(world.ability) ? world.ability : (entry.abilities[0] as string);
  const ability = mon.ability !== null && entry.abilities.includes(mon.ability) ? mon.ability : worldAbility;
  let item: string;
  if (mon.mega) item = mon.mega_stone ?? "";
  else if (mon.item_known) item = mon.item ?? "";
  else if (mon.item_lost) item = "";
  else item = world.item ?? "";

  const natureName = vocabulary.natures.find((n) => n.id === world.nature)?.name;
  if (natureName === undefined) throw new AssumedSetError("the world's nature is not in the team vocabulary");
  const set: PokemonSet = {
    name: "",
    species: dex.species.get(worldMega ? world.species : world.forme).name,
    item: item === "" ? "" : dex.items.get(item).name,
    ability: dex.abilities.get(ability).name,
    moves: world.moves.map((move) => dex.moves.get(move).name),
    nature: natureName,
    gender: mon.gender ?? world.gender ?? "",
    evs: { ...world.stat_points } as StatTable,
    ivs: defaultIvs(),
    level: mon.level ?? world.level,
  };
  const problems = engine.sim.TeamValidator.get(format_id).validateSet(structuredClone(set), {}) ?? [];
  if (problems.length > 0) throw new AssumedSetError(`no legal set holds what was revealed: ${problems.join("; ")}`);
  const paste = exportPaste(engine, [set]);
  const back = importPaste(engine, paste);
  if (back.length !== 1 || exportPaste(engine, back) !== paste) throw new AssumedSetError("the assumed set is not in the engine's canonical paste form");
  return { set, paste, revealed_moves: revealed, filler_moves: [], unplaceable_moves: unplaceable, shown_hp_percent: shownPercent(mon), mega: mon.mega };
}
