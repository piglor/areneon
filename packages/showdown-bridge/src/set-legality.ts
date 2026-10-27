// The `SetLegality` port of the opponent sampler (ADR-025 section 4, review finding N-D), implemented with
// the pinned engine. A sampled opponent set is legal when the target format's own validator takes it
// (`TeamValidator.validateSet`, the only legality authority) and it is the engine's canonical paste (the
// set survives the import round trip unchanged, as for every team the probe is asked about). A set the
// validator refuses is not an error here: the sampler redraws the world.
//
// `packages/opponent-model` is engine-free and declares the port; this module only has to be assignable
// to it, so the types below are the structural shape of its `WorldSet`. A Mega slot is built the way the
// player's Team Preview shows it (and the team generator writes it): the preview species holding the stone.
// A Mega Evolution that has happened fixes the Mega ability, which the base forme cannot hold; the set is
// then checked with the species' first listed ability, so the validator still judges the item, moves,
// nature, Stat Points and level (the belief already checked the Mega ability against the Mega forme's list).

import type { Engine, PokemonSet, StatTable } from "./engine.ts";
import { exportPaste, importPaste } from "./teams.ts";
import type { TeamVocabulary } from "./team-vocabulary.ts";

/** The part of an opponent-model `WorldSet` the validator needs. */
export interface LegalityWorldSet {
  readonly species: string;
  readonly forme: string;
  readonly item: string | null;
  readonly ability: string;
  readonly nature: string;
  readonly stat_points: Readonly<Record<"hp" | "atk" | "def" | "spa" | "spd" | "spe", number>>;
  readonly moves: readonly string[];
  readonly level: number;
  readonly gender: string | null;
}

export interface SetLegalityPort {
  legal(set: LegalityWorldSet): boolean;
}

/** Showdown's own default IV; the Champions mod ignores IVs (Stat Points replace EVs). */
const IV_DEFAULT = 31;

export function createSetLegality(context: { readonly engine: Engine; readonly vocabulary: TeamVocabulary }): SetLegalityPort {
  const { engine, vocabulary } = context;
  const formatId = vocabulary.format_id;
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(formatId).mod);
  const validator = engine.sim.TeamValidator.get(formatId);
  const roster = new Map(vocabulary.roster.map((r) => [r.id, r]));
  const natures = new Map(vocabulary.natures.map((n) => [n.id, n.name]));

  const build = (set: LegalityWorldSet): PokemonSet | null => {
    const entry = roster.get(set.species);
    const nature = natures.get(set.nature);
    if (entry === undefined || nature === undefined) return null;
    const stone = roster.get(set.forme)?.required_item !== undefined && set.forme !== set.species;
    const ability = stone && !entry.abilities.includes(set.ability) ? entry.abilities[0] : set.ability;
    const ivs: StatTable = { hp: IV_DEFAULT, atk: IV_DEFAULT, def: IV_DEFAULT, spa: IV_DEFAULT, spd: IV_DEFAULT, spe: IV_DEFAULT };
    return {
      name: "",
      species: dex.species.get(set.species).name,
      item: set.item === null ? "" : dex.items.get(set.item).name,
      ability: dex.abilities.get(ability as string).name,
      moves: set.moves.map((move) => dex.moves.get(move).name),
      nature,
      gender: set.gender ?? "",
      evs: { ...set.stat_points } as StatTable,
      ivs,
      level: set.level,
    };
  };

  return {
    legal(set) {
      const built = build(set);
      if (built === null) return false;
      if ((validator.validateSet(structuredClone(built), {}) ?? []).length > 0) return false;
      try {
        const paste = exportPaste(engine, [built]);
        return exportPaste(engine, importPaste(engine, paste)) === paste;
      } catch {
        return false;
      }
    },
  };
}
