// The scenario world builder (ADR-026 Evidence 1, option W3, slice W part 1).
//
// A WORLD is a scenario battle (ADR-016: the target format minus the non-mechanics rules, started at once)
// that holds the four brought Pokemon of each side and the state the player knows, built from the player's
// own view only: the own request, the own team as brought (`ActorContext.team`), the shared reader's public
// state and a FULL opponent team (its own input type; a later slice maps ADR-025's sampled `OpponentWorld`
// onto it). The engine cannot start from a public state, so the opponent's sets are an input, never read
// from anywhere here. The builder takes no live battle, `BattleRecord`, omniscient log, hidden verdict or
// seed: its input type has none of them (a type test shows it).
//
// This part writes the E-class state items of the ADR's closed table, each through the engine's own method
// and nothing else: HP (`sethp`), fainted Pokemon (`faint`, `faintMessages`), major status (`setStatus`, the
// way the speed probe declares an observed fact), stat stages (as the speed probe writes them), Mega
// Evolution (`runMegaEvo`), a forme change shown in the log (`formeChange`), own PP (`deductPP` to the
// request), weather, terrain, Trick Room and other field conditions, and side conditions with their layers
// (`setWeather`, `setTerrain`, `addPseudoWeather`, `addSideCondition`, the source holding its world item so
// the engine's own duration callback gives the base length). It is absolute: whatever a switch-in effect
// set at the scenario start is replaced.
//
// Every state item this part does NOT write (W-class writers, X/T/R classification and the writer file are
// part 2) makes the world a counted `state_unwritten` refusal, never a silent omission: a volatile, a toxic
// stage, a changed item or ability, an own move lock, an Illusion or phantom entry, a switch-in effect that
// left state behind. Three checks run on every world: V1 (each set passes the target format's
// `validateSet`, is the engine's canonical paste and keeps every revealed fact and Item Clause), V3 (the
// opponent as the player would see it, rendered by the engine, equals the reader's state) and V5 (same seat
// and nicknames). V2 and V4 are part 2. Every refusal is typed, carries fixed words only and is counted.

import type { PublicFlag, PublicMon, PublicState } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { assumedHp } from "./damage-aware-assumed-set.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import type { Engine, PokemonSet, SimBattle } from "./engine.ts";
import { startScenario } from "./mechanics-scenario.ts";
import { exportPaste, importPaste } from "./teams.ts";

/** Why a world was not built. Every code is counted; none carries a set, a stat or a name from an input. */
export const WORLD_REFUSAL_CODES = [
  "not_move_request",
  "reader_flagged",
  "own_team_mismatch",
  "opponent_team_mismatch",
  "v1_set_invalid",
  "v1_not_canonical",
  "v1_revealed_fact",
  "v1_item_clause",
  "state_unwritten",
  "start_failed",
  "engine_refused",
  "v3_mismatch",
  "v5_mismatch",
] as const;
export type WorldRefusalCode = (typeof WORLD_REFUSAL_CODES)[number];

/** The state this part does not write (parts of the ADR's W, X and R classes), as the kind of the first one found. */
export const UNWRITTEN_KINDS = ["volatile", "toxic", "item_changed", "ability_changed", "own_move_lock", "illusion", "start_trigger", "bench_status"] as const;
export type UnwrittenKind = (typeof UNWRITTEN_KINDS)[number];

export interface WorldRefusal {
  code: WorldRefusalCode;
  /** Fixed words, at most naming a field or a position; never a value from the input. */
  detail: string;
  /** Set exactly for `state_unwritten`. */
  unwritten?: UnwrittenKind;
}

/** One opponent Pokemon of the world: the set, and which roster entry of the reader's state it is (the shown identifier). */
export interface WorldOpponentMember {
  /** Index into `state.opponent.roster`. */
  roster_index: number;
  /** A whole set. Its nickname is replaced by the shown one when the Pokemon has been seen. */
  set: Readonly<PokemonSet>;
}

/** A full opponent team: the Pokemon the opponent brought (`state.opponent.teamsize` of them), each as a roster entry and a set. */
export interface WorldOpponent {
  members: readonly WorldOpponentMember[];
}

/** A Pokemon of the world: the side and its place in that side's `side.pokemon` order (the own request's order; the opponent's actives, then its bench). */
export interface WorldPokemonRef {
  side: "own" | "opponent";
  position: number;
}

/**
 * Which Pokemon set a condition, per the log (the interface slice L's world facts will fill). The source's
 * held item gives the engine's duration callback its base length (Heat Rock, Light Clay). A condition with no
 * entry falls back to the first Pokemon of the side it belongs to (the own side for the field) and is counted
 * in `defaulted_sources`.
 */
export interface WorldSources {
  weather?: WorldPokemonRef;
  terrain?: WorldPokemonRef;
  /** Field conditions other than weather and terrain (Trick Room), by engine id. */
  field?: Readonly<Record<string, WorldPokemonRef>>;
  /** Side conditions by engine id. */
  own_side?: Readonly<Record<string, WorldPokemonRef>>;
  opponent_side?: Readonly<Record<string, WorldPokemonRef>>;
}

/** Everything a world is built from: the player's own view and a full opponent team. Nothing else is accepted. */
export interface WorldInput {
  /** The player's own move request, as the engine sent it. */
  request: EngineRequest;
  /** The player's own team as brought (`ActorContext.team`). */
  team: readonly Readonly<PokemonSet>[];
  /** The shared reader's state at this request. */
  state: PublicState;
  /** The reader's flagged lines up to this request; any one refuses the world. */
  flags: readonly PublicFlag[];
  opponent: WorldOpponent;
  sources?: WorldSources;
}

export interface WorldContext {
  engine: Engine;
  /** The target format (its `TeamValidator` judges the sets). */
  format_id: string;
  /** The scenario format id (`checkScenario`). */
  scenario_id: string;
  /** How a shown percentage becomes hit points (ADR-018 `assumedHp`). The Champions display is `floor(100 * hp / max)`, which `ceil` inverts for a maximum of 100 or more. */
  hp_rounding: DamageAwareConfig["hp_rounding"];
}

/** A built world: the scenario battle (for the part that plays it) and what was resampled or defaulted. */
export interface ScenarioWorld {
  battle: SimBattle;
  /** The player's seat. */
  seat: "p1" | "p2";
  /** Own Pokemon's nicknames in `side.pokemon` order. */
  own_order: readonly string[];
  /** The reader's roster index of each opponent Pokemon in `side.pokemon` order. */
  opponent_order: readonly number[];
  /** State the world resamples through the engine's own draw (X class); counted. Today: a sleep counter. */
  resampled: readonly { side: "own" | "opponent"; position: number; item: "sleep" }[];
  /** Conditions whose source Pokemon was defaulted. */
  defaulted_sources: number;
}

export type WorldResult = { kind: "world"; world: ScenarioWorld } | { kind: "refused"; refusal: WorldRefusal };

export interface WorldCounts {
  requests: number;
  built: number;
  refused: number;
  refusals: Record<WorldRefusalCode, number>;
  unwritten: Record<UnwrittenKind, number>;
  /** Built worlds that resampled a sleep counter. */
  resampled_sleep: number;
  /** Conditions whose source was defaulted, over built worlds. */
  defaulted_sources: number;
}

// ---- the engine objects this builder touches, narrowed here so the shared engine types stay as they are ----

interface WMon {
  name: string;
  hp: number;
  maxhp: number;
  status: string;
  fainted: boolean;
  level: number;
  gender: string;
  item: string;
  ability: string;
  baseAbility: string;
  boosts: Record<string, number>;
  species: { id: string; isMega?: boolean };
  moveSlots: { id: string; pp: number; maxpp: number }[];
  storedStats: Record<string, number>;
  volatiles: Record<string, unknown>;
  canMegaEvo: unknown;
  faint(): unknown;
  setStatus(status: string, source: null, sourceEffect: null, ignoreImmunities: boolean): unknown;
  sethp(hp: number): unknown;
  deductPP(move: string, amount: number): number;
  formeChange(species: string, source: null, isPermanent: boolean): unknown;
  getHealth(): { shared: string };
}
interface WSide {
  id: string;
  pokemon: WMon[];
  addSideCondition(id: string, source: WMon): unknown;
  sideConditions: Record<string, { layers?: number } | undefined>;
}
interface WBattle {
  sides: WSide[];
  ended: boolean;
  field: {
    weather: string;
    terrain: string;
    setWeather(id: string, source: WMon): unknown;
    clearWeather(): unknown;
    setTerrain(id: string, source: WMon): unknown;
    clearTerrain(): unknown;
    addPseudoWeather(id: string, source: WMon): unknown;
    pseudoWeather: Record<string, unknown>;
  };
  actions: { runMegaEvo(pokemon: WMon): boolean };
  dex: { conditions: { get(id: string): { exists: boolean; effectType: string } } };
  faintMessages(): unknown;
}

/** What the engine says an id is (`Weather`, `Terrain`, `Status`, `Condition`), or `none` for an id it does not have. */
const conditionKind = (battle: WBattle, id: string): string => {
  const found = battle.dex.conditions.get(id);
  return found.exists ? found.effectType : "none";
};

class Refuse extends Error {
  readonly refusal: WorldRefusal;
  constructor(code: WorldRefusalCode, detail: string, unwritten?: UnwrittenKind) {
    super(detail);
    this.refusal = unwritten === undefined ? { code, detail } : { code, detail, unwritten };
  }
}

const need = (ok: boolean, code: WorldRefusalCode, detail: string): void => {
  if (!ok) throw new Refuse(code, detail);
};
const unwritten = (kind: UnwrittenKind, detail: string): never => {
  throw new Refuse("state_unwritten", detail, kind);
};

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const STAGES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"] as const;
const SLOT_POSITION: Record<string, number> = { a: 0, b: 1 };
const SLOT_LETTER = ["a", "b"] as const;
const OWN_CONDITION = /^(\d+)\/(\d+)(?: ([a-z]+))?$/;
const SHOWN_HEALTH = /^(\d+)\/100/;
/** The permanent forme a Pokemon is shown in (its details), else the set's. */
const permanentOf = (mon: PublicMon | undefined, set: PokemonSet): string => mon?.species ?? toId(set.species);
const SEATS = ["p1", "p2"] as const;

/** One own Pokemon of the world: the request's entry and the brought set it is. */
interface OwnEntry {
  name: string;
  set: PokemonSet;
  hp: number;
  maxhp: number;
  status: string | null;
  shown: PublicMon | undefined;
}

const emptyCounts = (): WorldCounts => ({
  requests: 0,
  built: 0,
  refused: 0,
  refusals: Object.fromEntries(WORLD_REFUSAL_CODES.map((code) => [code, 0])) as Record<WorldRefusalCode, number>,
  unwritten: Object.fromEntries(UNWRITTEN_KINDS.map((kind) => [kind, 0])) as Record<UnwrittenKind, number>,
  resampled_sleep: 0,
  defaulted_sources: 0,
});

export interface WorldBuilder {
  /** Build one world, or refuse it with a counted reason. A defect (a thrown error) is not a refusal. */
  build(input: WorldInput): WorldResult;
  /** The counts so far (a copy). */
  counts(): WorldCounts;
}

export function createWorldBuilder(context: WorldContext): WorldBuilder {
  const { engine, format_id, scenario_id } = context;
  const counts = emptyCounts();
  const validator = engine.sim.TeamValidator.get(format_id);
  const ruleTable = engine.sim.Dex.formats.getRuleTable(engine.sim.Dex.formats.get(format_id));
  const itemClause = ruleTable.has("itemclause");

  /** V1 for one set (the shape part): the validator takes it and it is the engine's canonical paste. */
  function checkSet(set: PokemonSet, own: boolean): void {
    const problems = validator.validateSet(structuredClone(set), {}) ?? [];
    need(problems.length === 0, "v1_set_invalid", own ? "an own set is refused by the validator" : "an opponent set is refused by the validator");
    const paste = exportPaste(engine, [set]);
    const back = importPaste(engine, paste);
    need(back.length === 1 && exportPaste(engine, back) === paste, "v1_not_canonical", "a set is not in the engine's canonical paste form");
  }

  /** V1 for an opponent set: every revealed fact of the shown Pokemon is kept. */
  function keepsRevealed(set: PokemonSet, mon: PublicMon): void {
    const species = toId(set.species);
    need([mon.species, mon.base_species].includes(species), "v1_revealed_fact", "an opponent set is for another species");
    const moves = set.moves.map(toId);
    need(mon.moves_revealed.every((move) => moves.includes(move)), "v1_revealed_fact", "a revealed move is not in the set");
    const item = toId(set.item);
    if (mon.mega) need(species === mon.base_species && item === mon.mega_stone, "v1_revealed_fact", "a Mega Evolved Pokemon's set is not its base forme with the stone");
    else if (mon.item_known && !mon.item_lost && !mon.item_received) need(item === (mon.item ?? ""), "v1_revealed_fact", "the revealed item is not the set's");
    if (mon.ability !== null && !mon.ability_received && !mon.mega) need(toId(set.ability) === mon.ability, "v1_revealed_fact", "the revealed ability is not the set's");
    if (mon.level !== null) need(set.level === mon.level, "v1_revealed_fact", "the shown level is not the set's");
    if (mon.gender !== null) {
      need(set.gender === "" || set.gender === mon.gender, "v1_revealed_fact", "the shown gender is not the set's");
      set.gender = mon.gender; // a set with no gender takes the shown one, like a nickname
    }
  }

  /** The own side from the request: each Pokemon's entry in request order. */
  function ownEntries(input: WorldInput): OwnEntry[] {
    const side = input.request.side;
    need(side.id === input.state.player, "v5_mismatch", "the request is for another seat than the state");
    need(side.pokemon.length > 0 && side.pokemon.length === input.state.own.teamsize, "own_team_mismatch", "the request lists another number of Pokemon than the team size");
    const used = new Set<PokemonSet>();
    return side.pokemon.map((entry) => {
      const name = entry.ident.slice(entry.ident.indexOf(": ") + 2);
      const matches = input.team.filter((set) => (set.name === "" ? set.species : set.name) === name);
      need(matches.length === 1 && !used.has(matches[0] as PokemonSet), "own_team_mismatch", "an own Pokemon matches no set, or more than one, of the team");
      const set = structuredClone(matches[0] as PokemonSet);
      used.add(matches[0] as PokemonSet);
      // A set brought without a gender had one drawn when the real battle started; the request's details say which.
      const shownGender = /, ([MF])(?:,|$)/.exec(entry.details)?.[1] ?? "";
      need(set.gender === "" || set.gender === shownGender || (shownGender === "" && set.gender === "N"), "own_team_mismatch", "an own Pokemon's gender differs from its set");
      if (shownGender !== "") set.gender = shownGender;
      const condition = OWN_CONDITION.exec(entry.condition);
      const fainted = entry.condition === "0 fnt";
      need(fainted || condition !== null, "own_team_mismatch", "an own condition has a shape this builder does not know");
      const [, hp = "0", max = "0", status] = condition ?? [];
      const maxhp = fainted ? 0 : Number(max);
      return { name, set, hp: fainted ? 0 : Number(hp), maxhp, status: status === undefined ? null : status, shown: input.state.own.roster.find((mon) => mon.name === name) };
    });
  }

  /** The opponent side in world order: the actives in their slots, then the bench in the reader's order. */
  function opponentOrder(input: WorldInput): { index: number; set: PokemonSet; mon: PublicMon }[] {
    const { roster, teamsize } = input.state.opponent;
    const { members } = input.opponent;
    need(members.length === teamsize && members.length > 0, "opponent_team_mismatch", "the opponent team has another size than the team size");
    const seen = new Set<number>();
    const placed = members.map((member) => {
      const mon = roster[member.roster_index];
      need(Number.isInteger(member.roster_index) && mon !== undefined && !seen.has(member.roster_index), "opponent_team_mismatch", "a member names no roster entry, or one twice");
      seen.add(member.roster_index);
      need(mon?.own === false && mon.phantom !== true && mon.brought !== false, "opponent_team_mismatch", "a member is not a Pokemon the opponent brought");
      return { index: member.roster_index, set: structuredClone(member.set) as PokemonSet, mon: mon as PublicMon };
    });
    need(roster.every((mon, index) => mon.brought !== true || seen.has(index)), "opponent_team_mismatch", "a Pokemon seen in play is missing from the team");
    const slots = placed.map((entry) => entry.mon.active_slot);
    need(slots.every((slot) => slot === null || slot in SLOT_POSITION) && new Set(slots.filter((slot) => slot !== null)).size === slots.filter((slot) => slot !== null).length, "opponent_team_mismatch", "the active slots are not a, b or empty, or are repeated");
    // The reader empties a slot when its Pokemon faints. The engine still holds that fainted Pokemon there, so a fainted member fills it (the player's view is the same whichever one does).
    const bySlot = SLOT_LETTER.map((letter) => placed.find((entry) => entry.mon.active_slot === letter));
    const spare = placed.filter((entry) => entry.mon.active_slot === null && entry.mon.fainted).sort((a, b) => a.index - b.index);
    const actives = bySlot.map((entry) => entry ?? spare.shift());
    need(actives.every((entry) => entry !== undefined), "opponent_team_mismatch", "an empty active slot has no fainted Pokemon to fill it");
    const filled = new Set(actives.map((entry) => entry?.index));
    const bench = placed.filter((entry) => !filled.has(entry.index)).sort((a, b) => a.index - b.index);
    return [...(actives as typeof placed), ...bench];
  }

  /** What this part does not write, refused with its kind. Reader and request facts only. */
  function checkUnwritten(input: WorldInput, own: readonly OwnEntry[], foes: readonly { set: PokemonSet; mon: PublicMon }[]): void {
    const active = input.request.active ?? [];
    for (const [index, slot] of active.entries()) {
      // A move with no PP left is marked disabled too; the world writes its PP, so that is not a lock.
      if (slot.moves.some((move) => move.disabled !== undefined && move.disabled !== false && move.pp !== 0)) unwritten("own_move_lock", `own active ${index} has a disabled move`);
    }
    for (const [index, entry] of own.entries()) {
      const requested = input.request.side.pokemon[index];
      const shown = entry.shown;
      if (requested !== undefined && toId(requested.item) !== toId(entry.set.item)) unwritten("item_changed", `own Pokemon ${index} holds another item than it was brought with`);
      if (requested !== undefined) {
        // A Mega Evolution replaces the permanent ability too, so a Mega Evolved Pokemon's request shows the Mega forme's.
        if (shown?.mega !== true) need(toId(requested.baseAbility) === toId(entry.set.ability), "own_team_mismatch", `own Pokemon ${index} has another ability than its set`);
        if (shown?.mega !== true && toId(requested.ability ?? requested.baseAbility) !== toId(requested.baseAbility)) unwritten("ability_changed", `own Pokemon ${index} has another ability than it was brought with`);
      }
      if (entry.status === "tox") unwritten("toxic", `own Pokemon ${index} is badly poisoned`);
      if (shown !== undefined && shown.volatiles.length > 0) unwritten("volatile", `own Pokemon ${index} has a volatile`);
      if (shown !== undefined && (shown.ability_suppressed || shown.ability_received)) unwritten("ability_changed", `own Pokemon ${index} has a changed ability`);
    }
    for (const [index, foe] of foes.entries()) {
      const mon = foe.mon;
      if (mon.status === "tox") unwritten("toxic", `opponent Pokemon ${index} is badly poisoned`);
      if (mon.volatiles.length > 0) unwritten("volatile", `opponent Pokemon ${index} has a volatile`);
      if (mon.item_lost || mon.item_received) unwritten("item_changed", `opponent Pokemon ${index} has lost or received an item`);
      if (mon.ability_received || mon.ability_suppressed) unwritten("ability_changed", `opponent Pokemon ${index} has a changed ability`);
    }
    if (input.state.opponent.roster.some((mon) => mon.phantom === true) || input.state.own.roster.some((mon) => mon.phantom === true)) unwritten("illusion", "the reader shows an Illusion pair");
  }

  /** The engine's Pokemon a reference names. */
  function sourceOf(battle: WBattle, seat: number, ref: WorldPokemonRef | undefined, fallbackSide: "own" | "opponent", defaulted: { n: number }): WMon {
    if (ref === undefined) defaulted.n += 1;
    const side = (ref ?? { side: fallbackSide, position: 0 }).side;
    const index = side === "own" ? seat : 1 - seat;
    const mon = battle.sides[index]?.pokemon[(ref ?? { position: 0 }).position];
    need(mon !== undefined, "engine_refused", "a condition source names no Pokemon of the world");
    return mon as WMon;
  }

  /** The forme a Pokemon is in: its Mega Evolution, else the permanent forme its details show, then the temporary one the log showed. Done for every Pokemon, a fainted one included (it keeps only the permanent one). */
  function setForme(battle: WBattle, mon: WMon, to: { mega: boolean; permanent: string; temporary: string | null }, where: string): void {
    if (to.mega) need(battle.actions.runMegaEvo(mon), "engine_refused", `${where} cannot Mega Evolve`);
    else if (mon.species.id !== to.permanent) {
      mon.formeChange(to.permanent, null, true);
      need(mon.species.id === to.permanent, "engine_refused", `${where} cannot take the forme shown`);
    }
    if (to.temporary !== null && !to.mega) {
      mon.formeChange(to.temporary, null, false);
      need(mon.species.id === to.temporary, "engine_refused", `${where} cannot take the temporary forme shown`);
    }
  }

  /** Write one living Pokemon's state. Returns whether a sleep counter was resampled. */
  function writeMon(battle: WBattle, mon: WMon, to: { hp: number | null; status: string | null; boosts: Record<string, number>; active: boolean; pp: ReadonlyMap<string, number> }, where: string): boolean {
    let slept = false;
    if (to.status !== null) {
      // The engine's setStatus refuses a Pokemon that is not in play: a status on the bench needs a reviewed writer (part 2).
      need(conditionKind(battle, to.status) === "Status", "engine_refused", `${where} has a status the engine does not have`);
      if (!to.active) unwritten("bench_status", `${where} is on the bench with a status`);
      mon.setStatus(to.status, null, null, true); // an observed fact: type immunities do not apply
      need(mon.status === to.status, "engine_refused", `${where} cannot take the status shown`);
      slept = to.status === "slp";
    }
    if (to.hp !== null && to.hp < mon.maxhp) {
      need(to.hp >= 1, "engine_refused", `${where} has no hit points left to write`);
      mon.sethp(to.hp);
      need(mon.hp === to.hp, "engine_refused", `${where} cannot take the hit points shown`);
    }
    for (const stage of STAGES) mon.boosts[stage] = to.boosts[stage] ?? 0;
    for (const slot of mon.moveSlots) {
      const left = to.pp.get(slot.id);
      if (left === undefined || left === slot.maxpp) continue;
      need(left >= 0 && left < slot.maxpp, "own_team_mismatch", `${where} has a PP count the engine's maximum does not allow`);
      mon.deductPP(slot.id, slot.maxpp - left);
      need(slot.pp === left, "engine_refused", `${where} cannot take the PP shown`);
    }
    return slept;
  }

  /** The opponent Pokemon as the player would see it, rendered by the engine, against the reader's state (V3). */
  function checkRendered(battle: WBattle, seat: number, foes: readonly { mon: PublicMon; set: PokemonSet }[]): void {
    const side = battle.sides[1 - seat] as WSide;
    for (const [position, foe] of foes.entries()) {
      const mon = side.pokemon[position] as WMon;
      const shown = foe.mon;
      const at = `opponent Pokemon ${position}`;
      const forme = (shown.fainted ? null : shown.temp_species) ?? shown.species ?? shown.base_species;
      // The engine resets a fainted Pokemon to the species of its set (a busted Disguise): the log's last forme is not kept.
      need(mon.species.id === forme || (shown.fainted && !shown.mega && mon.species.id === toId(foe.set.species)), "v3_mismatch", `${at}: species`);
      need(mon.fainted === shown.fainted, "v3_mismatch", `${at}: fainted`);
      need((mon.status === "" ? null : mon.status) === shown.status, "v3_mismatch", `${at}: status`);
      // A fainted Pokemon may hold a slot the reader has already emptied.
      need((position < SLOT_LETTER.length ? SLOT_LETTER[position] : null) === shown.active_slot || (shown.fainted && shown.active_slot === null), "v3_mismatch", `${at}: slot`);
      need(STAGES.every((stage) => (mon.boosts[stage] ?? 0) === (shown.boosts[stage] ?? 0)), "v3_mismatch", `${at}: stat stages`);
      need((mon.species.isMega === true) === shown.mega, "v3_mismatch", `${at}: Mega Evolution`);
      if (shown.level !== null) need(mon.level === shown.level, "v3_mismatch", `${at}: level`);
      if (shown.gender !== null) need(mon.gender === shown.gender, "v3_mismatch", `${at}: gender`);
      if (shown.hp_num !== null) {
        const health = SHOWN_HEALTH.exec(mon.getHealth().shared);
        need(shown.hp_den === 100 && (shown.fainted ? mon.hp === 0 : health !== null && Number(health[1]) === shown.hp_num), "v3_mismatch", `${at}: hit points`);
      }
      if (shown.item_known && !shown.item_lost) need(toId(mon.item) === (shown.item ?? ""), "v3_mismatch", `${at}: item`);
      // The reader keeps the ability a Pokemon showed before it Mega Evolved.
      if (shown.ability !== null && !shown.mega) need(toId(mon.ability) === shown.ability, "v3_mismatch", `${at}: ability`);
      const moves = mon.moveSlots.map((slot) => slot.id);
      need(shown.moves_revealed.every((move) => moves.includes(move)), "v3_mismatch", `${at}: revealed moves`);
    }
  }

  function make(input: WorldInput): ScenarioWorld {
    need(input.request.active !== undefined && input.request.wait !== true && input.request.teamPreview !== true && input.request.forceSwitch === undefined, "not_move_request", "the request is not a move request");
    need(input.flags.length === 0, "reader_flagged", "the reader flagged a line");
    const { state } = input;
    const seatIndex = SEATS.indexOf(state.player as "p1" | "p2");
    need(seatIndex >= 0, "v5_mismatch", "the state is for no seat of a two-player battle");
    const own = ownEntries(input);
    const foes = opponentOrder(input);
    const requestActive = input.request.side.pokemon.map((entry) => entry.active);
    need(requestActive.every((active, index) => active === index < 2), "own_team_mismatch", "the request's active Pokemon are not the first two");
    // V1.
    for (const entry of own) checkSet(entry.set, true);
    for (const foe of foes) {
      foe.set.name = foe.mon.name ?? foe.set.name;
      keepsRevealed(foe.set, foe.mon);
      checkSet(foe.set, false);
    }
    if (itemClause) {
      const items = foes.map((foe) => toId(foe.set.item)).filter((item) => item !== "");
      need(new Set(items).size === items.length, "v1_item_clause", "two opponent sets hold one item");
    }
    checkUnwritten(input, own, foes);

    // Start: the own side on the player's seat.
    const teams = { p1: [] as PokemonSet[], p2: [] as PokemonSet[] };
    teams[SEATS[seatIndex] as "p1" | "p2"] = own.map((entry) => entry.set);
    teams[SEATS[1 - seatIndex] as "p1" | "p2"] = foes.map((foe) => foe.set);
    let battle: WBattle;
    try {
      battle = startScenario(engine, scenario_id, teams) as unknown as WBattle;
    } catch {
      throw new Refuse("start_failed", "the engine could not start the scenario battle");
    }
    const ownSide = battle.sides[seatIndex] as WSide;
    const foeSide = battle.sides[1 - seatIndex] as WSide;
    need(ownSide.id === state.player && ownSide.pokemon.length === own.length && foeSide.pokemon.length === foes.length, "start_failed", "the scenario battle has other sides than declared");

    // Switch-in effects of the start may have changed an item, an ability or left a volatile: they are not restored here.
    for (const [index, entry] of own.entries()) {
      const mon = ownSide.pokemon[index] as WMon;
      if (toId(mon.item) !== toId(entry.set.item) || toId(mon.ability) !== toId(entry.set.ability) || Object.keys(mon.volatiles).length > 0) unwritten("start_trigger", `a switch-in effect changed own Pokemon ${index}`);
    }
    for (const [index, foe] of foes.entries()) {
      const mon = foeSide.pokemon[index] as WMon;
      if (toId(mon.item) !== toId(foe.set.item) || toId(mon.ability) !== toId(foe.set.ability) || Object.keys(mon.volatiles).length > 0) unwritten("start_trigger", `a switch-in effect changed opponent Pokemon ${index}`);
    }

    // Formes first (a fainted Pokemon keeps the Mega forme it fainted in), then the fainted through the engine, then everything else is absolute.
    own.forEach((entry, index) => setForme(battle, ownSide.pokemon[index] as WMon, { mega: entry.shown?.mega === true, permanent: permanentOf(entry.shown, entry.set), temporary: entry.shown?.temp_species ?? null }, `own Pokemon ${index}`));
    foes.forEach((foe, index) => setForme(battle, foeSide.pokemon[index] as WMon, { mega: foe.mon.mega, permanent: permanentOf(foe.mon, foe.set), temporary: foe.mon.temp_species }, `opponent Pokemon ${index}`));
    const down: WMon[] = [];
    own.forEach((entry, index) => {
      if (entry.hp === 0) down.push(ownSide.pokemon[index] as WMon);
    });
    foes.forEach((foe, index) => {
      if (foe.mon.fainted) down.push(foeSide.pokemon[index] as WMon);
    });
    for (const mon of down) mon.faint();
    if (down.length > 0) battle.faintMessages();
    need(!battle.ended, "start_failed", "the world ended the battle");
    need(down.every((mon) => mon.fainted), "engine_refused", "the engine did not faint a Pokemon");

    const resampled: { side: "own" | "opponent"; position: number; item: "sleep" }[] = [];
    for (const [index, entry] of own.entries()) {
      const mon = ownSide.pokemon[index] as WMon;
      if (mon.fainted) continue;
      const pp = new Map<string, number>(Object.entries(entry.shown?.pp ?? {}));
      for (const move of input.request.active?.[index]?.moves ?? []) if (move.pp !== undefined) pp.set(move.id, move.pp);
      const slept = writeMon(battle, mon, { hp: entry.hp, status: entry.status, boosts: entry.shown?.boosts ?? {}, active: index < 2, pp }, `own Pokemon ${index}`);
      if (slept) resampled.push({ side: "own", position: index, item: "sleep" });
    }
    for (const [index, foe] of foes.entries()) {
      const mon = foeSide.pokemon[index] as WMon;
      if (mon.fainted) continue;
      const percent = foe.mon.hp_num;
      const hp = percent === null ? null : assumedHp(percent, mon.maxhp, context.hp_rounding);
      const slept = writeMon(battle, mon, { hp, status: foe.mon.status, boosts: foe.mon.boosts, active: index < 2, pp: new Map() }, `opponent Pokemon ${index}`);
      if (slept) resampled.push({ side: "opponent", position: index, item: "sleep" });
    }

    // Own stats against the request: the same set in the same place.
    for (const [index, entry] of own.entries()) {
      const mon = ownSide.pokemon[index] as WMon;
      const stats = input.request.side.pokemon[index]?.stats as unknown as Record<string, number>;
      need(mon.maxhp === entry.maxhp || entry.hp === 0, "own_team_mismatch", `own Pokemon ${index} has another maximum hit points than the request`);
      need(Object.entries(stats).every(([stat, value]) => mon.storedStats[stat] === value), "own_team_mismatch", `own Pokemon ${index} has other stats than the request`);
    }

    // The field, absolute: what a switch-in set is cleared first.
    const defaulted = { n: 0 };
    const sources = input.sources ?? {};
    battle.field.clearWeather();
    battle.field.clearTerrain();
    if (state.weather !== null) {
      need(conditionKind(battle, state.weather) === "Weather", "engine_refused", "the weather shown is not an engine weather");
      battle.field.setWeather(state.weather, sourceOf(battle, seatIndex, sources.weather, "own", defaulted));
      need(battle.field.weather === state.weather, "engine_refused", "the engine refused the weather shown");
    }
    if (state.terrain !== null) {
      need(conditionKind(battle, state.terrain) === "Terrain", "engine_refused", "the terrain shown is not an engine terrain");
      battle.field.setTerrain(state.terrain, sourceOf(battle, seatIndex, sources.terrain, "own", defaulted));
      need(battle.field.terrain === state.terrain, "engine_refused", "the engine refused the terrain shown");
    }
    for (const id of state.field_conditions) {
      need(conditionKind(battle, id) !== "none", "engine_refused", "a field condition shown is not an engine condition");
      battle.field.addPseudoWeather(id, sourceOf(battle, seatIndex, sources.field?.[id], "own", defaulted));
      need(battle.field.pseudoWeather[id] !== undefined, "engine_refused", "the engine refused a field condition shown");
    }
    const sides = [
      { shown: state.own.conditions, side: ownSide, from: sources.own_side, whose: "own" as const },
      { shown: state.opponent.conditions, side: foeSide, from: sources.opponent_side, whose: "opponent" as const },
    ];
    for (const { shown, side, from, whose } of sides) {
      for (const [id, layers] of Object.entries(shown)) {
        need(conditionKind(battle, id) !== "none", "engine_refused", "a side condition shown is not an engine condition");
        need(Number.isInteger(layers) && layers >= 1, "engine_refused", "a side condition has no layer count");
        for (let layer = 0; layer < layers; layer += 1) {
          side.addSideCondition(id, sourceOf(battle, seatIndex, from?.[id], whose, defaulted));
        }
        const held = side.sideConditions[id];
        need(held !== undefined && (held.layers === undefined ? layers === 1 : held.layers === layers), "engine_refused", "the engine refused a side condition or its layers");
      }
    }

    // V5 and V3.
    need(own.every((entry, index) => (ownSide.pokemon[index] as WMon).name === entry.name), "v5_mismatch", "an own nickname differs from the request's");
    need(foes.every((foe, index) => foe.mon.name === null || (foeSide.pokemon[index] as WMon).name === foe.mon.name), "v5_mismatch", "an opponent nickname differs from the shown one");
    checkRendered(battle, seatIndex, foes);

    counts.defaulted_sources += defaulted.n;
    return {
      battle: battle as unknown as SimBattle,
      seat: state.player as "p1" | "p2",
      own_order: own.map((entry) => entry.name),
      opponent_order: foes.map((foe) => foe.index),
      resampled,
      defaulted_sources: defaulted.n,
    };
  }

  return {
    build(input) {
      counts.requests += 1;
      try {
        const world = make(input);
        counts.built += 1;
        if (world.resampled.length > 0) counts.resampled_sleep += 1;
        return { kind: "world", world };
      } catch (error) {
        if (!(error instanceof Refuse)) throw error;
        counts.refused += 1;
        counts.refusals[error.refusal.code] += 1;
        if (error.refusal.unwritten !== undefined) counts.unwritten[error.refusal.unwritten] += 1;
        return { kind: "refused", refusal: error.refusal };
      }
    },
    counts: () => structuredClone(counts),
  };
}
