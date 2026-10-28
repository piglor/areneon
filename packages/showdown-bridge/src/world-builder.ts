// The scenario world builder (ADR-026 Evidence 1, option W3, slices W part 1 and part 2).
//
// A WORLD is a scenario battle (ADR-016: the target format minus the non-mechanics rules, started at once)
// that holds the four brought Pokemon of each side and the state the player knows, built from the player's
// own view only: the own request, the own team as brought (`ActorContext.team`), the shared reader's public
// state, the world facts of the player's own channel and a FULL opponent team (its own input type; a later
// slice maps ADR-025's sampled `OpponentWorld` onto it). The engine cannot start from a public state, so the
// opponent's sets are an input, never read from anywhere here. The builder takes no live battle,
// `BattleRecord`, omniscient log, hidden verdict or seed: its input type has none of them (a type test).
//
// Every state item is classified by the reviewed record (`WorldWriters`, `world-state.ts`). Part 1 writes the
// E items through the engine's own methods: HP (`sethp`), fainted Pokemon (`faint`, `faintMessages`), major
// status (`setStatus`), stat stages, Mega Evolution (`runMegaEvo`), a forme change (`formeChange`), own PP
// (`deductPP`), weather, terrain, field conditions and side conditions with their layers. Part 2 adds the W
// writers fed by the world facts (Toxic stage, bench status, `activeTurns`, the Protect counter, the last move
// and the choice lock, side faint totals, the remaining turns of timed conditions and admitted volatiles, the
// turn number, a consumed item, a start trigger's restore), counts the X items the engine resamples (sleep,
// confusion, the opponent's PP, the hidden bench order), lets the engine's own turn rollover reset the T items
// and refuses the R items. The world is absolute: whatever a switch-in effect set at the scenario start is
// replaced.
//
// Five checks run on every world. V1: each set passes the target format's `validateSet`, is the engine's
// canonical paste and keeps every revealed fact and Item Clause. V2: the world's own request and candidate
// list equal the real ones. V3: the opponent as the engine renders it for the player equals the reader's
// state. V4: field and side conditions equal the reader's and their remaining turns are consistent with the
// log. V5: same seat and nicknames. Every refusal is typed, carries the check and the state class, says
// whether drawing the opponent's sets again can fix it, carries fixed words only and is counted.

import type { PublicFlag, PublicMon, PublicState, WorldFacts } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import { assumedHp } from "./damage-aware-assumed-set.ts";
import type { DamageAwareConfig } from "./damage-aware-config.ts";
import type { Engine, PokemonSet, SimBattle } from "./engine.ts";
import { startScenario } from "./mechanics-scenario.ts";
import { exportPaste, importPaste } from "./teams.ts";
import { type Remainder, v2Candidates, v2Request, v4Conditions, v4Durations } from "./world-checks.ts";
import { type WBattle, type WMon, type WSide, STAGES, conditionKind, toId } from "./world-engine.ts";
import { Refuse, WORLD_CHECKS, WORLD_REFUSAL_CODES, type WorldCheck, type WorldRefusal, type WorldRefusalCode, need } from "./world-refusal.ts";
import {
  type Classified,
  type Foe,
  type MonPlan,
  type OwnEntry,
  type Resample,
  type StatePlan,
  addVolatileAt,
  classify,
  flagItem,
  holds,
  refuseItem,
  remainingTurns,
  restoreStart,
  rollTurn,
  statePlan,
  writeBenchStatus,
  writeChoiceLock,
  writeItemRemoved,
  writeLastMove,
  writeProtectCounter,
  writeSide,
  writeToxicStage,
} from "./world-state.ts";
import { STATE_CLASSES, STATE_ITEM_IDS, type StateClass, type StateItemId, type WorldWriters } from "./world-writers.ts";

export { WORLD_CHECKS, WORLD_REFUSAL_CODES };
export type { Resample, WorldCheck, WorldRefusal, WorldRefusalCode };

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
  /** The world facts of the player's own channel at this request (null when the facts module refused: the world is refused). */
  facts: WorldFacts | null;
  opponent: WorldOpponent;
}

export interface WorldContext {
  engine: Engine;
  /** The target format (its `TeamValidator` judges the sets). */
  format_id: string;
  /** The scenario format id (`checkScenario`). */
  scenario_id: string;
  /** How a shown percentage becomes hit points (ADR-018 `assumedHp`). The Champions display is `floor(100 * hp / max)`, which `ceil` inverts for a maximum of 100 or more. */
  hp_rounding: DamageAwareConfig["hp_rounding"];
  /** The reviewed record of the state classes and writers (`config/search/world-writers.v1.json`). */
  writers: WorldWriters;
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
  /** State the world resamples through the engine's own draw or declares approximate (X class); counted. */
  resampled: readonly Resample[];
  /** The W writers that wrote something in this world. */
  written: readonly StateItemId[];
  /** Conditions whose source Pokemon the log does not name. */
  defaulted_sources: number;
  /** The remaining turns the writers computed from the log (V4 checked each is at least 1). */
  remainders: readonly Remainder[];
}

export type WorldResult = { kind: "world"; world: ScenarioWorld } | { kind: "refused"; refusal: WorldRefusal };

export interface WorldCounts {
  requests: number;
  built: number;
  refused: number;
  refusals: Record<WorldRefusalCode, number>;
  /** Refusals by the check that failed (V1 to V5). */
  checks: Record<WorldCheck, number>;
  /** Refusals by the state class involved. */
  classes: Record<StateClass, number>;
  /** Refusals by check and class, as `V2:W`, `V1:-`, `-:R` (`-` is none). */
  check_class: Record<string, number>;
  /** Refusals of a state item (`state_unwritten`) by item. */
  unwritten: Record<StateItemId, number>;
  /** Built worlds in which the item was resampled or declared approximate (X class). */
  resampled: Record<Resample["item"], number>;
  /** Built worlds in which the W writer wrote. */
  written: Record<StateItemId, number>;
  /** Conditions whose source was defaulted, over built worlds. */
  defaulted_sources: number;
}

const SLOT_POSITION: Record<string, number> = { a: 0, b: 1 };
const SLOT_LETTER = ["a", "b"] as const;
const OWN_CONDITION = /^(\d+)\/(\d+)(?: ([a-z]+))?$/;
const SHOWN_HEALTH = /^(\d+)\/100/;
/** The permanent forme a Pokemon is shown in (its details), else the set's. */
const permanentOf = (mon: PublicMon | undefined, set: PokemonSet): string => mon?.species ?? toId(set.species);
const SEATS = ["p1", "p2"] as const;
const X_ITEMS = ["sleep_counter", "confusion_duration", "opponent_pp", "bench_order"] as const;

const zeros = <K extends string>(keys: readonly K[]): Record<K, number> => Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;

const emptyCounts = (): WorldCounts => ({
  requests: 0,
  built: 0,
  refused: 0,
  refusals: zeros(WORLD_REFUSAL_CODES),
  checks: zeros(WORLD_CHECKS),
  classes: zeros(STATE_CLASSES),
  check_class: {},
  unwritten: zeros(STATE_ITEM_IDS),
  resampled: zeros(X_ITEMS),
  written: zeros(STATE_ITEM_IDS),
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
  const plan: StatePlan = statePlan(context.writers);
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

  /** The engine's Pokemon a condition's source names: by the nickname the log shows, else the first Pokemon of the side it belongs to (counted). */
  function sourceOf(battle: WBattle, seat: number, source: { side: "own" | "opponent"; name: string } | null | undefined, fallbackSide: "own" | "opponent", defaulted: { n: number }): WMon {
    const found = source == null ? undefined : battle.sides[source.side === "own" ? seat : 1 - seat]?.pokemon.find((mon) => mon.name === source.name);
    if (found === undefined) defaulted.n += 1;
    const mon = found ?? battle.sides[fallbackSide === "own" ? seat : 1 - seat]?.pokemon[0];
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

  /** Write one living Pokemon's E state: status, hit points, stages and PP. A status on the bench is the bench writer's. */
  function writeMon(battle: WBattle, mon: WMon, to: { hp: number | null; status: string | null; boosts: Record<string, number>; active: boolean }, where: string, written: Set<StateItemId>): void {
    if (to.status !== null) {
      need(conditionKind(battle, to.status) === "Status", "engine_refused", `${where} has a status the engine does not have`);
      if (!to.active) {
        writeBenchStatus(battle, mon, to.status);
        written.add("bench_status");
      } else mon.setStatus(to.status, null, null, true); // an observed fact: type immunities do not apply
      need(mon.status === to.status, "engine_refused", `${where} cannot take the status shown`);
    }
    if (to.hp !== null && to.hp < mon.maxhp) {
      need(to.hp >= 1, "engine_refused", `${where} has no hit points left to write`);
      mon.sethp(to.hp);
      need(mon.hp === to.hp, "engine_refused", `${where} cannot take the hit points shown`);
    }
    for (const stage of STAGES) mon.boosts[stage] = to.boosts[stage] ?? 0;
  }

  /** Own PP, down to the request's count (a fainted Pokemon keeps the PP it had). */
  function writePP(mon: WMon, pp: ReadonlyMap<string, number>, where: string): void {
    for (const slot of mon.moveSlots) {
      const left = pp.get(slot.id);
      if (left === undefined || left === slot.maxpp) continue;
      need(left >= 0 && left < slot.maxpp, "own_team_mismatch", `${where} has a PP count the engine's maximum does not allow`);
      mon.deductPP(slot.id, slot.maxpp - left);
      need(slot.pp === left, "engine_refused", `${where} cannot take the PP shown`);
    }
  }

  /** The W state of one living Pokemon, written after its E state: Toxic stage, the item gone, last move, choice lock, Protect counter, volatiles. */
  function writeWorking(battle: WBattle, mon: WMon, monPlan: MonPlan, turn: number, where: string, written: Set<StateItemId>, remainders: Remainder[]): void {
    if (monPlan.toxic_stage !== null) {
      writeToxicStage(mon, monPlan.toxic_stage);
      written.add("toxic_stage");
    }
    if (monPlan.remove_item) {
      writeItemRemoved(battle, mon);
      written.add("item_removed");
    }
    const facts = monPlan.facts;
    if (facts?.last_move != null) {
      holds(plan, "last_move", `${where} has a last move`);
      writeLastMove(battle, mon, facts.last_move);
      written.add("last_move");
      if (battle.dex.items.get(mon.item).isChoice === true) {
        holds(plan, "choice_lock", `${where} holds a Choice item and has chosen a move`);
        if (writeChoiceLock(battle, mon, plan.conditionOf("choice_lock"), facts.last_move)) written.add("choice_lock");
      }
    }
    if (facts !== null && facts.protect_streak > 0) {
      holds(plan, "protect_counter", `${where} has used a stalling move`);
      writeProtectCounter(mon, plan.conditionOf("protect_counter"), facts.protect_streak);
      written.add("protect_counter");
    }
    for (const volatile of monPlan.volatiles) {
      addVolatileAt(battle, mon, volatile.id, volatile.began?.moved_before ?? true);
      need(mon.volatiles[volatile.id] !== undefined, "engine_refused", `${where} cannot take a volatile shown`);
      if (volatile.began !== null) {
        const remaining = remainingTurns(mon.volatiles[volatile.id], volatile.began, turn);
        if (remaining !== null) remainders.push({ what: "volatile", remaining });
        written.add("volatile_duration");
      }
    }
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
    if (input.flags.length > 0) {
      if (flagItem(plan, input.flags)) refuseItem(plan, "unavailable_mechanics", "the channel shows a mechanic this format does not have");
      throw new Refuse("reader_flagged", "the reader flagged a line");
    }
    const { state, facts } = input;
    need(facts !== null, "facts_unavailable", "the world facts refused the channel");
    const worldFacts = facts as WorldFacts;
    need(worldFacts.turn === state.turn && state.turn >= 1, "facts_mismatch", "the world facts are for another turn than the state");
    const seatIndex = SEATS.indexOf(state.player as "p1" | "p2");
    need(seatIndex >= 0, "v5_mismatch", "the state is for no seat of a two-player battle");
    const own = ownEntries(input);
    const foes: Foe[] = opponentOrder(input).map(({ index, set, mon }) => ({ index, set, mon }));
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
      need(new Set(items).size === items.length, "v1_item_clause", "two opponent sets hold one item", { redraw: true });
    }
    if (state.opponent.roster.some((mon) => mon.phantom === true) || state.own.roster.some((mon) => mon.phantom === true)) refuseItem(plan, "illusion_transform", "the reader shows an Illusion pair");
    const classified: Classified = classify({ request: input.request, state, own, foes, facts: worldFacts, plan });

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
    const written = new Set<StateItemId>();
    const remainders: Remainder[] = [];

    // Switch-in effects of the start may have changed an item, an ability or left a volatile. An item or ability that an admitted trigger changed is restored; a volatile never is.
    const afterStart = (mon: WMon, set: PokemonSet, whose: string, index: number): void => {
      if (Object.keys(mon.volatiles).length > 0) refuseItem(plan, "start_trigger_restore", `a switch-in effect left a volatile on ${whose} Pokemon ${index}`);
      const itemChanged = toId(mon.item) !== toId(set.item);
      const abilityChanged = toId(mon.ability) !== toId(set.ability);
      if (!itemChanged && !abilityChanged) return;
      holds(plan, "start_trigger_restore", `a switch-in effect changed ${whose} Pokemon ${index}`);
      const covered = (!itemChanged || plan.admittedTrigger(toId(set.item))) && (!abilityChanged || plan.admittedTrigger(toId(set.ability)));
      if (!covered) refuseItem(plan, "start_trigger_restore", `a switch-in effect changed ${whose} Pokemon ${index} and no trigger is admitted for it`);
      restoreStart(battle, mon, set);
      written.add("start_trigger_restore");
    };
    own.forEach((entry, index) => afterStart(ownSide.pokemon[index] as WMon, entry.set, "own", index));
    foes.forEach((foe, index) => afterStart(foeSide.pokemon[index] as WMon, foe.set, "opponent", index));

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
    // The engine puts a fainted Pokemon back to its set's species (a busted Disguise); the real one keeps the forme it fainted in, which the player's request shows.
    own.forEach((entry, index) => {
      const mon = ownSide.pokemon[index] as WMon;
      if (mon.fainted && entry.shown?.mega !== true) setForme(battle, mon, { mega: false, permanent: permanentOf(entry.shown, entry.set), temporary: null }, `own Pokemon ${index}`);
    });

    for (const [index, entry] of own.entries()) {
      const mon = ownSide.pokemon[index] as WMon;
      const pp = new Map<string, number>(Object.entries(entry.shown?.pp ?? {}));
      for (const move of input.request.active?.[index]?.moves ?? []) if (move.pp !== undefined) pp.set(move.id, move.pp);
      writePP(mon, pp, `own Pokemon ${index}`);
      if (mon.fainted) continue;
      writeMon(battle, mon, { hp: entry.hp, status: entry.status, boosts: entry.shown?.boosts ?? {}, active: index < 2 }, `own Pokemon ${index}`, written);
    }
    for (const [index, foe] of foes.entries()) {
      const mon = foeSide.pokemon[index] as WMon;
      if (mon.fainted) continue;
      const percent = foe.mon.hp_num;
      const hp = percent === null ? null : assumedHp(percent, mon.maxhp, context.hp_rounding);
      writeMon(battle, mon, { hp, status: foe.mon.status, boosts: foe.mon.boosts, active: index < 2 }, `opponent Pokemon ${index}`, written);
    }

    // Own stats against the request: the same set in the same place.
    for (const [index, entry] of own.entries()) {
      const mon = ownSide.pokemon[index] as WMon;
      const stats = input.request.side.pokemon[index]?.stats as unknown as Record<string, number>;
      need(mon.maxhp === entry.maxhp || entry.hp === 0, "own_team_mismatch", `own Pokemon ${index} has another maximum hit points than the request`);
      need(Object.entries(stats).every(([stat, value]) => mon.storedStats[stat] === value), "own_team_mismatch", `own Pokemon ${index} has other stats than the request`);
    }

    // The W state of the Pokemon in play and on the bench.
    const rolled: { mon: WMon; turns: number; actions: number }[] = [];
    for (const [sideName, side, plans] of [["own", ownSide, classified.own], ["opponent", foeSide, classified.foes]] as const) {
      for (const monPlan of plans) {
        const mon = side.pokemon[monPlan.index] as WMon;
        if (monPlan.remove_item && mon.fainted) {
          writeItemRemoved(battle, mon);
          written.add("item_removed");
        }
        if (mon.fainted) continue;
        writeWorking(battle, mon, monPlan, state.turn, `${sideName} Pokemon ${monPlan.index}`, written, remainders);
        if (monPlan.facts !== null) rolled.push({ mon, turns: monPlan.facts.active_turns, actions: monPlan.facts.move_actions });
      }
    }
    holds(plan, "active_turns", "the world holds Pokemon in play");
    if (rolled.length > 0) written.add("active_turns");
    for (const [sideName, side, sideFacts] of [["own", ownSide, worldFacts.own], ["opponent", foeSide, worldFacts.opponent]] as const) {
      if (sideFacts.total_fainted > 0) holds(plan, "total_fainted", `the ${sideName} side has lost a Pokemon`);
      if (sideFacts.fainted_last_turn) holds(plan, "fainted_last_turn", `the ${sideName} side lost a Pokemon last turn`);
      writeSide(side, sideFacts);
      if (sideFacts.total_fainted > 0) written.add("total_fainted");
      if (sideFacts.fainted_last_turn) written.add("fainted_last_turn");
    }

    // The field, absolute: what a switch-in set is cleared first. Each timed condition keeps its source's base length less the turns since it began.
    const defaulted = { n: 0 };
    const timed = (state_: { duration?: number } | undefined, began: { started_turn: number; after_residual: boolean } | undefined, what: Remainder["what"]): void => {
      need(began !== undefined, "facts_mismatch", "the world facts do not date a condition the state shows");
      const remaining = remainingTurns(state_, began as { started_turn: number; after_residual: boolean }, state.turn);
      if (remaining === null) return;
      holds(plan, "condition_duration", `a ${what} condition counts down`);
      remainders.push({ what, remaining });
      written.add("condition_duration");
    };
    battle.field.clearWeather();
    battle.field.clearTerrain();
    if (state.weather !== null) {
      need(conditionKind(battle, state.weather) === "Weather", "engine_refused", "the weather shown is not an engine weather");
      need(worldFacts.weather?.id === state.weather, "facts_mismatch", "the world facts show another weather than the state");
      battle.field.setWeather(state.weather, sourceOf(battle, seatIndex, worldFacts.weather?.source, "own", defaulted));
      need(battle.field.weather === state.weather, "engine_refused", "the engine refused the weather shown");
      timed(battle.field.weatherState, worldFacts.weather ?? undefined, "weather");
    }
    if (state.terrain !== null) {
      need(conditionKind(battle, state.terrain) === "Terrain", "engine_refused", "the terrain shown is not an engine terrain");
      need(worldFacts.terrain?.id === state.terrain, "facts_mismatch", "the world facts show another terrain than the state");
      battle.field.setTerrain(state.terrain, sourceOf(battle, seatIndex, worldFacts.terrain?.source, "own", defaulted));
      need(battle.field.terrain === state.terrain, "engine_refused", "the engine refused the terrain shown");
      timed(battle.field.terrainState, worldFacts.terrain ?? undefined, "terrain");
    }
    for (const id of state.field_conditions) {
      need(conditionKind(battle, id) !== "none", "engine_refused", "a field condition shown is not an engine condition");
      battle.field.addPseudoWeather(id, sourceOf(battle, seatIndex, worldFacts.field_conditions[id]?.source, "own", defaulted));
      need(battle.field.pseudoWeather[id] !== undefined, "engine_refused", "the engine refused a field condition shown");
      timed(battle.field.pseudoWeather[id], worldFacts.field_conditions[id], "field");
    }
    const sides = [
      { shown: state.own.conditions, side: ownSide, began: worldFacts.own.conditions, whose: "own" as const },
      { shown: state.opponent.conditions, side: foeSide, began: worldFacts.opponent.conditions, whose: "opponent" as const },
    ];
    for (const { shown, side, began, whose } of sides) {
      for (const [id, layers] of Object.entries(shown)) {
        need(conditionKind(battle, id) !== "none", "engine_refused", "a side condition shown is not an engine condition");
        need(Number.isInteger(layers) && layers >= 1, "engine_refused", "a side condition has no layer count");
        for (let layer = 0; layer < layers; layer += 1) {
          side.addSideCondition(id, sourceOf(battle, seatIndex, began[id]?.source, whose, defaulted));
        }
        const held = side.sideConditions[id];
        need(held !== undefined && (held.layers === undefined ? layers === 1 : held.layers === layers), "engine_refused", "the engine refused a side condition or its layers");
        timed(held, began[id], "side");
      }
    }

    // The turn: the engine's own rollover resets the per-turn fields (T), counts the turns in play and makes the request.
    holds(plan, "turn_number", "the world is at a turn");
    written.add("turn_number");
    rollTurn(battle, state.turn, rolled);
    need(!battle.ended, "start_failed", "the world ended the battle");

    // V5 and V3.
    need(own.every((entry, index) => (ownSide.pokemon[index] as WMon).name === entry.name), "v5_mismatch", "an own nickname differs from the request's");
    need(foes.every((foe, index) => foe.mon.name === null || (foeSide.pokemon[index] as WMon).name === foe.mon.name), "v5_mismatch", "an opponent nickname differs from the shown one");
    checkRendered(battle, seatIndex, foes);

    // V4.
    const layersOf = (side: WSide): Record<string, number> => Object.fromEntries(Object.entries(side.sideConditions).filter(([, held]) => held !== undefined).map(([id, held]) => [id, held?.layers ?? 1]));
    const failure4 =
      v4Conditions(
        { weather: battle.field.weather === "" ? null : battle.field.weather, terrain: battle.field.terrain === "" ? null : battle.field.terrain, field: Object.keys(battle.field.pseudoWeather).filter((id) => battle.field.pseudoWeather[id] !== undefined), own: layersOf(ownSide), opponent: layersOf(foeSide) },
        { weather: state.weather, terrain: state.terrain, field: state.field_conditions, own: state.own.conditions, opponent: state.opponent.conditions },
      ) ?? v4Durations(remainders);
    // V2.
    const worldRequest = ownSide.activeRequest;
    need(worldRequest !== null && worldRequest !== undefined, "v2_request", "the world made no request");
    const failure2 = v2Request(worldRequest, input.request) ?? v2Candidates(worldRequest, input.request);
    for (const failure of [failure2, failure4]) {
      if (failure === null) continue;
      const extra: { item?: StateItemId; state_class?: StateClass; redraw: boolean } = { redraw: failure.redraw };
      if (failure.item !== null) extra.item = failure.item;
      if (failure.state_class !== null) extra.state_class = failure.state_class === "E" ? "E" : plan.classOf(failure.item as StateItemId);
      throw new Refuse(failure.code, failure.detail, extra);
    }

    counts.defaulted_sources += defaulted.n;
    return {
      battle: battle as unknown as SimBattle,
      seat: state.player as "p1" | "p2",
      own_order: own.map((entry) => entry.name),
      opponent_order: foes.map((foe) => foe.index),
      resampled: classified.resampled,
      written: [...written].sort(),
      defaulted_sources: defaulted.n,
      remainders,
    };
  }

  return {
    build(input) {
      counts.requests += 1;
      try {
        const world = make(input);
        counts.built += 1;
        for (const item of world.written) counts.written[item] += 1;
        for (const item of new Set(world.resampled.map((entry) => entry.item))) counts.resampled[item] += 1;
        return { kind: "world", world };
      } catch (error) {
        if (!(error instanceof Refuse)) throw error;
        const { refusal } = error;
        counts.refused += 1;
        counts.refusals[refusal.code] += 1;
        if (refusal.check !== null) counts.checks[refusal.check] += 1;
        if (refusal.state_class !== null) counts.classes[refusal.state_class] += 1;
        if (refusal.check !== null || refusal.state_class !== null) {
          const key = `${refusal.check ?? "-"}:${refusal.state_class ?? "-"}`;
          counts.check_class[key] = (counts.check_class[key] ?? 0) + 1;
        }
        if (refusal.code === "state_unwritten" && refusal.item !== null) counts.unwritten[refusal.item] += 1;
        return { kind: "refused", refusal };
      }
    },
    counts: () => structuredClone(counts),
  };
}
