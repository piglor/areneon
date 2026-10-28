import { beforeAll, describe, expect, expectTypeOf, it } from "vitest";
import type { PublicState, WorldFacts } from "@areneon/public-state";
import type { EngineRequest } from "@areneon/schema/types/v1/engine-request.ts";
import type { Engine, PokemonSet, SimBattle } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type HiddenVerdicts, loadEngine, type BattleRecord, type OmniscientLog, type PlayerChannel } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import {
  WORLD_REFUSAL_CODES,
  type WorldBuilder,
  type WorldContext,
  type WorldInput,
  type WorldOpponentMember,
  createWorldBuilder,
} from "../src/world-builder.ts";
import { STATE_ITEM_IDS, loadWorldWriters } from "../src/world-writers.ts";
import { type Case, cases, toId } from "./world-builder-support.ts";

let engine: Engine;
let context: WorldContext;
beforeAll(() => {
  engine = loadEngine();
  context = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: "ceil", writers: loadWorldWriters(engine, GOLDEN_FORMAT) };
});

const withState = (input: WorldInput, change: (state: PublicState) => void): WorldInput => {
  const state = structuredClone(input.state);
  change(state);
  return { ...input, state };
};
/** Change the reader's state and the world facts together (a condition the state shows is one the facts date). */
const withWorld = (input: WorldInput, change: (state: PublicState, facts: WorldFacts) => void): WorldInput => {
  const state = structuredClone(input.state);
  const facts = structuredClone(input.facts) as WorldFacts;
  change(state, facts);
  return { ...input, state, facts };
};

/** The refused state items the input really carries, read from the reader's state, the request and the facts alone (not from the builder). */
function present(input: WorldInput, rows: Record<string, readonly string[]>): Set<string> {
  const items = new Set<string>();
  const { state, request, facts } = input;
  const roster = [...state.own.roster, ...state.opponent.roster];
  const row = (id: string): string | undefined => Object.entries(rows).find(([, ids]) => ids.includes(id))?.[0];
  for (const mon of roster) {
    for (const id of mon.volatiles) {
      const named = row(id);
      if (named !== undefined) items.add(named);
      else if (id !== "confusion") items.add("unadmitted_id");
    }
  }
  const brought = new Map(input.team.map((s) => [toId(s.name === "" ? s.species : s.name), toId(s.item)]));
  if (state.opponent.roster.some((m) => m.item_received)) items.add("item_replaced");
  for (const p of request.side.pokemon) {
    const was = brought.get(toId(p.ident.slice(p.ident.indexOf(": ") + 2)));
    if (p.item !== "" && was !== p.item) items.add("item_replaced");
  }
  if (roster.some((m) => m.ability_received || m.ability_suppressed) || request.side.pokemon.some((p) => p.ability !== undefined && p.ability !== p.baseAbility)) items.add("ability_changed");
  if (roster.some((m) => m.phantom === true)) items.add("illusion_transform");
  if ((request.active ?? []).some((a, i) => a.moves.length < (request.side.pokemon[i]?.moves.length ?? 0))) items.add("locked_moves");
  for (const side of [facts?.own, facts?.opponent]) {
    for (const active of Object.values(side?.active ?? {})) {
      const named = active?.last_move == null ? undefined : row(active.last_move);
      if (named === "slot_conditions" || named === "locked_moves") items.add(named);
    }
  }
  for (const p of request.side.pokemon.slice(0, 2)) {
    const named = row(p.baseAbility);
    if (named === "truant_slow_start" || named === "illusion_transform") items.add(named);
  }
  for (const member of input.opponent.members) {
    const mon = state.opponent.roster[member.roster_index];
    const named = mon !== undefined && !mon.fainted ? row(toId(member.set.ability)) : undefined;
    // A hiding ability counts anywhere on the living team (the disguise cannot be told from the bench); a turn-parity ability only in play.
    if (named === "illusion_transform" || (named === "truant_slow_start" && mon?.active_slot !== null)) items.add(named);
  }
  return items;
}

describe("worlds at every move request of the committed channels", () => {
  it("build, pass all five checks, and are refused only for state the record classifies R, facts that refuse or an Illusion", () => {
    const builder = createWorldBuilder(context);
    const all = cases();
    expect(all.length).toBeGreaterThan(50);
    const rows = Object.fromEntries(context.writers.items.filter((item) => item.ids !== undefined && item.id !== "unavailable_mechanics").map((item) => [item.id, item.ids as string[]]));
    const times: number[] = [];
    const builtTimes: number[] = [];
    const other: string[] = [];
    let trapped = 0;
    for (const { id, input } of all) {
      const before = performance.now();
      const result = builder.build(input);
      times.push(performance.now() - before);
      if (result.kind === "world") builtTimes.push(times[times.length - 1] as number);
      if (result.kind === "refused") {
        // An Illusion disguise that never broke is attributed to the Pokemon it was shown as (the reader's known limit): the true team contradicts a revealed move.
        const disguised = result.refusal.code === "v1_revealed_fact" && result.refusal.detail.includes("revealed move") && input.opponent.members.some((member) => toId(member.set.ability) === "illusion");
        const noFacts = result.refusal.code === "facts_unavailable" && input.facts === null;
        if (result.refusal.code === "state_unwritten") {
          expect(STATE_ITEM_IDS).toContain(result.refusal.item);
          expect(result.refusal.state_class).toBe("R");
          // The refusal is justified: the input really carries state of that kind (a start trigger is the engine's, seen only after the start).
          if (result.refusal.item !== "start_trigger_restore") expect(present(input, rows), `${id}: ${result.refusal.detail}`).toContain(result.refusal.item);
        } else if (result.refusal.code === "v2_request" && result.refusal.detail.includes("trapping flags")) {
          // The own request says it is trapped and the world has nothing that traps it (a Mean Look or Block volatile the reader never shows, or an ability the sampled opponent does not hold): evidence, so a redraw can try again.
          expect(result.refusal).toMatchObject({ check: "V2", state_class: null, redraw: true });
          trapped += 1;
        } else if (result.refusal.code !== "reader_flagged" && !disguised && !noFacts) other.push(`${id}: ${result.refusal.code} ${result.refusal.detail}`);
      } else {
        expect(result.world.battle.turn).toBe(input.state.turn);
        expect(result.world.own_order.length).toBe(input.state.own.teamsize);
        expect(result.world.opponent_order.length).toBe(input.state.opponent.teamsize);
      }
    }
    expect(other).toEqual([]);
    const counts = builder.counts();
    expect(counts.refusals.v2_request).toBe(trapped);
    expect(counts.requests).toBe(all.length);
    expect(counts.built + counts.refused).toBe(all.length);
    expect(counts.built).toBeGreaterThanOrEqual(800);
    expect(Object.values(counts.refusals).reduce((a, b) => a + b, 0)).toBe(counts.refused);
    expect(Object.values(counts.unwritten).reduce((a, b) => a + b, 0)).toBe(counts.refusals.state_unwritten);
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    const builtMean = builtTimes.reduce((a, b) => a + b, 0) / builtTimes.length;
    // The measured cost of a four-a-side world (start, V1 on eight sets, the writes, the turn rollover, V2 to V5); journaled on the ticket.
    console.log(`world builder: ${counts.built} built of ${counts.requests}; refusals ${JSON.stringify(counts.refusals)}; unwritten ${JSON.stringify(counts.unwritten)}; check_class ${JSON.stringify(counts.check_class)}; resampled ${JSON.stringify(counts.resampled)}; written ${JSON.stringify(counts.written)}; mean ${builtMean.toFixed(1)} ms per built world, ${mean.toFixed(1)} ms per request, max ${Math.max(...times).toFixed(1)} ms`);
    expect(mean).toBeLessThan(2000);
  });

  it("does not change its inputs", () => {
    const { input } = cases().find((c) => createWorldBuilder(context).build(c.input).kind === "world") as Case;
    const before = JSON.stringify(input);
    createWorldBuilder(context).build(input);
    expect(JSON.stringify(input)).toBe(before);
  });
});

/** A case that builds, to mutate. */
let base: Case;
const builds = (builder: WorldBuilder, input: WorldInput) => builder.build(input);
beforeAll(() => {
  const builder = createWorldBuilder(context);
  // A world with a living opponent active in slot a, a benched living member, an unseen member and no field state.
  base = cases().find((c) => {
    const roster = c.input.state.opponent.roster;
    const members = new Set(c.input.opponent.members.map((m) => m.roster_index));
    return (
      builder.build(c.input).kind === "world" &&
      roster.some((m) => m.active_slot === "a" && !m.fainted) &&
      c.input.opponent.members.some((m) => roster[m.roster_index]?.active_slot === null && roster[m.roster_index]?.fainted === false) &&
      roster.some((_, index) => !members.has(index)) &&
      c.input.state.weather === null && c.input.state.terrain === null &&
      c.input.state.own.roster.some((m) => m.name !== null && m.active_slot !== null)
    );
  }) as Case;
});

describe("the state items written through the engine's own methods", () => {
  const world = (input: WorldInput) => {
    const result = createWorldBuilder(context).build(input);
    if (result.kind !== "world") throw new Error(`refused: ${result.refusal.code} ${result.refusal.detail}`);
    return result.world;
  };
  const battleOf = (input: WorldInput) => world(input).battle as unknown as {
    sides: { pokemon: { hp: number; maxhp: number; status: string; fainted: boolean; boosts: Record<string, number>; moveSlots: { id: string; pp: number; maxpp: number }[] }[]; sideConditions: Record<string, { layers?: number }> }[];
    field: { weather: string; terrain: string; pseudoWeather: Record<string, unknown>; weatherState: { duration: number } };
  };
  const seatIndex = () => (base.input.state.player === "p1" ? 0 : 1);

  it("writes weather, terrain, Trick Room, screens, Tailwind and hazards with layers, and the turns they have left", () => {
    const began = (facts: WorldFacts, turns: number) => ({ started_turn: facts.turn - turns, after_residual: false, source: null });
    const input = withWorld(base.input, (state, facts) => {
      state.weather = "raindance";
      state.terrain = "electricterrain";
      state.field_conditions = ["trickroom"];
      state.own.conditions = { reflect: 1, tailwind: 1 };
      state.opponent.conditions = { spikes: 3, stealthrock: 1 };
      facts.weather = { id: "raindance", ...began(facts, 1) };
      facts.terrain = { id: "electricterrain", ...began(facts, 2) };
      facts.field_conditions = { trickroom: began(facts, 3) };
      facts.own.conditions = { reflect: { ...began(facts, 1), layers: 1 }, tailwind: { ...began(facts, 1), layers: 1 } };
      facts.opponent.conditions = { spikes: { ...began(facts, 1), layers: 3 }, stealthrock: { ...began(facts, 1), layers: 1 } };
    });
    const battle = battleOf(input) as ReturnType<typeof battleOf> & { sides: { sideConditions: Record<string, { duration?: number }> }[]; field: { pseudoWeather: Record<string, { duration?: number }>; terrainState: { duration: number } } };
    expect(battle.field.weather).toBe("raindance");
    expect(battle.field.terrain).toBe("electricterrain");
    expect(Object.keys(battle.field.pseudoWeather)).toContain("trickroom");
    const own = battle.sides[seatIndex()];
    const foe = battle.sides[1 - seatIndex()];
    expect(Object.keys(own?.sideConditions ?? {}).sort()).toEqual(["reflect", "tailwind"]);
    expect(foe?.sideConditions.spikes?.layers).toBe(3);
    expect(foe?.sideConditions.stealthrock).toBeDefined();
    // The engine's own base length (the source's item reads in its callback) less the turns since the log started it, never a fresh full duration.
    const fresh = battleOf(withWorld(base.input, (state, facts) => {
      state.weather = "raindance";
      facts.weather = { id: "raindance", started_turn: facts.turn, after_residual: false, source: null };
    }));
    expect(fresh.field.weatherState.duration).toBe(5);
    expect(battle.field.weatherState.duration).toBe(4);
    expect(battle.field.terrainState.duration).toBe(3);
    expect(battle.field.pseudoWeather.trickroom?.duration).toBe(2);
    expect(own?.sideConditions.tailwind?.duration).toBe(3);
  });

  it("clears what a switch-in set when the state shows none", () => {
    const battle = battleOf(base.input);
    expect(battle.field.weather).toBe("");
    expect(battle.field.terrain).toBe("");
  });

  it("gives a condition the length its source's item sets: Heat Rock on the source named by the log, not the default", () => {
    const sun = withWorld(base.input, (state, facts) => {
      state.weather = "sunnyday";
      facts.weather = { id: "sunnyday", started_turn: facts.turn, after_residual: false, source: null };
    });
    const defaulted = createWorldBuilder(context).build(sun);
    expect(defaulted.kind).toBe("world");
    if (defaulted.kind !== "world") return;
    expect(defaulted.world.defaulted_sources).toBe(1);
    const plain = (defaulted.world.battle as unknown as { field: { weatherState: { duration: number } } }).field.weatherState.duration;
    // The own Pokemon at position 1 holds Heat Rock in the world: the log names it as the source.
    const team = base.input.team.map((set) => structuredClone(set) as PokemonSet);
    const request = structuredClone(base.input.request);
    const name = request.side.pokemon[1]?.ident.slice(request.side.pokemon[1].ident.indexOf(": ") + 2) as string;
    const set = team.find((candidate) => (candidate.name === "" ? candidate.species : candidate.name) === name) as PokemonSet;
    const used = new Set(team.map((s) => toId(s.item)));
    if (used.has("heatrock")) return;
    set.item = "Heat Rock";
    (request.side.pokemon[1] as { item: string }).item = "heatrock";
    const named = withWorld({ ...sun, team, request }, (_state, facts) => {
      facts.weather = { id: "sunnyday", started_turn: facts.turn, after_residual: false, source: { side: "own", name } };
    });
    const rocked = createWorldBuilder(context).build(named);
    expect(rocked.kind).toBe("world");
    if (rocked.kind !== "world") return;
    expect(rocked.world.defaulted_sources).toBe(0);
    const longer = (rocked.world.battle as unknown as { field: { weatherState: { duration: number } } }).field.weatherState.duration;
    expect(longer).toBeGreaterThan(plain);
  });

  it("writes major status, stat stages and counts a sleep counter as resampled", () => {
    const input = withState(base.input, (state) => {
      const mon = state.opponent.roster.find((m) => m.active_slot === "a" && !m.fainted);
      if (mon === undefined) throw new Error("no opponent active");
      mon.status = "slp";
      mon.boosts = { ...mon.boosts, atk: 2, spe: -1 };
    });
    const builder = createWorldBuilder(context);
    const result = builder.build(input);
    expect(result.kind).toBe("world");
    if (result.kind !== "world") return;
    expect(result.world.resampled.filter((entry) => entry.item === "sleep_counter")).toEqual([{ side: "opponent", position: 0, item: "sleep_counter" }]);
    expect(builder.counts().resampled.sleep_counter).toBe(1);
    const foe = (result.world.battle as unknown as { sides: { pokemon: { status: string; boosts: Record<string, number> }[] }[] }).sides[1 - seatIndex()]?.pokemon[0];
    expect(foe?.status).toBe("slp");
    expect(foe?.boosts.atk).toBe(2);
    expect(foe?.boosts.spe).toBe(-1);
  });

  it("writes opponent hit points with assumedHp and the engine shows them back as the reader's percentage", () => {
    const input = withState(base.input, (state) => {
      const mon = state.opponent.roster.find((m) => m.active_slot === "a" && !m.fainted);
      if (mon === undefined) throw new Error("no opponent active");
      mon.hp_num = 37;
    });
    const foe = battleOf(input).sides[1 - seatIndex()]?.pokemon[0];
    expect(Math.floor((100 * (foe?.hp ?? 0)) / (foe?.maxhp ?? 1))).toBe(37);
  });

  it("writes own PP to the request", () => {
    const request = structuredClone(base.input.request);
    const move = (request.active?.[0] as NonNullable<EngineRequest["active"]>[number]).moves[0] as { id: string; pp?: number; maxpp?: number };
    move.pp = 1;
    const slots = battleOf({ ...base.input, request }).sides[seatIndex()]?.pokemon[0]?.moveSlots;
    expect(slots?.find((slot) => slot.id === move.id)?.pp).toBe(1);
  });
});

describe("the checks and the typed, counted refusals", () => {
  const refusal = (input: WorldInput, builder: WorldBuilder = createWorldBuilder(context)) => {
    const result = builder.build(input);
    if (result.kind !== "refused") throw new Error("expected a refusal");
    return result.refusal;
  };

  it("refuses a request that is not a move request", () => {
    expect(refusal({ ...base.input, request: { ...base.input.request, forceSwitch: [true, false] } }).code).toBe("not_move_request");
    expect(refusal({ ...base.input, request: { side: base.input.request.side, teamPreview: true } }).code).toBe("not_move_request");
  });

  it("refuses while the reader has flagged a line", () => {
    expect(refusal({ ...base.input, flags: [{ index: 1, line: "|x|", reason: "r" }] }).code).toBe("reader_flagged");
  });

  it("refuses an own team that does not match the request", () => {
    expect(refusal({ ...base.input, team: base.input.team.slice(1).concat(structuredClone(base.input.team[1] as PokemonSet)) }).code).toBe("own_team_mismatch");
    const stats = structuredClone(base.input.request);
    (stats.side.pokemon[0] as { stats: { atk: number } }).stats.atk += 1;
    expect(refusal({ ...base.input, request: stats })).toMatchObject({ code: "own_team_mismatch" });
  });

  it("refuses an opponent team that does not fit the reader's state", () => {
    const members = base.input.opponent.members;
    expect(refusal({ ...base.input, opponent: { members: members.slice(1) } }).code).toBe("opponent_team_mismatch");
    expect(refusal({ ...base.input, opponent: { members: [{ ...(members[0] as WorldOpponentMember), roster_index: 99 }, ...members.slice(1)] } }).code).toBe("opponent_team_mismatch");
    expect(refusal({ ...base.input, opponent: { members: [members[0] as WorldOpponentMember, members[0] as WorldOpponentMember, ...members.slice(2)] } }).code).toBe("opponent_team_mismatch");
    const shown = withState(base.input, (state) => {
      const unseen = state.opponent.roster.find((m) => m.brought === null);
      if (unseen !== undefined) unseen.brought = false;
      const inMembers = new Set(members.map((m) => m.roster_index));
      state.opponent.roster.forEach((m, index) => {
        if (inMembers.has(index) && m.brought === null) m.brought = false;
      });
    });
    if (shown.state.opponent.roster.some((m, index) => m.brought === false && members.some((x) => x.roster_index === index))) expect(refusal(shown).code).toBe("opponent_team_mismatch");
    const slot = withState(base.input, (state) => {
      for (const m of state.opponent.roster) if (m.active_slot !== null) m.active_slot = "c";
    });
    expect(refusal(slot).code).toBe("opponent_team_mismatch");
  });

  it("refuses a set the validator refuses, or that is not canonical (V1)", () => {
    const members = structuredClone([...base.input.opponent.members]);
    (members[0] as { set: PokemonSet }).set = { ...(members[0] as WorldOpponentMember).set, moves: ["Splash", "Splash", "Splash", "Splash", "Splash"] };
    expect(refusal({ ...base.input, opponent: { members } }).code).toBe("v1_set_invalid");
    const loose = structuredClone([...base.input.opponent.members]);
    (loose[0] as { set: PokemonSet }).set = { ...(loose[0] as WorldOpponentMember).set, ivs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } };
    expect(["v1_not_canonical", "v1_set_invalid"]).toContain(refusal({ ...base.input, opponent: { members: loose } }).code);
  });

  it("refuses a set that drops a revealed fact (V1)", () => {
    const revealedIndex = base.input.state.opponent.roster.findIndex((m) => m.moves_revealed.length > 0 && m.brought === true);
    const input = withState(base.input, (state) => {
      const mon = state.opponent.roster[revealedIndex === -1 ? 0 : revealedIndex];
      if (mon !== undefined && mon.moves_revealed.length === 0) mon.moves_revealed = ["splash"];
      if (mon !== undefined) mon.moves_revealed = [...mon.moves_revealed, "splash"];
    });
    expect(refusal(input).code).toBe("v1_revealed_fact");
    expect(refusal(withState(base.input, (state) => { (state.opponent.roster[base.input.opponent.members[0]?.roster_index as number] as { level: number }).level = 7; })).code).toBe("v1_revealed_fact");
    expect(refusal(withState(base.input, (state) => { (state.opponent.roster[base.input.opponent.members[0]?.roster_index as number] as { ability: string }).ability = "noability"; })).code).toBe("v1_revealed_fact");
  });

  it("refuses two opponent sets holding one item (Item Clause)", () => {
    const members = structuredClone([...base.input.opponent.members]);
    const item = (members[0] as WorldOpponentMember).set.item;
    const second = members[1] as { set: PokemonSet };
    second.set = { ...second.set, item: item === "" ? "Leftovers" : item };
    (members[0] as { set: PokemonSet }).set = { ...(members[0] as WorldOpponentMember).set, item: second.set.item };
    const result = createWorldBuilder(context).build({ ...base.input, opponent: { members } });
    expect(result.kind).toBe("refused");
    if (result.kind === "refused") expect(["v1_item_clause", "v1_set_invalid", "v1_revealed_fact"]).toContain(result.refusal.code);
  });

  const activeFoe = () => base.input.state.opponent.roster.findIndex((m) => m.active_slot === "a");

  it("refuses state the record classifies R, with its item and class", () => {
    const foe = activeFoe();
    const item = (change: (state: PublicState) => void) => {
      const r = refusal(withState(base.input, change));
      expect(r.code).toBe("state_unwritten");
      expect(r.state_class).toBe("R");
      expect(r.redraw).toBe(false);
      return r.item;
    };
    expect(item((s) => { (s.opponent.roster[foe] as { volatiles: string[] }).volatiles = ["taunt"]; })).toBe("unadmitted_id");
    expect(item((s) => { (s.opponent.roster[foe] as { volatiles: string[] }).volatiles = ["substitute"]; })).toBe("substitute");
    expect(item((s) => { (s.opponent.roster[foe] as { volatiles: string[] }).volatiles = ["futuresight"]; })).toBe("slot_conditions");
    expect(item((s) => { (s.opponent.roster[foe] as { volatiles: string[] }).volatiles = ["transform"]; })).toBe("illusion_transform");
    expect(item((s) => { (s.opponent.roster[foe] as { item_received: boolean }).item_received = true; })).toBe("item_replaced");
    expect(item((s) => { (s.opponent.roster[foe] as { ability_suppressed: boolean }).ability_suppressed = true; })).toBe("ability_changed");
    const members = new Set(base.input.opponent.members.map((m) => m.roster_index));
    const outside = base.input.state.opponent.roster.findIndex((_, index) => !members.has(index));
    expect(item((s) => { (s.opponent.roster[outside] as { phantom: boolean }).phantom = true; })).toBe("illusion_transform");
    const bench = base.input.opponent.members.find((m) => base.input.state.opponent.roster[m.roster_index]?.active_slot === null && base.input.state.opponent.roster[m.roster_index]?.fainted === false);
    expect(item((s) => { (s.opponent.roster[bench?.roster_index as number] as { volatiles: string[] }).volatiles = ["confusion"]; })).toBe("unadmitted_id");
    const locked = structuredClone(base.input.request);
    (locked.active?.[0] as NonNullable<EngineRequest["active"]>[number]).moves = (locked.active?.[0] as NonNullable<EngineRequest["active"]>[number]).moves.slice(0, 1);
    expect(refusal({ ...base.input, request: locked })).toMatchObject({ code: "state_unwritten", item: "locked_moves", state_class: "R" });
    const swapped = structuredClone(base.input.request);
    (swapped.side.pokemon[0] as { item: string }).item = "leftovers";
    expect(refusal({ ...base.input, request: swapped })).toMatchObject({ code: "state_unwritten", item: "item_replaced" });
    const ability = structuredClone(base.input.request);
    (ability.side.pokemon[0] as { ability: string }).ability = "wonderguard";
    expect(refusal({ ...base.input, request: ability })).toMatchObject({ code: "state_unwritten", item: "ability_changed" });
    const volatile = withState(base.input, (s) => { (s.own.roster.find((m) => m.name !== null && m.active_slot !== null) as { volatiles: string[] }).volatiles = ["substitute"]; });
    expect(refusal(volatile)).toMatchObject({ code: "state_unwritten", item: "substitute" });
  });

  it("refuses a Pokemon that can hide as another, and a turn-parity ability, in play", () => {
    const hide = structuredClone(base.input.request);
    (hide.side.pokemon[0] as { baseAbility: string }).baseAbility = "illusion";
    expect(refusal({ ...base.input, request: hide })).toMatchObject({ code: "state_unwritten", item: "illusion_transform", redraw: false });
    const truant = structuredClone(base.input.request);
    (truant.side.pokemon[1] as { baseAbility: string }).baseAbility = "truant";
    expect(refusal({ ...base.input, request: truant })).toMatchObject({ code: "state_unwritten", item: "truant_slow_start" });
    // The opponent's sampled ability is judged by the record's ids: name the active foe's ability in the rows.
    const foeSet = base.input.opponent.members.find((m) => base.input.state.opponent.roster[m.roster_index]?.active_slot === "a") as WorldOpponentMember;
    const ability = toId(foeSet.set.ability);
    const named = (item: "illusion_transform" | "truant_slow_start") => {
      const writers = structuredClone(context.writers);
      for (const row of writers.items as { id: string; ids?: string[] }[]) if (row.ids !== undefined && row.id !== "unavailable_mechanics") row.ids = row.ids.filter((id) => id !== ability);
      const target = writers.items.find((row) => row.id === item) as unknown as { ids: string[] };
      target.ids = [...target.ids, ability].sort();
      return createWorldBuilder({ ...context, writers });
    };
    expect(refusal(base.input, named("illusion_transform"))).toMatchObject({ code: "state_unwritten", item: "illusion_transform", redraw: true });
    expect(refusal(base.input, named("truant_slow_start"))).toMatchObject({ code: "state_unwritten", item: "truant_slow_start", redraw: true });
  });

  it("refuses a last move whose effect outlasts the turn, and mechanics the format does not have", () => {
    const wish = withWorld(base.input, (_s, facts) => {
      (facts.opponent.active.a as { last_move: string }).last_move = "wish";
    });
    expect(refusal(wish)).toMatchObject({ code: "state_unwritten", item: "slot_conditions" });
    const rampage = withWorld(base.input, (_s, facts) => {
      (facts.own.active.a as { last_move: string }).last_move = "outrage";
    });
    expect(refusal(rampage)).toMatchObject({ code: "state_unwritten", item: "locked_moves" });
    expect(refusal({ ...base.input, flags: [{ index: 1, line: "|-terastallize|p1a: X|Fire", reason: "r" }] })).toMatchObject({ code: "state_unwritten", item: "unavailable_mechanics" });
    expect(refusal({ ...base.input, flags: [{ index: 1, line: "|-terastallize|p1a: X|Fire", reason: "r" }, { index: 2, line: "|x|", reason: "r" }] }).code).toBe("reader_flagged");
  });

  it("refuses when the world facts are missing, for another turn, or name another Pokemon", () => {
    expect(refusal({ ...base.input, facts: null }).code).toBe("facts_unavailable");
    expect(refusal(withWorld(base.input, (_s, facts) => { facts.turn += 1; })).code).toBe("facts_mismatch");
    expect(refusal(withWorld(base.input, (_s, facts) => { (facts.own.active.a as { name: string }).name = "Nobody"; })).code).toBe("facts_mismatch");
    expect(refusal(withWorld(base.input, (_s, facts) => { facts.opponent.active.a = null; })).code).toBe("facts_mismatch");
  });

  it("refuses a field or a status the engine will not take", () => {
    const foe = activeFoe();
    const dated = (s: PublicState, facts: WorldFacts, id: string) => { facts.weather = { id, started_turn: facts.turn, after_residual: false, source: null }; facts.terrain = { id, started_turn: facts.turn, after_residual: false, source: null }; void s; };
    expect(refusal(withWorld(base.input, (s, f) => { s.weather = "notaweather"; dated(s, f, "notaweather"); })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s, f) => { s.terrain = "raindance"; dated(s, f, "raindance"); })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s) => { s.weather = "raindance"; })).code).toBe("facts_mismatch");
    expect(refusal(withWorld(base.input, (s) => { s.terrain = "electricterrain"; })).code).toBe("facts_mismatch");
    expect(refusal(withState(base.input, (s) => { s.own.conditions = { spikes: 0 }; })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s, f) => { s.own.conditions = { reflect: 2 }; f.own.conditions = { reflect: { started_turn: f.turn, after_residual: false, source: null, layers: 2 } }; })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s, f) => { s.own.conditions = { notacondition: 1 }; f.own.conditions = { notacondition: { started_turn: f.turn, after_residual: false, source: null, layers: 1 } }; })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s, f) => { s.field_conditions = ["notacondition"]; f.field_conditions = { notacondition: { started_turn: f.turn, after_residual: false, source: null } }; })).code).toBe("engine_refused");
    expect(refusal(withWorld(base.input, (s) => { s.own.conditions = { reflect: 1 }; })).code).toBe("facts_mismatch");
    expect(refusal(withState(base.input, (s) => { (s.opponent.roster[foe] as { status: string }).status = "notastatus"; })).code).toBe("engine_refused");
    const bad = structuredClone(base.input.request);
    (bad.side.pokemon[0] as { condition: string }).condition = "1/2 notastatus";
    expect(refusal({ ...base.input, request: bad }).code).toBe("engine_refused");
  });

  it("refuses a Mega the set cannot explain (V1) and a Mega Evolution the engine refuses", () => {
    const foe = activeFoe();
    const mega = withState(base.input, (s) => {
      const mon = s.opponent.roster[foe] as { mega: boolean; mega_stone: string };
      mon.mega = true;
      mon.mega_stone = "leftovers";
    });
    expect(refusal(mega).code).toBe("v1_revealed_fact");
    // The stone is the set's but the species has no Mega forme: the engine refuses to evolve it.
    const members = structuredClone([...base.input.opponent.members]);
    const index = members.findIndex((m) => m.roster_index === foe);
    (members[index] as { set: PokemonSet }).set = { ...(members[index] as WorldOpponentMember).set, item: "Leftovers" };
    const noStone = withState({ ...base.input, opponent: { members } }, (s) => {
      const mon = s.opponent.roster[foe] as { mega: boolean; mega_stone: string };
      mon.mega = true;
      mon.mega_stone = "leftovers";
    });
    expect(refusal(noStone).code).toBe("engine_refused");
  });

  it("refuses a world whose engine rendering is not the reader's state (V3)", () => {
    const foe = activeFoe();
    expect(refusal(withState(base.input, (s) => { (s.opponent.roster[foe] as { hp_den: number }).hp_den = 48; })).code).toBe("v3_mismatch");
    // A rounding that is not the inverse of the Champions display (floor of 100 * hp / max) is caught by V3, not hidden.
    const floor = createWorldBuilder({ ...context, hp_rounding: "floor" });
    const codes = new Set<string>();
    for (let percent = 1; percent < 100; percent += 1) {
      const result = floor.build(withState(base.input, (s) => { (s.opponent.roster[foe] as { hp_num: number }).hp_num = percent; }));
      codes.add(result.kind === "world" ? "world" : result.refusal.code);
    }
    expect([...codes].sort()).toEqual(["v3_mismatch", "world"]);
    // And `ceil` round-trips every shown percentage.
    const ceil = createWorldBuilder(context);
    for (let percent = 1; percent <= 100; percent += 1) {
      const result = ceil.build(withState(base.input, (s) => { (s.opponent.roster[foe] as { hp_num: number }).hp_num = percent; }));
      expect(result.kind, `percent ${percent}`).toBe("world");
    }
  });

  it("refuses another seat (V5)", () => {
    expect(refusal(withState(base.input, (s) => { s.player = "p3"; })).code).toBe("v5_mismatch");
    expect(refusal({ ...base.input, request: { ...base.input.request, side: { ...base.input.request.side, id: base.input.request.side.id === "p1" ? "p2" : "p1" } } }).code).toBe("v5_mismatch");
  });

  it("refuses with start_failed when the scenario battle cannot start", () => {
    const bad: WorldContext = { ...context, scenario_id: "gen9championsvgc2026regmc@@@!Not A Rule" };
    expect(refusal(base.input, createWorldBuilder(bad)).code).toBe("start_failed");
  });

  it("counts every refusal by code and by unwritten kind, with every code present and a copy returned", () => {
    const builder = createWorldBuilder(context);
    expect(builder.counts().refusals).toEqual(Object.fromEntries(WORLD_REFUSAL_CODES.map((code) => [code, 0])));
    expect(builder.counts().unwritten).toEqual(Object.fromEntries(STATE_ITEM_IDS.map((id) => [id, 0])));
    builds(builder, base.input);
    builds(builder, { ...base.input, flags: [{ index: 1, line: "|x|", reason: "r" }] });
    const counts = builder.counts();
    expect(counts).toMatchObject({ requests: 2, built: 1, refused: 1 });
    expect(counts.refusals.reader_flagged).toBe(1);
    counts.built = 99;
    expect(builder.counts().built).toBe(1);
  });
});

describe("the privacy boundary of the world builder (ADR-026, ADR-008 section 5)", () => {
  it("takes no live battle, BattleRecord, omniscient log, hidden verdicts, channel or seed: each is a compile error", () => {
    const neverRun = (battle: SimBattle, record: BattleRecord, log: OmniscientLog, verdicts: HiddenVerdicts[], channel: PlayerChannel): void => {
      const builder = createWorldBuilder(context);
      // @ts-expect-error: a live battle is not a world input
      void builder.build(battle);
      // @ts-expect-error: nor is a BattleRecord
      void builder.build(record);
      // @ts-expect-error: nor the omniscient log
      void builder.build(log);
      // @ts-expect-error: nor hidden verdicts
      void builder.build(verdicts);
      // @ts-expect-error: nor a player's channel
      void builder.build(channel);
      // @ts-expect-error: nor a seed
      void builder.build("sodium,00000000000000000000000000000000");
      // @ts-expect-error: the input has no field for a live battle
      void builder.build({ ...base.input, battle });
      // @ts-expect-error: nor for the omniscient log
      void builder.build({ ...base.input, omniscient_log: log });
    };
    expect(neverRun).toBeTypeOf("function");
    expectTypeOf<keyof WorldInput>().toEqualTypeOf<"request" | "team" | "state" | "flags" | "facts" | "opponent">();
  });
});
