/* eslint-disable @typescript-eslint/no-explicit-any */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { type HistoryFacts, type HistoryFactsData, HistoryFactsReader, type PokemonHistory } from "@areneon/public-state";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { goldenHistoryFactsData } from "../src/history-facts-data.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { type CombatantFacts, type ResolvedHistory } from "../src/mechanics-history.ts";
import { declaredOutcomeScenario } from "../src/mechanics-outcome.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { loadRandomCorpus, pendingFirstActor, playRandomBattle, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import type { Actor } from "../src/runner.ts";
import { statePlan } from "../src/world-state.ts";
import { loadWorldWriters } from "../src/world-writers.ts";

// ADR-022 erratum E-22.5.3, slice H2, required test: the history the probe declares equals the live engine's own fields. Every committed battle
// is replayed on the pinned engine with a test-only spy on the live battle (as `history-facts-parity.test.ts` does); at each request of both seats the
// history facts of the seat's own channel are declared in a scenario battle built from the live Pokemon's sets, and the scenario's engine fields
// are compared with the live battle's. Volatiles are not declared here (they carry random lengths); the hand-worked tests declare them.

let engine: Engine;
let data: HistoryFactsData;
let scenarioId: string;
let stall: string;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  data = goldenHistoryFactsData();
  scenarioId = checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT));
  stall = statePlan(loadWorldWriters(engine, GOLDEN_FORMAT)).conditionOf("protect_counter");
});

const MAX = Number(process.env.ARENEON_HISTORY_LIVE_MAX ?? "100000");
const exercised = new Map<string, number>();
const hit = (what: string, when = true): void => {
  if (when) exercised.set(what, (exercised.get(what) ?? 0) + 1);
};
let requestsDeclared = 0;
let requestsSkipped = 0;

const ZERO_BOOSTS = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const NO_SIDE_CONDITIONS = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
const FIELD = { weather: null, terrain: null, trick_room: false, sides: [NO_SIDE_CONDITIONS, NO_SIDE_CONDITIONS] } as never;

const slotOf = (viewer: number, entry: string): string => `p${(entry.startsWith("own") ? viewer : 1 - viewer) + 1}${entry.split(":")[1]}`;

/** The scenario's history of the live battle's Pokemon in play, read from the facts of one seat. Null when a side's positions do not line up with the scenario's (a side with a gap). */
function historyOf(facts: HistoryFacts, live: any, seat: number): { sets: any[][]; history: ResolvedHistory; liveMons: any[][]; undeclared: Set<string> } | null {
  const sets: any[][] = [[], []];
  const liveMons: any[][] = [[], []];
  const combatants: ResolvedHistory["combatants"] = [];
  let place = 0;
  for (const sideIndex of [0, 1]) {
    const label = sideIndex === seat ? "own" : "opponent";
    const side = facts[label];
    const active = live.sides[sideIndex].active as any[];
    for (const [index, mon] of active.entries()) {
      if (mon === null || mon === undefined || mon.fainted) {
        // a gap before a Pokemon in play would shift the scenario's positions
        if (active.slice(index + 1).some((other) => other && !other.fainted)) return null;
        continue;
      }
      const letter = String.fromCharCode(97 + index);
      const entry = side.active[letter] as PokemonHistory | null | undefined;
      if (entry === null || entry === undefined || entry.unknown) return null;
      const roster = side.roster[mon.name];
      const f: CombatantFacts = {
        move_actions: entry.move_actions,
        stall_counter: entry.stall_counter ?? 0,
        times_attacked: entry.times_attacked,
        hurt_this_turn: entry.hurt_this_turn,
        stats_raised_this_turn: entry.stats_raised_this_turn,
        stats_lowered_this_turn: entry.stats_lowered_this_turn,
      };
      if (entry.attackers_this_turn !== null) {
        f.attacked_by = entry.attackers_this_turn.map((position) => ({ side: Number(slotOf(seat, position)[1]) - 1, position: slotOf(seat, position).charCodeAt(2) - 97, damage: 1 }));
      }
      if (entry.last_move_failed !== null) f.last_move_failed = entry.last_move_failed;
      if (roster !== undefined && roster.ate_berry !== null) f.ate_berry = roster.ate_berry;
      if (roster !== undefined && roster.last_item !== "unsettled") f.last_item = roster.last_item;
      if (entry.moves_used !== null) f.moves_used = entry.moves_used.filter((id) => mon.moveSlots.some((slot: any) => slot.id === id));
      sets[sideIndex]?.push(structuredClone(mon.set));
      liveMons[sideIndex]?.push(mon);
      combatants.push({ side: sideIndex, position: (sets[sideIndex] as any[]).length - 1, place: place++, facts: f });
    }
  }
  if (sets[0]?.length === 0 || sets[1]?.length === 0) return null;
  // an attacker must be a declared Pokemon of the scenario
  // an attacker that has left play is no declared Pokemon of the scenario: its entry is not declared and not compared
  const undeclared = new Set<string>();
  for (const c of combatants) {
    if ((c.facts.attacked_by ?? []).some((a) => sets[a.side]?.[a.position] === undefined)) {
      delete c.facts.attacked_by;
      undeclared.add(`${c.side}/${c.position}`);
    }
  }
  const slots = (label: "own" | "opponent") =>
    Object.entries(facts[label].pending_slot).flatMap(([letter, byId]) => Object.entries(byId).map(([id, began]) => ({ position: letter.charCodeAt(0) - 97, id, started_turn: began.started_turn, after_residual: began.after_residual })));
  const sideSlots = (label: "own" | "opponent", sideIndex: number) => slots(label).filter((slot) => slot.position < (sets[sideIndex] as any[]).length);
  const history: ResolvedHistory = {
    turn: facts.turn,
    field_last_move: facts.field_last_move,
    sides: [0, 1].map((sideIndex) => {
      const label = sideIndex === seat ? "own" : "opponent";
      return { total_fainted: facts[label].total_fainted, pending_slots: sideSlots(label, sideIndex) };
    }),
    combatants,
  };
  return { sets, history, liveMons, undeclared };
}

const engineAttackers = (p: any): string[] => p.attackedBy.filter((a: any) => a.thisTurn && typeof a.damageValue === "number").map((a: any) => String(a.slot)).sort();

/** The scenario's fields against the live battle's; returns the differences. */
function compareDeclared(scenario: any, live: any, declared: NonNullable<ReturnType<typeof historyOf>>, where: string): string[] {
  const diffs: string[] = [];
  const check = (what: string, got: unknown, want: unknown): void => {
    hit(`compared ${what.split(" ").at(-1)}`);
    if (JSON.stringify(got) !== JSON.stringify(want)) diffs.push(`${where}: ${what}: declared ${JSON.stringify(got)} live ${JSON.stringify(want)}`);
  };
  check("field_last_move", scenario.lastMove?.id ?? null, live.lastMove?.id ?? null);
  for (const sideIndex of [0, 1]) {
    check(`side ${sideIndex} totalFainted`, scenario.sides[sideIndex].totalFainted, live.sides[sideIndex].totalFainted);
    declared.liveMons[sideIndex]?.forEach((l: any, position: number) => {
      const s = scenario.sides[sideIndex].pokemon[position];
      const at = `side ${sideIndex} position ${position} ${l.name}`;
      check(`${at} activeMoveActions`, s.activeMoveActions, l.activeMoveActions);
      check(`${at} stall`, s.volatiles[stall]?.counter ?? 0, l.volatiles[stall]?.counter ?? 0);
      check(`${at} timesAttacked`, s.timesAttacked, l.timesAttacked);
      check(`${at} hurtThisTurn`, s.hurtThisTurn !== null, l.hurtThisTurn !== null && l.hurtThisTurn !== undefined);
      check(`${at} statsRaisedThisTurn`, !!s.statsRaisedThisTurn, !!l.statsRaisedThisTurn);
      check(`${at} statsLoweredThisTurn`, !!s.statsLoweredThisTurn, !!l.statsLoweredThisTurn);
      if (!declared.undeclared.has(`${sideIndex}/${position}`)) check(`${at} attackedBy`, engineAttackers(s), engineAttackers(l));
      const combatant = declared.history.combatants.find((c) => c.side === sideIndex && c.position === position);
      if (combatant?.facts.last_move_failed !== undefined) check(`${at} moveLastTurnResult`, s.moveLastTurnResult === false, l.moveLastTurnResult === false);
      if (combatant?.facts.ate_berry !== undefined) check(`${at} ateBerry`, !!s.ateBerry, !!l.ateBerry);
      if (combatant?.facts.last_item !== undefined) check(`${at} lastItem`, s.lastItem ?? "", l.lastItem ?? "");
      if (combatant?.facts.moves_used !== undefined) check(`${at} moveSlots.used`, s.moveSlots.filter((m: any) => m.used).map((m: any) => m.id).sort(), l.moveSlots.filter((m: any) => m.used && combatant.facts.moves_used?.includes(m.id)).map((m: any) => m.id).sort());
    });
    for (const slot of declared.history.sides[sideIndex]?.pending_slots ?? []) check(`side ${sideIndex} slot ${slot.id}`, !!scenario.sides[sideIndex].slotConditions[slot.position]?.[slot.id], !!live.sides[sideIndex].slotConditions[slot.position]?.[slot.id]);
  }
  return diffs;
}

interface Spec extends RandomBattleFile {
  actors?: Record<"p1" | "p2", "random-legal" | "pending-first">;
}

/** Replays one battle and, at every request of both seats, declares the seat's history in a scenario and compares it with the live battle. */
async function replay(spec: Spec): Promise<string[]> {
  let captured: any = null;
  const Base = engine.sim.BattleStream as unknown as new () => { battle: unknown };
  const capture = (stream: unknown): void => {
    captured = stream;
  };
  class Spy extends Base {
    constructor() {
      super();
      capture(this);
    }
  }
  const spy: Engine = { ...engine, sim: { ...engine.sim, BattleStream: Spy as never } };
  const readers = { p1: new HistoryFactsReader("p1", data), p2: new HistoryFactsReader("p2", data) };
  const fed = { p1: 0, p2: 0 };
  const diffs: string[] = [];
  const actorFor = (seed: number): Actor => {
    const player = spec.actor_seeds.p1 === seed ? "p1" : "p2";
    const kind = spec.actors?.[player] ?? "random-legal";
    const inner = kind === "pending-first" ? pendingFirstActor(seed) : randomLegalActor(seed);
    return (request, ctx) => {
      if (ctx !== undefined && captured?.battle != null) {
        const lines = ctx.channel as readonly string[];
        for (; fed[player] < lines.length - 1; fed[player]++) readers[player].feed(lines[fed[player]] as string);
        try {
          const facts = readers[player].facts();
          const declared = historyOf(facts, captured.battle, player === "p1" ? 0 : 1);
          if (declared === null) requestsSkipped += 1;
          else {
            const sides = declared.sets.map((side) => side.map(() => ({ set: "", boosts: ZERO_BOOSTS, status: null, hp: null, mega: false })));
            const scenario = declaredOutcomeScenario(engine, scenarioId, { sides: sides as never, actions: [] as never, state: FIELD }, declared.sets.flat(), { resolved: declared.history, stall });
            requestsDeclared += 1;
            diffs.push(...compareDeclared(scenario, captured.battle, declared, `${spec.id}/${player}/turn ${facts.turn}`));
          }
        } catch (error) {
          if ((error as { code?: string }).code === undefined) throw error; // a refusal of the facts module is skipped, anything else is a defect
          requestsSkipped += 1;
        }
      }
      return inner(request, ctx);
    };
  };
  await playRandomBattle(spy, spec, actorFor);
  return diffs;
}

const single = (dir: string): Spec => JSON.parse(readFileSync(join(GOLDEN_DIR, dir, "battle.json"), "utf8")) as Spec;

describe("the history the probe declares equals the live engine's fields", () => {
  it("at every request of the random-play battles", async () => {
    const all: string[] = [];
    for (const spec of (loadRandomCorpus() as Spec[]).slice(0, MAX)) all.push(...(await replay(spec)));
    if (all.length > 0) console.log(`${all.length} differences, first:\n${all.slice(0, 12).join("\n")}`);
    expect(all).toEqual([]);
  }, 1_200_000);

  it("at every request of the Baton Pass, Fling and revival battles", async () => {
    const all: string[] = [];
    for (const dir of ["baton-pass", "baton-pass-volatiles", "fling", "revival"].filter((d) => existsSync(join(GOLDEN_DIR, d, "battle.json")))) all.push(...(await replay(single(dir))));
    expect(all).toEqual([]);
  }, 600_000);

  it("declared many facts, and exercised the states the facts exist for", () => {
    console.log(`history live: ${requestsDeclared} requests declared, ${requestsSkipped} skipped; compared ${JSON.stringify([...exercised])}`);
    expect(requestsDeclared).toBeGreaterThan(100);
    for (const what of ["compared activeMoveActions", "compared timesAttacked", "compared hurtThisTurn", "compared totalFainted", "compared field_last_move"]) expect(exercised.get(what) ?? 0, what).toBeGreaterThan(0);
  });
});
