/* eslint-disable @typescript-eslint/no-explicit-any */
// The test-only fork oracle (ADR-026 Evidence 1, B2 and N-A, slice F). This module is the ONLY place the fork of a
// live battle and the copy of its random-number state exist: it lives under test/, and `fidelity-isolation.test.ts`
// scans every shipped source for its identifiers. A player at play time has neither the live battle nor its seed.
//
// A decision is captured while `runBattle` plays a committed seeded A1 battle (a spy on the engine's BattleStream
// reads the live battle at each own move request, as `world-writers-live.test.ts` does). The fork is
// `Battle.fromJSON(toJSON())` of the live battle, as `CloneJudge` makes it. The oracle world is built by the
// production world builder from the player's channel, the own team and the opponent's TRUE brought sets; just
// before the choices its random state is replaced by a copy of the fork's. F1 plays the recorded joint choices on
// both, F2 plays top-k own candidates against a fixed sample of the opponent's legal replies, and the player-view
// lines of the turn, the next own request and the random state after the turn are compared.

import { enumerateCandidates } from "../../src/choices.ts";
import { type Engine, type PokemonSet } from "../../src/engine.ts";
import { randomLegalActor } from "../../src/random-legal.ts";
import { decodeRequest } from "../../src/request.ts";
import { type Actor, type BattleRecord, runBattle } from "../../src/runner.ts";
import { pendingFirstActor, RANDOM_MAX_TURNS } from "../../src/random-corpus.ts";
import { GOLDEN_FORMAT } from "../../src/golden.ts";
import { type WorldContext, type WorldInput, createWorldBuilder } from "../../src/world-builder.ts";
import { firstDifference, plainRequest } from "../../src/world-checks.ts";
import type { StateItemId } from "../../src/world-writers.ts";
import { inputAtLastRequest } from "../world-builder-support.ts";

type Player = "p1" | "p2";
const other = (p: Player): Player => (p === "p1" ? "p2" : "p1");

/** What a battle needs to be replayed with a spy. `actors` defaults to random-legal for both seats. */
export interface OracleSpec {
  id: string;
  seed: string;
  actor_seeds: Record<Player, number>;
  teams: Record<Player, PokemonSet[]>;
  actors?: Record<Player, "random-legal" | "pending-first">;
}

/** One own move decision of a replayed battle: the serialised live battle (the fork) and the world input. */
export interface Decision {
  battle_id: string;
  player: Player;
  turn: number;
  request_seq: number;
  input: WorldInput;
  /** `toJSON()` of the live battle at the request, before any choice of this turn was applied. */
  fork: object;
  /** The own and the opponent's accepted choice of this turn (null when the battle recorded none). */
  recorded: { own: string; foe: string } | null;
}

export type Kind = "outcome" | "draw_order" | "request" | "choice";

/** The result of playing one joint choice on the fork and on the world. */
export interface Comparison {
  equal: boolean;
  /** Why not: the first kind that differs (choice acceptance, lines, random state after the turn, next request). */
  kind: Kind | null;
  /** Fixed words naming where the first difference is; never a line of the battle. */
  where: string | null;
  /** The state differences between fork and world before the turn (tags), computed only for a differing turn. */
  delta: string[];
  /** Lines the fork's turn produced that name an action of a Pokemon (`|move|`, `|cant|`), by player-view side and nickname. */
  actors: string[];
  /** The joint choice played (own, then the opponent's as the fork takes it): for a human reading a difference. */
  choices: [string, string];
}

/** The player-view of a slice of a battle log: the secret half of a split for the seat's own lines, else the public half. */
export function playerView(lines: readonly string[], seat: Player): string[] {
  const out: string[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] as string;
    if (line.startsWith("|split|")) {
      const side = line.slice("|split|".length);
      out.push((side === seat ? lines[i + 1] : lines[i + 2]) as string);
      i += 2;
    } else if (!line.startsWith("|t:|")) out.push(line);
  }
  return out;
}

/** Rebuild a battle from its serialised form. */
export const revive = (engine: Engine, state: object): any => engine.sim.Battle.fromJSON(JSON.parse(JSON.stringify(state)));

/** The random-state copy: the world draws exactly what the fork will. Exists only in tests. */
export function copyRandomState(world: any, fork: any): void {
  world.prng = fork.prng.clone();
}

/** Translate the opponent's `switch N` (a position in the fork's side order) into the world's order, by nickname. */
export function translateChoice(choice: string, from: any, to: any): string {
  return choice
    .split(",")
    .map((part) => {
      const m = /^(\s*)switch (\d+)(\s*)$/.exec(part);
      if (m === null) return part;
      const name = from.pokemon[Number(m[2]) - 1]?.name as string | undefined;
      const at = to.pokemon.findIndex((mon: any) => mon.name === name);
      return `${m[1]}switch ${at + 1}${m[3]}`;
    })
    .join(",");
}

interface TurnResult {
  accepted: boolean;
  lines: string[];
  request: unknown;
  seed: string;
  ended: boolean;
}

function playTurn(battle: any, seat: Player, own: string, foe: string): TurnResult {
  const before = battle.log.length;
  for (const side of battle.sides) side.clearChoice();
  const a = battle.choose(seat, own);
  const b = a ? battle.choose(other(seat), foe) : false;
  const side = battle.sides.find((s: any) => s.id === seat);
  return {
    accepted: a && b,
    lines: playerView(battle.log.slice(before), seat),
    request: side.activeRequest === null || side.activeRequest === undefined ? null : plainRequest(side.activeRequest),
    seed: battle.prng.getSeed(),
    ended: battle.ended,
  };
}

/** Play one joint choice on a fresh fork and a fresh world (both revived from their serialised state) and compare. */
export function compare(engine: Engine, d: Decision, worldState: object, own: string, foe: string): Comparison {
  const fork = revive(engine, d.fork);
  const world = revive(engine, worldState);
  const foeSide = fork.sides.find((s: any) => s.id === other(d.player));
  const worldFoe = world.sides.find((s: any) => s.id === other(d.player));
  // Translate before the fork plays: a switch reorders its side.
  const worldFoeChoice = translateChoice(foe, foeSide, worldFoe);
  const f = playTurn(fork, d.player, own, foe);
  const w = playTurn(world, d.player, own, worldFoeChoice);
  const actors = f.lines.filter((l) => /^\|(move|cant)\|/.test(l)).map((l) => l.split("|").slice(2, 3)[0] as string);
  const make = (kind: Kind | null, where: string | null): Comparison => ({ equal: kind === null, kind, where, delta: kind === null ? [] : stateDelta(revive(engine, d.fork), revive(engine, worldState), d.player), actors, choices: [own, foe] });
  if (f.accepted !== w.accepted) return make("choice", f.accepted ? "the world refuses a choice the real battle takes" : "the world takes a choice the real battle refuses");
  if (!f.accepted) return make(null, null);
  const at = f.lines.findIndex((line, i) => line !== w.lines[i]);
  if (f.lines.length !== w.lines.length || at >= 0) return make("outcome", `player-view lines differ at line ${at < 0 ? Math.min(f.lines.length, w.lines.length) : at} of the turn`);
  if (f.seed !== w.seed) return make("draw_order", "the random state after the turn differs");
  if (!f.ended || !w.ended) {
    const path = firstDifference(f.request as never, w.request as never);
    if (path !== null) return make("request", `the next own request differs at ${path.replace(/\/\d+/g, "/#")}`);
  }
  return make(null, null);
}

/** Candidates of a request as the engine offers them (non-pending only: no pre-submission verdict is claimed for a pending one). */
export function candidatesOf(request: unknown): string[] {
  return enumerateCandidates(decodeRequest(JSON.stringify(request))).candidates.filter((c) => !c.pending).map((c) => c.choice);
}

/** `count` entries spread evenly over the list in order (all of it when shorter): the fixed sample of F2. */
export function stride<T>(list: readonly T[], count: number): T[] {
  if (list.length <= count) return [...list];
  return Array.from({ length: count }, (_, i) => list[Math.floor((i * list.length) / count)] as T);
}

export { createWorldBuilder };
export type { WorldContext };

const ACTORS = { "random-legal": randomLegalActor, "pending-first": pendingFirstActor } as const;

/**
 * Replay a committed battle with a spy and return its own move decisions (both seats) with the fork state and the
 * recorded joint choices. The battle is played by the production `runBattle`.
 */
export async function captureDecisions(engine: Engine, spec: OracleSpec): Promise<{ decisions: Decision[]; record: BattleRecord }> {
  let captured: any = null;
  const capture = (stream: unknown): void => {
    captured = stream;
  };
  const Base = engine.sim.BattleStream as unknown as new () => { battle: unknown };
  class Spy extends Base {
    constructor() {
      super();
      capture(this);
    }
  }
  const spy: Engine = { ...engine, sim: { ...engine.sim, BattleStream: Spy as never } };
  const raw: { player: Player; turn: number; request_seq: number; input: WorldInput; fork: object }[] = [];
  const turns = new Map<string, number>();
  const kinds = spec.actors ?? { p1: "random-legal", p2: "random-legal" };
  const actorFor = (player: Player): Actor => {
    const inner = ACTORS[kinds[player]](spec.actor_seeds[player]);
    return (request, ctx) => {
      if (ctx !== undefined && request.phase === "move" && captured?.battle != null) {
        const live = captured.battle as any;
        turns.set(`${player}:${request.request_seq}`, live.turn);
        const names = new Set(live.sides.find((s: any) => s.id === other(player)).pokemon.map((m: any) => m.name));
        const foeSets = (spec.teams[other(player)] as PokemonSet[]).filter((set) => names.has(set.name === "" ? set.species : set.name));
        const input = inputAtLastRequest(player, ctx.channel, spec.teams[player], foeSets);
        if (input !== null) raw.push({ player, turn: live.turn, request_seq: request.request_seq, input, fork: JSON.parse(JSON.stringify(live.toJSON())) });
      }
      return inner(request, ctx);
    };
  };
  const record = await runBattle({
    engine: spy,
    formatId: GOLDEN_FORMAT,
    seed: spec.seed,
    teams: spec.teams,
    actors: { p1: actorFor("p1"), p2: actorFor("p2") },
    maxTurns: RANDOM_MAX_TURNS,
  });
  // The accepted choice of a seat in a turn is its last unrefused one.
  const final = new Map<string, string>();
  for (const c of record.choices) {
    const turn = turns.get(`${c.player}:${c.request_seq}`);
    if (turn !== undefined && c.refused === undefined) final.set(`${c.player}:${turn}`, c.choice);
  }
  // A refused choice is followed by a retry in the same turn: only the last decision of a seat and turn made the accepted choice.
  const last = new Map(raw.map((r, i) => [`${r.player}:${r.turn}`, i]));
  const decisions = raw.filter((r, i) => last.get(`${r.player}:${r.turn}`) === i).map((r): Decision => {
    const own = final.get(`${r.player}:${r.turn}`);
    const foe = final.get(`${other(r.player)}:${r.turn}`);
    return { battle_id: spec.id, player: r.player, turn: r.turn, request_seq: r.request_seq, input: r.input, fork: r.fork, recorded: own !== undefined && foe !== undefined ? { own, foe } : null };
  });
  return { decisions, record };
}

export type { StateItemId };

const NOISE = new Set(["effectOrder", "source", "sourceSlot", "sourceEffect", "target", "id", "name", "effectType", "noCopy"]);
const bare = (state: unknown): string => JSON.stringify(state, (key, value) => (NOISE.has(key) ? undefined : value));

/**
 * The state differences between the fork and the world just before the turn, as tags (fixed words, no line of the
 * battle): the classification of a divergent turn. `opponent:` tags are on the side the player cannot see.
 * Fields that only record history (`attackedBy`, `lastDamage`, `effectOrder`) are not state the turn reads, except
 * through the items below, so they are not compared.
 */
export function stateDelta(fork: any, world: any, seat: Player): string[] {
  const tags = new Set<string>();
  for (const fs of fork.sides) {
    const ws = world.sides.find((s: any) => s.id === fs.id);
    const who = fs.id === seat ? "own" : "opponent";
    if (fs.totalFainted !== ws.totalFainted) tags.add(`${who}:total_fainted`);
    if (!!fs.faintedLastTurn !== !!ws.faintedLastTurn) tags.add(`${who}:fainted_last_turn`);
    const conditions = (sc: Record<string, any>): string => bare(Object.fromEntries(Object.entries(sc).sort()));
    if (conditions(fs.sideConditions) !== conditions(ws.sideConditions)) tags.add(`${who}:side_conditions`);
    const slots = (side: any): string => JSON.stringify(side.active.map((m: any) => (m === null || m === undefined ? null : [m.name, m.fainted])));
    if (slots(fs) !== slots(ws)) tags.add(`${who}:active_slots`);
    for (const fm of fs.pokemon) {
      const wm = ws.pokemon.find((m: any) => m.name === fm.name);
      if (wm === undefined) {
        tags.add(`${who}:missing_pokemon`);
        continue;
      }
      if (fm.fainted !== wm.fainted) tags.add(`${who}:fainted`);
      // A fainted Pokemon takes no part in the turn: nothing else of it is compared.
      if (fm.fainted && wm.fainted) continue;
      if (fm.hp !== wm.hp || fm.maxhp !== wm.maxhp) tags.add(`${who}:hp`);
      if (fm.status !== wm.status) tags.add(`${who}:status`);
      if (fm.item !== wm.item) tags.add(`${who}:item`);
      if (fm.ability !== wm.ability) tags.add(`${who}:ability`);
      if (bare(fm.boosts) !== bare(wm.boosts)) tags.add(`${who}:boosts`);
      if (fm.moveSlots.some((slot: any, i: number) => slot.id !== wm.moveSlots[i]?.id || slot.pp !== wm.moveSlots[i]?.pp)) tags.add(`${who}:pp`);
      if (!fm.isActive && !wm.isActive) continue;
      // A benched Pokemon's status counter is reset when it switches in, so only a Pokemon in play is compared.
      if (fm.status !== "" && fm.status === wm.status && bare(fm.statusState) !== bare(wm.statusState)) tags.add(fm.status === "tox" ? `${who}:status_state` : `${who}:status_state_${fm.status}`);
      if (fm.activeTurns !== wm.activeTurns || fm.activeMoveActions !== wm.activeMoveActions) tags.add(`${who}:active_turns`);
      if ((fm.lastMove?.id ?? null) !== (wm.lastMove?.id ?? null)) tags.add(`${who}:last_move`);
      for (const id of new Set([...Object.keys(fm.volatiles), ...Object.keys(wm.volatiles)])) {
        if (bare(fm.volatiles[id]) !== bare(wm.volatiles[id])) tags.add(`${who}:volatile:${id}`);
      }
    }
  }
  const [ff, wf] = [fork.field, world.field];
  if (ff.weather !== wf.weather) tags.add("field:weather");
  else if (ff.weather !== "" && bare(ff.weatherState) !== bare(wf.weatherState)) tags.add("field:weather_duration");
  if (ff.terrain !== wf.terrain) tags.add("field:terrain");
  else if (ff.terrain !== "" && bare(ff.terrainState) !== bare(wf.terrainState)) tags.add("field:terrain_duration");
  if (bare(Object.fromEntries(Object.entries(ff.pseudoWeather).sort())) !== bare(Object.fromEntries(Object.entries(wf.pseudoWeather).sort()))) tags.add("field:conditions");
  return [...tags].sort();
}
