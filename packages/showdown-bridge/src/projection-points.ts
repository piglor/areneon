/* eslint-disable @typescript-eslint/no-explicit-any */
// The trace reader of the battle-state projection (ADR-030 groundwork G4, #1464): a committed battle played again with the G2
// recorder (`recordBattleTrace`), which also hands over the engine's `toJSON()` at the end of every step. A point is one step:
// the engine state (trimmed to the keys the battle-state spec reads) and how far each seat's channel had got; a decision is a
// request of a seat at that step that asks for a choice, with the seat's real query (its observation as `observe` builds it from
// the Showdown channel, and the request's candidates with their pending flags from the trace's mask). The reference packer and
// the projection are Python (`ml/areneon_ml/projection`): the seat bookkeeping is the reducer's, which only Python has.
//
// This is the one module of the projection path that touches the recorder and the runner (the type rule of G4): a point holds no
// omniscient log, record or hidden verdict (`ProjectionPoint`'s fields; a type test in `battle-state-projection.python.test.ts`).
// The state holds both sides' hidden information: an evaluation artifact, never an Episode or actor input.
import { fileURLToPath } from "node:url";
import type { BattleStateSpec } from "./battle-state-spec.ts";
import { coveredKeys } from "./battle-state-spec.ts";
import { recordBattleTrace, type Seat, type TraceSpec } from "./battle-trace.ts";
import type { Engine, PokemonSet } from "./engine.ts";
import { GOLDEN_FORMAT } from "./golden.ts";
import type { Observation } from "./observation.ts";
import { observe, playerChannel } from "./player-channel.ts";

/** Record v2 of the battle-state spec, the projection's record (v1 lacks what the projection reads: see its tests). */
export const SPEC_V2_PATH = fileURLToPath(new URL(`../../../config/engine-state/${GOLDEN_FORMAT}/battle-state-spec.v2.json`, import.meta.url));

export class ProjectionPointError extends Error {
  override name = "ProjectionPointError";
}

/** A seat's decision at a point: its real query. */
export interface ProjectionDecision {
  seat: Seat;
  query: { observation: Observation; candidates: { choice: string; pending: boolean }[] };
}

/** One step of a battle: the engine state at its end and each seat's channel length then. */
export interface ProjectionPoint {
  step: number;
  cut: Record<Seat, number>;
  state: unknown;
  decisions: ProjectionDecision[];
}

export interface ProjectionBattle {
  battle_id: string;
  teams: Record<Seat, PokemonSet[]>;
  /** Each seat's whole channel; a point's channel is its first `cut` lines. */
  channels: Record<Seat, string[]>;
  points: ProjectionPoint[];
}

const SEATS: readonly Seat[] = ["p1", "p2"];
const isMap = (v: unknown): v is Record<string, any> => typeof v === "object" && v !== null && !Array.isArray(v);
const pick = (value: unknown, keep: ReadonlySet<string>, also: readonly string[] = []): any => {
  if (!isMap(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [key, inner] of Object.entries(value)) if (keep.has(key) || also.includes(key)) out[key] = inner;
  return out;
};
const values = (map: unknown, f: (v: unknown) => unknown): unknown => (isMap(map) ? Object.fromEntries(Object.entries(map).map(([k, v]) => [k, f(v)])) : map);

/**
 * The engine state trimmed to what the spec reads: at each classified level the keys a field reads, plus the keys that join the
 * levels (a battle's sides and field, a side's id, Pokemon and active list, a Pokemon's set name and species, which the packer
 * joins a Pokemon to its roster entry by). The packer reads nothing else, so trimming changes no tensor.
 */
export function trimState(spec: BattleStateSpec, state: any): any {
  const covered = coveredKeys(spec);
  const keys = (level: string): Set<string> => new Set([...covered].filter((k) => k.startsWith(`${level}.`)).map((k) => k.slice(level.length + 1)));
  const [battle, side, pokemon, field, slot, effect] = ["battle", "side", "pokemon", "field", "move_slot", "effect"].map(keys) as Set<string>[];
  const eff = (v: unknown): unknown => pick(v, effect as Set<string>);
  const mon = (m: any): any => {
    const out = pick(m, pokemon as Set<string>, ["set", "details", "isActive", "position"]);
    if (isMap(out.set)) out.set = { name: out.set.name, species: out.set.species };
    for (const key of ["statusState", "itemState", "abilityState", "speciesState"]) if (key in out) out[key] = eff(out[key]);
    if ("volatiles" in out) out.volatiles = values(out.volatiles, eff);
    for (const key of ["moveSlots", "baseMoveSlots"]) if (Array.isArray(out[key])) out[key] = out[key].map((s: unknown) => pick(s, slot as Set<string>, ["id"]));
    return out;
  };
  const out = pick(state, battle as Set<string>, ["sides", "field"]);
  const f = pick(state.field, field as Set<string>);
  for (const key of ["weatherState", "terrainState"]) if (key in f) f[key] = eff(f[key]);
  if ("pseudoWeather" in f) f.pseudoWeather = values(f.pseudoWeather, eff);
  out.field = f;
  out.sides = (state.sides as any[]).map((s) => {
    const kept = pick(s, side as Set<string>, ["id", "pokemon", "active"]);
    if ("sideConditions" in kept) kept.sideConditions = values(kept.sideConditions, eff);
    if (Array.isArray(kept.slotConditions)) kept.slotConditions = kept.slotConditions.map((c: unknown) => values(c, eff));
    kept.pokemon = (s.pokemon as any[]).map(mon);
    return kept;
  });
  return out;
}

/**
 * Play `spec` again with the recorder and read every point and decision. Fails closed (ProjectionPointError) on a decision whose
 * request is not the last line its seat got in the step: the state at the step's end would then be later than the request.
 */
export async function projectionBattle(engine: Engine, spec: TraceSpec, record_spec: BattleStateSpec): Promise<ProjectionBattle> {
  const states: unknown[] = [];
  const { trace, record } = await recordBattleTrace(engine, spec, {
    onStep: (_step, battle) => {
      const json = JSON.parse(JSON.stringify(battle.toJSON()));
      states.push(trimState(record_spec, json));
    },
  });
  if (states.length !== trace.steps.length) throw new ProjectionPointError(`${spec.battle_id}: the recorder gave ${states.length} states for ${trace.steps.length} steps`);
  const channels: Record<Seat, string[]> = { p1: [], p2: [] };
  const points: ProjectionPoint[] = [];
  for (const [k, step] of trace.steps.entries()) {
    for (const seat of SEATS) channels[seat].push(...step.channels[seat]);
    const decisions: ProjectionDecision[] = [];
    for (const seat of SEATS) {
      for (const request of step.requests[seat]) {
        if (request.mask === undefined) continue;
        const lines = channels[seat];
        if (!(lines.at(-1) ?? "").startsWith("|request|") || step.requests[seat].at(-1) !== request) {
          throw new ProjectionPointError(`${spec.battle_id}: a decision of ${seat} at step ${k} is not the last line its seat got in the step`);
        }
        const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: seat, lines: [...lines] });
        const observation = await observe({ channel, team: { schema_version: 1, data_class: "synthetic", team: structuredClone(spec.teams[seat]) as never }, manifest: record.manifest });
        decisions.push({ seat, query: { observation, candidates: request.mask.map((m) => ({ choice: m.choice, pending: m.verdict !== "legal" })) } });
      }
    }
    points.push({ step: k, cut: { p1: channels.p1.length, p2: channels.p2.length }, state: states[k], decisions });
  }
  return { battle_id: spec.battle_id, teams: structuredClone(spec.teams), channels, points };
}
