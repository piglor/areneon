// The oracle side of the differential harness at level 1, the mechanics-query level (ADR-030 groundwork G1, #1461): the
// pinned probe as the oracle backend (through `batchOf`, the batch seam's call), and the facts of a query the harness's entity
// rule reads, parsed from the query's own sets with the pinned engine (`importPaste`) and its declared state.
//
// The bridge stays the engine's owner and does NOT import the harness (`@areneon/differential`): the shapes below are declared
// structurally, and the composition root (packages/team-dataset) joins the two, as it does for the opponent model.
import { toId } from "@areneon/public-state";
import type { Engine } from "./engine.ts";
import type { MechanicsConfig } from "./mechanics-config.ts";
import type { DamagePath } from "./mechanics-damage.ts";
import { probeMechanics } from "./mechanics-probe.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery } from "./mechanics-query.ts";
import { MECHANICS_ADAPTER_FORMAT, type MechanicsResult } from "./mechanics-result.ts";
import { batchOf, type Settled } from "./probe-batch.ts";
import type { RulesSnapshot } from "./rules.ts";
import { importPaste } from "./teams.ts";

/** A backend as the harness's port declares it: a batch in, the outcomes in input order. */
export interface OracleBackend {
  readonly id: string;
  readonly backend_format: number;
  readonly backend_commit: string;
  run(batch: readonly ParsedMechanicsQuery[]): Settled[];
}

/** One Pokemon of a query, in engine ids (the harness's `MonFacts`). */
export interface QueryMon {
  species: string;
  ability: string;
  item: string | null;
  moves: string[];
  status: string | null;
  volatiles: string[];
}

/** What a mechanics query holds, in engine ids (the harness's `MechanicsDescriptor`). */
export interface QueryFacts {
  kind: ParsedMechanicsQuery["query"]["kind"];
  mons: QueryMon[];
  used_moves: string[];
  weather: string | null;
  terrain: string | null;
  trick_room: boolean;
  side_conditions: string[];
  slot_conditions: string[];
}

export interface OracleOptions {
  config: MechanicsConfig;
  snapshot?: RulesSnapshot;
  damage_path?: DamagePath;
  /** The name reports give it (default `showdown-probe`). */
  id?: string;
}

/**
 * The pinned probe as a backend: every query of a batch answered by `probeMechanics` on `engine`, in query order, equal queries
 * once (`batchOf`). Its format is the probe's adapter format and its commit the engine's. A second `Engine` object (another
 * `loadEngine()`) has its own per-engine probe tables, so two oracles over two engine objects are two instances.
 */
export function probeOracle(engine: Engine, options: OracleOptions): OracleBackend {
  const context = { engine, config: options.config, ...(options.snapshot === undefined ? {} : { snapshot: options.snapshot }), ...(options.damage_path === undefined ? {} : { damage_path: options.damage_path }) };
  const run = batchOf((query) => probeMechanics(context, query).result);
  return { id: options.id ?? "showdown-probe", backend_format: MECHANICS_ADAPTER_FORMAT, backend_commit: engine.commit, run: (batch) => run(batch) };
}

type Fact<T> = { kind: "known"; value: T } | { kind: "unknown"; ends: T[] };
type Combatant = Extract<ParsedMechanicsQuery["query"], { kind: "damage" }>["attacker"];
type FieldState = Extract<ParsedMechanicsQuery["query"], { kind: "damage" }>["state"];
type History = Extract<ParsedMechanicsQuery["query"], { kind: "damage" }>["history"];

/** Every value a fact can hold: a known one, or each end of an unknown one. */
const values = <T>(fact: Fact<T> | undefined): T[] => (fact === undefined ? [] : fact.kind === "known" ? [fact.value] : fact.ends);

const SIDE_CONDITIONS = ["aurora_veil", "light_screen", "reflect", "tailwind"] as const;

/** The facts of a query, read with the pinned engine. One object per engine: it keeps the sets it parsed. */
export function queryFacts(engine: Engine): { describe(query: ParsedMechanicsQuery): QueryFacts; queryHash(query: ParsedMechanicsQuery): string; errorKey(error: unknown): string } {
  const parsed = new Map<string, { species: string; ability: string; item: string; moves: string[] }>();
  const setOf = (paste: string) => {
    let set = parsed.get(paste);
    if (set === undefined) {
      const [one] = importPaste(engine, paste);
      const s = one as NonNullable<typeof one>;
      set = { species: s.species, ability: s.ability, item: s.item, moves: s.moves };
      parsed.set(paste, set);
    }
    return set;
  };
  const megaOf = (formatId: string, species: string, item: string): string => {
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(formatId).mod);
    const name = dex.species.get(species);
    const stone = dex.items.get(item).megaStone;
    return stone?.[name.name] ?? stone?.[name.baseSpecies] ?? species;
  };
  const mon = (formatId: string, paste: string, combatant?: Combatant): QueryMon => {
    const set = setOf(paste);
    const species = combatant?.mega === true ? megaOf(formatId, set.species, set.item) : set.species;
    const volatiles = values(combatant?.history?.volatiles as Fact<{ id: string }[]> | undefined).flatMap((list) => list.map((v) => v.id));
    return {
      species: toId(species),
      ability: toId(set.ability),
      item: set.item === "" ? null : toId(set.item),
      moves: set.moves.map(toId),
      status: combatant?.status ?? null,
      volatiles: [...new Set(volatiles)].sort(),
    };
  };
  const field = (state: FieldState, history: History) => ({
    weather: state.weather,
    terrain: state.terrain,
    trick_room: state.trick_room,
    side_conditions: SIDE_CONDITIONS.filter((name) => state.sides.some((side) => side[name])),
    slot_conditions: [...new Set((history?.sides ?? []).flatMap((side) => values(side.pending_slots as Fact<{ id: string }[]> | undefined).flatMap((list) => list.map((s) => s.id))))].sort(),
  });
  return {
    describe(query) {
      const q = query.query;
      const f = query.format_id;
      if (q.kind === "stats") return { kind: q.kind, mons: [mon(f, q.set)], used_moves: [], weather: null, terrain: null, trick_room: false, side_conditions: [], slot_conditions: [] };
      if (q.kind === "damage") return { kind: q.kind, mons: [mon(f, q.attacker.set, q.attacker), mon(f, q.defender.set, q.defender)], used_moves: [q.move], ...field(q.state, q.history) };
      const mons = q.sides.flat().map((c) => mon(f, c.set, c));
      const actions = q.kind === "move_outcome" ? [q.actions[q.asked] as (typeof q.actions)[number]] : q.actions;
      const used = [...new Set(actions.flatMap((a) => (a.move === null ? [] : [a.move])))].sort();
      return { kind: q.kind, mons, used_moves: used, ...field(q.state, q.history) };
    },
    queryHash: mechanicsQueryHash,
    // A refusal's class and words: the probe's refusals are fixed words of the probe (a `MechanicsProbeError`, a `ProbeRefusal`).
    errorKey: (error) => `${(error as Error).name}:${(error as Error).message}`,
  };
}

export type { MechanicsResult };
