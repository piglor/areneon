/* eslint-disable @typescript-eslint/no-explicit-any */
// Running the oracle over the committed corpora and collecting TurnRecords for the report (ADR-026 slice F).

import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type Engine, loadEngine, PINNED_SHOWDOWN_COMMIT, type PokemonSet } from "../../src/engine.ts";
import { jcs } from "../../src/canonical-json.ts";
import type { Inputs } from "./report.ts";
import { GOLDEN_FORMAT } from "../../src/golden.ts";
import { loadMechanicsConfig } from "../../src/mechanics-config.ts";
import { checkScenario } from "../../src/mechanics-scenario.ts";
import { loadRandomCorpus, RANDOM_DIR, UNAVAILABLE_DIR } from "../../src/random-corpus.ts";
import { FIDELITY_DIR, loadFidelityTargets } from "../../src/search-fidelity-corpus.ts";
import { type WorldContext, createWorldBuilder } from "../../src/world-builder.ts";
import { type WorldWriters, loadWorldWriters, STATE_ITEMS } from "../../src/world-writers.ts";
import { type Decision, captureDecisions, type OracleSpec } from "./oracle.ts";
import { type Arm, evaluate, type Options, type TurnRecord } from "./evaluate.ts";

export const CORPORA = ["random", "random-choice-lock", "unavailable", "search-fidelity"] as const;
export type Corpus = (typeof CORPORA)[number];

export interface RunOptions extends Options {
  corpora: readonly Corpus[];
  /** At most this many battles of each corpus (all when absent); the CI subset. */
  battles?: number | undefined;
  /** Also test every volatile id and start-trigger id the corpora show (admission arms). */
  candidates: boolean;
  /** A volatile id is tried only when at least this many decisions show it (the prefilter keeps the full run affordable). */
  min_decisions?: number;
}

export interface Candidate {
  kind: "volatile" | "start_trigger";
  id: string;
  /** The class the arm admits it with. */
  class: "E" | "W";
}

export interface RunData {
  corpora: Record<string, { battles: number; decisions: number }>;
  turns: TurnRecord[];
  /** Refusals of the base arm by `code:item`. */
  refusals: Record<string, number>;
  /** Decisions whose recorded choices were missing (not compared in F1). */
  decisions: number;
  candidates: Candidate[];
  options: { k: number; m: number };
  ms: number;
}

const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/**
 * The committed battles of a corpus. `random-choice-lock` is the random corpus replayed with a Choice Scarf on one
 * Pokemon of each side (the committed teams hold none in play), the rule `world-writers-live.test.ts` uses: the
 * choice-lock writer needs exercising turns the natural corpora lack. The battles are the same seeds on those teams.
 */
function specsOf(corpus: Corpus, limit: number | undefined): OracleSpec[] {
  const dir = corpus === "random" || corpus === "random-choice-lock" ? undefined : corpus === "unavailable" ? UNAVAILABLE_DIR : FIDELITY_DIR;
  let all = loadRandomCorpus(dir) as unknown as (OracleSpec & { actors?: OracleSpec["actors"] })[];
  if (corpus === "random-choice-lock") {
    all = all.map((spec, index) => {
      const teams = structuredClone(spec.teams);
      for (const side of ["p1", "p2"] as const) {
        const team = teams[side];
        for (const set of team) if (toId(set.item) === "choicescarf") set.item = "";
        (team[(index + (side === "p1" ? 0 : 3)) % team.length] as PokemonSet).item = "Choice Scarf";
      }
      return { ...spec, id: `${spec.id}+scarf`, teams };
    });
  }
  return limit === undefined ? all : all.slice(0, limit);
}

/** The ids of the active Pokemon of a side carrying a volatile, by name. */
const carriersOf = (id: string) => (d: Decision): string[] => {
  const out: string[] = [];
  for (const side of [d.input.state.own, d.input.state.opponent]) {
    for (const index of Object.values(side.slots)) {
      const mon = index === null ? undefined : side.roster[index];
      if (mon !== undefined && mon.name !== null && mon.volatiles.includes(id)) out.push(mon.name);
    }
  }
  return out;
};

/** The ids of the volatiles shown on an active Pokemon at a decision. */
function shownVolatiles(d: Decision): Set<string> {
  const out = new Set<string>();
  for (const side of [d.input.state.own, d.input.state.opponent]) {
    for (const index of Object.values(side.slots)) {
      const mon = index === null ? undefined : side.roster[index];
      for (const id of mon?.volatiles ?? []) out.add(id);
    }
  }
  return out;
}

const withAdmitted = (base: WorldWriters, admitted: WorldWriters["admitted"]): WorldWriters => ({ ...structuredClone(base), admitted: [...admitted].sort((a, b) => (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : a.id < b.id ? -1 : a.id > b.id ? 1 : 0)) });

export async function runFidelity(options: RunOptions, engine: Engine = loadEngine()): Promise<RunData> {
  const t0 = Date.now();
  // The arms start from the committed record with no id admitted, so the evidence never depends on the admissions it supports.
  const writers: WorldWriters = { ...structuredClone(loadWorldWriters(engine, GOLDEN_FORMAT)), admitted: [] };
  const context: WorldContext = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: "ceil", writers };
  const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as any;
  const refused = new Set(writers.items.flatMap((item) => item.ids ?? []));
  const confusion = writers.items.find((item) => item.id === "confusion_duration")?.condition;
  const volatileClass = (id: string): "E" | "W" | null => {
    const c = dex.conditions.get(id);
    if (!c.exists || c.effectType !== "Condition" || refused.has(id) || id === confusion) return null;
    return c.duration !== undefined || c.durationCallback !== undefined ? "W" : "E";
  };
  void STATE_ITEMS;
  const specs = options.corpora.flatMap((corpus) => specsOf(corpus, options.battles).map((spec) => ({ corpus, spec })));
  // Every ability and item id a brought set holds: the start-trigger probe admits them all at once.
  const triggerIds = new Set<string>();
  for (const { spec } of specs) for (const team of Object.values(spec.teams)) for (const set of team) for (const id of [toId(set.ability), toId(set.item)]) if (id !== "") triggerIds.add(id);
  const probeIds = [...triggerIds].filter((id) => ["abilities", "items"].some((table) => dex[table].get(id).exists)).sort();
  const base: Arm = { name: "base", writers, id: null };
  const probe: Arm = { name: "trigger-probe", writers: withAdmitted(writers, probeIds.map((id) => ({ kind: "start_trigger" as const, id, class: "W" as const }))), id: null };
  const builders = new Map<string, ReturnType<typeof createWorldBuilder>>();
  const volatileArms = new Map<string, Arm>();
  const triggerArms = new Map<string, Arm>();
  const data: RunData = { corpora: {}, turns: [], refusals: {}, decisions: 0, candidates: [], options: { k: options.k, m: options.m }, ms: 0 };
  const seen = new Map<string, Candidate>();
  const minDecisions = options.min_decisions ?? 1;
  const shownCount = new Map<string, number>();

  for (const { corpus, spec } of specs) {
    const { decisions } = await captureDecisions(engine, spec);
    const stats = (data.corpora[corpus] ??= { battles: 0, decisions: 0 });
    stats.battles += 1;
    stats.decisions += decisions.length;
    data.decisions += decisions.length;
    for (const d of decisions) {
      const o = evaluate(engine, context, base, d, options, builders);
      if (o.kind === "refused") {
        const key = `${o.refusal.code}:${o.refusal.item ?? "-"}`;
        data.refusals[key] = (data.refusals[key] ?? 0) + 1;
      } else data.turns.push(o.record);
      if (!options.candidates) continue;
      for (const id of shownVolatiles(d)) {
        const cls = volatileClass(id);
        if (cls === null) continue;
        shownCount.set(id, (shownCount.get(id) ?? 0) + 1);
        let arm = volatileArms.get(id);
        if (arm === undefined) {
          arm = { name: `volatile:${id}`, writers: withAdmitted(writers, [{ kind: "volatile", id, class: cls }]), id: `volatile:${id}`, carriers: carriersOf(id) };
          volatileArms.set(id, arm);
          seen.set(`volatile:${id}`, { kind: "volatile", id, class: cls });
        }
        if ((shownCount.get(id) ?? 0) < minDecisions && minDecisions > 1) continue;
        const r = evaluate(engine, context, arm, d, options, builders);
        if (r.kind === "turn") data.turns.push(r.record);
      }
      // A start trigger: the probe arm admits every id; where it restored something, each in-play id is tried alone.
      const p = evaluate(engine, context, probe, d, { k: 0, m: 0 }, builders);
      if (p.kind === "turn" && p.written.includes("start_trigger_restore")) {
        const fork = d.fork as any;
        const inPlay = new Set<string>();
        for (const side of fork.sides) for (const mon of side.pokemon.filter((m: any) => m.isActive ?? true).slice(0, d.input.request.active?.length ?? 2)) for (const id of [toId(mon.set?.ability ?? ""), toId(mon.set?.item ?? "")]) if (probeIds.includes(id)) inPlay.add(id);
        for (const id of inPlay) {
          let arm = triggerArms.get(id);
          if (arm === undefined) {
            arm = { name: `trigger:${id}`, writers: withAdmitted(writers, [{ kind: "start_trigger", id, class: "W" }]), id: `trigger:${id}` };
            triggerArms.set(id, arm);
            seen.set(`trigger:${id}`, { kind: "start_trigger", id, class: "W" });
          }
          const r = evaluate(engine, context, arm, d, options, builders);
          // Only a world in which the single admitted id covered the restore is evidence for it.
          if (r.kind === "turn" && r.written.includes("start_trigger_restore")) data.turns.push(r.record);
        }
      }
    }
  }
  data.candidates = [...seen.values()].sort((a, b) => (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : a.id < b.id ? -1 : 1));
  data.ms = Date.now() - t0;
  void loadFidelityTargets;
  return data;
}

/** The version of the oracle, its arms and its rules; bump it when a number the report holds would change meaning. */
export const FIDELITY_FORMAT = 1;

/** What the report depends on, so that a committed report can be checked for staleness without a run. */
export function reportInputs(engine: Engine, writers: WorldWriters, corpora: readonly Corpus[] = CORPORA): Inputs {
  const hash = createHash("sha256");
  for (const corpus of corpora) {
    const dir = corpus === "random" || corpus === "random-choice-lock" ? RANDOM_DIR : corpus === "unavailable" ? UNAVAILABLE_DIR : FIDELITY_DIR;
    for (const name of readdirSync(dir).filter((n) => /^b\d+\.json$/.test(n)).sort()) hash.update(`${corpus}/${name}\n`).update(readFileSync(join(dir, name)));
  }
  const record = { ...writers, admitted: [] };
  return {
    fidelity_format: FIDELITY_FORMAT,
    showdown_commit: PINNED_SHOWDOWN_COMMIT,
    corpora_sha256: hash.digest("hex"),
    base_record_sha256: createHash("sha256").update(jcs(record as never)).digest("hex"),
  };
}
