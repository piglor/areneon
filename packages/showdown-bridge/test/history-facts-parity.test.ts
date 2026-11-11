/* eslint-disable @typescript-eslint/no-explicit-any */
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { HistoryFactsReader, type HistoryFacts, type HistoryFactsData, type PokemonHistory } from "@areneon/public-state";
import { GOLDEN_DIR, goldenBattle } from "../src/golden.ts";
import { goldenHistoryFactsData } from "../src/history-facts-data.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { loadRandomCorpus, pendingFirstActor, playRandomBattle, type RandomBattleFile } from "../src/random-corpus.ts";
import { randomLegalActor } from "../src/random-legal.ts";
import type { Actor } from "../src/runner.ts";

// ADR-022 erratum E-22.5.2, slice H1, the required parity test. Every committed battle is replayed on the pinned
// engine with a test-only spy on the live battle (as `world-writers-live.test.ts` does): at every request of
// both seats the history facts of that seat's own channel are compared with the engine's own fields, for both
// sides' Pokemon. A fact marked `unknown` (Illusion) is skipped and counted; a refusal is counted. The
// negative control shifts one fact by one and must be seen.

let engine: Engine;
let data: HistoryFactsData;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
  data = goldenHistoryFactsData();
});

const FORMAT_DIR = GOLDEN_DIR;
/** `ARENEON_HISTORY_PARITY_MAX` limits the battles of each corpus (local runs on a shared machine); CI runs them all. */
const MAX = Number(process.env.ARENEON_HISTORY_PARITY_MAX ?? "100000");
/** `ARENEON_HISTORY_PARITY_SKIP`: a comma list of corpora to leave out locally. */
const SKIP = new Set((process.env.ARENEON_HISTORY_PARITY_SKIP ?? "").split(",").filter((s) => s !== ""));

const exercised = new Map<string, number>();
const hit = (what: string, when = true): void => {
  if (when) exercised.set(what, (exercised.get(what) ?? 0) + 1);
};
const refusals = new Map<string, number>();
let requestsChecked = 0;
let unknownSkipped = 0;

type Diff = string;

/** What the engine's `attackedBy` says about this turn's attackers, as positions seen from the viewer: counted hits only (a numeric damage value). */
function engineAttackers(p: any, viewerSide: string): string[] {
  return p.attackedBy
    .filter((a: any) => a.thisTurn && typeof a.damageValue === "number")
    .map((a: any) => `${String(a.slot).slice(0, 2) === viewerSide ? "own" : "opponent"}:${String(a.slot).slice(2)}`)
    .sort();
}

/** Compares the facts with the live battle; returns the differences. */
export function compare(facts: HistoryFacts, live: any, viewer: "p1" | "p2", where: string): Diff[] {
  const diffs: Diff[] = [];
  const check = (what: string, fact: unknown, engineValue: unknown): void => {
    if ((fact === null && engineValue !== null && /(attackers_this_turn|last_move_failed|ate_berry)$/.test(what)) || fact === "unsettled") {
      // a fact the lines cannot settle is null, never guessed
      hit(`unsettled ${what.split(" ").at(-1)}`);
      return;
    }
    hit(`compared ${what.split(" ").at(-1)}`);
    if (JSON.stringify(fact) !== JSON.stringify(engineValue)) diffs.push(`${where}: ${what}: fact ${JSON.stringify(fact)} engine ${JSON.stringify(engineValue)}`);
  };
  check("field_last_move", facts.field_last_move, live.lastMove?.id ?? null);
  const seat = viewer === "p1" ? 0 : 1;
  for (const [label, sideIndex] of [["own", seat], ["opponent", 1 - seat]] as const) {
    const side = facts[label];
    const liveSide = live.sides[sideIndex];
    check(`${label} total_fainted`, side.total_fainted, liveSide.totalFainted);
    for (const [letter, entry] of Object.entries(side.active)) {
      const index = letter.charCodeAt(0) - 97;
      const p = liveSide.active[index];
      const at = `${label}:${letter} ${entry?.name}`;
      if (entry === null || p === null || p === undefined || p.fainted) continue;
      if (entry.unknown) {
        unknownSkipped += 1;
        continue;
      }
      const f = entry as Extract<PokemonHistory, { unknown: false }>;
      check(`${at} move_actions`, f.move_actions, p.activeMoveActions);
      check(`${at} stall_counter`, f.stall_counter, p.volatiles.stall?.counter ?? null);
      check(`${at} times_attacked`, f.times_attacked, p.timesAttacked);
      check(`${at} hurt_this_turn`, f.hurt_this_turn, p.hurtThisTurn !== null && p.hurtThisTurn !== undefined);
      check(`${at} attackers_this_turn`, f.attackers_this_turn, engineAttackers(p, viewer));
      check(`${at} last_move_failed`, f.last_move_failed, p.moveLastTurnResult === false);
      check(`${at} stats_raised_this_turn`, f.stats_raised_this_turn, !!p.statsRaisedThisTurn);
      check(`${at} stats_lowered_this_turn`, f.stats_lowered_this_turn, !!p.statsLoweredThisTurn);
      if (label === "own") check(`${at} moves_used`, f.moves_used, p.moveSlots.filter((m: any) => m.used).map((m: any) => m.id).sort());
      else expect(f.moves_used, "an opponent's moves_used is not known").toBeNull();
      for (const id of Object.keys(f.volatile_started)) {
        const v = p.volatiles[id];
        if (v === undefined) continue; // an id the engine names differently (a derived name): not compared
        hit("volatile_started present");
      }
    }
    // Until its `|replace|` an Illusion shows another Pokemon's name: the roster facts are attributed to the Pokemon shown (as every reader's are), not to the one that holds the item.
    const disguised = liveSide.pokemon.some((m: any) => m.illusion);
    if (disguised) hit("roster facts skipped under an Illusion");
    for (const [name, roster] of disguised ? [] : Object.entries(side.roster)) {
      const p = liveSide.pokemon.find((m: any) => m.name === name);
      if (p === undefined || roster.ate_berry === null) continue;
      check(`${label} ${name} ate_berry`, roster.ate_berry, !!p.ateBerry);
      check(`${label} ${name} last_item`, roster.last_item, p.lastItem === "" || p.lastItem === undefined ? null : p.lastItem);
    }
    for (const slotId of data.slot_conditions.map((c) => c.id)) {
      for (let index = 0; index < liveSide.slotConditions.length; index++) {
        const letter = String.fromCharCode(97 + index);
        check(`${label}:${letter} pending ${slotId}`, side.pending_slot[letter]?.[slotId] !== undefined, !!liveSide.slotConditions[index][slotId]);
      }
    }
  }
  return diffs;
}

function note(facts: HistoryFacts): void {
  for (const side of [facts.own, facts.opponent]) {
    hit("total_fainted >= 2", side.total_fainted >= 2);
    for (const a of Object.values(side.active)) {
      if (a === null || a.unknown) continue;
      hit("move_actions >= 3", a.move_actions >= 3);
      hit("stall_counter", a.stall_counter !== null);
      hit("stall_counter > 3", (a.stall_counter ?? 0) > 3);
      hit("times_attacked >= 1", a.times_attacked >= 1);
      hit("times_attacked >= 3", a.times_attacked >= 3);
      hit("hurt_this_turn", a.hurt_this_turn);
      hit("attackers_this_turn", (a.attackers_this_turn?.length ?? 0) > 0);
      hit("last_move_failed", a.last_move_failed === true);
      hit("stats_raised_this_turn", a.stats_raised_this_turn);
      hit("stats_lowered_this_turn", a.stats_lowered_this_turn);
      hit("moves_used", (a.moves_used?.length ?? 0) > 0);
      hit("volatile_started", Object.keys(a.volatile_started).length > 0);
    }
    hit("ate_berry", Object.values(side.roster).some((r) => r.ate_berry === true));
    hit("last_item", Object.values(side.roster).some((r) => r.last_item !== null));
    hit("pending_slot", Object.keys(side.pending_slot).length > 0);
  }
  hit("field_last_move", facts.field_last_move !== null);
}

interface Spec extends RandomBattleFile {
  actors?: Record<"p1" | "p2", "random-legal" | "pending-first">;
}

/** Replays one battle and compares every request of both seats; `mutate` lets the negative control shift a fact. */
async function replay(spec: Spec, mutate?: (facts: HistoryFacts) => void): Promise<Diff[]> {
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
  const diffs: Diff[] = [];
  const actorFor = (seed: number): Actor => {
    const player = spec.actor_seeds.p1 === seed ? "p1" : "p2";
    const kind = spec.actors?.[player] ?? "random-legal";
    const inner = kind === "pending-first" ? pendingFirstActor(seed) : randomLegalActor(seed);
    return (request, ctx) => {
      if (ctx !== undefined && captured?.battle != null) {
        const lines = ctx.channel as readonly string[];
        // the channel ends in the request being answered: facts are read before that line
        for (; fed[player] < lines.length - 1; fed[player]++) readers[player].feed(lines[fed[player]] as string);
        requestsChecked += 1;
        try {
          const facts = readers[player].facts();
          mutate?.(facts);
          note(facts);
          diffs.push(...compare(facts, captured.battle, player, `${spec.id}/${player}/turn ${facts.turn}`));
        } catch (error) {
          const code = (error as { code?: string }).code ?? "error";
          refusals.set(code, (refusals.get(code) ?? 0) + 1);
        }
      }
      return inner(request, ctx);
    };
  };
  await playRandomBattle(spy, spec, actorFor);
  return diffs;
}

const corpus = (dir: string): Spec[] => {
  const full = join(FORMAT_DIR, dir);
  return existsSync(full) ? (loadRandomCorpus(full) as Spec[]).slice(0, MAX) : [];
};
const single = (dir: string): Spec => JSON.parse(readFileSync(join(FORMAT_DIR, dir, "battle.json"), "utf8")) as Spec;

async function sweep(name: string, specs: Spec[]): Promise<void> {
  if (SKIP.has(name)) return;
  const all: Diff[] = [];
  for (const spec of specs) all.push(...(await replay(spec)));
  if (process.env.ARENEON_HISTORY_PARITY_DUMP !== undefined) appendFileSync(process.env.ARENEON_HISTORY_PARITY_DUMP, all.map((d) => `${name} ${d}\n`).join(""));
  if (all.length > 0) console.log(`${name}: ${all.length} differences, first:\n${all.slice(0, 12).join("\n")}`);
  expect(all, `${name}: facts equal the engine's fields`).toEqual([]);
}

describe("history facts equal the live engine's fields at every request of both seats", () => {
  it("random-play corpus", async () => sweep("random", loadRandomCorpus().slice(0, MAX) as Spec[]), 600_000);
  it("unavailable corpus (pending-first play)", async () => sweep("unavailable", corpus("unavailable").map((s) => ({ ...s, actors: { p1: "pending-first", p2: "pending-first" } as const }))), 600_000);
  it("the Illusion, Baton Pass, Fling, revival and side-guard battles", async () => sweep("special", ["illusion", "baton-pass", "baton-pass-volatiles", "fling", "revival", "side-guard"].filter((d) => existsSync(join(FORMAT_DIR, d, "battle.json"))).map(single)), 600_000);
  it("the targeted fidelity corpus", async () => sweep("fidelity", corpus("search-fidelity")), 3_600_000);
  it("the Transform battle (a recorded game with no seeds: its channels give facts with no refusal, Transform's copy of timesAttacked included)", () => {
    const b = single("transform");
    for (const player of ["p1", "p2"] as const) {
      const reader = new HistoryFactsReader(player, data);
      for (const line of b.channels[player].lines) reader.feed(line);
      const facts = reader.facts();
      note(facts);
      hit("transform channel read", Object.values(facts.own.active).concat(Object.values(facts.opponent.active)).some((a) => a !== null));
    }
    expect(exercised.get("transform channel read")).toBeGreaterThan(0);
  });

  it("the golden battle", async () => {
    if (SKIP.has("golden")) return;
    const record = await goldenBattle(engine);
    // the golden battle's own channels: facts are computed and checked for refusals only (its actors are not the random ones the spy replays)
    for (const player of ["p1", "p2"] as const) {
      const reader = new HistoryFactsReader(player, data);
      for (const line of record.channels[player]) reader.feed(line);
      expect(() => reader.facts()).not.toThrow();
    }
  }, 60_000);

  it("compared many facts, and exercised the states the facts exist for", () => {
    console.log(`history-facts parity: ${requestsChecked} requests, refusals ${JSON.stringify([...refusals])}, unknown skipped ${unknownSkipped}; exercised ${JSON.stringify([...exercised])}`);
    if (process.env.ARENEON_HISTORY_PARITY_DUMP !== undefined) appendFileSync(`${process.env.ARENEON_HISTORY_PARITY_DUMP}.stats`, `${requestsChecked} requests; refusals ${JSON.stringify([...refusals])}; ${JSON.stringify([...exercised])}\n`);
    if (SKIP.size > 0 || MAX < 100000) return;
    expect(requestsChecked).toBeGreaterThan(2000);
    for (const what of ["move_actions >= 3", "stall_counter", "times_attacked >= 1", "hurt_this_turn", "attackers_this_turn", "last_move_failed", "stats_raised_this_turn", "stats_lowered_this_turn", "moves_used", "ate_berry", "last_item", "total_fainted >= 2", "field_last_move"]) {
      expect(exercised.get(what) ?? 0, what).toBeGreaterThan(0);
    }
  });

  it("has teeth: one fact computed off by one is seen", async () => {
    if (SKIP.size > 0) return;
    const spec = loadRandomCorpus()[0] as Spec;
    const diffs = await replay(spec, (facts) => {
      for (const a of Object.values(facts.own.active)) if (a !== null && !a.unknown) a.times_attacked += 1;
    });
    expect(diffs.length).toBeGreaterThan(0);
    expect(diffs.some((d) => d.includes("times_attacked"))).toBe(true);
  }, 120_000);
});
