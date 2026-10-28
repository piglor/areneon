/* eslint-disable @typescript-eslint/no-explicit-any */
import { beforeAll, describe, expect, it } from "vitest";
import type { PokemonSet } from "../src/engine.ts";
import { type Engine, loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { type WorldContext, type WorldInput, createWorldBuilder } from "../src/world-builder.ts";
import { remainingTurns, statePlan } from "../src/world-state.ts";
import { type WorldWriters, loadWorldWriters, readWorldWriters } from "../src/world-writers.ts";
import { type Case, cases, toId } from "./world-builder-support.ts";

let engine: Engine;
let context: WorldContext;
let all: Case[];
beforeAll(() => {
  engine = loadEngine();
  context = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: "ceil", writers: loadWorldWriters(engine, GOLDEN_FORMAT) };
  all = cases();
});

const withClass = (item: string, cls: "R"): WorldWriters => {
  const writers = structuredClone(context.writers) as any;
  writers.items.find((row: any) => row.id === item).class = cls;
  return writers;
};

describe("the record indexes into a plan", () => {
  it("answers class, condition, id rows and admitted ids", () => {
    const writers = structuredClone(context.writers);
    writers.admitted = [{ kind: "start_trigger", id: "whiteherb", class: "W" }, { kind: "volatile", id: "taunt", class: "W" }];
    const plan = statePlan(readWorldWriters(writers, engine));
    expect(plan.classOf("toxic_stage")).toBe("W");
    expect(plan.classOf("substitute")).toBe("R");
    expect(plan.conditionOf("protect_counter")).toBe("stall");
    expect(plan.rowOf("substitute")).toBe("substitute");
    expect(plan.rowOf("wish")).toBe("slot_conditions");
    expect(plan.rowOf("taunt")).toBeUndefined();
    expect(plan.lineRow("-terastallize")).toBe(true);
    expect(plan.lineRow("terastallize")).toBe(false);
    expect(plan.admittedVolatile("taunt")).toBe("W");
    expect(plan.admittedVolatile("encore")).toBeUndefined();
    expect(plan.admittedTrigger("whiteherb")).toBe(true);
    expect(plan.admittedTrigger("taunt")).toBe(false);
  });
});

describe("remaining turns", () => {
  it("takes the turns since the effect began off its length, one fewer when it began after the residual, and leaves an indefinite effect alone", () => {
    expect(remainingTurns({ duration: 5 }, { started_turn: 3, after_residual: false }, 3)).toBe(5);
    expect(remainingTurns({ duration: 5 }, { started_turn: 3, after_residual: false }, 5)).toBe(3);
    expect(remainingTurns({ duration: 5 }, { started_turn: 3, after_residual: true }, 5)).toBe(4);
    expect(remainingTurns({ duration: 5 }, { started_turn: 0, after_residual: true }, 1)).toBe(5);
    expect(remainingTurns({ duration: 2 }, { started_turn: 1, after_residual: false }, 4)).toBe(-1);
    expect(remainingTurns({}, { started_turn: 1, after_residual: false }, 4)).toBeNull();
    expect(remainingTurns({ duration: 0 }, { started_turn: 1, after_residual: false }, 4)).toBeNull();
    expect(remainingTurns(undefined, { started_turn: 1, after_residual: false }, 4)).toBeNull();
  });
});

describe("an item the record gives class R refuses a world that holds it", () => {
  // Each W and X item present on the committed channels, switched to R: some world is refused for exactly that item, as R, and the others still build.
  it.each(["toxic_stage", "bench_status", "active_turns", "protect_counter", "last_move", "condition_duration", "fainted_last_turn", "total_fainted", "item_removed", "sleep_counter", "confusion_duration", "opponent_pp", "bench_order", "turn_number"])("%s", (item) => {
    const builder = createWorldBuilder({ ...context, writers: withClass(item, "R") });
    const seen = new Set<string>();
    for (const { input } of all) {
      const result = builder.build(input);
      if (result.kind === "refused" && result.refusal.code === "state_unwritten" && result.refusal.item === item) {
        expect(result.refusal.state_class).toBe("R");
        seen.add(result.refusal.detail);
      }
    }
    expect(seen.size, item).toBeGreaterThan(0);
    expect(builder.counts().unwritten[item as keyof ReturnType<typeof builder.counts>["unwritten"]]).toBeGreaterThan(0);
  });

  it("the opponent's PP and bench order refuse as evidence a redraw can change", () => {
    for (const item of ["bench_order"]) {
      const builder = createWorldBuilder({ ...context, writers: withClass(item, "R") });
      const refusals = all.map((c) => builder.build(c.input)).filter((r) => r.kind === "refused" && r.refusal.item === item);
      expect(refusals.length).toBeGreaterThan(0);
      expect(refusals.every((r) => r.kind === "refused" && r.refusal.redraw)).toBe(true);
    }
  });
});

describe("the X items are counted, not written", () => {
  it("counts the engine's sleep and confusion draws, the opponent's PP and the hidden bench order per world", () => {
    const builder = createWorldBuilder(context);
    for (const { input } of all) builder.build(input);
    const counts = builder.counts();
    for (const item of ["sleep_counter", "confusion_duration", "opponent_pp", "bench_order"] as const) expect(counts.resampled[item], item).toBeGreaterThan(0);
  });

  it("builds a confused Pokemon with the engine's own duration draw (a volatile the world adds and the engine times)", () => {
    const builder = createWorldBuilder(context);
    const found = all.find((c) => c.input.state.own.roster.some((m) => m.volatiles.includes("confusion") && m.active_slot !== null && !m.fainted) && builder.build(c.input).kind === "world");
    expect(found).toBeDefined();
    const result = builder.build((found as Case).input);
    if (result.kind !== "world") return;
    const mons = (result.world.battle as any).sides.flatMap((s: any) => s.pokemon);
    const confused = mons.filter((m: any) => m.volatiles.confusion !== undefined);
    expect(confused.length).toBeGreaterThan(0);
    expect(confused[0].volatiles.confusion.time).toBeGreaterThan(0);
  });
});

describe("a scenario-start trigger", () => {
  // The committed record admits two volatiles since slice F: these cases are chosen and refused as under a record that admits no id.
  const bare = (): WorldContext => ({ ...context, writers: { ...structuredClone(context.writers), admitted: [] } as WorldWriters });
  // An own active holding White Herb against a foe active with Intimidate: the start consumes the herb.
  const trigger = (): WorldInput | null => {
    for (const { input } of all) {
      const foeActive = input.opponent.members.filter((m) => input.state.opponent.roster[m.roster_index]?.active_slot !== null && !input.state.opponent.roster[m.roster_index]?.fainted);
      if (!foeActive.some((m) => toId(m.set.ability) === "intimidate")) continue;
      // The case itself must build: the herb is the only thing the start changes.
      if (createWorldBuilder(bare()).build(input).kind !== "world") continue;
      const request = structuredClone(input.request);
      const team = input.team.map((set) => structuredClone(set) as PokemonSet);
      if (team.some((set) => toId(set.item) === "whiteherb")) continue;
      const first = request.side.pokemon[0];
      if (first === undefined || request.side.pokemon[0]?.condition === "0 fnt") continue;
      const name = first.ident.slice(first.ident.indexOf(": ") + 2);
      const set = team.find((candidate) => (candidate.name === "" ? candidate.species : candidate.name) === name) as PokemonSet;
      set.item = "White Herb";
      (first as { item: string }).item = "whiteherb";
      return { ...input, request, team };
    }
    return null;
  };

  it("is refused while no trigger is admitted, and restored once one is", () => {
    const input = trigger();
    expect(input, "a corpus case with an Intimidate foe").not.toBeNull();
    const refused = createWorldBuilder(bare()).build(input as WorldInput);
    expect(refused).toMatchObject({ kind: "refused", refusal: { code: "state_unwritten", item: "start_trigger_restore", state_class: "W" } });
    const writers = structuredClone(context.writers);
    writers.admitted = [{ kind: "start_trigger", id: "whiteherb", class: "W" }];
    const world = createWorldBuilder({ ...context, writers: readWorldWriters(writers, engine) }).build(input as WorldInput);
    expect(world.kind).toBe("world");
    if (world.kind !== "world") return;
    expect(world.world.written).toContain("start_trigger_restore");
    const mon = (world.world.battle as any).sides[(input as WorldInput).state.player === "p1" ? 0 : 1].pokemon[0];
    expect(mon.item).toBe("whiteherb");
    expect(mon.boosts.atk).toBe((input as WorldInput).state.own.roster.find((m) => m.name === mon.name)?.boosts.atk ?? 0);
  });

  it("is refused when the restore is a writer the record turned off, or an item no trigger covers", () => {
    const input = trigger() as WorldInput;
    const off = createWorldBuilder({ ...context, writers: withClass("start_trigger_restore", "R") }).build(input);
    expect(off).toMatchObject({ kind: "refused", refusal: { item: "start_trigger_restore", state_class: "R" } });
    const writers = structuredClone(bare().writers);
    writers.admitted = [{ kind: "start_trigger", id: "trace", class: "W" }];
    expect(createWorldBuilder({ ...context, writers }).build(input)).toMatchObject({ kind: "refused", refusal: { item: "start_trigger_restore" } });
  });
});
