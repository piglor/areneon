import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { DEFAULT_SHOWDOWN_DIR } from "../src/engine.ts";
import {
  directEndItems,
  eatenBerries,
  eatTags,
  exportHistoryFactsData,
  externalActionSources,
  gemTags,
  goldenHistoryFactsData,
  HISTORY_FACTS_DIR,
  HistoryFactsDataError,
  historyFactsDataHash,
  historyFactsFile,
  itemUseSources,
  lastItemClearers,
  nullCantReasons,
  protectBreakers,
  readHistoryFactsData,
  slotConditions,
  stallCounter,
  stealEatTags,
} from "../src/history-facts-data.ts";
import { type FactsTable } from "../src/world-facts-data.ts";

// ADR-022 erratum E-22.5.2, slice H1: the engine data the history facts take, read from the pinned source (no rule in
// code), and the exported file the Python reducer loads.

const engine = loadEngine();
const table = JSON.parse(readFileSync(join(GOLDEN_DIR, "engine-data.json"), "utf8")) as FactsTable;
const path = join(HISTORY_FACTS_DIR, historyFactsFile(GOLDEN_FORMAT));
const data = goldenHistoryFactsData();

describe("the history facts' data of the pinned engine", () => {
  it("reads the rules the facts need from the engine's own source", () => {
    expect(data.stall_counter).toEqual({ start: 3, factor: 3, max: 729 });
    expect(data.action_sources).toEqual(["ability: Dancer", "move: Round"]);
    expect(data.external_action_sources).toEqual(["ability: Dancer"]);
    expect(data.item_use_sources).toEqual(["move: Fling"]);
    expect(data.eat_tags).toEqual(["eat", "weaken"]);
    expect(data.steal_eat_tags).toEqual(["stealeat"]);
    expect(data.gem_tags).toEqual(["gem"]);
    expect(data.direct_end_items).toEqual(["airballoon"]);
    expect(data.last_item_restorers).toEqual(["ability: Harvest", "move: Recycle"]);
    expect(data.last_item_takers).toEqual(["ability: Pickup"]);
    expect(data.null_cant_reasons).toEqual(["recharge"]);
    expect(data.protect_breaker_moves).toEqual(expect.arrayContaining(["feint"]));
    expect(data.eaten_berries).toEqual(expect.arrayContaining(["oranberry", "sitrusberry"]));
    expect(data.slot_conditions).toEqual([
      { id: "futuremove", moves: ["futuresight"], residual_turns: 3 },
      { id: "wish", moves: ["wish"], residual_turns: 2 },
    ]);
    expect(data.stall_moves).toEqual(expect.arrayContaining(["protect", "detect", "endure"]));
    expect(data.volatile_not_copied).toEqual(expect.arrayContaining(["torment"]));
    expect(data.shed_tail_copies).toEqual(["substitute"]);
  });

  it("the exported file the Python reducer loads equals a fresh export byte for byte", () => {
    const fresh = exportHistoryFactsData(engine.commit, GOLDEN_FORMAT, data);
    expect(`${JSON.stringify(fresh, null, 2)}\n`).toBe(readFileSync(path, "utf8"));
    const { history_facts_data_hash: hash, ...body } = fresh;
    expect(historyFactsDataHash(body)).toBe(hash);
  });
});

describe("a source that no longer has the shape the reader expects is a typed failure", () => {
  const tmp = mkdtempSync(join(tmpdir(), "history-facts-data-"));
  afterAll(() => rmSync(tmp, { recursive: true, force: true }));

  /** A copy of the engine's data files the reader reads, with one text replaced. */
  function doctored(name: string, file: string, from: RegExp | string, to: string, more: [string, RegExp | string, string][] = []): string {
    const dir = join(tmp, name);
    mkdirSync(join(dir, "data"), { recursive: true });
    mkdirSync(join(dir, "sim"), { recursive: true });
    for (const f of ["data/conditions.ts", "data/moves.ts", "data/items.ts", "data/abilities.ts", "sim/pokemon.ts", "sim/battle-actions.ts"]) cpSync(join(DEFAULT_SHOWDOWN_DIR, f), join(dir, f));
    for (const [f, a, b] of [[file, from, to] as [string, RegExp | string, string], ...more]) {
      const text = readFileSync(join(dir, f), "utf8");
      const changed = text.replace(a, b);
      expect(changed).not.toBe(text);
      writeFileSync(join(dir, f), changed);
    }
    return dir;
  }

  it("the stall condition's constants", () => {
    expect(() => stallCounter(doctored("stall", "data/conditions.ts", "counterMax: 729", "counterMaximum: 729"))).toThrow(/stall counterMax/);
    expect(stallCounter(DEFAULT_SHOWDOWN_DIR)).toEqual(data.stall_counter);
  });

  it("the Wish and Future Sight conditions", () => {
    expect(() => slotConditions(doctored("wish", "data/moves.ts", /getOverflowedTurnCount\(\) <= this\.effectState\.startingTurn/, "getOverflowedTurnCount() < this.effectState.startingTurn"), table)).toThrow(HistoryFactsDataError);
    expect(() => slotConditions(doctored("future", "data/conditions.ts", /getOverflowedTurnCount\(\) < this\.effectState\.endingTurn\) return/, "getOverflowedTurnCount() > this.effectState.endingTurn) return"), table)).toThrow(HistoryFactsDataError);
    expect(() => slotConditions(doctored("ending", "data/conditions.ts", "endingTurn = (this.turn - 1) + 2", "endingTurn = this.turn + 2"), table)).toThrow(/futuremove ending turn/);
    expect(slotConditions(DEFAULT_SHOWDOWN_DIR, { ...table, moves: [] })).toEqual([]);
  });

  it("the eat tag of the engine's own eat step", () => {
    expect(eatTags(doctored("eat", "sim/pokemon.ts", "'[eat]'", "'[munch]'"))).toEqual(["munch", "weaken"]);
    expect(() => eatTags(doctored("noeat", "sim/pokemon.ts", /eatItem\(force[\s\S]*?\n\t\}\n/, "eatItem(force?: boolean) {}\n"))).toThrow(/eatItem end-item tag/);
    // an item table without items names no direct end and no eaten berry
    const bare = readHistoryFactsData({ terrains: table.terrains, moves: table.moves, abilities: table.abilities }, GOLDEN_FORMAT);
    expect(bare.direct_end_items).toEqual([]);
    expect(bare.eaten_berries).toEqual([]);
  });

  it("an extractor that finds nothing in a source that has it at this pin fails closed", () => {
    const empty = /no .* found in the engine source/;
    const dancer = /externalMove: true/g;
    expect(() => externalActionSources(doctored("external", "sim/battle-actions.ts", dancer, "externalMove: false"), table)).toThrow(empty);
    expect(() => itemUseSources(doctored("itemuse", "data/moves.ts", /pokemon\.lastItem = item\.id;/g, "pokemon.lastItem = '';"), table)).toThrow(empty);
    expect(() => nullCantReasons(doctored("nullcant", "data/conditions.ts", /this\.add\('cant', pokemon, 'recharge'\)/g, "this.add('cant', pokemon, reason)"))).toThrow(HistoryFactsDataError);
    expect(() => directEndItems(doctored("direct", "data/items.ts", /target\.item = ''/g, "target.item = null"), table)).toThrow(empty);
    expect(() => eatenBerries(doctored("berries", "data/items.ts", /isBerry: true/g, "isBerry: false"), table)).toThrow(empty);
    expect(() => lastItemClearers(doctored("clearers", "data/moves.ts", /\.lastItem = '';/g, ".lastItem = null;", [["data/abilities.ts", /\.lastItem = '';/g, ".lastItem = null;"]]), table)).toThrow(empty);
    expect(() => protectBreakers(doctored("breakers", "data/moves.ts", /breaksProtect: true/g, "breaksProtect: false"), table)).toThrow(empty);
    expect(() => stealEatTags(doctored("steal", "data/moves.ts", /'\[move\] (Bug Bite|Pluck)'/g, "'[moved] $1'"))).toThrow(empty);
    expect(() => gemTags(doctored("gem", "sim/pokemon.ts", /`\[move\] \$\{this\.battle\.activeMove!\.name\}`/g, "`[moved] x`"))).toThrow(empty);
    // the one live extraction of each stays untouched
    expect(gemTags(DEFAULT_SHOWDOWN_DIR)).toEqual(["gem"]);
  });
});
