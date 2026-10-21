import { createRequire } from "node:module";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { FakeDocument, FakeNode } from "./dom.ts";
import { WEB, fixtures, fullDex, loadView } from "./support.ts";

const { view } = loadView();
const dex = fullDex();
const text = (node: FakeNode) => node.textContent;

function state(extra: Record<string, unknown> = {}, mon: Record<string, unknown> = {}) {
  const base = {
    own: true, name: "Nick", base_species: "", species: null, temp_species: null, level: null, gender: null,
    shiny: false, hp_num: null, hp_den: null, status: null, fainted: false, brought: true, active_slot: "a",
    boosts: { atk: 0 }, volatiles: [], moves_revealed: [], item: null, item_known: false, item_lost: false,
    ability: null, ability_suppressed: false, mega: false, mega_stone: null, stats: null, moves_own: null,
    pp: {}, can_mega_evo: false, ...mon,
  };
  const side = (player: string, roster: unknown[]) => ({ player, username: null, teamsize: null, roster, slots: { a: roster.length ? 0 : null, b: null }, conditions: {} });
  return {
    player: "p1", format_name: null, gametype: "doubles", turn: 0, started: false, request_kind: null, weather: null,
    terrain: null, field_conditions: [], winner: null, own: side("p1", [base]), opponent: side("p2", []), labels: {}, ...extra,
  };
}

function board() {
  const doc = new FakeDocument();
  const frames: ((t: number) => void)[] = [];
  const b = view.createBoard(doc, { dex: {}, animate: () => true, raf: (f: (t: number) => void) => frames.push(f), now: () => 0 } as any, "p1");
  return { doc, b, frames };
}

describe("the board on states that are not a real battle", () => {
  it("shows a Pokemon with nothing known, an HP of zero over zero, and no types", () => {
    const { b } = board();
    b.update(state(), false);
    const n = b.svg.byClass("nm").map(text);
    expect(n[2]).toBe("Nick".length ? "?" : "");
    expect(b.svg.byClass("hpt").map(text)[2]).toBe("HP ?");
    expect(b.svg.byClass("ty-unknown").length).toBeGreaterThan(0);
    b.update(state({}, { species: "zzz", hp_num: 5, hp_den: 0, level: 5 }), false);
    expect(b.svg.byClass("lv").map(text)[2]).toBe("L5");
  });
  it("shows a Pokemon known only by its base species and a forme it changed into", () => {
    const { b } = board();
    b.update(state({}, { base_species: "venusaur", hp_num: 1, hp_den: 2, species: "venusaur", temp_species: "charizard", gender: "F" }), false);
    expect(b.svg.byClass("lv").map(text)[2]).toContain("as charizard");
    expect(b.svg.byClass("nm").map(text)[2]).toContain("♀");
  });
  it("draws weather, terrain, field and side conditions, and bench blanks when a team size is known", () => {
    const { b } = board();
    const s: any = state({ started: true, weather: "raindance", terrain: "grassyterrain", field_conditions: ["trickroom"] });
    s.opponent.conditions = { tailwind: 2 };
    s.opponent.teamsize = 4;
    s.own.teamsize = 4;
    b.update(s, false);
    expect(b.svg.byClass("badge").map(text)).toEqual(expect.arrayContaining(["raindance", "grassyterrain", "trickroom", "tailwind ×2"]));
    expect(b.svg.byClass("blank").length).toBe(4 + 3);
  });
  it("lets an earlier easing stop when a later one starts", () => {
    const { b, frames } = board();
    b.update(state({}, { species: "x", hp_num: 100, hp_den: 100 }), false);
    b.update(state({}, { species: "x", hp_num: 50, hp_den: 100 }), true);
    b.update(state({}, { species: "x", hp_num: 20, hp_den: 100 }), true);
    const widths = () => b.svg.byClass("hp").map((n) => n.getAttribute("width"));
    for (const f of frames.splice(0)) f(100000);
    const final = widths();
    expect(final[2]).toBe("22");
    // an empty slot clears its bar and token
    b.update(state(), false);
  });
  it("clears a slot a Pokemon has left", () => {
    const { b } = board();
    b.update(state({}, { species: "x" }), false);
    const empty: any = state();
    empty.own.slots = { a: null, b: null };
    b.update(empty, false);
    expect(b.svg.byClass("slot").some((n) => n.getAttribute("class") === "slot empty")).toBe(true);
  });
});

describe("the decision panel on every shape of request", () => {
  const doc = new FakeDocument();
  const side = (n: number) => ({ pokemon: Array.from({ length: n }, (_, i) => ({ ident: `p1: M${i}`, condition: "10/10", active: i < 2 })) });
  const panel = (request: unknown, decision: unknown = null) => text(view.decisionPanel(doc, request, decision as any, {}));

  it("handles no request, a wait and a forced switch", () => {
    expect(panel(null)).toContain("No request yet.");
    expect(panel({ wait: true, side: side(2) })).toContain("Waiting for the other player");
    const forced = panel({ forceSwitch: [true, false], side: side(4) }, { choice: "switch 3, pass", request_seq: 1, policy: "p" });
    expect(forced).toContain("Must switch.");
    expect(forced).toContain("Switch to M2");
    expect(forced).toContain("Pass");
  });
  it("handles a Team Preview with and without a size or a chosen order", () => {
    expect(panel({ teamPreview: true, side: side(3) })).toContain("bring ?");
    const chosen = panel({ teamPreview: true, maxChosenTeamSize: 2, side: side(3) }, { choice: "team 31", request_seq: 1, policy: "p" });
    expect(chosen).toContain("slot 1");
    expect(chosen).toContain("bring 2");
  });
  it("handles moves without a count, disabled moves, a missing slot and a choice it cannot read", () => {
    const request = { active: [{ moves: [{ move: "Splash", disabled: true }] }, { moves: [] }, { moves: [] }], side: side(1) };
    const out = panel(request, { choice: "move 1 mega, switch 9, rest", request_seq: 1, policy: "p" });
    expect(out).toContain("Splash disabled");
    expect(out).toContain("mega");
    expect(out).toContain("Switch to #9");
    expect(out).toContain("Choice: rest");
    expect(out).toContain("Slot c: ?");
    expect(panel({ active: [{ moves: [{ move: "X", pp: 1, maxpp: 2 }] }], side: side(2) }, { choice: "default", request_seq: 1, policy: "p" })).toContain("Choice: default");
    expect(panel({ active: [], forceSwitch: [], side: { pokemon: [] } })).not.toContain("Slot");
    expect(panel({ active: [{ moves: [] }], side: {} })).toContain("Slot a: ?");
  });
});

describe("building the replay with little data and with defaults", () => {
  const channel = fixtures().find((f) => f.id === "golden/p1")!;
  const data = (views: unknown[]) => ({ views, dex: undefined as unknown });
  it("works with no options, no dex, no notes, no team and a set with no nickname", () => {
    const doc = new FakeDocument();
    const root = doc.createElement("div");
    const page = view.init(doc, root, data([{ player: "p1", label: "x", lines: channel.lines, decisions: [], result: "r", team: [{ name: "Venusaur", species: "Venusaur", moves: [] }] }]) as any);
    page.next();
    page.play();
    page.pause();
    page.show(page.last);
    expect(text(root.byClass("set")[0]!)).toContain("no item");
    const bare = view.init(new FakeDocument(), new FakeDocument().createElement("div"), data([{ player: "p1", label: "x", lines: [], decisions: [], result: "r" }]) as any);
    expect(bare.last).toBe(0);
  });
  it("makes elements with null children", () => {
    const doc = new FakeDocument();
    expect(view.make(doc, "p", null, [null, "x"]).textContent).toBe("x");
  });
});

describe("the keyboard and the timer", () => {
  function page() {
    const doc = new FakeDocument();
    const root = doc.createElement("div");
    const timers: (() => void)[] = [];
    const f = fixtures().find((x) => x.id === "golden/p1")!;
    const p = view.init(doc, root, { title: "t", views: [{ player: "p1", label: "p1", lines: f.lines, team: [], decisions: [], result: "r" }], dex, links: [], notes: [] }, {
      setTimeout: (fn: () => void) => timers.push(fn), clearTimeout: () => undefined,
    });
    return { doc, p, timers };
  }
  it("steps with the arrow keys and plays with the space bar", () => {
    const { doc, p, timers } = page();
    doc.fire("keydown", { key: "ArrowRight" });
    const a = p.position();
    expect(a).toBeGreaterThan(0);
    doc.fire("keydown", { key: "ArrowLeft" });
    expect(p.position()).toBeLessThan(a);
    let prevented = 0;
    doc.fire("keydown", { key: " ", target: { tagName: "DIV" }, preventDefault: () => prevented++ });
    expect(prevented).toBe(1);
    expect(timers).toHaveLength(1);
    doc.fire("keydown", { key: " ", target: { tagName: "DIV" } }); // pause again, no preventDefault on the event
    doc.fire("keydown", { key: " ", target: { tagName: "BUTTON" } }); // a focused button keeps its own space bar
    doc.fire("keydown", { key: " ", target: null });
    doc.fire("keydown", { key: "x" });
    expect(timers).toHaveLength(1);
  });
  it("ignores a timer that fires after a pause", () => {
    const { p, timers } = page();
    p.play();
    const stale = timers[0]!;
    p.pause();
    const at = p.position();
    stale();
    expect(p.position()).toBe(at);
  });
});

describe("aligning two channels that do not match", () => {
  it("pairs private lines of one kind, skips the other's, and falls back to a nearby match", () => {
    expect(view.align(["|turn|1", "|request|{}"], ["|turn|1"])).toEqual([0, 1, 1]);
    expect(view.align(["|request|{}", "|turn|1"], ["|request|{}", "|error|x", "|turn|1"])).toEqual([0, 1, 3]);
    // the other channel has an extra line first: found within a few lines
    expect(view.align(["|move|p1a: A|B"], ["|-damage|p1a: A|1/2", "|-damage|p1a: A|1/2", "|move|p1a: A|B"])).toEqual([0, 3]);
    // no match anywhere: the other channel stays where it was, and the end goes to the end
    expect(view.align(["|move|p1a: A|B", "|turn|2"], ["|turn|1"])).toEqual([0, 0, 1]);
    expect(view.align([], [])).toEqual([0]);
  });
});

describe("booting with and without a document and a window", () => {
  it("takes a window with no location or matchMedia", () => {
    const f = fixtures().find((x) => x.id === "golden/p1")!;
    const doc = new FakeDocument();
    const block = doc.createElement("script");
    block.textContent = JSON.stringify({ title: "t", views: [{ player: "p1", label: "p", lines: f.lines, team: [], decisions: [], result: "r" }], dex, links: [], notes: [] });
    doc.ids.set("replay-data", block);
    doc.ids.set("replay-root", doc.createElement("div"));
    const win: Record<string, unknown> = { requestAnimationFrame: (fn: (t: number) => void) => fn(0), performance: { now: () => 0 }, setTimeout: () => 1, clearTimeout: () => undefined };
    const page = view.boot(doc, win);
    expect(page.position()).toBe(0);
    page.next(); // the easing runs through the window's frame callback and clock
    expect(view.boot(doc, { ...win, location: { hash: "#f=2" } }).position()).toBe(2);
  });

  it("boots itself when the page has a document and a window", () => {
    const require = createRequire(import.meta.url);
    const path = join(WEB, "view.cjs");
    const g = globalThis as Record<string, unknown>;
    const seen: string[] = [];
    g.document = { getElementById: (id: string) => { seen.push(id); return null; } };
    g.window = {};
    delete require.cache[path];
    try {
      require(path);
    } finally {
      delete g.document;
      delete g.window;
      delete require.cache[path];
      require(path);
    }
    expect(seen).toEqual(["replay-data", "replay-root"]);
  });
});
