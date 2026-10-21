import { describe, expect, it } from "vitest";
import { FakeDocument, FakeNode } from "./dom.ts";
import { fixtures, fullDex, loadView } from "./support.ts";

const { view } = loadView();
const dex = fullDex();

function pageFor(ids: string[], decisions: Record<string, any[]> = {}) {
  const all = fixtures();
  const views = ids.map((id) => {
    const f = all.find((x) => x.id === id)!;
    return { player: f.player, label: `${f.player}'s own channel`, lines: f.lines, team: f.team as any[], decisions: decisions[id] ?? [], result: "tested" };
  });
  const doc = new FakeDocument();
  const root = doc.createElement("div");
  const timers: { fn: () => void; ms: number }[] = [];
  const frames: ((t: number) => void)[] = [];
  let clock = 0;
  let reduced = false;
  const page = view.init(doc, root, { title: "t", views, dex, links: [], notes: [] }, {
    raf: (f: (t: number) => void) => { frames.push(f); },
    now: () => clock,
    setTimeout: (fn: () => void, ms: number) => { timers.push({ fn, ms }); return timers.length; },
    clearTimeout: () => { timers.length = 0; },
    reducedMotion: () => reduced,
  });
  return {
    doc, root, page, timers, frames,
    setReduced: (v: boolean) => { reduced = v; },
    runFrames: (t: number) => { clock = t; const f = frames.splice(0); f.forEach((g) => g(t)); },
  };
}

const texts = (node: FakeNode, cls: string) => node.byClass(cls).map((n) => n.textContent);
const widths = (node: FakeNode) => node.byClass("hp").map((n) => Number(n.getAttribute("width")));

describe("the replay view", () => {
  it("builds from two own channels, with controls and one panel each", () => {
    const { root, page } = pageFor(["golden/p1", "golden/p2"]);
    expect(page.views).toHaveLength(2);
    expect(root.byClass("viewpanel")).toHaveLength(2);
    expect(page.last).toBeGreaterThan(50);
    expect(page.controls.slider.getAttribute("max")).toBe(String(page.last));
  });

  it("shows the Team Preview, then the actives, then the result", () => {
    const { root, page } = pageFor(["golden/p1"]);
    expect(texts(root, "turn")[0]).toBe("Team Preview");
    page.show(page.last);
    expect(texts(root, "turn")[0]).toMatch(/^Turn \d+ · finished$/);
    expect(texts(root, "caption")[0]).toMatch(/won the battle|tie/);
  });

  it("an own Pokemon shows exact HP and the opponent a percentage only", () => {
    const { root, page } = pageFor(["golden/p1"]);
    const turn1 = page.turns[0].frame;
    page.show(turn1);
    const hp = texts(root, "hpt");
    expect(hp[0]).toMatch(/^\d+%$/); // opponent, top row
    expect(hp[2]).toMatch(/^\d+\/\d+ \(\d+%\)$/); // own, bottom row
  });

  it("steps by event, jumps to a turn and scrubs", () => {
    const { page } = pageFor(["golden/p1"]);
    page.next();
    const first = page.position();
    expect(first).toBeGreaterThan(0);
    page.next();
    expect(page.position()).toBeGreaterThan(first);
    page.prev();
    expect(page.position()).toBe(first);
    const jump = page.controls.jump;
    const target = page.turns[1];
    jump.value = String(target.frame);
    jump.fire("change");
    expect(page.position()).toBe(target.frame);
    page.controls.slider.value = "3";
    page.controls.slider.fire("input");
    expect(page.position()).toBe(3);
    page.controls.toEnd.fire("click");
    expect(page.position()).toBe(page.last);
    page.controls.toStart.fire("click");
    expect(page.position()).toBe(0);
  });

  it("plays on a timer, at the chosen speed, and stops at the end", () => {
    const t = pageFor(["golden/p1"]);
    t.page.controls.play.fire("click");
    expect(t.timers[0]!.ms).toBe(1100);
    t.page.controls.speed.value = "4";
    t.page.controls.speed.fire("change");
    let guard = 0;
    while (t.timers.length > 0 && guard++ < 10000) t.timers.pop()!.fn();
    expect(t.page.position()).toBe(t.page.last);
    expect(t.page.controls.play.textContent).toBe("▶");
    t.page.controls.play.fire("click"); // from the end: starts again from the start
    expect(t.page.position()).toBe(0);
    t.page.controls.play.fire("click"); // pause
    expect(t.timers).toHaveLength(0);
  });

  it("eases an HP bar between states and ends exactly on the state", () => {
    const t = pageFor(["golden/p1"]);
    const d = t.page.runs[0].frames.findIndex((f: any) => f.kind === "-damage");
    t.page.show(d - 1);
    const before = widths(t.root);
    t.frames.length = 0;
    t.page.controls.fwd.fire("click");
    expect(t.page.position()).toBe(d);
    expect(t.frames.length).toBeGreaterThan(0);
    t.runFrames(200); // the start time is 0 and the easing takes 450 ms
    const mid = widths(t.root);
    t.runFrames(100000);
    const end = widths(t.root);
    expect(mid.join()).not.toBe(end.join());
    expect(before.join()).not.toBe(end.join());
    // the end of the easing equals the state's own value: jumping straight there shows the same widths
    const direct = pageFor(["golden/p1"]);
    direct.page.show(d);
    expect(widths(direct.root)).toEqual(end);
  });

  it("does not ease when the toggle is off or the reader prefers reduced motion", () => {
    for (const mode of ["toggle", "reduced"]) {
      const t = pageFor(["golden/p1"]);
      const d = t.page.runs[0].frames.findIndex((f: any) => f.kind === "-damage");
      if (mode === "toggle") {
        t.page.controls.anim.checked = false;
        t.page.controls.anim.fire("change");
      } else {
        t.setReduced(true);
      }
      t.page.show(d - 1);
      t.frames.length = 0;
      t.page.controls.fwd.fire("click");
      expect(t.frames).toHaveLength(0);
      const direct = pageFor(["golden/p1"]);
      direct.page.show(d);
      expect(widths(t.root)).toEqual(widths(direct.root));
    }
  });

  it("keeps two channels at the same moment", () => {
    const { page } = pageFor(["random/b00/p1", "random/b00/p2"]);
    const [a, b] = page.runs;
    for (const at of [0, 40, 100, 200, page.last]) {
      page.show(at);
      const turnA = a.frames[at].state.turn;
      const j = page.maps[1][at];
      expect(b.frames[Math.min(j, b.frames.length - 1)].state.turn).toBe(turnA);
    }
    expect(page.maps[1][page.last]).toBe(b.frames.length - 1);
  });

  it("selects which channel to show", () => {
    const { root, page } = pageFor(["golden/p1", "golden/p2"]);
    page.controls.view.value = "1";
    page.controls.view.fire("change");
    expect(root.byClass("viewpanel").map((p) => p.hidden)).toEqual([true, false]);
    page.controls.view.value = "all";
    page.controls.view.fire("change");
    expect(root.byClass("viewpanel").map((p) => p.hidden)).toEqual([false, false]);
  });

  it("shows the request's options with the chosen one marked, and the policy", () => {
    const all = fixtures().find((f) => f.id === "golden/p1")!;
    const seq = all.lines.map((l, i) => (l.startsWith("|request|") ? i : -1)).filter((i) => i >= 0);
    const decisions = [
      { choice: "team 1234", request_seq: 1, policy: "random-legal@v1" },
      { choice: "move 2 1, switch 3", request_seq: 2, policy: "random-legal@v1" },
    ];
    const t = pageFor(["golden/p1"], { "golden/p1": decisions });
    t.page.show(seq[1]! + 1);
    const chosen = texts(t.root, "chosen-line");
    expect(chosen[0]).toBe("Chosen: move 2 1, switch 3  (policy random-legal@v1)");
    const marked = t.root.byClass("chosen").map((n) => n.textContent);
    expect(marked.length).toBeGreaterThanOrEqual(2);
    expect(marked[0]).toMatch(/→ target 1/);
    t.page.show(seq[0]! + 1);
    expect(texts(t.root, "chosen-line")[0]).toMatch(/^Chosen: team 1234/);
    expect(t.root.byClass("chosen").length).toBeGreaterThan(0);
    t.page.show(0);
    expect(texts(t.root, "decisionbox")[0]).toContain("No request yet.");
  });

  it("shows a request with no recorded decision, a wait and a forced switch", () => {
    const t = pageFor(["unavailable/b00/p1"]);
    const reqs = t.page.runs[0].requests;
    expect(reqs.length).toBeGreaterThan(3);
    const frames = t.page.runs[0].frames;
    const kinds = new Set<string>();
    frames.forEach((f: any, i: number) => {
      if (f.kind === "request") {
        t.page.show(i);
        const text = t.root.byClass("decision")[0]!.textContent;
        const r = reqs[f.request - 1];
        if (r.wait) expect(text).toContain("Waiting for the other player");
        else expect(text).toContain("No decision is recorded");
        kinds.add(r.wait ? "wait" : r.teamPreview ? "preview" : r.forceSwitch ? "force" : "move");
      }
    });
    expect(kinds.has("move")).toBe(true);
  });

  it("parses a recorded choice", () => {
    expect(view.parseChoice("move 2 1, switch 3")).toEqual([
      { kind: "move", index: 2, target: "1", extra: "", raw: "move 2 1" },
      { kind: "switch", index: 3, target: "", extra: "", raw: "switch 3" },
    ]);
    expect(view.parseChoice("move 1 -2 mega")[0]).toMatchObject({ kind: "move", index: 1, target: "-2", extra: "mega" });
    expect(view.parseChoice("move 1 mega")[0]).toMatchObject({ kind: "move", target: "", extra: "mega" });
    expect(view.parseChoice("team 3145")[0]).toMatchObject({ kind: "team", target: "3145" });
    expect(view.parseChoice("pass, default")).toMatchObject([{ kind: "pass" }, { kind: "other" }]);
  });

  it("refuses to build a style or event-handler attribute", () => {
    const doc = new FakeDocument();
    expect(() => view.make(doc, "div", { style: "color:red" })).toThrow();
    expect(() => view.make(doc, "div", { onclick: "x()" })).toThrow();
    expect(() => view.make(doc, "div", { onError: "x()" })).toThrow();
    expect(view.make(doc, "div", { class: "ok" }, ["text"]).textContent).toBe("text");
  });

  it("refuses a replay with no view", () => {
    const doc = new FakeDocument();
    expect(() => view.init(doc, doc.createElement("div"), { title: "x", views: [], dex: {}, links: [], notes: [] })).toThrow();
  });

  it("boots from the page's one data block and opens at a frame given in the link", () => {
    const f = fixtures().find((x) => x.id === "golden/p1")!;
    const data = { title: "t", views: [{ player: "p1", label: "p1", lines: f.lines, team: [], decisions: [], result: "r" }], dex, links: [], notes: [] };
    const make = (hash: string) => {
      const doc = new FakeDocument();
      const block = doc.createElement("script");
      block.textContent = JSON.stringify(data);
      const root = doc.createElement("div");
      doc.ids.set("replay-data", block);
      doc.ids.set("replay-root", root);
      const timers: (() => void)[] = [];
      const win = {
        location: { hash },
        requestAnimationFrame: (fn: (t: number) => void) => fn(0),
        performance: { now: () => 0 },
        setTimeout: (fn: () => void) => timers.push(fn),
        clearTimeout: () => undefined,
        matchMedia: () => ({ matches: true }),
      };
      return { page: view.boot(doc, win), doc, root, win };
    };
    expect(make("").page.position()).toBe(0);
    expect(make("#f=12").page.position()).toBe(12);
    expect(make("#f=zz").page.position()).toBe(0);
    const booted = make("#f=3");
    booted.page.next(); // reduced motion is on in this window: the easing is skipped
    expect(booted.page.position()).toBeGreaterThan(3);
    booted.page.play();
    booted.page.pause();
    expect(view.boot(new FakeDocument(), {})).toBeNull();
  });

  it("keeps the jump menu on the turn the position is in", () => {
    const { page } = pageFor(["golden/p1"]);
    const t = page.turns[2];
    page.show(t.frame + 1);
    expect(page.controls.jump.value).toBe(String(t.frame));
    page.show(0);
    expect(page.controls.jump.value).toBe("0");
  });

  it("shows every kind of request the corpus has", () => {
    const kinds = new Set<string>();
    for (const id of ["unavailable/b00/p1", "unavailable/b00/p2", "unavailable/b01/p1", "random/b00/p1", "random/b03/p2"]) {
      const t = pageFor([id]);
      for (const r of t.page.runs[0].requests) kinds.add(r.wait ? "wait" : r.teamPreview ? "preview" : r.forceSwitch ? "force" : "move");
      t.page.runs[0].frames.forEach((fr: any, i: number) => { if (fr.kind === "request") t.page.show(i); });
    }
    expect([...kinds].sort()).toEqual(["force", "move", "preview", "wait"]);
  });
});
