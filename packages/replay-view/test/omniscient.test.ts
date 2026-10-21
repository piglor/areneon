import { beforeAll, describe, expect, it } from "vitest";
import { loadEngine, type Engine } from "../../showdown-bridge/src/index.ts";
import { loadRandomCorpus, playRandomBattle } from "../../showdown-bridge/src/random-corpus.ts";
import { FakeDocument, FakeNode } from "./dom.ts";
import { fullDex, loadView, type Interpreted } from "./support.ts";

const { replay, view } = loadView();
const dex = fullDex();

/** The one known, labelled limit of the interpreter: an Illusion twin shown in two positions. */
const TWIN = "a Pokemon shown in two positions (an Illusion twin) is not reconciled";

const omniscient = (lines: string[]): Interpreted => replay.interpretChannel("p1", lines, dex, true);

/** A little log in the engine's omniscient form: exact HP for both sides, no requests, no split markers. */
const SMALL = [
  "|gametype|doubles",
  "|player|p1|Ann|",
  "|player|p2|Bo|",
  "|gen|9",
  "|clearpoke",
  "|poke|p1|Venusaur, L50, F|",
  "|poke|p1|Charizard, L50, F|",
  "|poke|p2|Venusaur, L50, M|",
  "|poke|p2|Charizard, L50, F|",
  "|teampreview|2",
  "|teamsize|p1|2",
  "|teamsize|p2|2",
  "|start",
  "|switch|p1a: Venusaur|Venusaur, L50, F|159/159",
  "|switch|p1b: Charizard|Charizard, L50, F|157/157",
  "|switch|p2a: Venusaur|Venusaur, L50, M|159/159",
  "|switch|p2b: Charizard|Charizard, L50, F|157/157",
  "|turn|1",
  "|move|p2b: Charizard|Flamethrower|p1a: Venusaur",
  "|-damage|p1a: Venusaur|7/159",
  "|-damage|p2a: Venusaur|0 fnt",
  "|faint|p2a: Venusaur",
  "|turn|2",
  "|win|Ann",
];

describe("the omniscient interpreter", () => {
  it("builds both rosters from the poke lines and keeps exact HP on both sides", () => {
    const run = omniscient(SMALL);
    expect(run.flags).toEqual([]);
    const state = run.frames.at(-1)!.state;
    expect(state.omniscient).toBe(true);
    expect(state.own.player).toBe("p1");
    expect(state.opponent.player).toBe("p2");
    expect(state.own.roster.map((m: any) => [m.species, m.hp_num, m.hp_den])).toEqual([
      ["venusaur", 7, 159],
      ["charizard", 157, 157],
    ]);
    expect(state.opponent.roster.map((m: any) => [m.species, m.hp_num, m.hp_den, m.fainted])).toEqual([
      ["venusaur", 0, 159, true],
      ["charizard", 157, 157, false],
    ]);
    expect(state.winner).toBe("p1");
    expect(state.turn).toBe(2);
    expect(run.requests).toEqual([]);
  });

  it("names the winner by seat, and a tie", () => {
    expect(omniscient([...SMALL.slice(0, -1), "|win|Bo"]).frames.at(-1)!.state.winner).toBe("p2");
    expect(omniscient([...SMALL.slice(0, -1), "|tie"]).frames.at(-1)!.state.winner).toBe("tie");
    expect(omniscient([...SMALL.slice(0, -1), "|win|Cy"]).flags.map((f) => f.reason)).toEqual(["'Cy' is neither player"]);
  });

  it("takes a |split| pair once: the secret line applies, the public line is dropped", () => {
    const lines = [...SMALL.slice(0, 19), "|split|p1", "|-damage|p1a: Venusaur|7/159", "|-damage|p1a: Venusaur|5/100", ...SMALL.slice(20)];
    const run = omniscient(lines);
    expect(run.flags).toEqual([]);
    const state = run.frames.at(-1)!.state;
    expect(state.own.roster[0].hp_num).toBe(7);
    expect(state.own.roster[0].hp_den).toBe(159);
    expect(run.frames.map((f) => f.kind)).toContain("split-public");
    // the same log without the pair gives the same final state
    expect(JSON.stringify(state)).toEqual(JSON.stringify(omniscient(SMALL).frames.at(-1)!.state));
    // a marker that names no side is flagged, and does not swallow the next lines
    const bad = omniscient([...SMALL.slice(0, 19), "|split|", ...SMALL.slice(19)]);
    expect(bad.flags.map((f) => f.reason)).toEqual(["a |split| marker that names no side"]);
  });

  it("refuses a request line, and still refuses a |split| in a player's own channel", () => {
    const request = JSON.stringify({ side: { id: "p1", pokemon: [] }, wait: true });
    expect(omniscient([...SMALL.slice(0, 13), `|request|${request}`]).flags.map((f) => f.reason)).toEqual([
      "a request in an omniscient log (a player's own half)",
    ]);
    const own = replay.interpretChannel("p1", ["|split|p1"], dex);
    expect(own.flags.map((f) => f.reason)).toEqual(["a |split| marker holds the opponent's secret half (refused)"]);
    expect(() => replay.createInterpreter({ player: "p2", omniscient: true } as any)).toThrow("an omniscient view is p1's");
  });

  it("takes an HP colour class on either side", () => {
    const lines = [...SMALL.slice(0, 19), "|-damage|p1a: Venusaur|79/159g", ...SMALL.slice(20)];
    const run = omniscient(lines);
    expect(run.flags).toEqual([]);
    expect(run.frames.at(-1)!.state.own.roster[0].hp_num).toBe(79);
  });

  it("the same non-omniscient reading still refuses an opponent's exact HP", () => {
    const own = replay.interpretChannel("p1", ["|player|p1|Ann|", "|player|p2|Bo|", "|gametype|doubles", "|clearpoke", "|poke|p2|Venusaur, L50, M|", "|teamsize|p2|1", "|start", "|switch|p2a: Venusaur|Venusaur, L50, M|159/159"], dex);
    expect(own.flags.map((f) => f.reason)).toEqual(["an opponent HP that is not a percentage (refused: a secret half)"]);
  });
});

let engine: Engine;
beforeAll(() => {
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

describe("the omniscient interpreter against the engine's own logs", () => {
  it("agrees with the record, and with each player's own view, on every committed random battle", async () => {
    const corpus = loadRandomCorpus();
    expect(corpus.length).toBeGreaterThan(3);
    for (const file of corpus) {
      const { id, seed, actor_seeds, teams } = file;
      const record = await playRandomBattle(engine, { id, seed, actor_seeds, teams });
      const lines = [...record.omniscient_log];
      const run = omniscient(lines);
      expect(run.flags.map((f) => f.reason).filter((r) => r !== TWIN), id).toEqual([]);
      const end = run.frames.at(-1)!.state;
      expect(end.turn, id).toBe(record.turns);
      expect(end.winner, id).toBe(record.winner ?? "tie");
      if (run.flags.length > 0) continue; // the one known limit: an Illusion twin is not reconciled
      for (const player of ["p1", "p2"] as const) {
        const mine = replay.interpretChannel(player, file.channels[player].lines, dex);
        const seen = mine.frames.at(-1)!.state;
        const omniOwn = player === "p1" ? end.own : end.opponent;
        const omniOther = player === "p1" ? end.opponent : end.own;
        for (const m of seen.own.roster) {
          const twin = omniOwn.roster.find((o: any) => o.name === m.name);
          if (twin === undefined || m.hp_num === null) continue;
          expect([twin.hp_num, twin.hp_den, twin.fainted, twin.status], `${id} ${player} ${m.name}`).toEqual([m.hp_num, m.hp_den, m.fainted, m.status]);
        }
        for (const m of seen.opponent.roster) {
          const twin = omniOther.roster.find((o: any) => o.name === m.name);
          if (twin === undefined || m.hp_num === null || twin.hp_num === null) continue;
          expect(twin.fainted, `${id} ${player} sees ${m.name}`).toBe(m.fainted);
          expect(Math.abs(Math.round((100 * twin.hp_num) / twin.hp_den) - m.hp_num), `${id} ${player} sees ${m.name}`).toBeLessThanOrEqual(1);
        }
      }
    }
  }, 120_000);
});

function omniPage(extra: Record<string, unknown> = {}) {
  const doc = new FakeDocument();
  const root = doc.createElement("div");
  const points = [
    { seat: "p1", frame: 10, kind: "preview", request_seq: 1, chosen: "team 12", refused: null, policy: "one", candidates: [{ choice: "team 12", pending: false }, { choice: "team 21", pending: false }] },
    { seat: "p2", frame: 10, kind: "preview", request_seq: 1, chosen: "team 21", refused: null, policy: "two", candidates: [{ choice: "team 21", pending: false }] },
    { seat: "p1", frame: 18, kind: "move", request_seq: 2, chosen: "move 1 1", refused: "[Unavailable choice] no", policy: "one", candidates: [{ choice: "move 1 1", pending: true }, { choice: "move 2 1", pending: false }] },
    { seat: "p1", frame: 18, kind: "move", request_seq: 3, chosen: "move 2 1", refused: null, policy: "one", candidates: [{ choice: "move 2 1", pending: false }] },
  ];
  const data = {
    title: "t",
    views: [
      {
        player: "p1", omniscient: true, label: "Omniscient log", lines: SMALL, team: [], decisions: [],
        teams: { p1: [{ species: "Venusaur", item: "x", ability: "y", moves: ["a"] }], p2: [{ name: "Nick", species: "Venusaur", item: "", ability: "y", moves: [] }] },
        seats: [{ seat: "p1", label: "p1: the candidate" }, { seat: "p2", label: "p2: the opponent" }],
        decision_points: points, result: "p1 won after 2 turns", ...extra,
      },
    ],
    dex, links: [], notes: ["omniscient"],
  };
  const page = view.init(doc, root, data, { raf: (f: any) => f(0), now: () => 0, setTimeout: () => 0, clearTimeout: () => undefined, reducedMotion: () => false });
  return { doc, root, page };
}

const texts = (node: FakeNode, cls: string) => node.byClass(cls).map((n) => n.textContent);

describe("the omniscient view", () => {
  it("is one panel with no channel selector, both teams and exact HP on both sides", () => {
    const { root, page } = omniPage();
    expect(root.byClass("viewpanel")).toHaveLength(1);
    expect(page.controls.view.parent).toBeNull();
    expect(root.find((n) => n.tagName === "SUMMARY").map((n) => n.textContent)).toContain("Both teams");
    expect(texts(root, "set")).toHaveLength(2);
    page.show(page.last);
    expect(texts(root, "hpt").some((t) => /^\d+\/\d+ \(\d+%\)$/.test(t))).toBe(true);
    expect(root.find((n) => n.tagName === "SVG" && (n.getAttribute("aria-label") ?? "").startsWith("Omniscient"))).toHaveLength(1);
    expect(texts(root, "who").join("|")).toBe("Bo (p2, exact)|Ann (p1, exact)");
    expect(texts(root, "turn")[0]).toBe("Turn 2 · finished");
  });

  it("shows both seats' decisions in force at the position, with candidates and a refusal", () => {
    const { root, page } = omniPage();
    expect(texts(root, "muted")).toContain("No decision yet.");
    page.show(10);
    let lines = texts(root, "chosen-line");
    expect(lines).toEqual(["Request 1 (preview): chose team 12  (policy one)", "Request 1 (preview): chose team 21  (policy two)"]);
    expect(root.find((n) => n.tagName === "SUMMARY").map((n) => n.textContent)).toContain("Candidates (2)");
    page.show(18);
    lines = texts(root, "chosen-line");
    expect(lines).toHaveLength(3);
    expect(lines[0]).toContain("refused by the engine: [Unavailable choice] no");
    expect(lines[1]).toBe("Request 3 (move): chose move 2 1  (policy one)");
    expect(lines[2]).toBe("Request 1 (preview): chose team 21  (policy two)");
    expect(root.byClass("pending").map((n) => n.textContent)).toEqual(["move 1 1  (pending)"]);
    expect(root.byClass("chosen")).toHaveLength(3);
  });

  it("rebuilds the decision panel only when a seat's decision changes", () => {
    const { root, page } = omniPage();
    page.show(10);
    const box = root.byClass("decisionbox")[0]!;
    const before = box.children[1];
    page.show(11);
    expect(box.children[1]).toBe(before);
    page.show(18);
    expect(box.children[1]).not.toBe(before);
    expect(root.find((n) => n.tagName === "H3").map((n) => n.textContent)).toContain("Decisions at turn 1");
  });

  it("text from the data stays text", () => {
    const hostile = "</script><img src=x onerror=alert(1)>";
    const { doc, root, page } = omniPage({ seats: [{ seat: "p1", label: hostile }, { seat: "p2", label: hostile }] });
    page.show(10);
    expect(texts(root, "slotopts").join("")).toContain(hostile);
    expect(doc.created).not.toContain("img");
    expect(doc.created).not.toContain("script");
  });
});
