import { describe, expect, it } from "vitest";
import { fixtures, fullDex, loadInterpreter } from "./support.ts";

const api = loadInterpreter();
const dex = fullDex();

const SPECIES = ["Venusaur", "Charizard", "Blastoise", "Pikachu", "Snorlax", "Gengar"];
const request = (names: string[], extra: Record<string, unknown> = {}) =>
  "|request|" +
  JSON.stringify({
    ...extra,
    side: {
      name: "Alice",
      id: "p1",
      pokemon: names.map((n, i) => ({
        ident: `p1: ${n}`,
        details: `${n}, L50, F`,
        condition: "150/150",
        active: i < 2,
        stats: { atk: 1, def: 2, spa: 3, spd: 4, spe: 5 },
        moves: ["tackle", "protect"],
        baseAbility: "overgrow",
        item: "leftovers",
        ability: "overgrow",
      })),
    },
  });
const HEADER = [
  "|gametype|doubles",
  "|player|p1|Alice||",
  "|player|p2|Bob||",
  "|tier|[Gen 9 Champions] VGC 2026 Reg M-C",
  "|clearpoke",
  ...SPECIES.map((n) => `|poke|p1|${n}, L50, F|`),
  ...SPECIES.map((n) => `|poke|p2|${n}, L50, M|`),
  "|teampreview|4",
  request(SPECIES, { teamPreview: true, maxChosenTeamSize: 4 }),
  "|teamsize|p1|4",
  "|teamsize|p2|4",
  "|start",
  "|switch|p1a: Venusaur|Venusaur, L50, F|150/150",
  "|switch|p1b: Charizard|Charizard, L50, F|150/150",
  "|switch|p2a: Venusaur|Venusaur, L50, M|100/100",
  "|switch|p2b: Charizard|Charizard, L50, F|100/100",
  "|turn|1",
];

function after(...lines: string[]) {
  const run = api.interpretChannel("p1", [...HEADER, ...lines], dex);
  return { run, state: run.frames.at(-1)!.state, flags: run.flags };
}
const opp = (s: any, name: string) => s.opponent.roster.find((m: any) => m.name === name);
const own = (s: any, name: string) => s.own.roster.find((m: any) => m.name === name);

describe("the header and the board it sets up", () => {
  it("reads both sides, the format and the actives", () => {
    const { state, flags } = after();
    expect(flags).toEqual([]);
    expect(state.format_name).toBe("[Gen 9 Champions] VGC 2026 Reg M-C");
    expect(state.own.username).toBe("Alice");
    expect(state.opponent.username).toBe("Bob");
    expect(state.own.roster).toHaveLength(6);
    expect(state.opponent.roster).toHaveLength(6);
    expect(state.own.slots).toEqual({ a: 0, b: 1 });
    expect(state.turn).toBe(1);
    expect(opp(state, "Venusaur")).toMatchObject({ hp_num: 100, hp_den: 100, active_slot: "a", brought: true });
    expect(own(state, "Venusaur")).toMatchObject({ hp_num: 150, hp_den: 150, own: true, item: "leftovers" });
  });
  it("hides the opponent's hidden forme and keeps blank species", () => {
    const run = api.interpretChannel("p1", ["|gametype|doubles", "|player|p1|A||", "|player|p2|B||", "|clearpoke", "|poke|p2|Urshifu-*, L50|"], dex);
    expect(run.flags).toEqual([]);
    expect(run.frames.at(-1)!.state.opponent.roster[0]).toMatchObject({ species: null });
  });
  it("sets singles and triples slots", () => {
    expect(api.interpretChannel("p1", ["|gametype|singles"]).frames.at(-1)!.state.own.slots).toEqual({ a: null });
    expect(api.interpretChannel("p1", ["|gametype|triples"]).frames.at(-1)!.state.opponent.slots).toEqual({ a: null, b: null, c: null });
    expect(api.interpretChannel("p1", ["|gametype|freeforall"]).flags).toHaveLength(1);
  });
});

describe("hit points, status and the winner", () => {
  it("damages, heals and faints", () => {
    const { state } = after(
      "|-damage|p2a: Venusaur|40/100",
      "|-heal|p1a: Venusaur|120/150|[from] item: Leftovers",
      "|-damage|p2b: Charizard|50/100g",
      "|faint|p2b: Charizard",
    );
    expect(opp(state, "Venusaur")).toMatchObject({ hp_num: 40, hp_den: 100 });
    expect(own(state, "Venusaur")).toMatchObject({ hp_num: 120, item: "leftovers", item_known: true });
    expect(opp(state, "Charizard")).toMatchObject({ fainted: true, hp_num: 0, active_slot: null });
    expect(state.opponent.slots).toEqual({ a: 0, b: null });
  });
  it("applies a status with its HP, cures and sets an HP without touching the status", () => {
    let { state } = after("|-status|p2a: Venusaur|par", "|-damage|p2a: Venusaur|70/100 par");
    expect(opp(state, "Venusaur").status).toBe("par");
    ({ state } = after("|-status|p2a: Venusaur|brn", "|-sethp|p2a: Venusaur|10/100"));
    expect(opp(state, "Venusaur")).toMatchObject({ status: "brn", hp_num: 10 });
    ({ state } = after("|-status|p2a: Venusaur|brn", "|-curestatus|p2a: Venusaur|brn"));
    expect(opp(state, "Venusaur").status).toBeNull();
    ({ state } = after("|-status|p2a: Venusaur|psn", "|-status|p2b: Charizard|psn", "|-cureteam|p2: Bob"));
    expect(state.opponent.roster.every((m: any) => m.status === null)).toBe(true);
  });
  it("reads a fainted HP without a denominator", () => {
    expect(opp(after("|-damage|p2a: Venusaur|0 fnt").state, "Venusaur")).toMatchObject({ fainted: true, hp_num: 0 });
  });
  it("finds the winner by name, and a tie", () => {
    expect(after("|win|Alice").state.winner).toBe("own");
    expect(after("|win|Bob").state.winner).toBe("opponent");
    expect(after("|tie").state.winner).toBe("tie");
    expect(after("|win|Carol").flags).toHaveLength(1);
  });
});

describe("boosts", () => {
  it("boosts, drops, sets, swaps, inverts, clears and copies", () => {
    let { state } = after("|-boost|p2a: Venusaur|atk|2", "|-unboost|p2a: Venusaur|spe|1", "|-setboost|p2b: Charizard|spa|6");
    expect(opp(state, "Venusaur").boosts).toMatchObject({ atk: 2, spe: -1 });
    expect(opp(state, "Charizard").boosts.spa).toBe(6);
    ({ state } = after("|-boost|p2a: Venusaur|atk|2", "|-swapboost|p2a: Venusaur|p2b: Charizard|atk, def"));
    expect(opp(state, "Charizard").boosts.atk).toBe(2);
    expect(opp(state, "Venusaur").boosts.atk).toBe(0);
    ({ state } = after("|-boost|p2a: Venusaur|atk|2", "|-invertboost|p2a: Venusaur"));
    expect(opp(state, "Venusaur").boosts.atk).toBe(-2);
    ({ state } = after("|-boost|p2a: Venusaur|atk|2", "|-boost|p2a: Venusaur|def|1", "|-clearnegativeboost|p2a: Venusaur"));
    expect(opp(state, "Venusaur").boosts).toMatchObject({ atk: 2, def: 1 });
    ({ state } = after("|-unboost|p2a: Venusaur|atk|2", "|-clearpositiveboost|p2a: Venusaur|p2b: Charizard|move: Haze"));
    expect(opp(state, "Venusaur").boosts.atk).toBe(-2);
    ({ state } = after("|-boost|p2a: Venusaur|atk|2", "|-clearboost|p2a: Venusaur"));
    expect(opp(state, "Venusaur").boosts.atk).toBe(0);
    ({ state } = after("|-boost|p2a: Venusaur|atk|2", "|-clearallboost"));
    expect(opp(state, "Venusaur").boosts.atk).toBe(0);
    ({ state } = after("|-boost|p2a: Venusaur|atk|3", "|-copyboost|p2b: Charizard|p2a: Venusaur|[from] move: Psych Up"));
    expect(opp(state, "Charizard").boosts.atk).toBe(3);
  });
  it("flags a boost out of range or of an unknown stat", () => {
    expect(after("|-setboost|p2a: Venusaur|atk|7").flags).toHaveLength(1);
    expect(after("|-boost|p2a: Venusaur|luck|1").flags).toHaveLength(1);
    expect(after("|-swapboost|p2a: Venusaur|p2b: Charizard|luck").flags).toHaveLength(1);
  });
  it("a switch out clears boosts, volatiles and a temporary forme", () => {
    const { state } = after(
      "|-boost|p2a: Venusaur|atk|2",
      "|-start|p2a: Venusaur|Substitute",
      "|-formechange|p2a: Venusaur|Charizard-Mega-X",
      "|switch|p2a: Pikachu|Pikachu, L50, M|100/100",
    );
    expect(opp(state, "Venusaur")).toMatchObject({ active_slot: null, volatiles: [], temp_species: null });
    expect(opp(state, "Venusaur").boosts.atk).toBe(0);
    expect(opp(state, "Pikachu").active_slot).toBe("a");
  });
});

describe("the field", () => {
  it("weather, terrain, field and side conditions", () => {
    let { state } = after("|-weather|RainDance", "|-fieldstart|move: Electric Terrain", "|-fieldstart|move: Trick Room", "|-sidestart|p2: Bob|move: Tailwind", "|-sidestart|p2: Bob|move: Tailwind");
    expect(state.weather).toBe("raindance");
    expect(state.terrain).toBe("electricterrain");
    expect(state.field_conditions).toEqual(["trickroom"]);
    expect(state.opponent.conditions).toEqual({ tailwind: 2 });
    expect(state.labels.raindance).toBe("RainDance");
    ({ state } = after("|-weather|RainDance", "|-weather|none", "|-fieldstart|move: Grassy Terrain", "|-fieldend|move: Grassy Terrain", "|-fieldstart|move: Trick Room", "|-fieldend|move: Trick Room", "|-sidestart|p2: Bob|Reflect", "|-sideend|p2: Bob|Reflect"));
    expect(state.weather).toBeNull();
    expect(state.terrain).toBeNull();
    expect(state.field_conditions).toEqual([]);
    expect(state.opponent.conditions).toEqual({});
    ({ state } = after("|-sidestart|p1: Alice|move: Tailwind", "|-swapsideconditions|"));
    expect(state.own.conditions).toEqual({});
    expect(state.opponent.conditions).toEqual({ tailwind: 1 });
    ({ state } = after("|-fieldstart|move: Electric Terrain", "|-fieldend|move: Psychic Terrain"));
    expect(state.terrain).toBe("electricterrain");
  });
  it("flags a side condition for a side that does not exist or is not a side", () => {
    expect(after("|-sidestart|p3: X|move: Tailwind").flags).toHaveLength(1);
    expect(after("|-sidestart|nobody|move: Tailwind").flags).toHaveLength(1);
  });
});

describe("volatiles, items, abilities and formes", () => {
  it("starts and ends a volatile, and reads an ability one", () => {
    const { state } = after("|-start|p2a: Venusaur|Substitute", "|-start|p2a: Venusaur|ability: Flash Fire", "|-start|p2b: Charizard|confusion", "|-end|p2b: Charizard|confusion");
    expect(opp(state, "Venusaur")).toMatchObject({ volatiles: ["flashfire", "substitute"], ability: "flashfire" });
    expect(opp(state, "Charizard").volatiles).toEqual([]);
    expect(after("|-start|p2a: Venusaur|<b>x</b>").flags).toHaveLength(1);
    expect(after("|-end|p2a: Venusaur|" + "x".repeat(70)).flags).toHaveLength(1);
  });
  it("an item or ability received from another Pokemon or effect is flagged, and the flag is sticky (ADR-025, N6)", () => {
    let { state } = after("|-item|p2a: Venusaur|Leftovers");
    expect(opp(state, "Venusaur")).toMatchObject({ item: "leftovers", item_received: false, ability_received: false });
    ({ state } = after("|-item|p2a: Venusaur|Leftovers|[from] move: Trick"));
    expect(opp(state, "Venusaur")).toMatchObject({ item: "leftovers", item_received: true });
    ({ state } = after("|-item|p2a: Venusaur|Leftovers|[from] move: Trick", "|-enditem|p2a: Venusaur|Leftovers"));
    expect(opp(state, "Venusaur")).toMatchObject({ item: null, item_lost: true, item_received: true });
    ({ state } = after("|-item|p2a: Venusaur|Leftovers|[from] ability: Frisk|[of] p1a: Venusaur|[identify]"));
    expect(opp(state, "Venusaur").item_received).toBe(false);
    ({ state } = after("|-item|p2a: Venusaur|Leftovers|[from] ability: Frisk|[of] p1b: Charizard|[identify]"));
    expect(opp(state, "Venusaur")).toMatchObject({ item: "leftovers", ability: null, ability_received: false });
    expect(own(state, "Charizard").ability).toBe("frisk");
    ({ state } = after("|-item|p1a: Venusaur|Leftovers|[from] ability: Frisk|[of] p2b: Charizard|[identify]"));
    expect(opp(state, "Charizard")).toMatchObject({ ability: "frisk", item: null, ability_received: false });
    expect(own(state, "Venusaur").ability).not.toBe("frisk");
    ({ state } = after("|-ability|p2a: Venusaur|Mummy|[from] ability: Mummy|[of] p2b: Charizard"));
    expect(opp(state, "Venusaur")).toMatchObject({ ability: "mummy", ability_received: true });
    expect(opp(state, "Charizard").ability_received).toBe(false);
    ({ state } = after("|-ability|p2a: Venusaur|Intimidate|boost"));
    expect(opp(state, "Venusaur").ability_received).toBe(false);
  });
  it("items and abilities: revealed, changed and lost", () => {
    let { state } = after("|-item|p2a: Venusaur|Leftovers", "|-ability|p2b: Charizard|Intimidate|boost");
    expect(opp(state, "Venusaur")).toMatchObject({ item: "leftovers", item_known: true });
    expect(opp(state, "Charizard").ability).toBe("intimidate");
    ({ state } = after("|-item|p2a: Venusaur|Leftovers", "|-enditem|p2a: Venusaur|Leftovers"));
    expect(opp(state, "Venusaur")).toMatchObject({ item: null, item_lost: true });
    // A Fling whose item was already gone names none (#1162); any other empty name is flagged.
    ({ state } = after("|-item|p2a: Venusaur|Leftovers", "|-enditem|p2a: Venusaur||[from] move: Fling"));
    expect(opp(state, "Venusaur")).toMatchObject({ item: null, item_lost: true });
    expect(after("|-enditem|p2a: Venusaur|").flags).toHaveLength(1);
    expect(after("|-enditem|p2a: Venusaur||[from] move: Knock Off").flags).toHaveLength(1);
    ({ state } = after("|-ability|p2a: Venusaur|Trace|[from] ability: Trace|[of] p2b: Charizard"));
    expect(opp(state, "Venusaur").ability).toBe("trace");
    ({ state } = after("|-ability|p2a: Venusaur|Intimidate|[from] ability: Trace|[of] p2b: Charizard"));
    expect(opp(state, "Charizard").ability).toBe("intimidate");
    ({ state } = after("|-endability|p2a: Venusaur"));
    expect(opp(state, "Venusaur").ability_suppressed).toBe(true);
    ({ state } = after("|-activate|p2a: Venusaur|move: Skill Swap|[of] p2b: Charizard"));
    expect(opp(state, "Venusaur").ability).toBeNull();
    ({ state } = after("|-status|p2a: Venusaur|brn|[from] item: Flame Orb"));
    expect(opp(state, "Venusaur").item).toBe("flameorb");
    ({ state } = after("|-damage|p2a: Venusaur|90/100|[from] ability: Rough Skin|[of] p1a: Venusaur"));
    expect(opp(state, "Venusaur").ability).toBeNull(); // with [of], whose ability it is is not stated
  });
  it("formes, megas and transforms", () => {
    let { state } = after("|detailschange|p2a: Charizard|Charizard-Mega-X, L50, F", "|-mega|p2a: Charizard|Charizard|Charizardite X");
    expect(opp(state, "Charizard")).toMatchObject({ species: "charizardmegax", mega: true, mega_stone: "charizarditex" });
    ({ state } = after("|-transform|p2a: Venusaur|p2b: Charizard"));
    expect(opp(state, "Venusaur")).toMatchObject({ temp_species: "charizard", volatiles: ["transform"] });
    ({ state } = after("|detailschange|p2a: Venusaur|Venusaur-Mega, L50, M|90/100"));
    expect(opp(state, "Venusaur")).toMatchObject({ species: "venusaurmega", hp_num: 90 });
  });
  it("moves reveal themselves once, and not when called by another effect", () => {
    const { state } = after("|move|p2a: Venusaur|Giga Drain|p1a: Venusaur", "|move|p2a: Venusaur|Giga Drain|p1a: Venusaur", "|move|p2a: Venusaur|Sleep Talk|p2a: Venusaur", "|move|p2a: Venusaur|Hidden Power|null|[from] move: Sleep Talk");
    expect(opp(state, "Venusaur").moves_revealed).toEqual(["gigadrain", "sleeptalk"]);
    expect(state.labels.gigadrain).toBe("Giga Drain");
  });
  it("swaps positions", () => {
    const { state } = after("|swap|p2a: Venusaur|1");
    expect(state.opponent.slots).toEqual({ a: 1, b: 0 });
    expect(opp(state, "Venusaur").active_slot).toBe("b");
    expect(after("|swap|p2a: Venusaur|5").flags).toHaveLength(1);
    expect(after("|swap|p2a: Pikachu|1").flags).toHaveLength(1);
    // a swap carries the disguise bookkeeping with the position
    expect(after("|swap|p2a: Venusaur|1", "|-damage|p2b: Venusaur|10/100").flags).toEqual([]);
  });
});

describe("requests", () => {
  it("builds the own side from the request and marks who is brought", () => {
    const { state } = after(request(SPECIES.slice(0, 4), { active: [{ moves: [{ move: "Tackle", id: "tackle", pp: 5, maxpp: 8 }], canMegaEvo: true }, { moves: [] }] }));
    expect(state.request_kind).toBe("move");
    expect(state.own.roster.filter((m: any) => m.brought)).toHaveLength(4);
    expect(own(state, "Venusaur")).toMatchObject({ pp: { tackle: 5 }, can_mega_evo: true, stats: { atk: 1, spe: 5 } });
    expect(state.labels.tackle).toBe("Tackle");
  });
  it("tells the phases apart and refuses the other player's request", () => {
    expect(after(request(SPECIES, { wait: true })).state.request_kind).toBe("wait");
    expect(after(request(SPECIES, { forceSwitch: [true, false] })).state.request_kind).toBe("forceSwitch");
    expect(after(request(SPECIES, {})).flags).toHaveLength(1);
    expect(after(request(SPECIES, { active: [] }).replace('"id":"p1"', '"id":"p2"')).flags).toHaveLength(1);
    expect(after("|request|not json").flags).toHaveLength(1);
    expect(after('|request|{"active":[]}').flags).toHaveLength(1);
    expect(after(request(SPECIES, { active: [{ moves: [] }, { moves: [] }, { moves: [] }, { moves: [] }, { moves: [] }, { moves: [] }, { moves: [] }] })).flags).toHaveLength(1);
  });
  it("adds an own Pokemon a request names that it had not seen", () => {
    const run = api.interpretChannel("p1", [request(["Zed"], { teamPreview: true })], dex);
    expect(run.flags).toEqual([]);
    expect(run.frames.at(-1)!.state.own.roster[0]).toMatchObject({ name: "Zed", own: true });
  });
  it("keeps the requests it took, in order", () => {
    const { run } = after(request(SPECIES, { wait: true }), "|", "|error|[Unavailable choice] Can't move");
    expect(run.requests).toHaveLength(2);
    expect(run.frames.at(-1)!.request).toBe(2);
    expect(run.frames.at(-1)!.caption).toMatch(/^Choice refused/);
  });
});

describe("what it will not take", () => {
  it("shows an unknown line kind verbatim and flagged, and goes on", () => {
    const { run } = after("|frobnicate|x|y", "|-damage|p2a: Venusaur|10/100");
    const bad = run.frames.find((f) => f.kind === "frobnicate")!;
    expect(bad.line).toBe("|frobnicate|x|y");
    expect(bad.flagged).toContain("no rule for line type");
    expect(run.frames.at(-1)!.state.opponent.roster.find((m: any) => m.name === "Venusaur").hp_num).toBe(10);
  });
  it("refuses a |split| marker and does not read it", () => {
    const { run } = after("|split|p2", "|-damage|p2a: Venusaur|10/100");
    expect(run.flags[0]!.reason).toContain("secret half");
  });
  it("refuses an opponent HP that is not out of 100", () => {
    const { run, state } = after("|-damage|p2a: Venusaur|123/150");
    expect(run.flags[0]!.reason).toContain("not a percentage");
    expect(opp(state, "Venusaur").hp_num).toBe(100);
  });
  it("flags malformed lines of every shape without throwing", () => {
    const bad = [
      "no bar",
      "|-damage|p2a: Venusaur",
      "|-damage|p2a: Nobody|10/100",
      "|-damage|p9a: Venusaur|10/100",
      "|-damage|p2a: Venusaur|abc",
      "|-damage|p2a: Venusaur|150/100",
      "|-damage|p2a: Venusaur|0/0",
      "|-damage|p2a: Venusaur|10/100 fnt",
      "|-damage|p2a: Venusaur|5/100g",
      "|-damage|p2a: Venusaur|0",
      "|switch|p2c: Venusaur|Venusaur, L50|100/100",
      "|switch|p2a: Venusaur|Venusaur, Lx|100/100",
      "|switch|p1a: Mew|Mew, L50|100/100",
      "|poke|p2|Pikachu, L50|bad",
      "|player|p7|x",
      "|turn|x",
      "|-boost|p2a: Venusaur|atk|x",
      "|tier",
      "|error|something else",
      "|-start|p2a: Venusaur|Substitute|extra|more",
      "|replace|p2a: Venusaur|Charizard, L50",
      "|replace|p2c: Venusaur|Charizard, L50",
      "|swap|p2a: Venusaur|x",
      "|-damage|p2a: Venusaur|10/100|[from] item: A|stray",
    ];
    for (const raw of bad) {
      const { run } = after(raw);
      expect(run.flags, raw).toHaveLength(1);
    }
  });
  it("refuses a team larger than the size it announced and a fourth position", () => {
    expect(after("|switch|p2a: Mew|Mew, L50, M|100/100").flags).toHaveLength(1); // six revealed, four announced
    const run = api.interpretChannel("p1", ["|gametype|doubles", "|player|p1|A||", "|player|p2|B||", "|teamsize|p2|1", "|switch|p2a: One|One, L50|100/100", "|switch|p2b: Two|Two, L50|100/100"], dex);
    expect(run.flags).toHaveLength(1);
  });
  it("only players p1 and p2 are interpreted", () => {
    expect(() => api.createInterpreter({ player: "p3" })).toThrow();
  });
});

// Each test below interprets every committed channel (88+ channels, 15,000+ lines). The corpus grows
// with every reader fixture, and a loaded CI runner has taken more than vitest's 5 s default for the
// first test (5.4 s on PR #203); the work is CPU-bound and has no wait to make event-driven, so the
// budget is explicit instead.
const CORPUS_TEST_TIMEOUT_MS = 30_000;

describe("the whole committed corpus", () => {
  it("interprets every line of every channel with no unexpected flag", () => {
    let lines = 0;
    let channels = 0;
    for (const f of fixtures()) {
      const run = api.interpretChannel(f.player, f.lines, dex);
      lines += f.lines.length;
      channels += 1;
      const bad = run.flags.filter((x) => !x.reason.startsWith("a Pokemon shown in two positions"));
      expect(bad, f.id).toEqual([]);
      expect(run.frames).toHaveLength(f.lines.length + 1);
    }
    expect(channels).toBeGreaterThanOrEqual(88);
    expect(lines).toBeGreaterThan(15000);
  }, CORPUS_TEST_TIMEOUT_MS);
  it("captions and tokens a frame for every line kind in the corpus", () => {
    const kinds = new Map<string, number>();
    for (const f of fixtures()) {
      for (const frame of api.interpretChannel(f.player, f.lines, dex).frames) {
        if (frame.kind) kinds.set(frame.kind, (kinds.get(frame.kind) ?? 0) + 1);
      }
    }
    expect(kinds.get("move")).toBeGreaterThan(100);
    expect(kinds.get("-damage")).toBeGreaterThan(100);
    expect(kinds.get("replace")).toBeGreaterThan(0);
  }, CORPUS_TEST_TIMEOUT_MS);
});

describe("Frisk in the committed corpus (#1171)", () => {
  it("credits the item to the target and the ability to the Frisk holder at every reveal", () => {
    const FRISK = /^\|-item\|(p[12])[ab]: ([^|]+)\|([^|]+)\|\[from\] ability: Frisk\|\[of\] (p[12])[ab]: ([^|]+)/;
    let reveals = 0;
    for (const f of fixtures()) {
      const run = api.interpretChannel(f.player, f.lines, dex);
      f.lines.forEach((line, i) => {
        const m = FRISK.exec(line);
        if (!m) return;
        reveals += 1;
        const side = (id: string) => (id === f.player ? "own" : "opponent");
        const find = (state: any, id: string, name: string) => state[side(id)].roster.find((x: any) => x.name === name);
        const before = run.frames[i]!.state;
        const after = run.frames[i + 1]!.state;
        const target = find(after, m[1]!, m[2]!);
        const frisker = find(after, m[4]!, m[5]!);
        expect(target.item, `${f.id} ${line}`).toBe(m[3]!.toLowerCase().replace(/[^a-z0-9]/g, ""));
        expect(frisker.ability, `${f.id} ${line}`).toBe("frisk");
        // The target's ability is not touched by the line.
        expect(target.ability, `${f.id} ${line}`).toEqual(find(before, m[1]!, m[2]!).ability);
      });
    }
    expect(reveals).toBeGreaterThan(10);
  }, CORPUS_TEST_TIMEOUT_MS);
});

describe("Illusion", () => {
  it("settles a replace: the real Pokemon takes the slot and the disguise is put back", () => {
    for (const side of ["p1", "p2"]) {
      const f = fixtures().find((x) => x.id === `illusion/battle/${side}`)!;
      const run = api.interpretChannel(f.player, f.lines, dex);
      const at = run.frames.findIndex((fr) => fr.kind === "replace");
      expect(at).toBeGreaterThan(0);
      expect(run.frames[at]!.caption).toMatch(/was really/);
    }
  });
});
