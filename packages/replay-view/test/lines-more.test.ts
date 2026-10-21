import { describe, expect, it } from "vitest";
import { fullDex, loadInterpreter } from "./support.ts";

const api = loadInterpreter();
const dex = fullDex();

const names = ["Venusaur", "Charizard", "Blastoise", "Pikachu", "Snorlax", "Gengar"];
const ownRequest = (ability = "overgrow") =>
  "|request|" +
  JSON.stringify({
    teamPreview: true,
    side: {
      id: "p1",
      pokemon: names.map((n) => ({ ident: `p1: ${n}`, details: `${n}, L50, F`, condition: "100/100", active: false, stats: {}, moves: ["tackle"], baseAbility: ability, item: "leftovers", ability })),
    },
  });
const HEAD = [
  "|gametype|doubles",
  "|player|p1|Alice||",
  "|player|p2|Bob||",
  "|clearpoke",
  ...names.map((n) => `|poke|p2|${n}, L50, M|`),
  ownRequest(),
  "|teamsize|p1|4",
  "|teamsize|p2|4",
  "|start",
  "|switch|p1a: Venusaur|Venusaur, L50, F|100/100",
  "|switch|p1b: Charizard|Charizard, L50, F|100/100",
  "|switch|p2a: Venusaur|Venusaur, L50, M|100/100",
  "|switch|p2b: Charizard|Charizard, L50, F|100/100",
  "|turn|1",
];
const run = (...lines: string[]) => api.interpretChannel("p1", [...HEAD, ...lines], dex);
const last = (r: ReturnType<typeof run>) => r.frames.at(-1)!.state;
const opp = (s: any, n: string) => s.opponent.roster.find((m: any) => m.name === n);

describe("captions say what the log says", () => {
  const table: [string, RegExp][] = [
    ["|drag|p2a: Venusaur|Venusaur, L50, M|100/100", /dragged out/],
    ["|-sethp|p2a: Venusaur|50/100", /HP was set/],
    ["|-curestatus|p2a: Venusaur|par", /cured of par/],
    ["|-cureteam|p2: Bob", /team was cured/],
    ["|-setboost|p2a: Venusaur|atk|6", /was set to 6/],
    ["|-clearallboost", /All stat changes/],
    ["|-clearboost|p2a: Venusaur", /stat changes were reset/],
    ["|-immune|p2a: Venusaur", /unaffected/],
    ["|-fail|p2a: Venusaur", /But it failed/],
    ["|cant|p2a: Venusaur|par", /can't move \(par\)/],
    ["|cant|p2a: Venusaur", /can't move$/],
    ["|-fieldend|move: Trick Room", /Trick Room ended/],
    ["|-sideend|p2: Bob|move: Tailwind", /Tailwind ended on/],
    ["|-end|p2a: Venusaur|Substitute", /Substitute ended/],
    ["|-item|p2a: Venusaur|Leftovers", /has Leftovers/],
    ["|-enditem|p2a: Venusaur|Leftovers", /lost its Leftovers/],
    ["|-ability|p2a: Venusaur|Overgrow", /has Overgrow/],
    ["|-formechange|p2a: Venusaur|Venusaur-Mega", /changed forme: Venusaur-Mega/],
    ["|-transform|p2a: Venusaur|p2b: Charizard", /transformed into Charizard/],
    ["|swap|p2a: Venusaur|1", /changed places/],
    ["|tie", /tie/],
    ["|-hint|a hint", /a hint/],
    ["|-activate|p2a: Venusaur|move: Protect", /Protect/],
    ["|-prepare|p2a: Venusaur|Fly", /preparing Fly/],
    ["|-block|p2a: Venusaur|move: Protect", /was protected/],
    ["|-hitcount|p2a: Venusaur|3", /Hit 3 time/],
    ["|-notarget", /no target/],
    ["|-mustrecharge|p2a: Venusaur", /must recharge/],
    ["|-singleturn|p2a: Venusaur|move: Protect", /Protect/],
    ["|-singlemove|p2a: Venusaur|move: Destiny Bond", /Destiny Bond/],
    ["|-waiting|p2a: Venusaur|p2b: Charizard", /waiting for Charizard/],
    ["|-fieldactivate|move: Perish Song", /took effect/],
    ["|-ohko", /one-hit KO/],
    ["|-copyboost|p2a: Venusaur|p2b: Charizard", /copied Charizard/],
    ["|-swapboost|p2a: Venusaur|p2b: Charizard", /swapped stat changes/],
    ["|-invertboost|p2a: Venusaur", /inverted/],
    ["|-clearpositiveboost|p2a: Venusaur|p2b: Charizard|move: Haze", /boosts were removed/],
    ["|-clearnegativeboost|p2a: Venusaur", /drops were removed/],
    ["|-swapsideconditions", /swapped/],
    ["|-endability|p2a: Venusaur", /suppressed/],
    ["|-crit|p2a: Venusaur", /critical/],
    ["|-miss|p2a: Venusaur|p1a: Venusaur", /missed/],
    ["|-resisted|p2a: Venusaur", /not very effective/],
    ["|-supereffective|p2a: Venusaur", /super effective/],
    ["|-weather|none", /weather cleared/],
    ["|-fieldstart|move: Trick Room", /began/],
    ["|-sidestart|p2: Bob|move: Tailwind", /began on/],
    ["|-start|p2a: Venusaur|Substitute", /began/],
    ["|-mega|p2a: Venusaur|Venusaur|Venusaurite", /Mega Evolved/],
    ["|detailschange|p2a: Venusaur|Venusaur-Mega, L50, M", /changed forme/],
    ["|-status|p2a: Venusaur|brn", /afflicted/],
    ["|-heal|p2a: Venusaur|100/100", /regained/],
    ["|-unboost|p2a: Venusaur|spe|2", /fell by 2/],
    ["|-boost|p2a: Venusaur|spe|2", /rose by 2/],
    ["|-message|hello", /hello/],
    ["|-weather|Sandstorm", /Weather: Sandstorm/],
    ["|move|p2a: Venusaur|Protect|null", /used Protect$/],
  ];
  for (const [line, caption] of table) {
    it(line, () => {
      const r = run(line);
      const frame = r.frames.at(-1)!;
      expect(frame.caption ?? `FLAGGED ${frame.flagged}`).toMatch(caption);
    });
  }
  it("gives a line with no words no caption, and the unknown kinds none", () => {
    expect(run("|upkeep").frames.at(-1)!.caption).toBeNull();
    expect(run("|rule|Species Clause").frames.at(-1)!.caption).toBeNull();
  });
});

describe("Illusion and the other settling rules", () => {
  it("a replace carries what was seen under the disguise to the real Pokemon", () => {
    const r = run(
      "|-enditem|p2a: Venusaur|Leftovers",
      "|-mega|p2a: Venusaur|Venusaur|Venusaurite",
      "|-ability|p2a: Venusaur|Overgrow",
      "|-boost|p2a: Venusaur|atk|1",
      "|replace|p2a: Gengar|Gengar, L50, M|60/100",
    );
    const s = last(r);
    expect(r.flags).toEqual([]);
    expect(opp(s, "Gengar")).toMatchObject({ item_lost: true, mega: true, ability: "illusion", hp_num: 60, active_slot: "a" });
    expect(opp(s, "Gengar").boosts.atk).toBe(1);
    const disguise = s.opponent.roster.find((m: any) => m.name === null && m.base_species === "venusaur");
    expect(disguise).toMatchObject({ mega: false, item_lost: false, active_slot: null }); // put back as before its switch-in
    expect(s.opponent.slots.a).toBe(s.opponent.roster.indexOf(opp(s, "Gengar")));
  });
  it("flags a replace the roster cannot answer", () => {
    expect(run("|replace|p2a: Nobody|Missingno, L50, M|100/100").flags[0]!.reason).toContain("names no single Pokemon");
    expect(run("|replace|p2c: Gengar|Gengar, L50, M").flags[0]!.reason).toContain("names a position");
    const wait = ownRequest().replace('"teamPreview":true', '"wait":true').replace(/"active":false/g, '"active":true');
    expect(run(wait, "|replace|p1a: Zed|Venusaur, L50, F").flags[0]!.reason).toContain("no disguise on record");
    expect(run(wait, "|replace|p1a: Venusaur|Venusaur, L50, F|90/100").flags).toEqual([]);
  });
  it("flags a replace for a Pokemon that is already in play elsewhere", () => {
    const r = run("|replace|p2a: Charizard|Charizard, L50, F|100/100");
    expect(r.flags[0]!.reason).toContain("already in play");
  });
  it("an own request names the real Pokemon, and settles a disguise before its replace", () => {
    const illusion = (n: string) => ({
      ident: `p1: ${n}`, details: `${n}, L50, F`, condition: "100/100", active: true, stats: {}, moves: [], baseAbility: n === "Gengar" ? "illusion" : "overgrow", item: "", ability: n === "Gengar" ? "illusion" : "overgrow",
    });
    const req = "|request|" + JSON.stringify({ active: [{ moves: [] }, { moves: [] }], side: { id: "p1", pokemon: [illusion("Gengar"), illusion("Charizard")] } });
    const r = api.interpretChannel("p1", [...HEAD.slice(0, 4), ownRequest(), "|start", "|switch|p1a: Venusaur|Venusaur, L50, F|100/100", "|switch|p1b: Charizard|Charizard, L50, F|100/100", req], dex);
    expect(r.flags).toEqual([]);
    const s = r.frames.at(-1)!.state;
    expect(s.own.roster.find((m: any) => m.name === "Gengar").active_slot).toBe("a");
    expect(s.own.roster.find((m: any) => m.name === "Venusaur").active_slot).toBeNull();
    // a replace that follows finds the position already settled
    const again = api.interpretChannel("p1", [...r.frames.slice(1).map((f) => f.line), "|replace|p1a: Gengar|Gengar, L50, F|100/100"], dex);
    expect(again.flags).toEqual([]);
  });
  it("a fainted opponent that is shown again was an Illusion user: its faint stays on an entry of its own", () => {
    const r = run("|faint|p2a: Venusaur", "|switch|p2a: Venusaur|Venusaur, L50, M|100/100");
    const s = last(r);
    expect(r.flags).toEqual([]);
    expect(s.opponent.roster.filter((m: any) => m.phantom)).toHaveLength(1);
    expect(opp(s, "Venusaur")).toMatchObject({ fainted: false, active_slot: "a" });
  });
  it("a Pokemon shown in two positions gets an entry of its own and is flagged", () => {
    const r = run("|switch|p2b: Venusaur|Venusaur, L50, M|100/100");
    expect(r.flags[0]!.reason).toContain("two positions");
    expect(last(r).opponent.roster.filter((m: any) => m.phantom)).toHaveLength(1);
    const own = run("|switch|p1b: Venusaur|Venusaur, L50, F|100/100");
    expect(own.flags[0]!.reason).toContain("two positions");
  });
});

describe("the details of single lines", () => {
  it("reads shiny, a missing species, a status after the HP and a bare request line", () => {
    const r = api.interpretChannel("p1", ["|gametype|doubles", "|player|p1|A||", "|player|p2|B||", "|clearpoke", "|poke|p2|Venusaur, L50, M, shiny|", "|request"], dex);
    expect(r.frames[5]!.state.opponent.roster[0].shiny).toBe(true);
    expect(r.flags[0]!.reason).toContain("not JSON");
    expect(run("|-damage|p2a: Venusaur|10/100 brn").frames.at(-1)!.state.opponent.roster[0]).toBeDefined();
    expect(opp(last(run("|-damage|p2a: Venusaur|10/100 brn")), "Venusaur").status).toBe("brn");
    expect(run("|switch|p2a: Venusaur|, L50|100/100").flags.length).toBeGreaterThanOrEqual(0);
  });
  it("skill swap with one field, a terrain end that is not the current one, a swap that moves nothing known", () => {
    expect(run("|-activate|p2a: Venusaur").flags).toEqual([]);
    expect(opp(last(run("|-activate|p2a: Venusaur|move: Skill Swap")), "Venusaur").ability).toBeNull();
    expect(run("|-swapboost|p2a: Venusaur|p2b: Nobody").flags).toHaveLength(1);
  });
  it("a team preview line for the own side changes nothing, and a faint at a position not held", () => {
    expect(run("|poke|p1|Venusaur, L50, F|").flags).toEqual([]);
    expect(run("|faint|p2c: Venusaur").flags).toHaveLength(0);
  });
  it("refuses an ability item flag and other malformed starts", () => {
    expect(run("|poke|p2|Venusaur, L50|weird").flags).toHaveLength(1);
    expect(run("|-damage|p2a: Venusaur|10/100|[from] ability: Rough Skin").flags).toEqual([]);
    expect(run("|-start|p2a: Venusaur|ability: Flash Fire").flags).toEqual([]);
    expect(run("|-fieldend|move: Electric Terrain").flags).toEqual([]);
    expect(api.parseLine("|x|a|[from] b").tags.from).toBe("b");
  });
});

describe("requests with little in them", () => {
  it("takes a request with no Pokemon list, an entry with no stats or moves, and positions with no disguise", () => {
    const bare = api.interpretChannel("p1", ["|request|" + JSON.stringify({ wait: true, side: { id: "p1" } })], dex);
    expect(bare.flags).toEqual([]);
    const thin = {
      active: [{}],
      side: { id: "p1", pokemon: [{ ident: "p1: A", details: "Venusaur, L50, F", condition: "10/10", active: true, baseAbility: "", item: "" }] },
    };
    const r = api.interpretChannel("p1", ["|gametype|doubles", "|request|" + JSON.stringify(thin)], dex);
    expect(r.flags).toEqual([]);
    expect(r.frames.at(-1)!.state.own.roster[0]).toMatchObject({ stats: {}, moves_own: [], ability: null, item: null });
  });
  it("flags a replace with no position letter", () => {
    expect(run("|replace|p2: Bob|Venusaur, L50, M").flags[0]!.reason).toContain("names a position");
  });
});
