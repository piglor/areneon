// #1171: Frisk's line is `|-item|TARGET|ITEM|[from] ability: Frisk|[of] SOURCE` (pinned engine, data/abilities.ts frisk).
// The item is the target's own; the ability is the Frisk holder's. The corpus battle b20 has one such reveal.
import { describe, expect, it } from "vitest";
import { FORMAT_DIR, fullDex, loadInterpreter, readJson } from "./support.ts";
import { join } from "node:path";

const api = loadInterpreter();
const dex = fullDex();
const b20 = readJson(join(FORMAT_DIR, "random", "b20.json"));
const roster = (player: string, side: "own" | "opponent") => {
  const run = api.interpretChannel(player, b20.channels[player].lines, dex);
  expect(run.flags).toEqual([]);
  return Object.fromEntries(run.frames.at(-1)!.state[side].roster.map((m: any) => [m.species, m]));
};

describe("Frisk in corpus battle b20 (Gourgeist holds Frisk, its reveal shows Houndstone's Aggronite)", () => {
  it("the opponent's view credits the item to Houndstone and Frisk to Gourgeist", () => {
    const seen = roster("p2", "opponent");
    expect(seen.gourgeist).toMatchObject({ ability: "frisk", ability_received: false, item: null });
    expect(seen.houndstone ?? {}).not.toHaveProperty("ability", "frisk");
    const own = roster("p2", "own");
    expect(own.houndstone).toMatchObject({ item: "aggronite", ability: "fluffy", item_received: false });
  });
  it("the Frisk holder's own view keeps its Frisk, and the target is shown with its item and no ability", () => {
    expect(roster("p1", "own").gourgeist).toMatchObject({ ability: "frisk" });
    expect(roster("p1", "opponent").houndstone).toMatchObject({ item: "aggronite", item_received: false, ability: null });
  });
});
