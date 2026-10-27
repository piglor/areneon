import { beforeAll, describe, expect, it } from "vitest";
import type { Engine } from "../src/engine.ts";
import { goldenTeamVocabulary } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { type LegalityWorldSet, createSetLegality } from "../src/set-legality.ts";
import type { TeamVocabulary } from "../src/team-vocabulary.ts";

let engine: Engine;
let vocabulary: TeamVocabulary;
beforeAll(() => {
  engine = loadEngine();
  vocabulary = goldenTeamVocabulary();
});

const none = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const venusaur = (change: Partial<LegalityWorldSet> = {}): LegalityWorldSet => ({
  species: "venusaur",
  forme: "venusaur",
  item: null,
  ability: "overgrow",
  nature: "adamant",
  stat_points: { ...none, hp: 2, atk: 32 },
  moves: ["protect", "gigadrain", "sludgebomb", "sleeppowder"],
  level: 50,
  gender: "M",
  ...change,
});

describe("the SetLegality port of the opponent sampler, with the pinned validator", () => {
  it("takes a legal set and refuses what validateSet refuses", () => {
    const port = createSetLegality({ engine, vocabulary });
    expect(port.legal(venusaur())).toBe(true);
    expect(port.legal(venusaur({ item: "leftovers" }))).toBe(true);
    expect(port.legal(venusaur({ ability: "thickfat" }))).toBe(false); // an ability the species cannot hold
    expect(port.legal(venusaur({ moves: ["protect", "closecombat"] }))).toBe(false); // a move outside its pool
    expect(port.legal(venusaur({ moves: [] }))).toBe(false); // a set needs a move
    expect(port.legal(venusaur({ stat_points: { ...none, atk: 32, spe: 32, hp: 32 } }))).toBe(false); // over the Stat Point limit
  });

  it("refuses a species or nature the vocabulary does not list, and an item the format bans", () => {
    const port = createSetLegality({ engine, vocabulary });
    expect(port.legal(venusaur({ species: "notaspecies" }))).toBe(false);
    expect(port.legal(venusaur({ nature: "nonature" }))).toBe(false);
    expect(port.legal(venusaur({ item: "notanitem" }))).toBe(false);
  });

  it("builds a Mega slot as the preview species holding the stone: the pre-Evolution ability and the evolved one both pass", () => {
    const port = createSetLegality({ engine, vocabulary });
    const mega = (ability: string) => venusaur({ forme: "venusaurmega", item: "venusaurite", ability });
    expect(port.legal(mega("chlorophyll"))).toBe(true); // not yet evolved: the preview form's ability
    expect(port.legal(mega("thickfat"))).toBe(true); // evolved: the Mega ability, checked with the base form's first ability
    expect(port.legal({ ...mega("thickfat"), moves: ["closecombat"] })).toBe(false);
    expect(port.legal({ ...mega("thickfat"), item: "leftovers" })).toBe(true); // the item is judged as given
  });

  it("is a function of the set: the same set gives the same verdict, and the caller's object is not changed", () => {
    const port = createSetLegality({ engine, vocabulary });
    const set = venusaur();
    const copy = structuredClone(set);
    expect(port.legal(set)).toBe(port.legal(set));
    expect(set).toEqual(copy);
  });
});
