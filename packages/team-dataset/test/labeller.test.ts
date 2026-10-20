import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { PokemonSet, TeamLabelSet } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { cellCount, cellLabels, cellOf, holds, LabelError, labelTeam, labelValue, MAX_CELLS, radices, teamFacts } from "../src/labeller.ts";
import { CONFIG_DIR, labelSet } from "./support.ts";

const set = (over: Partial<PokemonSet>): PokemonSet =>
  ({ name: "", species: "Pikachu", item: "", ability: "Static", moves: ["Thunderbolt"], nature: "Serious", gender: "", evs: {}, ivs: {}, level: 50, ...over }) as PokemonSet;

// A small label set, written for the test only (the real one is data under config/teams/).
const tiny: TeamLabelSet = {
  schema_version: 1,
  label_set_id: "tiny",
  labels: [
    {
      label_id: "speed",
      values: [
        { value: "both", when: { kind: "all_of", of: [{ kind: "has_move", moves: ["trickroom"] }, { kind: "has_move", moves: ["tailwind"] }] } },
        { value: "tr", when: { kind: "has_move", moves: ["trickroom"] } },
        { value: "tw", when: { kind: "has_move", moves: ["tailwind"] } },
      ],
      default: "neither",
    },
    { label_id: "held", values: [{ value: "yes", when: { kind: "has_item", items: ["leftovers", "sitrusberry"] } }], default: "no" },
    { label_id: "who", values: [{ value: "pika", when: { kind: "has_species", species: ["pikachu"] } }], default: "other" },
    { label_id: "abil", values: [{ value: "intim", when: { kind: "has_ability", abilities: ["intimidate"] } }], default: "none" },
  ],
};

describe("the labeller reads a label-set record", () => {
  it("reads each leaf kind from the team's own sets, by id", () => {
    const team = [set({ species: "Mr. Mime", item: "Leftovers", ability: "Filter", moves: ["Trick Room", "Fake Out"] }), set({ ability: "Intimidate" })];
    const facts = teamFacts(team);
    expect(holds({ kind: "has_species", species: ["mrmime"] }, facts)).toBe(true);
    expect(holds({ kind: "has_species", species: ["raichu"] }, facts)).toBe(false);
    expect(holds({ kind: "has_move", moves: ["trickroom"] }, facts)).toBe(true);
    expect(holds({ kind: "has_item", items: ["sitrusberry", "leftovers"] }, facts)).toBe(true);
    expect(holds({ kind: "has_ability", abilities: ["intimidate"] }, facts)).toBe(true);
    expect(holds({ kind: "has_ability", abilities: ["levitate"] }, facts)).toBe(false);
  });

  it("reads all_of as every leaf, the first matching value, else the default", () => {
    const tr = set({ moves: ["Trick Room"] });
    const tw = set({ species: "Whimsicott", moves: ["Tailwind"] });
    const label = tiny.labels[0] as TeamLabelSet["labels"][number];
    expect(labelValue(label, teamFacts([tr]))).toBe("tr");
    expect(labelValue(label, teamFacts([tw]))).toBe("tw");
    // both moves on different members: all_of is about the team, not one member
    expect(labelValue(label, teamFacts([tr, tw]))).toBe("both");
    expect(labelValue(label, teamFacts([set({})]))).toBe("neither");
    // first match wins: a team with both also satisfies "tr", but "both" is listed first
    expect(labelValue(label, teamFacts([set({ moves: ["Trick Room", "Tailwind"] })]))).toBe("both");
  });

  it("ignores an empty item, and labels a team in the label set's order", () => {
    expect(teamFacts([set({ item: "" })]).items.size).toBe(0);
    const labels = labelTeam(tiny, [set({ item: "Leftovers", ability: "Intimidate", moves: ["Tailwind"] })]);
    expect(labels).toEqual([
      { label_id: "speed", value: "tw" },
      { label_id: "held", value: "yes" },
      { label_id: "who", value: "pika" },
      { label_id: "abil", value: "intim" },
    ]);
  });
});

describe("base cells", () => {
  it("number a label vector as a mixed-radix digit string, first label most significant", () => {
    expect(radices(tiny)).toEqual([4, 2, 2, 2]);
    expect(cellCount(tiny)).toBe(32);
    const value = (label_id: string, value: string) => ({ label_id, value });
    expect(cellOf(tiny, [value("speed", "both"), value("held", "yes"), value("who", "pika"), value("abil", "intim")])).toBe(0);
    expect(cellOf(tiny, [value("speed", "neither"), value("held", "no"), value("who", "other"), value("abil", "none")])).toBe(31);
    expect(cellOf(tiny, [value("speed", "tr"), value("held", "yes"), value("who", "other"), value("abil", "intim")])).toBe(10);
  });

  it("round-trips every cell of the committed 64-cell label set", () => {
    expect(cellCount(labelSet)).toBe(64);
    for (let cell = 0; cell < 64; cell++) expect(cellOf(labelSet, cellLabels(labelSet, cell))).toBe(cell);
    expect(cellLabels(labelSet, 0).map((l) => l.value)).toEqual(["both", "yes", "yes", "yes", "yes"]);
    expect(cellLabels(labelSet, 63).map((l) => l.value)).toEqual(["neither", "no", "no", "no", "no"]);
  });

  it("refuses wrong labels and cells outside the space", () => {
    const labels = cellLabels(tiny, 3);
    expect(() => cellOf(tiny, labels.slice(1))).toThrow(LabelError);
    expect(() => cellOf(tiny, [{ label_id: "held", value: "yes" }, ...labels.slice(1)])).toThrow(/labels\[0\] is held, not speed/);
    expect(() => cellOf(tiny, [{ label_id: "speed", value: "sideways" }, ...labels.slice(1)])).toThrow(/sideways is not one of its values/);
    for (const cell of [-1, 32, 1.5]) expect(() => cellLabels(tiny, cell)).toThrow(/outside the label space/);
  });

  it("refuses a label space too big for a stratum number", () => {
    const wide = { ...tiny, labels: Array.from({ length: 17 }, (_, k) => ({ label_id: `l${k}`, values: [{ value: "y", when: { kind: "has_move" as const, moves: ["protect"] } }], default: "n" })) as unknown as TeamLabelSet["labels"] };
    expect(() => cellCount(wide)).toThrow(new RegExp(`more than ${MAX_CELLS}`));
  });
});

describe("the committed label set", () => {
  it("holds the data, not the code: no id of labels.json appears as a literal in the package source", () => {
    const ids = new Set<string>();
    const walk = (value: unknown): void => {
      if (Array.isArray(value)) value.forEach(walk);
      else if (typeof value === "object" && value !== null) Object.entries(value).forEach(([key, v]) => (["species", "moves", "items", "abilities"].includes(key) ? (v as string[]).forEach((id) => ids.add(id)) : walk(v)));
    };
    walk(JSON.parse(readFileSync(join(CONFIG_DIR, "labels.json"), "utf8")));
    expect(ids.size).toBeGreaterThan(80);
    const src = new URL("../src/", import.meta.url).pathname;
    const files = ["labeller.ts", "draw.ts", "plan.ts", "generate.ts", "proposal.ts", "binner.ts", "check-labels.ts", "command.ts"];
    for (const file of files) {
      const text = readFileSync(join(src, file), "utf8");
      for (const id of ids) expect(text, `${file} mentions ${id}`).not.toMatch(new RegExp(`["'\`]${id}["'\`]`));
    }
  });

  it("has the four-valued speed-control label of ADR-013", () => {
    const speed = labelSet.labels[0] as TeamLabelSet["labels"][number];
    expect([...speed.values.map((v) => v.value), speed.default].sort()).toEqual(["both", "neither", "tailwind", "trick_room"]);
  });
});
