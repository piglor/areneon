import { canonicalTeamHash, exportPaste, importPaste, type PokemonSet, readTeamRecord, teamHash, validateTeam } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { canonicalOrder, Drawer } from "../src/draw.ts";
import { cellOf, labelTeam } from "../src/labeller.ts";
import { engine, drawerFor, format, KEY, labelSet, small } from "./support.ts";
import { syntheticTeam } from "../../showdown-bridge/src/golden.ts";

const stream = { config_hash: "d".repeat(64), plan_id: "draw-test" };
const draws = Array.from({ length: 120 }, (_, k) => drawerFor().draw(stream, k));
const accepted = draws.flatMap((d) => (d.status === "accepted" ? [d] : []));
const rejected = draws.flatMap((d) => (d.status === "rejected" ? [d] : []));

/** A drawer whose proposal is fixed, to show how each gate refuses. */
const fixed = (change: (team: PokemonSet[]) => void): Drawer => {
  const team = structuredClone(syntheticTeam());
  change(team);
  return new Drawer({ engine, format, proposal: { propose: () => team } as never, labelSet, generationKey: KEY });
};

describe("one draw (ADR-013 section 2)", () => {
  it("accepts most draws and rejects the rest, each only on the engine's word", () => {
    expect(accepted.length).toBeGreaterThan(90);
    expect(rejected.length).toBeGreaterThan(0);
    for (const r of rejected) {
      expect(r.reason).toBe("invalid");
      expect(r.detail).toMatch(/\S/);
    }
  });

  it("keeps a team only if the engine's paste round trip and the pinned validator accept it", () => {
    for (const d of accepted) {
      const sets = importPaste(engine, d.paste);
      expect(exportPaste(engine, sets)).toBe(d.paste);
      expect(validateTeam(engine, format, structuredClone(sets))).toEqual({ valid: true, problems: [] });
    }
  });

  it("puts the team in canonical order, so both hashes agree and the bridge's reader accepts it as a record", () => {
    for (const d of accepted.slice(0, 20)) {
      const sets = importPaste(engine, d.paste);
      expect(d.team_hash).toBe(teamHash(sets));
      expect(d.canonical_team_hash).toBe(canonicalTeamHash(sets));
      expect(d.team_hash).toBe(d.canonical_team_hash);
      const record = {
        schema_version: 1,
        data_class: "synthetic",
        team_hash: d.team_hash,
        canonical_team_hash: d.canonical_team_hash,
        paste: d.paste,
        origin: { dataset_id: "x", split: "dev", index: 0, draw_counter: d.k },
        label_set_hash: "0".repeat(64),
        labels: d.labels,
      };
      expect(readTeamRecord(engine, record).canonical_team_hash).toBe(d.canonical_team_hash);
    }
  });

  it("labels the team with the label set and numbers its cell", () => {
    for (const d of accepted) {
      expect(d.labels).toEqual(labelTeam(labelSet, importPaste(engine, d.paste)));
      expect(d.cell).toBe(cellOf(labelSet, d.labels));
    }
    expect(new Set(accepted.map((d) => d.cell)).size).toBeGreaterThan(4);
  });

  it("is a pure function of the key, the plan and the counter", () => {
    const again = drawerFor().draw(stream, 7);
    expect(again).toEqual(draws[7]);
    expect(drawerFor().draw({ ...stream, plan_id: "other" }, 7)).not.toEqual(draws[7]);
    expect(drawerFor(small(), "22".repeat(32)).draw(stream, 7)).not.toEqual(draws[7]);
    const pilot = drawerFor().pilot(stream.config_hash, 7);
    expect(pilot).not.toEqual(draws[7]);
    expect(drawerFor().pilot(stream.config_hash, 7)).toEqual(pilot);
  });

  it("rejects a paste the engine's import does not reproduce, and one whose export differs", () => {
    const imported = fixed((t) => ((t[0] as PokemonSet).moves = ["Not A Move\nSPs: 4 HP"])).draw(stream, 0);
    expect(imported).toMatchObject({ status: "rejected", reason: "round_trip" });
    expect((imported as { detail: string }).detail).toMatch(/not reproduced by the pinned engine's export/);
    const changed = fixed((t) => ((t[0] as PokemonSet).item = "Leftovers\nShiny: Yes")).draw(stream, 0);
    expect(changed).toMatchObject({ status: "rejected", reason: "round_trip", detail: "the export of the imported team differs from the paste" });
  });

  it("rejects what the validator refuses, with its own message", () => {
    const twice = fixed((t) => (t[1] = structuredClone(t[0] as PokemonSet))).draw(stream, 0);
    expect(twice).toMatchObject({ status: "rejected", reason: "invalid" });
    expect((twice as { detail: string }).detail).toMatch(/Species Clause|Item Clause/);
    expect(fixed(() => {}).draw(stream, 0).status).toBe("accepted");
  });

  it("gives a rejected draw the canonical hash of what was proposed", () => {
    const twice = fixed((t) => (t[1] = structuredClone(t[0] as PokemonSet))).draw(stream, 0);
    const original = syntheticTeam();
    const proposed = [original[0] as PokemonSet, original[0] as PokemonSet, ...original.slice(2)];
    expect(twice.canonical_team_hash).toBe(canonicalTeamHash(proposed));
  });
});

describe("canonicalOrder", () => {
  it("sorts by species id and breaks a tie by the set's JCS bytes", () => {
    const [a, b] = [syntheticTeam()[0], { ...(syntheticTeam()[0] as PokemonSet), moves: ["Tackle"] }] as [PokemonSet, PokemonSet];
    expect(canonicalOrder([b, a])).toEqual(canonicalOrder([a, b]));
    const names = canonicalOrder(syntheticTeam()).map((s) => s.species.toLowerCase());
    expect(names).toEqual([...names].sort());
  });
});

describe("a failure that is not the engine refusing a paste", () => {
  it("is not hidden as a rejection", () => {
    // the engine's namespaces carry non-enumerable members, so wrap them by prototype
    const teams = Object.create(engine.sim.Teams, { import: { value: () => { throw new RangeError("the importer broke"); } } });
    const broken = { ...engine, sim: Object.create(engine.sim, { Teams: { value: teams } }) };
    const drawer = new Drawer({ engine: broken as never, format, proposal: { propose: () => syntheticTeam() } as never, labelSet, generationKey: KEY });
    expect(() => drawer.draw(stream, 0)).toThrow(RangeError);
  });
});
