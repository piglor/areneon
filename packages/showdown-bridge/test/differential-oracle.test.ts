/* eslint-disable @typescript-eslint/no-explicit-any */
// The oracle side of the differential harness (#1461): the probe as a batched backend and the facts of a query, read with the
// pinned engine from the query's own sets and declared state.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { probeOracle, queryFacts } from "../src/differential-oracle.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadMechanicsConfig, withMoveOutcome } from "../src/mechanics-config.ts";
import { mechanicsQueryHash, type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { MECHANICS_ADAPTER_FORMAT } from "../src/mechanics-result.ts";
import { loadRulesSnapshot } from "../src/rules-load.ts";

const engine = loadEngine();
const config = withMoveOutcome(loadMechanicsConfig(GOLDEN_FORMAT));
const sample = (name: string): ParsedMechanicsQuery => readMechanicsQuery(JSON.parse(readFileSync(new URL(`../../../fixtures/contracts/v1/mechanics-query/valid/${name}.json`, import.meta.url), "utf8")));
const megaCase = (): ParsedMechanicsQuery => {
  const cases = JSON.parse(readFileSync(new URL("../../../fixtures/mechanics/gen9championsvgc2026regmc/damage/cases.json", import.meta.url), "utf8")) as { cases: { id: string; query: unknown }[] };
  return readMechanicsQuery((cases.cases.find((c) => c.id === "mega") as { query: unknown }).query);
};
const edit = (query: ParsedMechanicsQuery, change: (q: any) => void): ParsedMechanicsQuery => {
  const copy = structuredClone(query) as any;
  change(copy);
  return readMechanicsQuery(copy);
};

describe("probeOracle", () => {
  it("answers a batch in order through the probe, equal queries once, a refusal kept as the query's outcome", () => {
    const oracle = probeOracle(engine, { config });
    expect(oracle).toMatchObject({ id: "showdown-probe", backend_format: MECHANICS_ADAPTER_FORMAT, backend_commit: engine.commit });
    const stats = sample("stats");
    const bad = edit(sample("damage"), (q) => (q.query.move = "notamove"));
    const out = oracle.run([stats, bad, stats]);
    expect(out.map((s) => s.ok)).toEqual([true, false, true]);
    expect(out[0]).toBe(out[2]);
    expect((out[0] as { result: { query_hash: string } }).result.query_hash).toBe(mechanicsQueryHash(stats));
    expect(probeOracle(engine, { config, id: "second", damage_path: "per-roll", snapshot: loadRulesSnapshot(engine.commit) }).id).toBe("second");
  });
});

describe("queryFacts", () => {
  const facts = queryFacts(engine);
  it("reads a stats query's one set", () => {
    const d = facts.describe(sample("stats"));
    expect(d).toMatchObject({ kind: "stats", used_moves: [], weather: null, terrain: null, trick_room: false, side_conditions: [], slot_conditions: [] });
    expect(d.mons).toHaveLength(1);
    expect(d.mons[0]?.moves.every((m) => /^[a-z0-9]+$/.test(m))).toBe(true);
  });
  it("reads a damage query's two Pokemon, its move, the field and the sides", () => {
    const query = edit(sample("damage"), (q) => {
      q.query.state.weather = "raindance";
      q.query.state.sides[1].reflect = true;
      q.query.attacker.status = "brn";
    });
    const d = facts.describe(query);
    expect(d.kind).toBe("damage");
    expect(d.mons).toHaveLength(2);
    expect(d.mons[0]?.status).toBe("brn");
    expect(d.used_moves).toEqual([query.query.kind === "damage" ? query.query.move : ""]);
    expect(d).toMatchObject({ weather: "raindance", side_conditions: ["reflect"] });
  });
  it("names an evolved Mega by the forme its stone makes, and a set with no item by null", () => {
    const d = facts.describe(megaCase());
    expect(d.mons[0]).toMatchObject({ species: "gengarmega", item: "gengarite", ability: "cursedbody" });
    expect(d.mons[1]?.item).toBeNull();
    const noStone = edit(megaCase(), (q) => (q.query.defender.mega = true));
    expect(facts.describe(noStone).mons[1]?.species).toBe("empoleon");
  });
  it("reads a move outcome's asked action, declared volatiles and slot conditions, known and unknown", () => {
    const query = sample("move-outcome-history");
    const d = facts.describe(query);
    expect(d).toMatchObject({ kind: "move_outcome", used_moves: ["protect"], slot_conditions: ["wish"] });
    expect(d.mons[0]?.volatiles.length).toBeGreaterThan(0);
    const unknown = edit(query, (q) => {
      const v = q.query.sides[0][0].history.volatiles;
      q.query.sides[0][0].history.volatiles = { kind: "unknown", ends: [v.value, []] };
      const s = q.query.history.sides[1].pending_slots;
      q.query.history.sides[1].pending_slots = { kind: "unknown", ends: [[], s.value] };
      q.query.asked = 1;
    });
    const u = facts.describe(unknown);
    expect(u.mons[0]?.volatiles).toEqual(d.mons[0]?.volatiles);
    expect(u.slot_conditions).toEqual(["wish"]);
    expect(u.used_moves).toEqual(["tackle"]);
  });
  it("reads a speed order's every action's move, a switch naming none", () => {
    const query = edit(sample("speed-order"), (q) => (q.query.actions[0].move = null));
    const d = facts.describe(query);
    expect(d.kind).toBe("speed_order");
    expect(d.used_moves.length).toBe(query.query.kind === "speed_order" ? query.query.actions.filter((a) => a.move !== null).length : -1);
    expect(facts.describe(sample("speed-order-history")).kind).toBe("speed_order");
    expect(facts.describe(sample("damage-history")).kind).toBe("damage");
  });
  it("hashes a query as the probe names it, and keys a refusal by its class and words", () => {
    expect(facts.queryHash(sample("stats"))).toBe(mechanicsQueryHash(sample("stats")));
    expect(facts.errorKey(new RangeError("x"))).toBe("RangeError:x");
  });
});
