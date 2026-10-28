/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { type CandidateFeatures, candidateFeatures, type FeatureConfigs } from "../src/candidate-features.ts";
import { jcs, type Json } from "../src/canonical-json.ts";
import type { PokemonSet } from "../src/engine.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { damagePathStats, scenarioStartStats } from "../src/mechanics-damage.ts";
import { type MechanicsContext, MechanicsProbeError, type ProbeOutcome, probeMechanics } from "../src/mechanics-probe.ts";
import { type ParsedMechanicsQuery, readMechanicsQuery } from "../src/mechanics-query.ts";
import { playerChannel } from "../src/player-channel.ts";
import { exportPaste, importPaste } from "../src/teams.ts";
import { capture, type Captured, config, corpus, engine, mechanics, snapshot, vocabulary } from "./damage-aware-support.ts";

// ADR-022 section 4, mitigation 4: the one-clone damage path answers with the same bytes as the per-roll path
// (33 clones) on every query the project holds: the hand-worked damage fixtures, the KO cases, the calc
// differential's sets, queries over the committed dataset teams, and the requests the feature builder featurises.

const root = new URL("../../../fixtures/", import.meta.url);
const read = (path: string): any => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const damage = read("mechanics/gen9championsvgc2026regmc/damage/cases.json") as { cases: { id: string; query: any }[] };
const ko = read("mechanics/gen9championsvgc2026regmc/ko/cases.json") as { probe_cases: { id: string; base_case: string; defender_hp: number }[] };

/** The result bytes (or the refusal's words) of one path. */
function answer(path: "one-clone" | "per-roll", context: Partial<MechanicsContext>, query: ParsedMechanicsQuery): string {
  try {
    const { result, diagnostics } = probeMechanics({ engine, config: mechanics, snapshot, ...context, damage_path: path }, query);
    return jcs({ result, diagnostics } as unknown as Json);
  } catch (error) {
    if (error instanceof MechanicsProbeError) return `refused: ${error.message}`;
    throw error;
  }
}

let compared = 0;
/** Both paths on one query; fails on any difference of bytes. */
function same(label: string, value: unknown, context: Partial<MechanicsContext> = {}): string {
  const query = readMechanicsQuery(value);
  const slow = answer("per-roll", context, query);
  const fast = answer("one-clone", context, query);
  compared += 1;
  expect(fast, label).toBe(slow);
  return fast;
}

const withDefender = (id: string, change: (defender: any) => void): unknown => {
  const query = structuredClone(damage.cases.find((c) => c.id === id)?.query);
  change(query.query.defender);
  return query;
};
const canonical = (paste: string): string => exportPaste(engine, importPaste(engine, paste));

beforeAll(() => {
  damagePathStats.one_clone = 0;
  damagePathStats.fell_back = 0;
  scenarioStartStats.started = 0;
  scenarioStartStats.reused = 0;
});

describe("the one-clone path equals the per-roll path", () => {
  it("on every hand-worked damage fixture case (the calc differential's damage set)", () => {
    for (const c of damage.cases) same(c.id, c.query);
  }, 60_000); // 12 cases x 2 paths, 33 clones each on the slow path

  it("on the KO cases (declared defender hit points)", () => {
    for (const c of ko.probe_cases) same(c.id, withDefender(c.base_case, (defender) => (defender.hp = c.defender_hp)));
  }, 60_000);

  it("on the ko_blockers: items and abilities whose use changes the defender", () => {
    const blockers = [...mechanics.ko_blockers.items.map((id) => ["item", id] as const), ...mechanics.ko_blockers.abilities.map((id) => ["ability", id] as const)];
    expect(blockers.length).toBeGreaterThan(20);
    const kinds = new Set<string>();
    for (const [kind, id] of blockers) {
      const paste = canonical(kind === "item" ? `Empoleon @ ${id}\nAbility: Torrent\nLevel: 50\nEVs: 32 HP / 32 SpA\nModest Nature\n- Surf\n- Protect\n` : `Empoleon\nAbility: ${id}\nLevel: 50\nEVs: 32 HP / 32 SpA\nModest Nature\n- Surf\n- Protect\n`);
      // The pinned validator and rule table decide whether the set is legal; a set it refuses is the same refusal on both paths.
      const outcome = same(`${kind} ${id}`, withDefender("single-target", (defender) => (defender.set = paste)));
      kinds.add(outcome.startsWith("refused") ? "refused" : "answered");
    }
    expect(kinds.has("answered")).toBe(true);
  }, 300_000); // about 27 blockers x 2 paths

  it("on a type-resist berry, which the move step eats: the battle changes, so the per-roll path answers", () => {
    const before = damagePathStats.fell_back;
    // Haban Berry halves a super effective Dragon move; the engine eats it inside the damage step.
    const berry = canonical("Garchomp @ Haban Berry\nAbility: Rough Skin\nLevel: 50\nEVs: 32 HP / 32 Atk\nAdamant Nature\n- Dragon Claw\n- Protect\n");
    const query = withDefender("single-target", (defender) => (defender.set = berry));
    (query as any).query.move = "dragonclaw";
    same("haban berry", query);
    expect(damagePathStats.fell_back).toBeGreaterThan(before);
  }, 60_000);

  it("on an immunity (sixteen zeros), which the engine's own steps stop before the damage loop", () => {
    const ghost = (damage.cases.find((c) => c.id === "mega")?.query as any).query.attacker.set as string;
    const query = structuredClone(damage.cases.find((c) => c.id === "single-target")?.query);
    query.query.move = "bodyslam";
    query.query.defender.set = ghost;
    same("immunity", query);
    expect(JSON.parse(answer("one-clone", {}, readMechanicsQuery(query))).result.answer.no_crit).toEqual(new Array(16).fill(0));
  }, 60_000);

  it("on moves the probe cannot answer: random base power, a charge turn, a status move, fixed damage and a missing move", () => {
    for (const move of ["magnitude", "bide", "solarbeam", "protect", "seismictoss", "nightshade", "sheercold", "bulletseed", "beatup", "present", "doublekick", "willowisp"]) {
      const query = structuredClone(damage.cases.find((c) => c.id === "single-target")?.query);
      query.query.move = move;
      same(move, query);
    }
    const query = structuredClone(damage.cases.find((c) => c.id === "single-target")?.query);
    query.query.move = "notamove";
    expect(same("unknown move", query)).toMatch(/^refused/);
  }, 60_000);

  it("on queries over: each damaging move into the first foe", () => {
    const dev = readFileSync(new URL("teams/gen9championsvgc2026regmc/datasets/fixtures-v1/teams-dev.jsonl", root), "utf8").trim().split("\n").map((l) => (JSON.parse(l) as { paste: string }).paste);
    const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(name: string): { id: string; category: string } } };
    let queries = 0;
    // ARENEON_PATHS_TEAMS widens the sweep for a one-off check (the committed default of 100 keeps the test short).
    const teams = Number(process.env.ARENEON_PATHS_TEAMS ?? 100);
    for (let team = 0; team < teams; team++) {
      const own = importPaste(engine, dev[team % dev.length] as string);
      const foe = importPaste(engine, dev[(team + 5) % dev.length] as string);
      const attacker = own[team % own.length] as PokemonSet;
      const defender = foe[(team * 2) % foe.length] as PokemonSet;
      for (const move of attacker.moves) {
        const id = dex.moves.get(move).id;
        if (dex.moves.get(move).category === "Status") continue;
        const combatant = (set: PokemonSet) => ({ set: exportPaste(engine, [set]), boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, status: null, hp: null, mega: false });
        const side = { aurora_veil: false, light_screen: false, reflect: false, tailwind: false };
        same(`team ${team} ${id}`, { schema_version: 1, data_class: "synthetic", format_id: GOLDEN_FORMAT, query: { kind: "damage", attacker: combatant(attacker), defender: combatant(defender), move: id, state: { weather: null, terrain: null, trick_room: false, sides: [side, side] } } });
        queries += 1;
      }
    }
    expect(queries).toBeGreaterThan(30);
  }, 600_000); // about 60 queries x 2 paths on the slow path's 33 clones
});

// Three seeded corpus battles (both seats of the first, second and fourth), as the builder's own tests use; the dataset sweep above stands in for the differential's stats sets, which are not damage queries.
describe("the feature builder's requests", () => {
  let requests: Captured[] = [];
  beforeAll(async () => {
    requests = [...(await capture(corpus[0] as never)), ...(await capture(corpus[1] as never)), ...(await capture(corpus[3] as never))];
  }, 120_000); // seeded real battles on the pinned engine

  it("give the same rows, byte for byte, whichever path the probe takes", () => {
    const via = (path: "one-clone" | "per-roll") => (context: MechanicsContext, query: ParsedMechanicsQuery): ProbeOutcome => probeMechanics({ ...context, damage_path: path }, query);
    const build = (c: Captured, path: "one-clone" | "per-roll"): CandidateFeatures => {
      const legal = c.request.legal as NonNullable<typeof c.request.legal>;
      const candidates = [...legal.choices, ...legal.pending].map((choice) => ({ choice, pending: legal.pending.includes(choice) }));
      const configs: FeatureConfigs = { vocabulary, config, mechanics, snapshot, data_class: "synthetic", probe: via(path) };
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player: c.context.player, lines: c.context.channel });
      return candidateFeatures(engine, configs, channel, c.context.team as never, candidates);
    };
    let rows = 0;
    for (const c of requests.filter((r) => r.request.phase === "move" && (r.request.legal?.choices.length ?? 0) >= 2)) {
      const fast = build(c, "one-clone");
      expect(JSON.stringify(fast), `request ${c.request.request_seq}`).toBe(JSON.stringify(build(c, "per-roll")));
      rows += fast.rows.length;
    }
    expect(rows).toBeGreaterThan(50);
    compared += rows;
  }, 900_000); // each request is built twice, the slow path with 33 clones per row
});

describe("the comparison was not vacuous", () => {
  it("compared many queries and answered most with the one-clone path", () => {
    expect(compared).toBeGreaterThan(100);
    expect(damagePathStats.one_clone).toBeGreaterThan(damagePathStats.fell_back);
    // The one-clone path takes its scenario from the kept start; the per-roll reference starts its own, so the equality above covers both.
    expect(scenarioStartStats.reused).toBeGreaterThan(scenarioStartStats.started);
  });
});
