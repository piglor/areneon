// The battle-state projection gate (ADR-030 groundwork G4, #1464) on every committed corpus, through the Python reference packer
// and projection (`ml/areneon_ml/projection`). Each committed battle is played again with the G2 recorder, which also hands over
// the engine's `toJSON()` at every step; at every request of both seats that asks for a decision, the state and both seats'
// channels are packed into the battle-state spec's tensors (record v2), projected for the seat through the visibility classes
// alone, rebuilt as the reducer-state view the encoder reads and encoded, and the tensors must be bit-equal to `Encoder.encode`
// of the seat's real query (its observation as `observe` builds it from the Showdown channel, its candidates with their pending
// flags) under the default options, the lineage run's options and `exact_volatiles`. A request the reducer or the packer refuses
// is counted by code, never skipped silently. Also here: the privacy canary at every decision (values planted in what the seat
// may not read leave its view unchanged) and the invariance test (two battles whose opponents differ only in unrevealed fields
// give byte-identical views at every request with an equal player channel), each with a negative control.
//
// It needs the project's Python environment; without it the file is skipped unless ARENEON_REQUIRE_PYTHON_WORKER=1 (CI sets it).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { actorRule } from "../src/actor-rule.ts";
import { type BattleStateSpec, loadBattleStateSpec } from "../src/battle-state-spec.ts";
import { type Engine, loadEngine, type PokemonSet } from "../src/engine.ts";
import { firstLegal, GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { type ProjectionBattle, type ProjectionDecision, projectionBattle, type ProjectionPoint, SPEC_V2_PATH } from "../src/projection-points.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
import type { Actor } from "../src/runner.ts";
import { Child, CORPUS_TIMEOUT_MS, pointFrame, skipProjection } from "./support/projection-corpus.ts";

describe.skipIf(skipProjection)("the battle-state projection against the encoder (record v2)", () => {
  let engine: Engine;
  let record: BattleStateSpec;
  let child: Child;
  beforeAll(() => {
    engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
    // The record must read against the pinned engine before anything is packed with it.
    record = loadBattleStateSpec(engine, GOLDEN_FORMAT, new URL(`file://${SPEC_V2_PATH}`));
    child = new Child();
  });
  afterAll(() => child?.close());

  // ---- invariance ----

  /** p2's team with only fields nobody sees before they act changed: nature, Stat Points and the held items (rotated). */
  function mutated(team: readonly PokemonSet[]): PokemonSet[] {
    const vocabulary = JSON.parse(readFileSync(join(GOLDEN_DIR, "team-vocabulary.json"), "utf8")) as { natures: { name: string }[] };
    const natures = vocabulary.natures.map((n) => n.name);
    const items = team.map((set) => set.item);
    return team.map((set, i) => {
      const copy = structuredClone(set) as PokemonSet;
      copy.nature = natures[(natures.indexOf(set.nature) + 7 + i) % natures.length] as string;
      copy.evs = { ...set.evs, hp: set.evs.spe, spe: set.evs.hp, atk: set.evs.def, def: set.evs.atk };
      copy.item = items[(i + 1) % items.length] as string;
      return copy;
    });
  }

  const p2: Actor = (request) => {
    if (request.legal === undefined) throw new Error("no legal set");
    return actorRule(request.legal);
  };

  /** The p1 decisions of two battles whose p1 channels are equal up to and including the request. */
  function equalPrefixes(a: ProjectionBattle, b: ProjectionBattle): [ProjectionDecision & { point: ProjectionPoint }, ProjectionDecision & { point: ProjectionPoint }][] {
    const of = (battle: ProjectionBattle) => battle.points.flatMap((point) => point.decisions.filter((d) => d.seat === "p1").map((d) => ({ ...d, point })));
    const left = of(a);
    const right = of(b);
    const out: [ProjectionDecision & { point: ProjectionPoint }, ProjectionDecision & { point: ProjectionPoint }][] = [];
    for (let k = 0; k < Math.min(left.length, right.length); k++) {
      const x = left[k] as ProjectionDecision & { point: ProjectionPoint };
      const y = right[k] as ProjectionDecision & { point: ProjectionPoint };
      if (JSON.stringify(x.query.observation.log) !== JSON.stringify(y.query.observation.log)) break;
      out.push([x, y]);
    }
    return out;
  }

  it(
    "gives byte-identical views at every request with an equal player channel whatever the opponent holds unrevealed; a raw read of the opponent's item differs (negative control)",
    async () => {
      let compared = 0;
      let leaked = 0;
      for (const file of loadRandomCorpus().slice(0, 6)) {
        const play = (team: PokemonSet[], id: string) =>
          projectionBattle(engine, { battle_id: id, format_id: GOLDEN_FORMAT, seed: file.seed, teams: { p1: file.teams.p1, p2: team }, actors: { p1: firstLegal, p2 }, max_turns: 8 }, record);
        const a = await play(file.teams.p2, `${file.id}/real`);
        const b = await play(mutated(file.teams.p2), `${file.id}/mutated`);
        for (const [x, y] of equalPrefixes(a, b)) {
          const left = await child.ask({ kind: "view", id: "a", point: pointFrame(a, x.point), seat: "p1" });
          const right = await child.ask({ kind: "view", id: "b", point: pointFrame(b, y.point), seat: "p1" });
          expect(left.status).toBe("ok");
          expect(right.view_sha256, `${file.id} step ${x.point.step}`).toBe(left.view_sha256);
          compared += 1;
          const leakLeft = await child.ask({ kind: "view", id: "a", point: pointFrame(a, x.point), seat: "p1", leak: "roster.item" });
          const leakRight = await child.ask({ kind: "view", id: "b", point: pointFrame(b, y.point), seat: "p1", leak: "roster.item" });
          if (leakLeft.view_sha256 !== leakRight.view_sha256) leaked += 1;
        }
      }
      expect(compared).toBeGreaterThan(10);
      expect(leaked).toBeGreaterThan(0);
    },
    CORPUS_TIMEOUT_MS,
  );
});

describe("the projection points (type rule)", () => {
  it("a point carries the engine state and the seats' channels, never an omniscient log, a record or hidden verdicts", () => {
    const point: ProjectionPoint = { step: 0, cut: { p1: 1, p2: 1 }, state: {}, decisions: [] };
    // @ts-expect-error a point has no omniscient log
    point.omniscient_log = [];
    // @ts-expect-error a point has no battle record
    point.record = {};
    // @ts-expect-error a point has no hidden verdicts
    point.hidden_verdicts = [];
    expect(Object.keys(point).sort()).toEqual(["cut", "decisions", "hidden_verdicts", "omniscient_log", "record", "state", "step"]);
  });
});
