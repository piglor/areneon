/* eslint-disable @typescript-eslint/no-explicit-any */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { type Engine, loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT } from "../src/golden.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { checkScenario } from "../src/mechanics-scenario.ts";
import { loadRandomCorpus, UNAVAILABLE_DIR } from "../src/random-corpus.ts";
import { FIDELITY_DIR } from "../src/search-fidelity-corpus.ts";
import { type WorldContext, createWorldBuilder } from "../src/world-builder.ts";
import { loadWorldWriters, type WorldWriters } from "../src/world-writers.ts";
import { type Comparison, captureDecisions, compare, copyRandomState, type Decision, type OracleSpec, playerView, revive, stateDelta, stride, translateChoice } from "./fidelity/oracle.ts";
import { evaluate, type TurnRecord } from "./fidelity/evaluate.ts";
import { blameOf, buildReport, type Inputs, N_EX, passes, type Report, renderReport } from "./fidelity/report.ts";
import { reportInputs, runFidelity } from "./fidelity/run.ts";

// ADR-026 slice F: the fork oracle, F1 (the taken turn), F2 (counterfactuals), the classification report and the
// admission rule. The oracle itself is test tooling (test/fidelity/); fidelity-isolation.test.ts proves nothing
// shipped can reach it. The full report is `node packages/showdown-bridge/test/fidelity/report-cli.ts` (about 5 minutes);
// the CI subset below runs the same code over the first battles of each corpus.

let engine: Engine;
let context: WorldContext;
let writers: WorldWriters;
beforeAll(() => {
  engine = loadEngine();
  writers = loadWorldWriters(engine, GOLDEN_FORMAT);
  context = { engine, format_id: GOLDEN_FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT)), hp_rounding: "ceil", writers };
});

const REPORT_DIR = join(GOLDEN_DIR, "search-fidelity");

interface Pick {
  corpus: "random" | "unavailable" | "search-fidelity";
  battle: string;
  player: "p1" | "p2";
  turn: number;
}

const specsOf = (corpus: Pick["corpus"]): OracleSpec[] => loadRandomCorpus(corpus === "random" ? undefined : corpus === "unavailable" ? UNAVAILABLE_DIR : FIDELITY_DIR) as unknown as OracleSpec[];

describe("primitives", () => {
  it("playerView keeps the secret half of a split for the seat and the public half for the other", () => {
    const log = ["|split|p1", "|-damage|p1a: A|10/20", "|-damage|p1a: A|50/100", "|move|p2a: B|Tackle|p1a: A", "|t:|123", "|split|p2", "|-heal|p2a: B|5/9", "|-heal|p2a: B|55/100"];
    expect(playerView(log, "p1")).toEqual(["|-damage|p1a: A|10/20", "|move|p2a: B|Tackle|p1a: A", "|-heal|p2a: B|55/100"]);
    expect(playerView(log, "p2")).toEqual(["|-damage|p1a: A|50/100", "|move|p2a: B|Tackle|p1a: A", "|-heal|p2a: B|5/9"]);
  });

  it("stride spreads a fixed sample over the canonical list and keeps a short list whole", () => {
    expect(stride([1, 2, 3], 5)).toEqual([1, 2, 3]);
    expect(stride([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3)).toEqual([0, 3, 6]);
  });

  it("translateChoice renames the opponent's switch targets by nickname and leaves the rest", () => {
    const from = { pokemon: [{ name: "A" }, { name: "B" }, { name: "C" }, { name: "D" }] };
    const to = { pokemon: [{ name: "A" }, { name: "B" }, { name: "D" }, { name: "C" }] };
    expect(translateChoice("switch 3, move 2 1 mega", from, to)).toBe("switch 4, move 2 1 mega");
    expect(translateChoice("move 1, switch 4", from, to)).toBe("move 1, switch 3");
  });
});

describe("blame: a state difference is classified against an item, an unmodelled state or an information limit", () => {
  const none = new Set<string>();
  it("maps the tags", () => {
    expect(blameOf("opponent:hp", none)).toMatchObject({ cause: "information", detail: "opponent_hp" });
    expect(blameOf("own:hp", none)).toMatchObject({ cause: "engine" });
    expect(blameOf("opponent:pp", none)).toMatchObject({ cause: "information", detail: "opponent_pp" });
    expect(blameOf("own:pp", none)).toMatchObject({ cause: "information", detail: "bench_pp" });
    expect(blameOf("opponent:volatile:mustrecharge", none)).toEqual({ cause: "unmodelled", item: null, detail: "mustrecharge" });
    expect(blameOf("own:volatile:stall", none)).toEqual({ cause: "writer", item: "protect_counter" });
    expect(blameOf("own:volatile:choicelock", none)).toEqual({ cause: "writer", item: "choice_lock" });
    expect(blameOf("opponent:volatile:taunt", new Set(["volatile:taunt"]))).toEqual({ cause: "writer", item: "volatile:taunt" });
    expect(blameOf("opponent:volatile:taunt", none).cause).toBe("unmodelled");
    expect(blameOf("field:weather_duration", none)).toEqual({ cause: "writer", item: "condition_duration" });
    expect(blameOf("own:side_conditions", none)).toEqual({ cause: "writer", item: "condition_duration" });
    expect(blameOf("opponent:status_state", none)).toEqual({ cause: "writer", item: "toxic_stage" });
    expect(blameOf("own:total_fainted", none)).toEqual({ cause: "writer", item: "total_fainted" });
    expect(blameOf("own:boosts", none)).toMatchObject({ cause: "engine", detail: "own_boosts" });
  });
});

const equal = (actors: string[] = ["p1a: A"]): Comparison => ({ equal: true, kind: null, where: null, delta: [], actors, choices: ["", ""] });
const differ = (delta: string[], actors: string[] = ["p1a: A"]): Comparison => ({ equal: false, kind: "outcome", where: "x", delta, actors, choices: ["", ""] });
const turn = (over: Partial<TurnRecord>): TurnRecord => ({ battle: "b", player: "p1", turn: 1, arm: "base", items: ["toxic_stage"], x: false, carriers: [], f1: equal(), f2: [equal()], ...over });
const INPUTS: Inputs = { fidelity_format: 1, showdown_commit: "c", corpora_sha256: "d", base_record_sha256: "r" };
const dataOf = (turns: TurnRecord[], candidates: any[] = []): any => ({ corpora: { random: { battles: 1, decisions: turns.length } }, turns, refusals: {}, decisions: turns.length, candidates, options: { k: 1, m: 1 }, ms: 0 });

describe("the report and the admission rule (N_ex)", () => {
  const rowOf = (report: Report, item: string) => report.writers.find((r) => r.item === item) as any;
  it("counts exercising turns and admits an item only at N_ex of them with no difference classified against it", () => {
    const turns = Array.from({ length: N_EX }, (_, i) => turn({ turn: i + 1 }));
    expect(rowOf(buildReport(dataOf(turns), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: true, f1: { exercised: N_EX, equal: N_EX, blamed: 0 } });
    // One short.
    expect(rowOf(buildReport(dataOf(turns.slice(1)), writers, INPUTS), "toxic_stage").admitted).toBe(false);
    // A difference classified against the item blames it; one classified elsewhere does not.
    const blamed = [...turns.slice(1), turn({ f1: differ(["own:status_state"]), f2: [equal()] })];
    expect(rowOf(buildReport(dataOf(blamed), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: false, f1: { exercised: N_EX, equal: N_EX - 1, blamed: 1 } });
    const elsewhere = [...turns.slice(1), turn({ f1: differ(["opponent:hp", "opponent:pp"]), f2: [equal()] })];
    expect(rowOf(buildReport(dataOf(elsewhere), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: true, f1: { equal: N_EX - 1, blamed: 0 } });
    // A difference no state difference names blames every item present: nothing is admitted on it.
    const unclassified = [...turns.slice(1), turn({ f1: differ([]), f2: [equal()] })];
    expect(rowOf(buildReport(dataOf(unclassified), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: false, f1: { blamed: 1 } });
  });

  it("a difference classified against a writer in a turn where the writer was absent is a missed write and stops the admission", () => {
    const turns = Array.from({ length: N_EX }, (_, i) => turn({ turn: i + 1 }));
    const missed = [...turns, turn({ items: ["last_move"], f1: differ(["own:status_state"]), f2: [equal()] })];
    expect(rowOf(buildReport(dataOf(missed), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: false, f1: { exercised: N_EX, blamed: 0, missed: 1 } });
  });

  it("requires F2 as well, keeps X-class turns out of every count and reports them apart", () => {
    const turns = Array.from({ length: N_EX }, (_, i) => turn({ turn: i + 1 }));
    const failF2 = [...turns.slice(1), turn({ f2: [differ(["own:status_state"])] })];
    expect(rowOf(buildReport(dataOf(failF2), writers, INPUTS), "toxic_stage")).toMatchObject({ admitted: false, f1: { blamed: 0 }, f2: { blamed: 1 } });
    const xs = [...turns.slice(1), turn({ x: true, f1: differ(["own:status_state"]), f2: [differ(["own:status_state"])] })];
    const report = buildReport(dataOf(xs), writers, INPUTS);
    expect(rowOf(report, "toxic_stage").f1.exercised).toBe(N_EX - 1);
    expect(report.f1.x).toMatchObject({ turns: 1, equal: 0 });
    expect(report.f1.compared).toMatchObject({ turns: N_EX - 1, equal: N_EX - 1 });
    expect(report.f2_pairs).toMatchObject({ compared: N_EX - 1, x: 1, x_equal: 0 });
  });

  it("an item counts a turn only when some Pokemon acted; an id only when its carrier did", () => {
    const quiet = Array.from({ length: N_EX }, () => turn({ f1: equal([]), f2: [equal([])] }));
    expect(rowOf(buildReport(dataOf(quiet), writers, INPUTS), "toxic_stage")).toBeUndefined();
    const carrier = (acted: string[]) => Array.from({ length: N_EX }, () => turn({ arm: "volatile:encore", items: ["volatile:encore"], carriers: ["Zed"], f1: equal(acted), f2: [equal(acted)] }));
    const cand = [{ kind: "volatile", id: "encore", class: "W" }];
    const elsewhere = buildReport(dataOf(carrier(["p1a: Other"]), cand), writers, INPUTS);
    expect(elsewhere.ids).toEqual([]);
    const there = buildReport(dataOf(carrier(["p2b: Zed"]), cand), writers, INPUTS);
    expect(there.ids[0]).toMatchObject({ item: "volatile:encore", admitted: true });
    expect(there.admissions).toEqual([{ kind: "volatile", id: "encore", class: "W" }]);
    expect(renderReport(there)).toContain("volatile encore (class W)");
  });

  it("a volatile's own difference blames it; the delta of another admitted id does not", () => {
    const cand = [{ kind: "volatile", id: "encore", class: "W" }];
    const mk = (delta: string[]) => Array.from({ length: N_EX }, (_, i) => turn({ arm: "volatile:encore", items: ["volatile:encore"], carriers: ["Zed"], f1: i === 0 ? differ(delta, ["p2b: Zed"]) : equal(["p2b: Zed"]), f2: [equal(["p2b: Zed"])] }));
    expect(buildReport(dataOf(mk(["opponent:volatile:encore"]), cand), writers, INPUTS).ids[0]).toMatchObject({ admitted: false, f1: { blamed: 1 } });
    expect(buildReport(dataOf(mk(["opponent:volatile:mustrecharge"]), cand), writers, INPUTS).ids[0]).toMatchObject({ admitted: true });
  });
});

describe("the oracle reproduces a real turn line for line (sanity gate)", () => {
  const picks = JSON.parse(readFileSync(join(REPORT_DIR, "oracle-sanity.json"), "utf8")) as { picks: Pick[] };
  const decisions: { pick: Pick; d: Decision }[] = [];
  let builder: ReturnType<typeof createWorldBuilder>;
  beforeAll(async () => {
    builder = createWorldBuilder(context);
    for (const group of [...new Set(picks.picks.map((p) => `${p.corpus}/${p.battle}`))]) {
      const [corpus, battle] = group.split("/") as [Pick["corpus"], string];
      const spec = specsOf(corpus).find((s) => s.id === battle) as OracleSpec;
      const found = (await captureDecisions(engine, spec)).decisions;
      for (const pick of picks.picks.filter((p) => p.corpus === corpus && p.battle === battle)) decisions.push({ pick, d: found.find((d) => d.player === pick.player && d.turn === pick.turn) as Decision });
    }
  }, 120_000);

  const world = (d: Decision, copy = true): { state: object; built: any } => {
    const result = builder.build(d.input);
    if (result.kind !== "world") throw new Error("a hand-picked turn must build");
    if (copy) copyRandomState(result.world.battle, revive(engine, d.fork));
    return { state: result.world.battle.toJSON(), built: result.world };
  };

  it("hand-picked E-only turns (no X item, no hidden volatile): F1 and the F2 samples are equal and play real lines", () => {
    expect(picks.picks.length).toBeGreaterThanOrEqual(12);
    expect(new Set(picks.picks.map((p) => p.corpus)).size).toBe(3);
    expect(new Set(picks.picks.map((p) => p.player)).size).toBe(2);
    for (const { pick, d } of decisions) {
      expect(d, `${pick.battle}/${pick.player}/${pick.turn} exists`).toBeDefined();
      const { state, built } = world(d);
      expect(built.resampled, "no X item").toEqual([]);
      const f1 = compare(engine, d, state, (d.recorded as any).own, (d.recorded as any).foe);
      expect(f1, `${pick.battle}/${pick.player}/${pick.turn} F1`).toMatchObject({ equal: true });
      expect(f1.actors.length, "a real turn: some Pokemon acted").toBeGreaterThan(0);
      const real = revive(engine, d.fork);
      const before = real.log.length;
      real.sides.forEach((s: any) => s.clearChoice());
      real.choose(d.player, (d.recorded as any).own);
      real.choose(d.player === "p1" ? "p2" : "p1", (d.recorded as any).foe);
      expect(playerView(real.log.slice(before), d.player).length, "the turn wrote lines").toBeGreaterThan(2);
    }
  }, 120_000);

  it("negative controls: without the random-state copy, or with a doctored world, the comparison differs", () => {
    let withoutCopy = 0;
    let doctored = 0;
    for (const { d } of decisions) {
      const a = world(d, false);
      if (!compare(engine, d, a.state, (d.recorded as any).own, (d.recorded as any).foe).equal) withoutCopy += 1;
      const b = world(d);
      const tampered = revive(engine, b.state);
      const foe = tampered.sides.find((s: any) => s.id !== d.player);
      for (const mon of foe.pokemon.slice(0, 2)) mon.boostBy({ spe: -6, atk: 6 });
      if (!compare(engine, d, tampered.toJSON(), (d.recorded as any).own, (d.recorded as any).foe).equal) doctored += 1;
    }
    expect(withoutCopy, "a world that draws from another stream differs somewhere").toBeGreaterThan(0);
    expect(doctored, "a world with the wrong stages differs somewhere").toBeGreaterThan(0);
  }, 120_000);

  it("stateDelta names what differs before the turn and nothing for an equal state", () => {
    const { d } = decisions[0] as { d: Decision };
    const { state } = world(d);
    expect(stateDelta(revive(engine, d.fork), revive(engine, state), d.player).filter((t) => !t.endsWith(":pp") && !t.endsWith(":hp"))).toEqual([]);
    const tampered = revive(engine, state);
    tampered.sides.find((s: any) => s.id !== d.player).pokemon[0].boostBy({ atk: 2 });
    expect(stateDelta(revive(engine, d.fork), tampered, d.player)).toContain("opponent:boosts");
    expect(stateDelta(revive(engine, d.fork), revive(engine, d.fork), d.player)).toEqual([]);
  });

  it("evaluate returns a typed refusal for a decision the record cannot build and counts it", () => {
    const refusing = { ...writers, items: writers.items.map((i) => (i.id === "last_move" ? { ...i, class: "R" as const } : i)) } as WorldWriters;
    const outcome = evaluate(engine, context, { name: "t", writers: refusing, id: null }, (decisions.find((x) => x.d.input.state.opponent.roster.some((m) => m.moves_revealed.length > 0)) ?? (decisions[0] as any)).d, { k: 1, m: 1 }, new Map());
    expect(["turn", "refused"]).toContain(outcome.kind);
  });
});

describe("the CI subset of the report", () => {
  it("runs the oracle over the first battles of every corpus, accounts for every decision and prints the report", async () => {
    // Explicit budget: about 25 s on a shared machine for these battles; the full report is the script.
    const data = await runFidelity({ corpora: ["random", "random-choice-lock", "unavailable", "search-fidelity"], battles: 4, candidates: true, k: 2, m: 2 }, engine);
    const report = buildReport(data, writers, reportInputs(engine, writers));
    const refused = Object.values(report.refusals).reduce((a, b) => a + b, 0);
    expect(report.built + refused).toBe(report.decisions);
    expect(report.f1.compared.turns + report.f1.x.turns).toBeLessThanOrEqual(report.built);
    expect(report.f1.compared.turns).toBeGreaterThan(20);
    expect(report.f2_pairs.compared + report.f2_pairs.x).toBeGreaterThan(100);
    for (const mode of ["f1", "f2"] as const) {
      for (const bucket of [report[mode].compared, report[mode].x]) {
        expect(bucket.equal).toBeLessThanOrEqual(bucket.turns);
        expect(Object.values(bucket.by_kind).reduce((a, b) => a + b, 0)).toBe(bucket.turns - bucket.equal);
      }
    }
    console.log(`fidelity report (CI subset: 4 battles of each corpus)\n${renderReport(report)}`);
  }, 300_000);
});

describe("the committed full report", () => {
  const committed = JSON.parse(readFileSync(join(REPORT_DIR, "fidelity-report.json"), "utf8")) as Report;

  it("is not stale: it was computed from these corpora, this pin and this record (rerun report-cli.ts with --json otherwise)", () => {
    expect(committed.inputs).toEqual(reportInputs(engine, writers));
  });

  it("supports exactly the ids the committed writers record admits (N_ex, F1 and F2, none blamed)", () => {
    for (const row of [...committed.writers, ...committed.ids]) {
      expect(row.admitted, row.item).toBe(passes(row.f1, row.f2));
    }
    expect(committed.admissions).toEqual(writers.admitted.map((a) => ({ kind: a.kind, id: a.id, class: a.class })));
    expect(committed.n_ex).toBe(N_EX);
  });

  it("every item the record lists as W with evidence passes it; items without enough evidence are R in the record", () => {
    const classOf = new Map(writers.items.map((i) => [i.id as string, i.class]));
    for (const row of committed.writers) expect(classOf.get(row.item) === "W", `${row.item}: W only with evidence`).toBe(row.admitted);
  });

  it("the markdown beside it is the report rendered", () => {
    expect(readFileSync(join(REPORT_DIR, "fidelity-report.md"), "utf8")).toBe(renderReport(committed));
  });
});
