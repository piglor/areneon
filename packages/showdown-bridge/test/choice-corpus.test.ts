import { beforeAll, describe, expect, it } from "vitest";
import {
  CASE_KINDS,
  type CaseKind,
  captureCase,
  type ChoiceCase,
  ChoiceCaseError,
  caseSpec,
  choiceUniverse,
  loadCorpus,
  missedActions,
  replayCase,
} from "../src/choice-corpus.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

const corpus = loadCorpus();
const cases = Object.entries(corpus);

describe("the golden choice corpus (ADR-006 rule 7)", () => {
  it("has a case of every kind", () => {
    const kinds = new Set(cases.map(([, c]) => c.kind));
    expect([...kinds].sort()).toEqual(Object.keys(CASE_KINDS).sort());
    for (const [file, c] of cases) expect(file).toBe(`${c.id}.json`);
  });

  it("each case's request is of its kind; phase-bound kinds match nothing else", () => {
    for (const [file, c] of cases) {
      const { request } = replayCase(engine, caseSpec(c));
      const matches = Object.entries(CASE_KINDS)
        .filter(([, matches]) => matches(request))
        .map(([kind]) => kind);
      expect(matches, file).toContain(c.kind);
      const forceSwitch = matches.filter((k) => k.startsWith("force-switch"));
      expect(forceSwitch.length > 0, file).toBe(request.phase === "forceSwitch");
      expect(matches.includes("preview"), file).toBe(request.phase === "preview");
    }
  });

  it("covers every target type in the format's move pool", () => {
    interface ModDex {
      mod(id: string): { moves: { all(): { target: string; isNonstandard: string | null }[] } };
    }
    const dex = engine.sim.Dex as unknown as ModDex;
    const { mod } = engine.sim.Dex.formats.get(GOLDEN_FORMAT);
    const pool = new Set(
      dex
        .mod(mod)
        .moves.all()
        .filter((m) => m.isNonstandard === null)
        .map((m) => m.target),
    );
    const seen = new Set<string>();
    for (const [, c] of cases) {
      const { request } = replayCase(engine, caseSpec(c));
      if (request.phase !== "move") continue;
      for (const active of request.request.active as { moves: { target?: string }[] }[]) {
        for (const move of active.moves) if (move.target !== undefined) seen.add(move.target);
      }
    }
    expect([...pool].filter((t) => !seen.has(t)).sort()).toEqual([]);
  });

  // Each case replays several thousand engine verdicts (the oracle): fast
  // locally, slow on a CI runner.
  it.each(cases)(
    "%s: re-captures identically, the engine accepts every non-pending candidate, and the oracle finds nothing missed",
    (_file, committed: ChoiceCase) => {
      const spec = caseSpec(committed);
      expect(captureCase(engine, spec)).toEqual(committed);
      const refused = committed.candidates.filter((c) => !c.pending && !c.legal);
      expect(refused).toEqual([]);
      expect(missedActions(engine, replayCase(engine, spec), spec.player)).toEqual([]);
      const pending = committed.candidates.some((c) => c.pending);
      expect(pending).toBe(committed.kind === "maybe-trapped" || committed.kind === "maybe-disabled");
    },
    60_000,
  );

  it("shows why pending candidates get no verdict: some are refused for hidden reasons", () => {
    for (const kind of ["maybe-trapped", "maybe-disabled"] satisfies CaseKind[]) {
      const [, c] = cases.find(([, x]) => x.kind === kind) as [string, ChoiceCase];
      expect(c.candidates.some((x) => x.pending && !x.legal), kind).toBe(true);
    }
  });
});

describe("the oracle", () => {
  it("finds the actions an enumerator drops", () => {
    const c = corpus["double-switch.json"] as ChoiceCase;
    const replayed = replayCase(engine, caseSpec(c));
    const offered = c.candidates.map((x) => x.choice).filter((x) => x !== "switch 3, switch 4");
    expect(missedActions(engine, replayed, c.player, offered)).toEqual(["switch 3, switch 4"]);
    const noMove1 = c.candidates.map((x) => x.choice).filter((x) => !x.startsWith("move 1 1,"));
    expect(missedActions(engine, replayed, c.player, noMove1).length).toBeGreaterThan(0);
  });
});

describe("corpus replay fails closed", () => {
  const preview = (): ChoiceCase => structuredClone(corpus["preview.json"] as ChoiceCase);

  it("on a logged choice the engine refuses", () => {
    const spec = { ...caseSpec(preview()), log: [{ player: "p1" as const, choice: "team 1123" }] };
    expect(() => replayCase(engine, spec)).toThrow(/refused logged p1 choice team 1123/);
  });

  it("when the player has nothing to choose", () => {
    const spec = { ...caseSpec(preview()), log: [{ player: "p1" as const, choice: "team 1234" }] };
    expect(() => replayCase(engine, spec)).toThrow(ChoiceCaseError);
  });

  it("when the request is not of the case's kind", () => {
    const spec = { ...caseSpec(preview()), kind: "mega" as const };
    expect(() => captureCase(engine, spec)).toThrow(/not a mega case/);
  });

  it("on a team the validator rejects", () => {
    const spec = caseSpec(preview());
    spec.teams = { ...spec.teams, p2: spec.teams.p2.slice(0, 5) };
    expect(() => replayCase(engine, spec)).toThrow(/p2 team: .*6 Pokémon/);
  });
});

describe("choiceUniverse", () => {
  it("knows only the phase and team size", () => {
    expect(choiceUniverse("preview", 6)).toHaveLength(1956);
    const move = choiceUniverse("move", 4);
    expect(move).toHaveLength(105 * 105);
    expect(move).toContain("move 4 -2 terastallize, switch 4");
    expect(move[0]).toBe("pass, pass");
  });
});
