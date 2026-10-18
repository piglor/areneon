import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import {
  GOLDEN_DIR,
  GOLDEN_FORMAT,
  GOLDEN_PHASES,
  GOLDEN_SEED,
  type RequestFixture,
  syntheticTeam,
} from "../src/golden.ts";
import {
  type ChoiceJudge,
  CloneJudge,
  decodeRequest,
  type Engine,
  EnumeratorDefectError,
  JudgeStateError,
  type LegalChoiceSet,
  legalChoiceSet,
  loadEngine,
  type PokemonSet,
  type SimBattle,
  type SimSide,
} from "../src/index.ts";

/** What the tests drive beyond the judge's narrow view of a battle. */
interface TestSide extends SimSide {
  requestState: string;
}
interface TestBattle extends SimBattle {
  sides: TestSide[];
  ended: boolean;
  turn: number;
  setPlayer(slot: string, options: { name: string; team: string }): void;
  choose(slot: string, input: string): boolean;
}
type TestBattleClass = new (options: { formatid: string; seed: string }) => TestBattle;

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

function start(seed: string, teams: Record<"p1" | "p2", PokemonSet[]>): TestBattle {
  const Battle = engine.sim.Battle as unknown as TestBattleClass;
  const battle = new Battle({ formatid: GOLDEN_FORMAT, seed });
  for (const player of ["p1", "p2"] as const) {
    battle.setPlayer(player, {
      name: player,
      team: engine.sim.Teams.pack(teams[player]),
    });
  }
  return battle;
}

/** After both sides pick `team 1234`: p1's first move request. */
function turnOne(p1: PokemonSet[] = syntheticTeam()): TestBattle {
  const battle = start(GOLDEN_SEED, { p1, p2: syntheticTeam() });
  battle.choose("p1", "team 1234");
  battle.choose("p2", "team 1234");
  return battle;
}

const decodeSide = (battle: TestBattle, i = 0): ReturnType<typeof decodeRequest> =>
  decodeRequest(JSON.stringify(battle.sides[i]?.activeRequest));

/** Synthetic golden team with Mega Stones swapped in (CC0, not a real team). */
function withStones(stones: Record<string, string>): PokemonSet[] {
  return syntheticTeam().map((set) => ({
    ...set,
    item: stones[set.species] ?? set.item,
  }));
}

const EVENT = / (mega|megax|megay|terastallize)$/;

/**
 * The pairs the enumerator leaves out of a move request, rebuilt from the
 * slot options it did offer: the same bench Pokémon switched in on both
 * slots, and an event (Mega, Tera) on both slots.
 */
function prunedPairs(set: LegalChoiceSet): string[] {
  const slot = (i: 0 | 1): string[] => [
    ...new Set(set.choices.map((c) => c.split(", ")[i] as string)),
  ];
  const [first, second] = [slot(0), slot(1)];
  const pairs: string[] = [];
  for (const a of first) {
    for (const b of second) {
      const sameSwitch = a.startsWith("switch") && a === b;
      const bothEvent =
        EVENT.test(a) && EVENT.test(b) && a.split(" ").at(-1) === b.split(" ").at(-1);
      const bothMega = EVENT.test(a) && EVENT.test(b) && a.includes(" mega") && b.includes(" mega");
      if (sameSwitch || bothEvent || bothMega) pairs.push(`${a}, ${b}`);
    }
  }
  return pairs;
}

interface Seen {
  sets: LegalChoiceSet[];
  phases: Set<string>;
  hashes: Set<string>;
  /** Left-out pairs the engine was asked about; every one was refused. */
  pruned: string[];
}

/**
 * Play a battle in process. At every request each side's legal choice set
 * is built with a CloneJudge, which throws if the engine refuses any
 * non-pending candidate. Every pair the enumerator left out is judged too
 * and must be refused. `pick` then chooses among the accepted candidates.
 */
function play(battle: TestBattle, pick: (set: LegalChoiceSet) => string, maxTurns = 40): Seen {
  const seen: Seen = {
    sets: [],
    phases: new Set(),
    hashes: new Set(),
    pruned: [],
  };
  const seq: Record<string, number> = {};
  while (!battle.ended && battle.turn < maxTurns) {
    let progressed = false;
    for (const side of battle.sides) {
      if (battle.ended || !side.requestState || side.isChoiceDone()) continue;
      const decoded = decodeRequest(JSON.stringify(side.activeRequest));
      if (decoded.phase === "wait") continue;
      seq[side.id] = (seq[side.id] ?? 0) + 1;
      const judge = new CloneJudge(engine, battle, side.id);
      const set = legalChoiceSet({
        request: { ...decoded, request_seq: seq[side.id] as number },
        showdown_commit: engine.commit,
        judge,
      });
      if (decoded.phase === "move") {
        const pruned = prunedPairs(set);
        const accepted = judge.judge(decoded.request_hash, pruned).filter((v) => v.legal);
        expect(accepted, "the engine accepts a pair the enumerator left out").toEqual([]);
        seen.pruned.push(...pruned);
      }
      seen.sets.push(set);
      seen.phases.add(decoded.phase);
      seen.hashes.add(decoded.request_hash);
      expect(battle.choose(side.id, pick(set)), `${side.id} turn ${battle.turn}`).toBe(true);
      progressed = true;
    }
    if (!progressed) break;
  }
  return seen;
}

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("the engine accepts every non-pending candidate", () => {
  it("on every request of the golden battle", () => {
    const team = syntheticTeam();
    const seen = play(start(GOLDEN_SEED, { p1: team, p2: team }), () => "default");
    expect([...seen.phases].sort()).toEqual(["forceSwitch", "move", "preview"]);
    // The same battle the golden request fixtures were captured from.
    for (const file of Object.keys(GOLDEN_PHASES)) {
      const fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, file), "utf8")) as RequestFixture;
      expect(seen.hashes, file).toContain(fixture.request_hash);
    }
    for (const set of seen.sets) {
      expect(set.showdown_commit).toBe(engine.commit);
      expect(set.choices.length).toBeGreaterThan(0);
    }
  });

  // Six full battles, each request judged on the engine: about 1 s locally,
  // over vitest's 5 s default on a slow CI runner.
  it("across seeded random battles with Mega Evolution and double switches", () => {
    const teams = {
      p1: withStones({ Charizard: "Charizardite Y", Venusaur: "Venusaurite" }),
      p2: withStones({ Gengar: "Gengarite", Blastoise: "Blastoisinite" }),
    };
    const all: Seen[] = [];
    for (let b = 0; b < 6; b++) {
      const next = random(b + 1);
      const seed = `sodium,${b.toString(16).padStart(32, "0")}`;
      all.push(
        play(start(seed, teams), (set) => {
          // Pending candidates are never picked here: only the engine may
          // judge them, at submission (ADR-006 rule 5).
          if (set.choices.length === 0) return "default";
          return set.choices[Math.floor(next() * set.choices.length)] as string;
        }),
      );
    }
    const offered = all.flatMap((s) => s.sets.flatMap((set) => set.choices));
    expect(offered.some((c) => / mega(x|y)?\b/.test(c))).toBe(true);
    expect(offered.some((c) => /^switch \d, switch \d$/.test(c))).toBe(true);
    expect(all.flatMap((s) => s.pruned).some((c) => /^switch (\d), switch \1$/.test(c))).toBe(true);
    expect(all.some((s) => s.sets.some((set) => set.request_seq > 10))).toBe(true);
  }, 60_000);

  it("with two Mega-capable actives: Mega on either slot, never both", () => {
    // Venusaur (slot 1) and Charizard (slot 2) lead, both holding a stone.
    const battle = turnOne(withStones({ Venusaur: "Venusaurite", Charizard: "Charizardite Y" }));
    const decoded = decodeSide(battle);
    const judge = new CloneJudge(engine, battle, "p1");
    const set = legalChoiceSet({
      request: { ...decoded, request_seq: 2 },
      showdown_commit: engine.commit,
      judge,
    });
    expect(set.choices).toContain("move 1 1 mega, move 1 1");
    expect(set.choices).toContain("move 1 1, move 1 1 mega");
    expect(set.choices.filter((c) => /mega.*, .*mega/.test(c))).toEqual([]);
    const pruned = prunedPairs(set).filter((c) => /mega.*, .*mega/.test(c));
    expect(pruned.length).toBeGreaterThan(0);
    expect(judge.judge(decoded.request_hash, pruned).filter((v) => v.legal)).toEqual([]);
  });
});

describe("CloneJudge", () => {
  const team = syntheticTeam();

  it("reports the engine's refusal and never touches the live battle", () => {
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    const before = JSON.stringify(battle.toJSON());
    const verdicts = new CloneJudge(engine, battle, "p1").judge(decodeSide(battle).request_hash, [
      "team 1123",
      "team 1234",
      "move 1 1, move 1 1",
      "team 6543",
    ]);
    expect(verdicts.map((v) => v.legal)).toEqual([false, true, false, true]);
    expect(verdicts[0]?.error).toMatch(/can only switch in once/);
    expect(verdicts[2]?.error).toMatch(/needs? a teampreview response/);
    expect(verdicts[1]?.error).toBe("");
    expect(JSON.stringify(battle.toJSON())).toBe(before);
    expect(battle.sides[0]?.isChoiceDone()).toBe(false);
  });

  it("calls an incomplete choice illegal", () => {
    const battle = turnOne();
    const [verdict] = new CloneJudge(engine, battle, "p1").judge(decodeSide(battle).request_hash, [
      "move 1 1",
    ]);
    expect(verdict).toEqual({
      choice: "move 1 1",
      legal: false,
      error: "incomplete choice",
    });
  });

  it("refuses to judge a request the battle has moved past", () => {
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    const preview = decodeSide(battle).request_hash;
    const judge = new CloneJudge(engine, battle, "p1");
    battle.choose("p1", "team 1234");
    battle.choose("p2", "team 1234");
    expect(() => judge.judge(preview, ["team 1234"])).toThrow(JudgeStateError);
  });

  it("refuses a player the battle does not have, and strictChoices battles", () => {
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    expect(() => new CloneJudge(engine, battle, "p3")).toThrow(/no side p3/);
    battle.strictChoices = true;
    expect(() => new CloneJudge(engine, battle, "p1")).toThrow(JudgeStateError);
  });

  it("refuses a side the engine has asked nothing of", () => {
    // Only p1 has joined, so the battle has not started and sent no request.
    const Battle = engine.sim.Battle as unknown as TestBattleClass;
    const battle = new Battle({ formatid: GOLDEN_FORMAT, seed: GOLDEN_SEED });
    battle.setPlayer("p1", { name: "p1", team: engine.sim.Teams.pack(team) });
    const judge = new CloneJudge(engine, battle, "p1");
    expect(() => judge.judge("0".repeat(64), [])).toThrow(/has no active request/);
  });
});

describe("legalChoiceSet", () => {
  const preview = (): ReturnType<typeof decodeRequest> & {
    request_seq: number;
  } => ({
    ...decodeSide(start(GOLDEN_SEED, { p1: syntheticTeam(), p2: syntheticTeam() })),
    request_seq: 1,
  });
  const run = (judge: ChoiceJudge | null): LegalChoiceSet =>
    legalChoiceSet({
      request: preview(),
      showdown_commit: engine.commit,
      judge,
    });

  it("fails on any enumerated choice the engine refuses", () => {
    const refuseAll: ChoiceJudge = {
      judge: (_hash, choices) => choices.map((choice) => ({ choice, legal: false, error: "no" })),
    };
    expect(() => run(refuseAll)).toThrow(EnumeratorDefectError);
    expect(() => run(refuseAll)).toThrow(/refused 360 enumerated choice\(s\): team 1234 \(no\)/);
  });

  it.each<[string, ChoiceJudge["judge"]]>([
    [
      "too few verdicts",
      (_hash, choices) => choices.slice(1).map((choice) => ({ choice, legal: true, error: "" })),
    ],
    [
      "verdicts out of order",
      (_hash, choices) =>
        [...choices].reverse().map((choice) => ({ choice, legal: true, error: "" })),
    ],
  ])("fails closed on %s", (_label, judge) => {
    expect(() => run({ judge })).toThrow(JudgeStateError);
  });

  it("asks the judge about the request it enumerated", () => {
    const hashes: string[] = [];
    const request = preview();
    legalChoiceSet({
      request,
      showdown_commit: engine.commit,
      judge: {
        judge: (hash, choices) => (
          hashes.push(hash),
          choices.map((choice) => ({ choice, legal: true, error: "" }))
        ),
      },
    });
    expect(hashes).toEqual([request.request_hash]);
  });

  it("without a judge (A2, advisor) offers the candidates unjudged", () => {
    const set = run(null);
    expect(set).toMatchObject({ request_seq: 1, pending: [] });
    expect(set.choices).toHaveLength(360);
  });

  it("never asks a judge about pending candidates", () => {
    const decoded = decodeSide(turnOne());
    // Pretend the foe may have a trapping ability: the engine would send this.
    const [first] = decoded.request.active as { maybeTrapped?: boolean }[];
    Object.assign(first ?? {}, { maybeTrapped: true });
    const asked: string[] = [];
    const judge: ChoiceJudge = {
      judge: (_hash, choices) => {
        asked.push(...choices);
        return choices.map((choice) => ({ choice, legal: true, error: "" }));
      },
    };
    const set = legalChoiceSet({
      request: { ...decoded, request_seq: 2 },
      showdown_commit: engine.commit,
      judge,
    });
    expect(set.pending.length).toBeGreaterThan(0);
    expect(set.pending.every((c) => c.startsWith("switch"))).toBe(true);
    expect(asked).toEqual(set.choices);
    expect(asked.some((c) => set.pending.includes(c))).toBe(false);
  });
});
