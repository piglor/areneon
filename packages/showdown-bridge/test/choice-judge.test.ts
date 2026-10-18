import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "../src/golden.ts";
import {
  type ChoiceJudge,
  CloneJudge,
  decodeRequest,
  type Engine,
  EnumeratorDefectError,
  type LegalChoiceSet,
  legalChoiceSet,
  loadEngine,
  type PokemonSet,
  type SimBattle,
} from "../src/index.ts";

let engine: Engine;
beforeAll(() => {
  engine = loadEngine();
});

function start(seed: string, teams: Record<"p1" | "p2", PokemonSet[]>): SimBattle {
  const battle = new engine.sim.Battle({ formatid: GOLDEN_FORMAT, seed });
  for (const player of ["p1", "p2"] as const) {
    battle.setPlayer(player, { name: player, team: engine.sim.Teams.pack(teams[player]) });
  }
  return battle;
}

/** Synthetic golden team with Mega Stones swapped in (CC0, not a real team). */
function withStones(stones: Record<string, string>): PokemonSet[] {
  return syntheticTeam().map((set) => ({ ...set, item: stones[set.species] ?? set.item }));
}

interface Seen {
  sets: LegalChoiceSet[];
  phases: Set<string>;
}

/**
 * Play a battle in process. At every request each side's legal choice set
 * is built with a CloneJudge, which throws if the engine refuses any
 * non-pending candidate; `pick` then chooses among the accepted ones.
 */
function play(
  battle: SimBattle,
  pick: (set: LegalChoiceSet) => string,
  maxTurns = 40,
): Seen {
  const seen: Seen = { sets: [], phases: new Set() };
  const seq = { p1: 0, p2: 0 } as Record<string, number>;
  while (!battle.ended && battle.turn < maxTurns) {
    let progressed = false;
    for (const side of battle.sides) {
      if (battle.ended || !side.requestState || side.isChoiceDone()) continue;
      const decoded = decodeRequest(JSON.stringify(side.activeRequest));
      if (decoded.phase === "wait") continue;
      seq[side.id] = (seq[side.id] ?? 0) + 1;
      const set = legalChoiceSet({
        request: { ...decoded, request_seq: seq[side.id] as number },
        showdown_commit: engine.commit,
        judge: new CloneJudge(engine, battle, side.id),
      });
      seen.sets.push(set);
      seen.phases.add(decoded.phase);
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
    for (const set of seen.sets) {
      expect(set.showdown_commit).toBe(engine.commit);
      expect(set.choices.length).toBeGreaterThan(0);
    }
  });

  it("across seeded random battles with Mega Evolution and double switches", () => {
    const teams = {
      p1: withStones({ Charizard: "Charizardite Y", Venusaur: "Venusaurite" }),
      p2: withStones({ Gengar: "Gengarite", Blastoise: "Blastoisinite" }),
    };
    const all: LegalChoiceSet[] = [];
    for (let b = 0; b < 6; b++) {
      const next = random(b + 1);
      const seed = `sodium,${b.toString(16).padStart(32, "0")}`;
      const seen = play(start(seed, teams), (set) => {
        // Pending candidates are never picked here: only the engine may
        // judge them, at submission (ADR-006 rule 5).
        if (set.choices.length === 0) return "default";
        return set.choices[Math.floor(next() * set.choices.length)] as string;
      });
      all.push(...seen.sets);
    }
    const offered = all.flatMap((set) => set.choices);
    expect(offered.some((c) => / mega(x|y)?\b/.test(c))).toBe(true);
    expect(offered.some((c) => /^switch \d, switch \d$/.test(c))).toBe(true);
    expect(all.some((set) => set.request_seq > 10)).toBe(true);
  });
});

describe("two Mega-capable actives", () => {
  it("offers Mega Evolution on either slot, never both", () => {
    // Venusaur (slot 1) and Charizard (slot 2) lead, both holding a stone.
    const p1 = withStones({ Venusaur: "Venusaurite", Charizard: "Charizardite Y" });
    const battle = start(GOLDEN_SEED, { p1, p2: syntheticTeam() });
    battle.choose("p1", "team 1234");
    battle.choose("p2", "team 1234");
    const decoded = decodeRequest(JSON.stringify(battle.sides[0]?.activeRequest));
    const set = legalChoiceSet({
      request: { ...decoded, request_seq: 2 },
      showdown_commit: engine.commit,
      judge: new CloneJudge(engine, battle, "p1"),
    });
    expect(set.choices).toContain("move 1 1 mega, move 1 1");
    expect(set.choices).toContain("move 1 1, move 1 1 mega");
    expect(set.choices.filter((c) => /mega.*, .*mega/.test(c))).toEqual([]);
    const [both] = new CloneJudge(engine, battle, "p1").judge(["move 1 1 mega, move 1 1 mega"]);
    expect(both?.legal).toBe(false);
  });
});

describe("CloneJudge", () => {
  const team = syntheticTeam();

  it("reports the engine's refusal and never touches the live battle", () => {
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    const before = JSON.stringify(battle.toJSON());
    const verdicts = new CloneJudge(engine, battle, "p1").judge([
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
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    battle.choose("p1", "team 1234");
    battle.choose("p2", "team 1234");
    const [verdict] = new CloneJudge(engine, battle, "p1").judge(["move 1 1"]);
    expect(verdict).toEqual({ choice: "move 1 1", legal: false, error: "incomplete choice" });
  });

  it("refuses a player the battle does not have", () => {
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    expect(() => new CloneJudge(engine, battle, "p3")).toThrow(/no side p3/);
  });
});

describe("legalChoiceSet", () => {
  const preview = (): ReturnType<typeof decodeRequest> & { request_seq: number } => {
    const team = syntheticTeam();
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    const side = battle.sides[0];
    return { ...decodeRequest(JSON.stringify(side?.activeRequest)), request_seq: 1 };
  };

  it("fails on any enumerated choice the engine refuses", () => {
    const refuseAll: ChoiceJudge = {
      judge: (choices) => choices.map((choice) => ({ choice, legal: false, error: "no" })),
    };
    const call = (): LegalChoiceSet =>
      legalChoiceSet({ request: preview(), showdown_commit: engine.commit, judge: refuseAll });
    expect(call).toThrow(EnumeratorDefectError);
    expect(call).toThrow(/refused 360 enumerated choice\(s\): team 1234 \(no\)/);
  });

  it("without a judge (A2, advisor) offers the candidates unjudged", () => {
    const request = preview();
    const set = legalChoiceSet({ request, showdown_commit: engine.commit, judge: null });
    expect(set).toMatchObject({ request_seq: 1, request_hash: request.request_hash, pending: [] });
    expect(set.choices).toHaveLength(360);
  });

  it("never asks a judge about pending candidates", () => {
    const team = syntheticTeam();
    const battle = start(GOLDEN_SEED, { p1: team, p2: team });
    battle.choose("p1", "team 1234");
    battle.choose("p2", "team 1234");
    const decoded = decodeRequest(JSON.stringify(battle.sides[0]?.activeRequest));
    // Pretend the foe may have a trapping ability: the engine would send this.
    const [first] = decoded.request.active as { maybeTrapped?: boolean }[];
    Object.assign(first ?? {}, { maybeTrapped: true });
    const asked: string[] = [];
    const judge: ChoiceJudge = {
      judge: (choices) => {
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
