import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { type Engine, loadEngine } from "../src/index.ts";
import { playerChannel } from "../src/player-channel.ts";
import { loadRandomCorpus, RANDOM_MANIFEST } from "../src/random-corpus.ts";
import {
  captureFidelityCorpus,
  checkUnreachable,
  FIDELITY_DIR,
  FIDELITY_SUMMARY,
  type FidelityBattleFile,
  FidelityError,
  type FidelityTarget,
  fidelityBattleFile,
  fidelitySummary,
  holdsTarget,
  loadFidelityTargets,
  occurrenceTurns,
  type PoolTeam,
  playFidelityBattle,
  readFidelityTargets,
} from "../src/search-fidelity-corpus.ts";
import { exportPaste, importPaste, validateTeam } from "../src/teams.ts";
import { resolveFormat } from "../src/formats.ts";

let engine: Engine;
beforeAll(() => {
  // SHOWDOWN_DIR: a worktree without its own build may point at another checkout.
  engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
});

const targets = loadFidelityTargets();
const corpus = loadRandomCorpus(FIDELITY_DIR) as unknown as FidelityBattleFile[];
const raw = JSON.parse(readFileSync(join(FIDELITY_DIR, "targets.json"), "utf8")) as Record<string, unknown>;

describe("the committed targeted fidelity corpus", () => {
  it("is the set its manifest names", () => {
    const manifest = JSON.parse(readFileSync(join(FIDELITY_DIR, RANDOM_MANIFEST), "utf8")) as {
      format: string;
      battles: { id: string; seed: string; turns: number; winner: string | null; actors: unknown }[];
    };
    expect(manifest.format).toBe(GOLDEN_FORMAT);
    expect(manifest.battles).toEqual(
      corpus.map((b) => ({ id: b.id, seed: b.seed, turns: b.turns, winner: b.winner, actors: b.actors })),
    );
  });

  it("re-plays every battle to the same file from its seed, teams and actor seeds", async () => {
    for (const committed of corpus) {
      const { id, seed, actor_seeds, teams, actors, origin } = committed;
      const spec = { id, seed, actor_seeds, teams, actors, origin };
      expect(fidelityBattleFile(spec, await playFidelityBattle(engine, spec)), id).toEqual(committed);
    }
    // Explicit budget: about 16 battles of at most 30 turns (a few seconds here; CI is slower).
  }, 120_000);

  it("holds only player-perspective channels and covers both actors on both seats", () => {
    for (const battle of corpus) {
      for (const player of ["p1", "p2"] as const) {
        expect(battle.channels[player].player).toBe(player);
        expect(() => playerChannel(battle.channels[player]), `${battle.id} ${player}`).not.toThrow();
      }
    }
    for (const seat of ["p1", "p2"] as const) {
      expect(new Set(corpus.map((b) => b.actors[seat]))).toEqual(new Set(["random-legal", "pending-first"]));
    }
  });

  it("commits the coverage summary that its channels and target list give", () => {
    const committed = JSON.parse(readFileSync(join(FIDELITY_DIR, FIDELITY_SUMMARY), "utf8"));
    expect(committed).toEqual(JSON.parse(JSON.stringify(fidelitySummary(corpus, targets))));
  });

  it("reaches the minimum number of turns of every target", () => {
    const summary = fidelitySummary(corpus, targets);
    for (const item of summary.items) expect(item.reached, `${item.id}: ${item.turns}/${item.min_turns}`).toBe(true);
    expect(summary.items.map((i) => i.id)).toEqual(targets.targets.map((t) => t.id));
  });

  it("has, for each target, a battle that occurs in and holds the targeted move or ability", () => {
    // Another effect can also produce a state (a status from a move outside the filter), so one holder per target is required.
    for (const target of targets.targets) {
      const hit = Object.keys(fidelitySummary(corpus, { ...targets, targets: [target] }).items[0]?.by_battle ?? {});
      const holders = corpus.filter((b) => hit.includes(b.id) && (holdsTarget(b.teams.p1, target) || holdsTarget(b.teams.p2, target)));
      expect(holders.length, target.id).toBeGreaterThan(0);
    }
  });

  it("names the origin of every team, and its teams pass the pinned validator", () => {
    const format = resolveFormat(engine, GOLDEN_FORMAT);
    for (const battle of corpus) {
      for (const seat of ["p1", "p2"] as const) {
        expect(battle.origin[seat].canonical_team_hash).toMatch(/^[0-9a-f]{64}$/);
        const team = importPaste(engine, exportPaste(engine, structuredClone(battle.teams[seat])));
        expect(validateTeam(engine, format, team), `${battle.id} ${seat}`).toEqual({ valid: true, problems: [] });
      }
    }
  });

  it("marks as unreachable only what the roster cannot hold", () => {
    checkUnreachable(targets.unreachable, goldenTeamVocabulary().roster);
    expect(() =>
      checkUnreachable([{ id: "x", description: "", moves: ["Fake Out"], abilities: [], reason: "" }], goldenTeamVocabulary().roster),
    ).toThrow(FidelityError);
  });
});

describe("occurrenceTurns", () => {
  const target: FidelityTarget = {
    id: "t",
    description: "",
    moves: ["Protect"],
    abilities: [],
    occurs: ["^\\|move\\|(p[12][ab]): [^|]*\\|Protect\\|"],
    consecutive: true,
    min_turns: 1,
  };
  const lines = [
    "|move|p1a: A|Protect|p1a: A", // turn 0, not a chain
    "|turn|1",
    "|move|p1a: A|Protect|p1a: A",
    "|turn|2",
    "|move|p1a: A|Protect||[still]", // chain from turn 1
    "|move|p2a: B|Protect|p2a: B", // not a chain
    "|turn|3",
    "|move|p2a: B|Protect|p2a: B", // chain from turn 2? no: B was at turn 2, so yes
    "|move|p1a: A|Tackle|p2a: B",
  ];

  it("counts a chain turn only when the same actor protected on the turn before", () => {
    expect(occurrenceTurns(lines, target)).toEqual([1, 2, 3]);
    expect(occurrenceTurns(lines.filter((l) => !l.includes("p1a: A|Protect||")), target)).toEqual([1, 3]);
  });

  it("counts a turn once and puts earlier lines in turn 0", () => {
    const { consecutive: _chain, ...rest } = target;
    const plain: FidelityTarget = { ...rest, occurs: ["^\\|move\\|"] };
    expect(occurrenceTurns(lines, plain)).toEqual([0, 1, 2, 3]);
    expect(occurrenceTurns(["|turn|4", "|move|a|b", "|move|a|c"], plain)).toEqual([4]);
  });

  it("treats a pattern without a group as one anonymous actor", () => {
    expect(occurrenceTurns(["|turn|1", "|move|x", "|turn|2", "|move|x"], { ...target, occurs: ["^\\|move\\|"] })).toEqual([2]);
  });
});

describe("readFidelityTargets", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const good = () => structuredClone(raw) as Record<string, any>;
  it("reads the committed record", () => {
    expect(readFidelityTargets(good()).targets.length).toBe(targets.targets.length);
  });
  it.each([
    ["a non-object", () => null],
    ["another version", () => ({ ...good(), schema_version: 2 })],
    ["a private class", () => ({ ...good(), data_class: "private" })],
    ["no targets", () => ({ ...good(), targets: undefined })],
    ["a repeated id", () => { const g = good(); g.targets[1].id = g.targets[0].id; return g; }],
    ["a target that is not an object", () => { const g = good(); g.targets[0] = 1; return g; }],
    ["a bad id", () => { const g = good(); g.targets[0].id = "Bad Id"; return g; }],
    ["nothing to filter by", () => { const g = good(); g.targets[0].moves = []; g.targets[0].abilities = []; return g; }],
    ["no pattern", () => { const g = good(); g.targets[0].occurs = []; return g; }],
    ["a bad flag", () => { const g = good(); g.targets[0].consecutive = "yes"; return g; }],
    ["no description", () => { const g = good(); g.targets[0].description = 3; return g; }],
    ["a bad minimum", () => { const g = good(); g.targets[0].min_turns = 0; return g; }],
    ["a bad list", () => { const g = good(); g.targets[0].moves = [""]; return g; }],
    ["a bad unreachable entry", () => { const g = good(); g.unreachable[0].reason = 1; return g; }],
    ["a bad attempts count", () => ({ ...good(), attempts_per_target: 0 })],
  ])("refuses %s", (_name, make) => {
    expect(() => readFidelityTargets(make())).toThrow(FidelityError);
  });
  it("refuses a pattern that is not a regular expression", () => {
    const g = good();
    g.targets[0].occurs = ["("];
    expect(() => readFidelityTargets(g)).toThrow(SyntaxError);
  });
});

describe("captureFidelityCorpus", () => {
  it("is a pure function of its inputs: two runs on a small pool give the same battles", async () => {
    const format = resolveFormat(engine, GOLDEN_FORMAT);
    const source = join(GOLDEN_DIR, "../../teams", GOLDEN_FORMAT, "datasets/fixtures-v1/teams-dev.jsonl");
    const pool: PoolTeam[] = readFileSync(source, "utf8")
      .split("\n")
      .filter((l) => l !== "")
      .slice(0, 30)
      .map((line) => {
        const record = JSON.parse(line) as { paste: string; canonical_team_hash: string };
        const team = importPaste(engine, record.paste);
        validateTeam(engine, format, team);
        return { team, origin: { dataset_id: "fixtures-v1", canonical_team_hash: record.canonical_team_hash } };
      });
    const small = readFidelityTargets({
      ...raw,
      attempts_per_target: 4,
      max_battles_per_target: 1,
      targets: [
        { ...(raw.targets as Record<string, unknown>[])[0], min_turns: 1 },
        { id: "none", description: "", moves: ["Hyperspace Fury"], abilities: [], occurs: ["^x"], min_turns: 1 },
      ],
    });
    const logs: string[] = [];
    const first = await captureFidelityCorpus(engine, pool, small, { seed_base: 0x7000, log: (l) => logs.push(l) });
    const again = await captureFidelityCorpus(engine, pool, small, { seed_base: 0x7000 });
    expect(again).toEqual(first);
    expect(first.length).toBeLessThanOrEqual(1);
    expect(logs.some((l) => l.startsWith("none: no team in the pool"))).toBe(true);
    // A target already met by the kept battles starts no search of its own.
    const met = await captureFidelityCorpus(engine, pool, { ...small, targets: [small.targets[0] as FidelityTarget, small.targets[0] as FidelityTarget].map((t, i) => ({ ...t, id: `${t.id}${i}` })) }, { seed_base: 0x7000 });
    expect(met.length).toBeLessThanOrEqual(1);
  }, 120_000);
});
