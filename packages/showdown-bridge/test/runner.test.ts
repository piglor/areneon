import { describe, expect, it } from "vitest";
import { engineDefault, GOLDEN_FORMAT, GOLDEN_SEED, goldenBattle, syntheticTeam } from "../src/golden.ts";
import {
  type Actor,
  BattleAbortedError,
  InvalidTeamError,
  loadEngine,
  replayActor,
  runBattle,
  UnsupportedFormatError,
} from "../src/index.ts";

const engine = loadEngine();
const team = syntheticTeam();
const options = (actors: { p1: Actor; p2: Actor }, overrides = {}) => ({
  engine,
  formatId: GOLDEN_FORMAT,
  seed: GOLDEN_SEED,
  teams: { p1: team, p2: team },
  actors,
  ...overrides,
});

describe("runBattle", () => {
  it("plays a legal doubles battle to a result", async () => {
    const record = await goldenBattle(engine);
    expect(["p1", "p2"]).toContain(record.winner);
    expect(record.turns).toBeGreaterThan(0);
    expect(record.manifest.showdown_commit).toBe(engine.commit);
    for (const player of ["p1", "p2"] as const) {
      const requests = record.requests[player];
      expect(requests.map((r) => r.request_seq)).toEqual(requests.map((_, i) => i + 1));
      expect(requests.every((r) => r.player === player && r.rqid === undefined)).toBe(true);
      expect(requests[0]?.phase).toBe("preview");
    }
    expect(record.omniscient_log.some((line) => line.startsWith("|t:|"))).toBe(false);
  });

  it("reproduces the same sim log from the same commit, seed and choices", async () => {
    const [a, b] = await Promise.all([goldenBattle(engine), goldenBattle(engine)]);
    expect(a.omniscient_log).toEqual(b.omniscient_log);
    expect(a.choices).toEqual(b.choices);
  });

  it("diverges under a different seed", async () => {
    const a = await goldenBattle(engine);
    const b = await runBattle(
      options({ p1: engineDefault, p2: engineDefault }, { seed: "sodium,ffeeddccbbaa99887766554433221100" }),
    );
    expect(b.omniscient_log).not.toEqual(a.omniscient_log);
  });

  it("passes each actor only its own requests", async () => {
    const seen: string[] = [];
    const spy: Actor = (request) => {
      seen.push(request.player);
      return "default";
    };
    await runBattle(options({ p1: spy, p2: engineDefault }));
    expect(new Set(seen)).toEqual(new Set(["p1"]));
  });

  it("replays explicit recorded choices to the same log", async () => {
    const reversed: Actor = (request) => (request.phase === "preview" ? "team 6543" : "default");
    const original = await runBattle(options({ p1: reversed, p2: engineDefault }));
    expect(original.choices[0]).toMatchObject({ player: "p1", request_seq: 1, choice: "team 6543" });
    const replayed = await runBattle(
      options({ p1: replayActor(original.choices, "p1"), p2: replayActor(original.choices, "p2") }),
    );
    expect(replayed.omniscient_log).toEqual(original.omniscient_log);
  });

  it("aborts a replay that runs out of recorded choices", async () => {
    await expect(
      runBattle(options({ p1: replayActor([], "p1"), p2: engineDefault })),
    ).rejects.toThrow(/no recorded choice/);
  });

  it("forces a tie at the turn limit", async () => {
    const record = await runBattle(options({ p1: engineDefault, p2: engineDefault }, { maxTurns: 1 }));
    expect(record.winner).toBeNull();
    expect(record.omniscient_log).toContain("|tie");
  });

  it("aborts on a choice the engine rejects", async () => {
    const bad: Actor = () => "move 9";
    await expect(runBattle(options({ p1: engineDefault, p2: bad }))).rejects.toThrow(
      BattleAbortedError,
    );
  });

  it("aborts when an actor throws", async () => {
    const broken: Actor = () => {
      throw new Error("policy crashed");
    };
    await expect(runBattle(options({ p1: broken, p2: engineDefault }))).rejects.toThrow(
      "policy crashed",
    );
  });

  it("wraps a non-Error thrown by an actor", async () => {
    const broken: Actor = () => Promise.reject("nope");
    await expect(runBattle(options({ p1: broken, p2: engineDefault }))).rejects.toThrow("nope");
  });

  it("rejects a team the validator rejects, with its reasons", async () => {
    const zeroed = team.map((set) => ({ ...set, evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } }));
    const run = runBattle(options({ p1: engineDefault, p2: engineDefault }, { teams: { p1: team, p2: zeroed } }));
    await expect(run).rejects.toThrow(InvalidTeamError);
    await expect(run).rejects.toMatchObject({ player: "p2", problems: expect.arrayContaining([expect.stringMatching(/0 Stat Points/)]) });
  });

  it("rejects an unsupported format before starting", async () => {
    await expect(
      runBattle(options({ p1: engineDefault, p2: engineDefault }, { formatId: "gen9championsbssregmc" })),
    ).rejects.toThrow(UnsupportedFormatError);
  });
});
