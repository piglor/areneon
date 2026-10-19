import { describe, expect, it } from "vitest";
import { type ChoiceCase, loadCorpus } from "../src/choice-corpus.ts";
import {
  engineDefault,
  firstLegal,
  GOLDEN_FORMAT,
  GOLDEN_SEED,
  goldenBattle,
  syntheticTeam,
} from "../src/golden.ts";
import {
  type Actor,
  BattleAbortedError,
  type BattleRecord,
  judgeSubmission,
  type LegalChoiceSet,
  type RequestRecord,
  InvalidTeamError,
  loadEngine,
  loadRulesSnapshot,
  NoCandidateError,
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
      options(
        { p1: engineDefault, p2: engineDefault },
        { seed: "sodium,ffeeddccbbaa99887766554433221100" },
      ),
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
    expect(original.choices[0]).toMatchObject({
      player: "p1",
      request_seq: 1,
      choice: "team 6543",
    });
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
    const record = await runBattle(
      options({ p1: engineDefault, p2: engineDefault }, { maxTurns: 1 }),
    );
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
    const zeroed = team.map((set) => ({
      ...set,
      evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
    }));
    const run = runBattle(
      options({ p1: engineDefault, p2: engineDefault }, { teams: { p1: team, p2: zeroed } }),
    );
    await expect(run).rejects.toThrow(InvalidTeamError);
    await expect(run).rejects.toMatchObject({
      player: "p2",
      problems: expect.arrayContaining([expect.stringMatching(/0 Stat Points/)]),
    });
  });

  it("resolves the format against a snapshot the caller loaded once", async () => {
    const snapshot = loadRulesSnapshot(engine.commit);
    const record = await runBattle(options({ p1: engineDefault, p2: engineDefault }, { snapshot }));
    expect(record.manifest).toEqual((await goldenBattle(engine)).manifest);
    const stale = loadRulesSnapshot("0".repeat(40));
    await expect(
      runBattle(options({ p1: engineDefault, p2: engineDefault }, { snapshot: stale })),
    ).rejects.toThrow(UnsupportedFormatError);
  });

  it("rejects an unsupported format before starting", async () => {
    await expect(
      runBattle(
        options({ p1: engineDefault, p2: engineDefault }, { formatId: "gen9championsbssregmc" }),
      ),
    ).rejects.toThrow(UnsupportedFormatError);
  });
});

describe("runBattle and legal choices (ADR-006)", () => {
  it("gives every actor request its engine-judged legal choice set", async () => {
    const record = await goldenBattle(engine);
    for (const player of ["p1", "p2"] as const) {
      for (const request of record.requests[player]) {
        if (request.phase === "wait") {
          expect(request.legal).toBeUndefined();
          continue;
        }
        expect(request.legal).toMatchObject({
          request_seq: request.request_seq,
          request_hash: request.request_hash,
          showdown_commit: engine.commit,
        });
        expect(request.legal?.choices.length).toBeGreaterThan(0);
      }
    }
    // firstLegal played each set's first choice.
    for (const { player, request_seq, choice } of record.choices) {
      expect(record.requests[player][request_seq - 1]?.legal?.choices[0]).toBe(choice);
    }
    expect(record.hidden_verdicts).toEqual([]);
  });

  it("firstLegal plays the first pending candidate when nothing is judged legal", () => {
    const request = {
      ...record0(),
      legal: { ...legal0(), choices: [], pending: ["switch 3, pass", "switch 4, pass"] },
    };
    expect(firstLegal(request)).toBe("switch 3, pass");
  });

  it("firstLegal refuses only an empty or missing legal choice set", () => {
    expect(() => firstLegal({ ...record0(), legal: legal0() })).toThrow(NoCandidateError);
    expect(() => firstLegal(record0())).toThrow(/p1: request 1 has no legal choice set/);
  });
});

describe("a pending switch refused at submission ([Unavailable choice])", () => {
  // The corpus case: foe Mega Gengar may have Shadow Tag, so p1's switches
  // are pending. p1 submits one; the engine refuses it for a hidden reason.
  const c = loadCorpus()["maybe-trapped.json"] as ChoiceCase;
  /** Replay the case's log for `player`, then `after`. */
  const scripted = (player: "p1" | "p2", after: Actor): Actor => {
    const queue = c.log.filter((e) => e.player === player).map((e) => e.choice);
    return (request) => queue.shift() ?? after(request);
  };
  /** p1 tries the first pending candidate once, then plays firstLegal. */
  const run = async (): Promise<{ record: BattleRecord; received: RequestRecord[] }> => {
    const received: RequestRecord[] = [];
    let tried = false;
    const p1 = scripted("p1", (request) => {
      received.push(structuredClone(request));
      const pending = request.legal?.pending ?? [];
      if (tried || pending.length === 0) return firstLegal(request);
      tried = true;
      return pending[0] as string;
    });
    const record = await runBattle(
      options({ p1, p2: scripted("p2", firstLegal) }, { seed: c.seed, teams: c.teams }),
    );
    return { record, received };
  };

  it("puts the refusal in the refused player's own channel only", async () => {
    const { record } = await run();
    const refusal = `|error|${record.choices.find((e) => e.refused !== undefined)?.refused}`;
    expect(record.channels.p1).toContain(refusal);
    expect(record.channels.p2.some((line) => line.startsWith("|error|"))).toBe(false);
    // Each channel carries only its own player's requests.
    for (const player of ["p1", "p2"] as const) {
      const sides = record.channels[player]
        .filter((line) => line.startsWith("|request|"))
        .map((line) => (JSON.parse(line.slice("|request|".length)) as { side: { id: string } }).side.id);
      expect(new Set(sides)).toEqual(new Set([player]));
    }
  });

  it("is answered as a new request, judged and recorded, and the battle completes", async () => {
    const { record } = await run();
    expect(record.winner === null || ["p1", "p2"].includes(record.winner)).toBe(true);
    const refused = record.choices.filter((e) => e.refused !== undefined);
    expect(refused).toHaveLength(1);
    const entry = refused[0] as BattleRecord["choices"][number];
    expect(entry.refused).toMatch(
      /^\[Unavailable choice\] Can't switch: The active Pokémon is trapped/,
    );
    // The engine's updated request is the next one, and the CloneJudge
    // accepted its hash: the streamed request and the clone agree.
    const next = record.requests.p1[entry.request_seq];
    expect(next?.request.update).toBe(true);
    expect(next?.request_seq).toBe(entry.request_seq + 1);
    expect(next?.legal?.request_hash).toBe(next?.request_hash);
    // The SubmissionJudge on that real engine reply.
    expect(judgeSubmission(entry.choice, entry.refused as string)).toMatchObject({
      legal: false,
      revealed: true,
    });
    // The hidden verdicts are recorded for evaluation, refusal included.
    const hidden = record.hidden_verdicts.find((h) => h.request_seq === entry.request_seq);
    expect(hidden?.verdicts.find((v) => v.choice === entry.choice)?.legal).toBe(false);
  });

  it("never shows an actor a verdict on a pending candidate", async () => {
    const { record, received } = await run();
    const errors = record.hidden_verdicts.flatMap((h) => h.verdicts.map((v) => v.error));
    const hiddenErrors = new Set(errors.filter((e) => e !== ""));
    expect(hiddenErrors.size).toBeGreaterThan(0);
    const seen = JSON.stringify(received);
    for (const error of hiddenErrors) expect(seen).not.toContain(error);
    for (const request of received) {
      expect(Object.keys(request.legal ?? {}).sort()).toEqual([
        "choices",
        "pending",
        "request_hash",
        "request_seq",
        "schema_version",
        "showdown_commit",
      ]);
    }
  });

  it("replays identically from the recorded choices, refused one included", async () => {
    const { record: original } = await run();
    const replayed = await runBattle(
      options(
        { p1: replayActor(original.choices, "p1"), p2: replayActor(original.choices, "p2") },
        { seed: c.seed, teams: c.teams },
      ),
    );
    expect(replayed.omniscient_log).toEqual(original.omniscient_log);
    expect(replayed.choices).toEqual(original.choices);
  });

  it("aborts when a choice outside the pending set is refused that way", async () => {
    const respelled: Actor = scripted("p1", (request) => {
      const pending = request.legal?.pending[0];
      // The same pending switch, spelled without the space: not a candidate.
      return pending === undefined ? firstLegal(request) : pending.replace(", ", ",");
    });
    const battle = runBattle(
      options({ p1: respelled, p2: scripted("p2", firstLegal) }, { seed: c.seed, teams: c.teams }),
    );
    await expect(battle).rejects.toThrow(/\[Unavailable choice\]/);
  });
});

describe("judgeSubmission (the SubmissionJudge)", () => {
  it("tells a revealing refusal from an invalid one, on real engine output", async () => {
    const bad: Actor = () => "move 9";
    const error = await runBattle(options({ p1: engineDefault, p2: bad })).catch((e: Error) => e);
    const text = (error as Error).message.replace(/^p2: /, "");
    expect(text).toMatch(/^\[Invalid choice\]/);
    expect(judgeSubmission("move 9", text)).toEqual({
      choice: "move 9",
      legal: false,
      error: text,
      revealed: false,
    });
    // The revealing side is classified on the engine's real refusal in the
    // maybe-trapped battle above.
  });
});

function record0(): RequestRecord {
  return { phase: "move", player: "p1", request_hash: "0".repeat(64), request: {}, request_seq: 1 };
}
function legal0(): LegalChoiceSet {
  return {
    schema_version: 1,
    request_seq: 1,
    request_hash: "0".repeat(64),
    showdown_commit: "x",
    choices: [],
    pending: [],
  };
}
