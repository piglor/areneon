import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine } from "../src/index.ts";
import { loadRandomCorpus, pendingFirstActor, playRandomBattle, RANDOM_MAX_TURNS, UNAVAILABLE_DIR } from "../src/random-corpus.ts";
import { readResimulationInput, ResimulationInputError, type ResimulationInput, resimulate } from "../src/resimulate.ts";

const engine = loadEngine();

async function played() {
  const spec = loadRandomCorpus(UNAVAILABLE_DIR)[0];
  if (spec === undefined) throw new Error("the unavailable corpus is empty");
  const record = await playRandomBattle(engine, spec, pendingFirstActor);
  const input: ResimulationInput = {
    format_id: GOLDEN_FORMAT,
    seed: spec.seed,
    max_turns: RANDOM_MAX_TURNS,
    teams: spec.teams,
    choices: record.choices.map(({ player, request_seq, choice }) => ({ player, request_seq, choice })),
  };
  return { record, input };
}

describe("resimulate", () => {
  it("reproduces a battle with refused choices: channels, log, result and decisions", async () => {
    const { record, input } = await played();
    expect(record.choices.some((c) => c.refused !== undefined)).toBe(true);
    const again = await resimulate(engine, readResimulationInput(JSON.parse(JSON.stringify(input))));
    expect(again.channels).toEqual(record.channels);
    expect(again.omniscient_log).toEqual([...record.omniscient_log]);
    expect(again.winner).toBe(record.winner);
    expect(again.turns).toBe(record.turns);
    expect(again.team_hashes).toEqual(record.team_hashes);
    expect(again.decisions.map(({ player, request_seq, choice, refused }) => ({ player, request_seq, choice, refused }))).toEqual(
      record.choices.map(({ player, request_seq, choice, refused }) => ({ player, request_seq, choice, refused })),
    );
    for (const decision of again.decisions) {
      expect(decision.candidates.length).toBeGreaterThan(0);
      expect(Object.keys(decision).every((k) => ["player", "request_seq", "choice", "refused", "candidates"].includes(k))).toBe(true);
    }
    // the input order does not matter
    const shuffled = await resimulate(engine, { ...input, choices: [...input.choices].reverse() });
    expect(shuffled.channels).toEqual(again.channels);
  });

  it("refuses choices that are not the battle's: a missing one, an extra one, a twin", async () => {
    const { input } = await played();
    await expect(resimulate(engine, { ...input, choices: input.choices.slice(0, -1) })).rejects.toThrow();
    const extra = { player: "p1" as const, request_seq: 999, choice: "default" };
    await expect(resimulate(engine, { ...input, choices: [...input.choices, extra] })).rejects.toThrow(ResimulationInputError);
    const [first] = input.choices;
    await expect(resimulate(engine, { ...input, choices: [...input.choices, { ...(first as object), choice: "x" } as never] })).rejects.toThrow(
      "two choices answer one request",
    );
    await expect(resimulate(engine, { ...input, max_turns: 1 })).rejects.toThrow();
  });
});

describe("readResimulationInput", () => {
  it("accepts the request and refuses every other shape without quoting a value", async () => {
    const { input } = await played();
    expect(readResimulationInput(JSON.parse(JSON.stringify(input)))).toEqual(input);
    const bad: unknown[] = [
      null,
      [],
      { ...input, extra: 1 },
      { ...input, format_id: "" },
      { ...input, format_id: 3 },
      { ...input, seed: "SECRET-seed" },
      { ...input, max_turns: 0 },
      { ...input, max_turns: 1.5 },
      { ...input, teams: { p1: input.teams.p1 } },
      { ...input, teams: { p1: input.teams.p1, p2: [] } },
      { ...input, choices: "x" },
      { ...input, choices: [null] },
      { ...input, choices: [{ player: "p3", request_seq: 1, choice: "x" }] },
      { ...input, choices: [{ player: "p1", request_seq: 0, choice: "x" }] },
      { ...input, choices: [{ player: "p1", request_seq: 1, choice: 4 }] },
      { ...input, choices: [{ player: "p1", request_seq: 1, choice: "x", extra: 1 }] },
    ];
    for (const value of bad) {
      expect(() => readResimulationInput(value)).toThrow(ResimulationInputError);
    }
    expect(() => readResimulationInput({ ...input, seed: "SECRET-seed" })).not.toThrow(/SECRET/);
  });
});
