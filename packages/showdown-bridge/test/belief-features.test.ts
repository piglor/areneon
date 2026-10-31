// The bridge side of the opponent sheet block (ADR-025 erratum E4): the structural port, the per-game counts and the
// refusals, with a stand-in port (the real model is joined in `packages/team-dataset`, which the bridge does not import).
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { OpponentSheet } from "@areneon/schema/types/v1/opponent-sheet.ts";
import {
  BeliefFeaturesError,
  type BeliefFeaturesPort,
  createSheetBuilder,
  opponentSheetProblem,
  refusedSharePpm,
  type SheetInput,
  type TeamVocabulary,
  type Engine,
} from "../src/index.ts";

const HASH = "d".repeat(64);
const unknownSlot = (species: string) => ({
  species,
  brought_ppm: 0,
  brought_source: "unknown" as const,
  item: { source: "unknown" as const, entries: [], other_ppm: 0 },
  ability: { source: "unknown" as const, entries: [], other_ppm: 0 },
  nature: { source: "unknown" as const, entries: [], other_ppm: 0 },
  moves: { source: "unknown" as const, entries: [], slots_known: 0 },
});
const block = (refusal?: NonNullable<OpponentSheet["source"]["refusal"]>): OpponentSheet => ({ source: { kind: "belief", hash: HASH, ...(refusal === undefined ? {} : { refusal }) }, slots: [unknownSlot("alpha")] });

const vocabulary = { format_id: "testformat" } as unknown as TeamVocabulary;
const engine = {} as Engine;
// A wait request: the trapping check declines it (null) without touching the engine.
const input = { request: { wait: true, side: { id: "p1", pokemon: [] } }, team: [], state: { opponent: { roster: [] }, player: "p1" }, flags: [] } as unknown as SheetInput;

function portOf(results: (OpponentSheet | Error)[], format = "testformat"): BeliefFeaturesPort & { calls: unknown[] } {
  const calls: unknown[] = [];
  return {
    identity_hash: "e".repeat(64),
    features_hash: HASH,
    prior_hash: "f".repeat(64),
    belief_config_hash: "a".repeat(64),
    prior_format_id: format,
    k: { item: 4, ability: 3, nature: 3, moves: 8 },
    calls,
    sheet(args) {
      calls.push(args);
      const next = results.shift() as OpponentSheet | Error;
      if (next instanceof Error) throw next;
      return next;
    },
  };
}

describe("the sheet builder", () => {
  it("hands the port the reader's opponent half, its flags and the trapping check (null for a request it declines) and returns the block", () => {
    const port = portOf([block()]);
    const builder = createSheetBuilder({ engine, vocabulary, port, scenario_id: "scenario" });
    expect(builder.build(input)).toEqual(block());
    expect(port.calls).toEqual([{ opponent: { roster: [] }, flags: [], trapping: null }]);
  });

  it("counts blocks and refusals per code, sorted by code; a refusal is a block, not a thrown error", () => {
    const builder = createSheetBuilder({ engine, vocabulary, port: portOf([block(), block("reader_flagged"), block("disguise_unsettled"), block("reader_flagged")]), scenario_id: "scenario" });
    for (let i = 0; i < 4; i++) builder.build(input);
    expect(builder.counts()).toEqual({ requests: 4, refused: 3, refusals: [{ code: "disguise_unsettled", count: 1 }, { code: "reader_flagged", count: 2 }] });
    expect(refusedSharePpm(builder.counts())).toBe(750000);
    expect(refusedSharePpm({ requests: 0, refused: 0 })).toBe(0);
  });

  it("a defect of the run propagates and is not counted as a refusal", () => {
    const builder = createSheetBuilder({ engine, vocabulary, port: portOf([new Error("format_mismatch")]), scenario_id: "scenario" });
    expect(() => builder.build(input)).toThrow("format_mismatch");
    expect(builder.counts()).toEqual({ requests: 0, refused: 0, refusals: [] });
  });

  it("refuses a prior of another format than the vocabulary's: never used silently", () => {
    expect(() => createSheetBuilder({ engine, vocabulary, port: portOf([], "otherformat"), scenario_id: "scenario" })).toThrow(BeliefFeaturesError);
  });

  it("a refused block passes the reader rules, as the block of an unrefused unknown field does", () => {
    expect(opponentSheetProblem(block("evidence_has_no_support"))).toBeNull();
    expect(opponentSheetProblem(block())).toBeNull();
  });
});

describe("the input is the player's own view and nothing else", () => {
  it("has no field for the opponent's team, the omniscient log, hidden verdicts, a record or the seed", () => {
    // @ts-expect-error the opponent's real team is not an input
    const team: SheetInput = { ...input, opponent_team: [] };
    // @ts-expect-error the omniscient log is not an input
    const log: SheetInput = { ...input, omniscient: [] };
    // @ts-expect-error hidden verdicts are not an input
    const verdicts: SheetInput = { ...input, hidden_verdicts: [] };
    // @ts-expect-error a battle record is not an input
    const record: SheetInput = { ...input, record: {} };
    // @ts-expect-error the seed is not an input
    const seed: SheetInput = { ...input, seed: "sodium,00" };
    expect([team, log, verdicts, record, seed]).toHaveLength(5);
  });

  it("the module imports no runner, omniscient, record or opponent-model module", () => {
    const code = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "src", "belief-features.ts"), "utf8");
    for (const m of code.matchAll(/^import\s[^;]*?from\s+"([^"]+)";/gms)) expect(m[1], m[0]).not.toMatch(/runner|omniscient|record|opponent-model|hidden|choice-judge/);
  });
});
