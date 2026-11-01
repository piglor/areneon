// The opponent sheet setup of one run's pair in the evaluation commands (ADR-027 slice 5): from that run's own config, a usage error when
// this checkout does not compute the hash it names, and the client option that carries the statement.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, readTrainingRunConfig } from "../src/index.ts";
import { frozenPairClient, pairSheetSetup, UsageError } from "../src/progress-cli.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const engine = loadEngine();
const tiny = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")) as { model: { encoder: { options?: unknown } } };
const named = (hash: string) => {
  const raw = structuredClone(tiny);
  raw.model.encoder.options = { opponent_sheet: { feature_set: "opponent-sheet/1", feature_set_hash: hash } };
  return readTrainingRunConfig(raw);
};

describe("pairSheetSetup", () => {
  it("is undefined for a config without the option", () => {
    expect(pairSheetSetup(engine, readTrainingRunConfig(structuredClone(tiny)))).toBeUndefined();
  });

  it("is a usage error for a missing factory and for a hash this checkout does not compute", () => {
    expect(() => pairSheetSetup(engine, named("a".repeat(64)))).toThrow(UsageError);
    const port = { features_hash: "b".repeat(64), prior_format_id: GOLDEN_FORMAT } as never;
    expect(() => pairSheetSetup(engine, named("a".repeat(64)), () => port)).toThrow(/computes as b{64}/);
  });

  it("builds the setup when the hash is the one the port computes, and the client option carries the statement", () => {
    const port = { features_hash: "b".repeat(64), prior_format_id: GOLDEN_FORMAT } as never;
    const setup = pairSheetSetup(engine, named("b".repeat(64)), () => port);
    expect(setup?.statement).toEqual({ feature_set: "opponent-sheet/1", feature_set_hash: "b".repeat(64) });
    const client = frozenPairClient({ python: "python", runDir: ".", pair: { preview: "1".repeat(64), battle: "2".repeat(64) }, encoderVersion: "e", engineDataFile: "d.json", engineDataHash: "e".repeat(64), max_line_bytes: 1000, handshake_timeout_ms: 1000, ...(setup === undefined ? {} : { sheet: setup.statement }) });
    expect(client.sheetSource).toBeNull(); // nothing stated until a worker says so
  });
});
