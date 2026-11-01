// ADR-022, `probe-damage/3`: the `belief_threats` array across the worker boundary and into the training stream (the reader rules
// of feature-rows.ts on the committed samples; the Python worker and the stream checker hold the same rules, tests/ml/test_worker_features.py).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { featureCoverageProblem, featureRowsProblem, type WireFeatureRows } from "../src/feature-rows.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";
import { loadEngine, readTrainingExample } from "../src/index.ts";
import { readPolicyQuery } from "../src/policy-protocol.ts";
import { SAMPLES_DIR } from "../src/samples.ts";

const engine = loadEngine();
type Sample = Record<string, unknown> & { candidate_features: WireFeatureRows & Record<string, unknown[]>; observation: { log: string[] }; candidates: { choice: string }[] };
const sample = (file: string): Sample => JSON.parse(readFileSync(`${SAMPLES_DIR}/v1/${file}.json`, "utf8")) as never;
const dex = engine.sim.Dex.mod(engine.sim.Dex.formats.get(GOLDEN_FORMAT).mod) as unknown as { moves: { get(id: string): { category: string } } };
const coverage = (query: Sample): string | null => featureCoverageProblem(query as never, query.candidate_features, (id) => dex.moves.get(id).category !== "Status");

describe("the reader rules of belief_threats", () => {
  it("the committed /3 samples read, and their arrays cover the request", () => {
    const query = sample("policy-query/valid/battle-features-3");
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toBeNull();
    expect(query.candidate_features.belief_threats?.length).toBeGreaterThan(0);
    expect(JSON.stringify(query.candidate_features.belief_threats)).toMatch(/inclusion_ppm/);
    readPolicyQuery({ ...query, query_id: 1 });
    readTrainingExample(sample("training-example/valid/battle-features-3"));
    const preview = sample("policy-query/valid/preview-features-3");
    expect(featureRowsProblem(preview.candidate_features)).toBeNull();
    expect(coverage(preview)).toBeNull();
  });

  it("belief_threats are present exactly under probe-damage/3", () => {
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-3-belief-under-2").candidate_features)).toMatch(/present exactly under probe-damage\/2/);
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-3-missing-belief-threats").candidate_features)).toMatch(/present exactly/);
    // a /2 value keeps reading
    expect(featureRowsProblem(sample("policy-query/valid/battle-features-2").candidate_features)).toBeNull();
  });

  it("belief_threats are sorted and unique by their key", () => {
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-3-belief-unsorted").candidate_features)).toMatch(/belief_threats must be sorted and unique/);
    const query = sample("policy-query/valid/battle-features-3");
    const rows = query.candidate_features.belief_threats as unknown[];
    expect(featureRowsProblem({ ...query.candidate_features, belief_threats: [rows[0], ...rows] } as never)).toMatch(/belief_threats must be sorted and unique/);
  });

  it("a belief threat may not repeat a move already in threats for that foe", () => {
    expect(featureRowsProblem(sample("policy-query/reader-invalid/features-3-belief-repeats-revealed").candidate_features)).toMatch(/repeats a move already in threats/);
    // the same move for ANOTHER foe is fine
    const query = sample("policy-query/valid/battle-features-3");
    const threat = (query.candidate_features.threats as { foe: number; move: string; own: number; mega: boolean }[])[0] as { foe: number; move: string; own: number; mega: boolean };
    const other = { ...(query.candidate_features.belief_threats?.[0] as object), foe: threat.foe === 1 ? 2 : 1, move: threat.move };
    const merged = [...(query.candidate_features.belief_threats as unknown as { own: number; mega: boolean; foe: number; move: string }[]), other as never].sort((a, b) => a.own - b.own || Number(a.mega) - Number(b.mega) || a.foe - b.foe || (a.move < b.move ? -1 : a.move > b.move ? 1 : 0));
    // (threats name this move for BOTH foes in the sample, so use the repeated-revealed sample's rule only for the foe it names)
    expect(featureRowsProblem({ ...query.candidate_features, belief_threats: merged } as never)).toMatch(/repeats a move/);
  });

  it("belief threats name only the own Pokemon and Mega variants the request implies", () => {
    const query = sample("policy-query/reader-invalid/features-3-belief-foreign-own");
    expect(featureRowsProblem(query.candidate_features)).toBeNull();
    expect(coverage(query)).toMatch(/belief_threats name an own Pokemon/);
  });
});
