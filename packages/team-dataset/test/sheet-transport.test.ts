// ADR-027 slice 4, ADR-025 erratum E4: the opponent sheet block across the worker boundary and into the training stream, with
// the real opponent model: the block a seat sends equals, byte for byte, the block the loader recomputes from the written Episode.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { BeliefError, refusedSheet } from "@areneon/opponent-model";
import {
  actorRule,
  createSheetBuilder,
  episodeFromRecord,
  GenerationError,
  loadMechanicsConfig,
  loadRulesSnapshot,
  loadTrainingSet,
  type PolicyClient,
  type PolicyQueryInput,
  policyGame,
  readPolicyQuery,
  readTrainingExample,
  readTrainingRunConfig,
  resolveFormat,
  sheetSetup,
  stampTeam,
  type TrainingRunConfig,
  type BeliefFeaturesPort,
} from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { jcs } from "../../showdown-bridge/src/canonical-json.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, goldenTeamVocabulary, syntheticFixtureHashes, syntheticTeam } from "../../showdown-bridge/src/golden.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { SAMPLES_DIR } from "../../showdown-bridge/src/samples.ts";
import { createBeliefFeatures } from "../src/belief-features.ts";
import { engine } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, GOLDEN_FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const port = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest });
const scenario_id = checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT));
const known = syntheticFixtureHashes(engine);
const sets = syntheticTeam();
const team = stampTeam(sets as PrivateTeamSpec["team"], known);
const POLICY = `policy:${"a".repeat(64)}`;
const budget = { preview_ms: 60_000, battle_ms: 60_000 };
const stated = { feature_set: "opponent-sheet/1", feature_set_hash: port.features_hash };

/** A stand-in worker that states the source, re-reads every query as the real client does and plays the actor rule. */
function standIn(sheetSource: typeof stated | null): PolicyClient & { queries: PolicyQueryInput[] } {
  const queries: PolicyQueryInput[] = [];
  return {
    queries,
    sheetSource,
    ask(_battle, query) {
      readPolicyQuery({ ...query, query_id: queries.length + 1 });
      queries.push(query);
      const open = query.candidates.filter((c) => !c.pending).map((c) => c.choice);
      const pending = query.candidates.filter((c) => c.pending).map((c) => c.choice);
      return Promise.resolve({ schema_version: 1, identity: query.identity, choice: actorRule({ choices: open, pending }), source_policy_id: query.policy_id, elapsed_ms: 0, deadline_outcome: "in_budget" });
    },
  };
}
const seat = (client: PolicyClient, using: BeliefFeaturesPort | null = port, record?: (seq: number, block: unknown) => void) => ({
  client,
  policy_id: POLICY,
  sampling: "argmax" as const,
  ...(using === null ? {} : { sheet: { builder: createSheetBuilder({ engine, vocabulary, port: using, scenario_id }), ...(record === undefined ? {} : { record }) } }),
});
const play = (p1: ReturnType<typeof seat>, p2: ReturnType<typeof seat>) =>
  policyGame({ engine, formatId: GOLDEN_FORMAT, seed: GOLDEN_SEED, game: 0, teams: { p1: sets, p2: sets }, known_synthetic: known, seats: { p1, p2 }, latency_budget: budget });

describe("a played game: the blocks sent equal the blocks the loader recomputes", () => {
  it("byte for byte, at every decision of both seats; a TrainingExample without the option carries none", async () => {
    const a = standIn(stated);
    const b = standIn(stated);
    const builders: ReturnType<typeof createSheetBuilder>[] = [];
    const recorded: unknown[] = [];
    const counted = (client: PolicyClient) => {
      const made = seat(client, port, (_seq, block) => void recorded.push(block));
      builders.push((made.sheet as NonNullable<typeof made.sheet>).builder);
      return made;
    };
    const { outcome, record } = await play(counted(a), counted(b));
    expect(outcome.kind).toBe("played");
    const played = record as NonNullable<typeof record>;
    const episodes = await Promise.all(
      (["p1", "p2"] as const).map((player) =>
        episodeFromRecord(played, player, { team, opponent_team: team, known_synthetic: known, behaviour_policy_id: POLICY, collection: { kind: "league", run_id: "league-1", seed: played.seed, opponent_policy_id: POLICY } }),
      ),
    );
    const { examples } = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY, sheet: { engine, vocabulary, port, scenario_id } });
    const sent = [...a.queries, ...b.queries];
    expect(sent.length).toBe(examples.length);
    expect(recorded.map((block) => jcs(block as never)).sort()).toEqual(sent.map((q) => jcs(q.opponent_sheet as never)).sort()); // the play-time sidecar hook sees what is sent
    let predicted = 0;
    for (const example of examples) {
      const query = sent.find((q) => q.observation.player === example.observation.player && q.identity.request_seq === example.observation.identity.request_seq) as PolicyQueryInput;
      expect(query, "a query for every example").toBeDefined();
      expect(query.opponent_sheet?.source).toEqual({ kind: "belief", hash: port.features_hash });
      expect(jcs(example.opponent_sheet as never)).toBe(jcs(query.opponent_sheet as never));
      readTrainingExample(example);
      if (example.opponent_sheet?.slots.some((s) => s.brought_source === "revealed")) predicted += 1;
    }
    expect(predicted).toBeGreaterThan(0); // after the leads are in play the block holds revealed slots
    // The seats counted their blocks; the golden mirror game has no refusal.
    const counts = builders.map((x) => x.counts());
    expect(counts.map((c) => c.requests)).toEqual([a.queries.length, b.queries.length]);
    expect(counts.every((c) => c.refused === 0)).toBe(true);
    const plain = await loadTrainingSet({ episodes, evaluation_manifests: [], known_synthetic: known, behaviour_policy_id: POLICY });
    expect(plain.examples.every((e) => !("opponent_sheet" in e))).toBe(true);
  }, 240_000);

  it("a worker that states no source gets no block, even from a seat that could build it", async () => {
    const a = standIn(null);
    const b = standIn(null);
    await play(seat(a), seat(b));
    expect([...a.queries, ...b.queries].every((q) => q.opponent_sheet === undefined)).toBe(true);
  }, 120_000);

  it("a source this seat builds that is not the stated one is a harness incident, not a forfeit", async () => {
    const { outcome } = await play(seat(standIn({ ...stated, feature_set_hash: "d".repeat(64) })), seat(standIn(null), null));
    expect(outcome).toMatchObject({ kind: "incident", incident: "policy_protocol" });
  }, 120_000);

  it("a refused belief is the unknown block with its code, counted, and the game goes on", async () => {
    const refusing: BeliefFeaturesPort = { ...port, sheet: ({ opponent }) => refusedSheet(opponent, "evidence_has_no_support", port.features_hash) };
    const a = standIn(stated);
    const { outcome } = await play(seat(a, refusing), seat(standIn(null), null));
    expect(outcome.kind).toBe("played");
    expect(a.queries.length).toBeGreaterThan(0);
    expect(a.queries.every((q) => q.opponent_sheet?.source.refusal === "evidence_has_no_support")).toBe(true);
  }, 120_000);

  it("a defect of the run is a crash forfeit, not a block", async () => {
    const broken: BeliefFeaturesPort = { ...port, sheet: () => { throw new BeliefError("format_mismatch", "x"); } };
    const { outcome } = await play(seat(standIn(stated), broken), seat(standIn(null), null));
    expect(outcome).toMatchObject({ kind: "forfeit", cause: "crash" });
  }, 120_000);
});

describe("sheetSetup (the run config names the source)", () => {
  const sample = JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/opponent-sheet.json"), "utf8")) as Record<string, unknown>;
  const configOf = (hash: string): TrainingRunConfig => {
    const copy = structuredClone(sample) as { model: { encoder: { options: { opponent_sheet: { feature_set_hash: string } } } } };
    copy.model.encoder.options.opponent_sheet.feature_set_hash = hash;
    return readTrainingRunConfig(copy);
  };
  const factory = (context: Parameters<typeof createBeliefFeatures>[0]) => createBeliefFeatures(context);

  it("builds the setup and the worker statement when the hash is the one the port computes", () => {
    const setup = sheetSetup(engine, configOf(port.features_hash), factory);
    expect(setup?.statement).toEqual(stated);
    expect(setup?.setup.scenario_id).toBe(scenario_id);
  });

  it("is refused for another hash, and for a script with no factory; a config without the option has none", () => {
    expect(() => sheetSetup(engine, configOf("0".repeat(64)), factory)).toThrow(GenerationError);
    expect(() => sheetSetup(engine, configOf(port.features_hash))).toThrow(/no belief features factory/);
    const plain = readTrainingRunConfig(JSON.parse(readFileSync(join(SAMPLES_DIR, "v1/training-run-config/valid/tiny.json"), "utf8")));
    expect(sheetSetup(engine, plain, factory)).toBeUndefined();
  });
});
