// The shared contract sample corpus (ADR-008 §2): valid and invalid instances
// of every contract, plus reader-invalid ones (ADR-009 §4: schema-valid, but a
// reader must refuse them), most built from the golden battle, which every language
// with contracts must classify the same way, plus their JCS digests. Captured
// by scripts/capture-samples.ts into fixtures/contracts/; CI compares a fresh
// capture with the committed files.

import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { legalChoiceSet } from "./choice-judge.ts";
import { engineDataHash, exportEngineData } from "./engine-data.ts";
import type { Engine } from "./engine.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { GOLDEN_FORMAT, goldenBattle, syntheticFixtureHashes, syntheticTeam } from "./golden.ts";
import { observe, recordedChannel } from "./player-channel.ts";
import { contractVersions } from "./policy-protocol.ts";
import { battleClass, stampTeam } from "./team-class.ts";
import { evaluationRunManifest, loadTrainingSet } from "./training-loader.ts";
import type { RequestRecord } from "./runner.ts";

export const SAMPLES_DIR = fileURLToPath(new URL("../../../fixtures/contracts", import.meta.url));
export const SAMPLE_HASHES = "hashes.json";

const TIMER = { starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 };

/** A copy of `value` with `change` applied: one mutation per invalid sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): unknown {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

/**
 * Every sample, keyed by its path under `fixtures/contracts/`:
 * `v1/<contract>/{valid,invalid,reader-invalid}/<case>.json`, plus
 * `hashes.json` mapping each sample's path to the SHA-256 of its JCS bytes.
 * A `reader-invalid` sample passes the schema; the contract's reader refuses it.
 */
export async function contractSamples(engine: Engine): Promise<Record<string, unknown>> {
  const record = await goldenBattle(engine);
  const known = syntheticFixtureHashes(engine);
  const team = stampTeam(syntheticTeam() as PrivateTeamSpec["team"], known);
  // The golden battle has all three phases (golden.test.ts pins GOLDEN_PHASES).
  const first = (phase: RequestRecord["phase"]) => record.requests.p1.find((r) => r.phase === phase) as RequestRecord;
  const preview = first("preview");
  const turn = first("move");
  const force = first("forceSwitch");
  const at = (seq: number) => recordedChannel(record, "p1", seq, battleClass(record.team_hashes, known));
  const context = {
    timer_source: "server" as const,
    timer_rules: TIMER,
    turn_seconds_left: 55,
    bank_seconds_left: 400,
    grace_seconds_left: 0,
    remaining_decisions_estimate: 8,
    safety_margin_ms: 3000,
    submit_within_ms: 20000,
    policy_budget_ms: 19000,
  };
  const legal = legalChoiceSet({ request: preview, showdown_commit: engine.commit, judge: null });
  const previewObservation = await observe({ channel: at(preview.request_seq), team, manifest: record.manifest });
  const battleObservation = await observe({
    channel: at(turn.request_seq),
    team,
    manifest: record.manifest,
    decision_context: context,
  });
  const episode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
  });
  const league = { kind: "league", run_id: "league-run-1", seed: record.seed, opponent_policy_id: "first-legal@golden" } as const;
  const evaluation = { kind: "evaluation", run_id: "eval-run-1", sealed: true, seed: record.seed } as const;
  const classed = { opponent_team_data_class: "synthetic" } as const;
  const collected = (collection: unknown) =>
    mutated(episode, (e) => ((e.provenance as Record<string, unknown>).collection = collection));
  const identity = { request_seq: turn.request_seq, request_hash: turn.request_hash };

  const leagueEpisode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
    collection: league,
  });
  const sealedEpisode = await episodeFromRecord(record, "p1", {
    team,
    opponent_team: team,
    known_synthetic: known,
    behaviour_policy_id: "first-legal@golden",
    collection: evaluation,
  });
  // Hash-only: all of a sealed run that may leave the isolated store (ADR-009 §4).
  const sealedManifest = await evaluationRunManifest({ run_id: "eval-run-1", sealed: true, episodes: [sealedEpisode] });
  const training = await loadTrainingSet({ episodes: [leagueEpisode], evaluation_manifests: [sealedManifest], known_synthetic: known, behaviour_policy_id: "first-legal@golden" });
  const exampleOf = (kind: "preview" | "battle") => training.examples.find((e) => e.observation.kind === kind) as (typeof training.examples)[number];
  // The stored shape before candidates and outcome: still valid (ADR-010 section 5).
  const legacyExample = mutated(exampleOf("battle"), (e) => {
    delete e.candidates;
    delete e.outcome;
  });
  const legacySet = mutated(training.set, (t) => {
    delete t.learner_ready;
    delete t.behaviour_policy_id;
  });
  const candidatesOf = (count: number) => Array.from({ length: count }, (_, i) => ({ choice: `team ${i}`, pending: false }));
  const [hashA, hashB] = ["a", "b"].map((c) => c.repeat(64)) as [string, string];

  // A slim table (full type list, two entries of each other vocabulary): the
  // complete one is fixtures/showdown/<format>/engine-data.json.
  const full = exportEngineData(engine, GOLDEN_FORMAT);
  const species = full.species.filter((x) => x.base_species === x.id).slice(0, 2);
  const moves = full.moves.slice(0, 2);
  const engineData = {
    ...full,
    species,
    moves,
    items: full.items.slice(0, 2),
    abilities: full.abilities.filter((a) => species.some((x) => x.abilities.includes(a.id))),
    move_targets: [...new Set(moves.map((m) => m.target))].sort(),
    move_flags: [...new Set(moves.flatMap((m) => m.flags))].sort(),
  };
  const [speciesA, speciesB] = engineData.species as [(typeof species)[number], (typeof species)[number]];

  // The policy worker's contracts (ADR-010 section 2). A query is the
  // training example's observation and candidates plus a sampling rule; its
  // A2 form adds an rqid and a decision context, which the encoder ignores.
  const policyId = `policy:${"c".repeat(64)}`;
  const queryOf = (kind: "preview" | "battle", sampling: unknown) => {
    const example = exampleOf(kind);
    return {
      schema_version: 1,
      query_id: 1,
      identity: example.observation.identity,
      observation: example.observation,
      candidates: (example.candidates as NonNullable<typeof example.candidates>).map(({ choice, pending }) => ({ choice, pending })),
      sampling,
      policy_id: policyId,
    };
  };
  const sampleBy = { kind: "sample", u: 4503599627370496 };
  const battleQuery = queryOf("battle", sampleBy);
  const previewQuery = queryOf("preview", { kind: "argmax" });
  const withRqid = (line: string): string =>
    line.startsWith("|request|") ? `|request|${JSON.stringify({ ...JSON.parse(line.slice("|request|".length)), rqid: 7 })}` : line;
  const battleQueryA2 = mutated(battleQuery, (q) => {
    q.query_id = 2;
    const observation = q.observation as { identity: Record<string, unknown>; log: string[]; decision_context?: unknown };
    observation.identity = { ...observation.identity, rqid: 7 };
    observation.log = observation.log.map(withRqid);
    observation.decision_context = context;
    q.identity = observation.identity;
    q.sampling = { kind: "argmax" };
  });
  const recommendation = {
    schema_version: 1,
    identity: battleQuery.identity,
    choice: battleQuery.candidates[0]?.choice,
    source_policy_id: policyId,
    elapsed_ms: 3,
    deadline_outcome: "in_budget",
  };
  const replyOf = (reply: unknown) => ({ schema_version: 1, query_id: 1, reply });
  const hello = { schema_version: 1, message: { kind: "hello", contracts: contractVersions() } };
  const checkpoints = [
    { role: "battle", checkpoint_id: hashA },
    { role: "preview", checkpoint_id: hashB },
  ];
  const ready = {
    schema_version: 1,
    message: {
      kind: "ready",
      contracts: contractVersions(),
      encoder_version: "stub-1",
      engine_data_hash: engineDataHash(full),
      checkpoints,
    },
  };
  const readyWith = (change: (m: Record<string, unknown>) => void) => mutated(ready, (r) => change(r.message as Record<string, unknown>));
  const helloWith = (change: (m: Record<string, unknown>) => void) => mutated(hello, (h) => change(h.message as Record<string, unknown>));
  const contractsOf = () => contractVersions() as unknown as Record<string, unknown>[];

  // A CheckpointRecord (ADR-010 section 5): ids are placeholders, the shape is what the corpus pins.
  const hexOf = (c: string): string => c.repeat(64);
  const checkpointRecord = {
    schema_version: 1,
    checkpoint_id: hexOf("1"),
    role: "battle",
    lineage_id: "main-0",
    parent: hexOf("2"),
    league_opponents: [hexOf("3"), hexOf("4")],
    training_set_hashes: [hexOf("5"), hexOf("6")],
    team_set_manifest_hash: hexOf("7"),
    run_config_hash: hexOf("8"),
    learner_state_hash: hexOf("9"),
    encoder_version: `enc1:${hexOf("e")}`,
    data_class: "synthetic",
    manifest: record.manifest,
  };
  const recordWith = (change: (r: Record<string, unknown>) => void) => mutated(checkpointRecord, change);

  const valid: Record<string, unknown> = {
    "engine-data/golden": engineData,
    "checkpoint-record/battle": checkpointRecord,
    "checkpoint-record/preview-fresh": recordWith((r) => {
      r.role = "preview";
      delete r.parent;
      r.league_opponents = [];
    }),
    "checkpoint-record/exploiter": recordWith((r) => {
      r.lineage_id = "exploiter-of-main-0";
      r.exploiter_target = hexOf("2");
      r.league_opponents = [hexOf("2")];
    }),
    "training-example/preview": exampleOf("preview"),
    "training-example/battle": exampleOf("battle"),
    "training-example/legacy-battle": legacyExample,
    "training-set/league": training.set,
    "training-set/legacy": legacySet,
    "evaluation-run-manifest/sealed": sealedManifest,
    "evaluation-run-manifest/open-empty": { schema_version: 1, run_id: "eval-run-2", sealed: false, episode_hashes: [] },
    "request-identity/a1": identity,
    "request-identity/server": { ...identity, rqid: 5 },
    "legal-choice-set/preview": legal,
    "decision-context/battle": context,
    "engine-request/preview": preview.request,
    "engine-request/turn": turn.request,
    "engine-request/force-switch": force.request,
    "player-channel-log/p1": { schema_version: 1, data_class: "synthetic", player: "p1", lines: record.channels.p1 },
    "private-team-spec/synthetic": team,
    "version-manifest/golden": record.manifest,
    "preview-observation/p1": previewObservation,
    "battle-observation/p1": battleObservation,
    "policy-query/battle": battleQuery,
    "policy-query/preview": previewQuery,
    "policy-query/battle-a2": battleQueryA2,
    "policy-reply/recommendation": replyOf({ kind: "recommendation", recommendation }),
    "policy-reply/unsupported-candidate": replyOf({ kind: "error", error: "unsupported_candidate", detail: "a candidate has a shape the worker does not know" }),
    "policy-reply/invalid-query": replyOf({ kind: "error", error: "invalid_query", detail: "the query's identity differs from its observation's identity" }),
    "policy-handshake/hello": hello,
    "policy-handshake/ready": ready,
    "recommendation/in-budget": {
      schema_version: 1,
      identity,
      choice: "move 1 1, move 1 2",
      source_policy_id: "first-legal@golden",
      elapsed_ms: 3,
      deadline_outcome: "in_budget",
    },
    "episode/p1": episode,
    "episode/league": leagueEpisode,
    "episode/evaluation-sealed": sealedEpisode,
  };
  // Schema-valid, refused by readEpisode (the sparring rule and the class rule).
  // No valid sparring sample exists: it must be private, and the data-class
  // guard refuses a committed "data_class": "private".
  const readerInvalid: Record<string, unknown> = {
    // Schema-valid, refused by the CheckpointRecord reader: the arrays are sorted and unique, and no id refers to the record itself.
    "checkpoint-record/unsorted-league-opponents": recordWith((r) => (r.league_opponents = [hexOf("4"), hexOf("3")])),
    "checkpoint-record/duplicate-training-sets": recordWith((r) => (r.training_set_hashes = [hexOf("5"), hexOf("5")])),
    "checkpoint-record/parent-is-self": recordWith((r) => (r.parent = r.checkpoint_id)),
    "checkpoint-record/opponent-is-self": recordWith((r) => (r.league_opponents = [r.checkpoint_id])),
    "checkpoint-record/exploiter-of-self": recordWith((r) => (r.exploiter_target = r.checkpoint_id)),
    "engine-data/unsorted-species": { ...engineData, species: [speciesB, speciesA] },
    "engine-data/duplicate-species": { ...engineData, species: [speciesA, speciesA] },
    "engine-data/dangling-type": mutated(engineData, (t) => ((t.species as { types: string[] }[])[0] as { types: string[] }).types = ["nosuchtype"]),
    "engine-data/short-damage-row": mutated(engineData, (t) => ((t.types as { damage_taken: number[] }[])[0] as { damage_taken: number[] }).damage_taken.pop()),
    // Schema-valid, refused by the readers: the hashes must be sorted and unique.
    "evaluation-run-manifest/unsorted": { schema_version: 1, run_id: "eval-run-3", sealed: false, episode_hashes: [hashB, hashA] },
    "evaluation-run-manifest/duplicate": { schema_version: 1, run_id: "eval-run-3", sealed: false, episode_hashes: [hashA, hashA] },
    // Schema-valid, refused by readTrainingSet: a learner_ready set must name its policy.
    "training-set/learner-ready-no-policy": mutated(training.set, (t) => delete t.behaviour_policy_id),
    // Schema-valid, refused by the stream check: choice must be among the candidates.
    "training-example/choice-not-in-candidates": { ...exampleOf("battle"), choice: "not a candidate" },
    // Schema-valid, refused by the worker's and the client's readers: the profile cannot compare fields.
    "policy-query/identity-differs": mutated(battleQuery, (q) => (q.identity = { ...battleQuery.identity, request_seq: battleQuery.identity.request_seq + 1 })),
    "policy-query/duplicate-candidate": mutated(battleQuery, (q) => ((q.candidates as unknown[])[1] = (battleQuery.candidates as unknown[])[0])),
    "policy-handshake/unsorted-contracts": helloWith((m) => (m.contracts = contractsOf().reverse())),
    "policy-handshake/duplicate-contracts": helloWith((m) => (m.contracts = [contractsOf()[0], contractsOf()[0]])),
    "policy-handshake/unsorted-checkpoints": readyWith((m) => (m.checkpoints = [...checkpoints].reverse())),
    "training-set/unsorted-episodes": { ...training.set, episode_hashes: [hashB, hashA] },
    "training-set/duplicate-manifests": { ...training.set, evaluation_manifest_hashes: [hashA, hashA] },
    "episode/sparring-synthetic": collected({ kind: "sparring", session_id: "sparring-session-1" }),
    "episode/class-below-opponent": collected({ ...league, opponent_team_data_class: "private" }),
  };
  const invalid: Record<string, unknown> = {
    "checkpoint-record/no-manifest": recordWith((r) => delete r.manifest),
    "checkpoint-record/short-checkpoint-id": recordWith((r) => (r.checkpoint_id = "abc")),
    "checkpoint-record/uppercase-parent": recordWith((r) => (r.parent = hexOf("A"))),
    "checkpoint-record/unknown-role": recordWith((r) => (r.role = "value")),
    "checkpoint-record/public-class": recordWith((r) => (r.data_class = "public")),
    "checkpoint-record/blank-lineage": recordWith((r) => (r.lineage_id = "")),
    "checkpoint-record/no-learner-state": recordWith((r) => delete r.learner_state_hash),
    "checkpoint-record/extra-field": recordWith((r) => (r.verdicts = [])),
    "checkpoint-record/next-version": recordWith((r) => (r.schema_version = 2)),
    "engine-data/damage-code-4": mutated(engineData, (t) => ((t.types as { damage_taken: number[] }[])[0] as { damage_taken: number[] }).damage_taken[0] = 4),
    "engine-data/three-types": mutated(engineData, (t) => ((t.species as { types: string[] }[])[0] as { types: string[] }).types = ["bug", "dark", "fire"]),
    "engine-data/unknown-category": mutated(engineData, (t) => ((t.moves as { category: string }[])[0] as { category: string }).category = "Typeless"),
    "engine-data/extra-field": { ...engineData, learnsets: [] },
    "engine-data/next-version": { ...engineData, schema_version: 2 },
    "engine-data/short-commit": { ...engineData, showdown_commit: "7244fe8" },
    "training-example/provenance": { ...exampleOf("battle"), provenance: { seed: record.seed } },
    "training-example/no-episode-hash": mutated(exampleOf("battle"), (e) => delete e.episode_hash),
    "training-example/omniscient-observation": mutated(exampleOf("battle"), (e) => {
      (e.observation as Record<string, unknown>).omniscient_log = ["|split|p2"];
    }),
    "training-example/short-episode-hash": { ...exampleOf("battle"), episode_hash: "abc" },
    "training-example/empty-candidates": { ...exampleOf("battle"), candidates: [] },
    "training-example/too-many-candidates": { ...exampleOf("battle"), candidates: candidatesOf(1025) },
    "training-example/candidate-no-pending": mutated(exampleOf("battle"), (e) => delete (e.candidates as Record<string, unknown>[])[0]?.pending),
    "training-example/candidate-verdict": mutated(exampleOf("battle"), (e) => ((e.candidates as Record<string, unknown>[])[0] as Record<string, unknown>).legal = true),
    "training-example/unknown-outcome": { ...exampleOf("battle"), outcome: "draw" },
    "training-set/learner-ready-false": { ...training.set, learner_ready: false },
    "training-set/empty-policy-id": { ...training.set, behaviour_policy_id: "" },
    "training-set/no-stream-hash": mutated(training.set, (t) => delete t.stream_sha256),
    "training-set/no-examples": { ...training.set, examples: 0 },
    "training-set/public-class": { ...training.set, data_class: "public" },
    "evaluation-run-manifest/no-run-id": { schema_version: 1, sealed: true, episode_hashes: [] },
    "evaluation-run-manifest/uppercase-hash": { ...sealedManifest, episode_hashes: [hashA.toUpperCase()] },
    "evaluation-run-manifest/sealed-flag-string": { ...sealedManifest, sealed: "yes" },
    "policy-query/no-candidates": { ...battleQuery, candidates: [] },
    "policy-query/too-many-candidates": { ...battleQuery, candidates: candidatesOf(1025) },
    "policy-query/seed": { ...battleQuery, seed: record.seed },
    "policy-query/candidate-verdict": mutated(battleQuery, (q) => ((q.candidates as Record<string, unknown>[])[0] as Record<string, unknown>).legal = true),
    "policy-query/unknown-sampling": { ...battleQuery, sampling: { kind: "temperature", t: 2 } },
    "policy-query/u-negative": { ...battleQuery, sampling: { kind: "sample", u: -1 } },
    "policy-query/argmax-with-u": { ...battleQuery, sampling: { kind: "argmax", u: 1 } },
    "policy-query/no-policy-id": mutated(battleQuery, (q) => delete q.policy_id),
    "policy-query/no-query-id": mutated(battleQuery, (q) => delete q.query_id),
    "policy-query/omniscient-observation": mutated(battleQuery, (q) => ((q.observation as Record<string, unknown>).omniscient_log = ["|split|p2"])),
    "policy-reply/unknown-error": replyOf({ kind: "error", error: "timeout", detail: "slow" }),
    "policy-reply/unknown-kind": replyOf({ kind: "retry" }),
    "policy-reply/extra-field": { ...replyOf({ kind: "recommendation", recommendation }), probabilities: [1] },
    "policy-reply/no-query-id": mutated(replyOf({ kind: "recommendation", recommendation }), (r) => delete r.query_id),
    "policy-reply/bad-outcome": replyOf({ kind: "recommendation", recommendation: { ...recommendation, deadline_outcome: "late" } }),
    "policy-handshake/hello-with-encoder": helloWith((m) => (m.encoder_version = "stub-1")),
    "policy-handshake/unknown-contract": helloWith((m) => (m.contracts = [{ ...contractsOf()[0], name: "Episode" }])),
    "policy-handshake/short-schema-hash": helloWith((m) => (m.contracts = [{ ...contractsOf()[0], schema_sha256: "abc" }])),
    "policy-handshake/ready-no-checkpoints": readyWith((m) => delete m.checkpoints),
    "policy-handshake/unknown-kind": helloWith((m) => (m.kind = "bye")),
    "request-identity/rqid-zero": { ...identity, rqid: 0 },
    "legal-choice-set/next-version": mutated(legal, (s) => (s.schema_version = 2)),
    "decision-context/negative-budget": { ...context, policy_budget_ms: -1 },
    "engine-request/z-move-offer": mutated(turn.request, (r) => {
      ((r.active as unknown[])[0] as Record<string, unknown>).canZMove = [];
    }),
    "player-channel-log/public-class": {
      schema_version: 1,
      data_class: "public",
      player: "p1",
      lines: record.channels.p1,
    },
    "private-team-spec/extra-set-field": mutated(team, (s) => {
      ((s.team as unknown[])[0] as Record<string, unknown>).secret = "x";
    }),
    "version-manifest/short-commit": { ...record.manifest, showdown_commit: "7244fe8" },
    "preview-observation/omniscient-log": { ...previewObservation, omniscient_log: ["|split|p2"] },
    "battle-observation/opponent-team": { ...battleObservation, opponent_team: team },
    "recommendation/unknown-outcome": {
      schema_version: 1,
      identity,
      choice: "move 1 1, move 1 2",
      source_policy_id: "first-legal@golden",
      elapsed_ms: 3,
      deadline_outcome: "late",
    },
    "episode/omniscient-log": { ...episode, omniscient_log: [] },
    "episode/collection-no-run-id": collected({ ...classed, kind: "league", seed: record.seed, opponent_policy_id: "first-legal@golden" }),
    "episode/collection-league-sealed": collected({ ...league, ...classed, sealed: false }),
    "episode/collection-evaluation-no-seed": collected({ ...classed, kind: "evaluation", run_id: "eval-run-1", sealed: true }),
    "episode/collection-bad-seed": collected({ ...league, ...classed, seed: "1,2,3,4" }),
    "episode/collection-no-opponent-class": collected(league),
    "episode/collection-bad-opponent-class": collected({ ...league, opponent_team_data_class: "public" }),
    "episode/collection-unknown-kind": collected({ ...classed, kind: "training", run_id: "league-run-1" }),
    "episode/collection-sparring-run-id": collected({ kind: "sparring", session_id: "s", run_id: "r" }),
  };

  const samples: Record<string, unknown> = {};
  for (const [kind, cases] of [["valid", valid], ["invalid", invalid], ["reader-invalid", readerInvalid]] as const) {
    for (const [key, value] of Object.entries(cases)) {
      const [contract, name] = key.split("/") as [string, string];
      samples[`v1/${contract}/${kind}/${name}.json`] = value;
    }
  }
  const hashes: Record<string, string> = {};
  for (const [path, value] of Object.entries(samples)) {
    hashes[path] = createHash("sha256").update(jcs(value as Json)).digest("hex");
  }
  samples[SAMPLE_HASHES] = hashes;
  return samples;
}
