// The shared contract sample corpus (ADR-008 §2): valid and invalid instances
// of every contract, most built from the golden battle, which every language
// with contracts must classify the same way, plus their JCS digests. Captured
// by scripts/capture-samples.ts into fixtures/contracts/; CI compares a fresh
// capture with the committed files.

import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import type { PrivateTeamSpec } from "@areneon/schema/types/v1/private-team-spec.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { legalChoiceSet } from "./choice-judge.ts";
import type { Engine } from "./engine.ts";
import { episodeFromRecord } from "./episode-io.ts";
import { goldenBattle, syntheticFixtureHashes, syntheticTeam } from "./golden.ts";
import { observe, recordedChannel } from "./player-channel.ts";
import { battleClass } from "./team-class.ts";
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
 * `v1/<contract>/{valid,invalid}/<case>.json`, plus `hashes.json` mapping
 * each sample's path to the SHA-256 of its JCS bytes.
 */
export async function contractSamples(engine: Engine): Promise<Record<string, unknown>> {
  const record = await goldenBattle(engine);
  const known = syntheticFixtureHashes(engine);
  const team: PrivateTeamSpec = {
    schema_version: 1,
    data_class: "synthetic",
    team: syntheticTeam() as PrivateTeamSpec["team"],
  };
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
  const collected = (collection: unknown) =>
    mutated(episode, (e) => ((e.provenance as Record<string, unknown>).collection = collection));
  const identity = { request_seq: turn.request_seq, request_hash: turn.request_hash };

  const valid: Record<string, unknown> = {
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
    "recommendation/in-budget": {
      schema_version: 1,
      identity,
      choice: "move 1 1, move 1 2",
      source_policy_id: "first-legal@golden",
      elapsed_ms: 3,
      deadline_outcome: "in_budget",
    },
    "episode/p1": episode,
    "episode/league": await episodeFromRecord(record, "p1", {
      team,
      opponent_team: team,
      known_synthetic: known,
      behaviour_policy_id: "first-legal@golden",
      collection: league,
    }),
    "episode/evaluation-sealed": await episodeFromRecord(record, "p1", {
      team,
      opponent_team: team,
      known_synthetic: known,
      behaviour_policy_id: "first-legal@golden",
      collection: evaluation,
    }),
    // Kept schema- and reader-valid until slice (b) (#826) adds the rule that a
    // sparring Episode must be private; it then moves to reader-invalid/sparring-synthetic.
    "episode/sparring": collected({ kind: "sparring", session_id: "sparring-session-1" }),
  };
  const invalid: Record<string, unknown> = {
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
    "episode/collection-no-run-id": collected({ kind: "league", seed: record.seed, opponent_policy_id: "first-legal@golden" }),
    "episode/collection-league-sealed": collected({ ...league, sealed: false }),
    "episode/collection-evaluation-no-seed": collected({ kind: "evaluation", run_id: "eval-run-1", sealed: true }),
    "episode/collection-bad-seed": collected({ ...league, seed: "1,2,3,4" }),
    "episode/collection-unknown-kind": collected({ kind: "training", run_id: "league-run-1" }),
    "episode/collection-sparring-run-id": collected({ kind: "sparring", session_id: "s", run_id: "r" }),
  };

  const samples: Record<string, unknown> = {};
  for (const [kind, cases] of [["valid", valid], ["invalid", invalid]] as const) {
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
