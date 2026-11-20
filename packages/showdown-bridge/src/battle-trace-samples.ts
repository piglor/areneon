// The BattleTrace share of the sample corpus (ADR-008 section 2, ADR-030 groundwork G2): small traces built by hand on the
// golden team (the recorder's own traces are long; its tests hold them to the contract). Reader-invalid traces pass the schema
// and break one rule of the core's reader each (`@areneon/differential`'s `readBattleTrace`, whose tests read these files).
import { PINNED_SHOWDOWN_COMMIT } from "./engine.ts";
import { GOLDEN_FORMAT, GOLDEN_SEED, syntheticTeam } from "./golden.ts";
import { teamHash } from "./team-class.ts";

type Cases = Record<string, unknown>;

const HASH = (c: string): string => c.repeat(64);
const team = syntheticTeam();
const lines = (...l: string[]): { p1: string[]; p2: string[] } => ({ p1: [...l], p2: [...l] });

/** A two-step trace: the start (Team Preview requests) and the step of both seats' orders, which ends the battle. */
function minimal(): Record<string, unknown> {
  const preview = (seq: number): Record<string, unknown> => ({ request_seq: seq, request_hash: HASH("a"), mask: [{ choice: "team 1234", verdict: "legal" }] });
  return {
    schema_version: 1,
    data_class: "synthetic",
    trace_format: 1,
    showdown_commit: PINNED_SHOWDOWN_COMMIT,
    format_id: GOLDEN_FORMAT,
    battle_id: "sample/minimal",
    seed: GOLDEN_SEED,
    teams: { p1: structuredClone(team), p2: structuredClone(team) },
    team_hashes: { p1: teamHash(team), p2: teamHash(team) },
    steps: [
      { actions: [], draws: [{ method: "random", args: ["16"], result: ["3"] }], channels: lines("|teampreview"), omniscient: ["|teampreview"], requests: { p1: [preview(1)], p2: [preview(1)] }, state_sha256: HASH("b") },
      {
        actions: [{ kind: "choice", seat: "p1", choice: "team 1234" }, { kind: "choice", seat: "p2", choice: "team 1234" }],
        draws: [{ method: "shuffle", args: ["2", "0", "2"], result: ["1", "0"] }, { method: "randomChance", args: ["1", "2"], result: ["true"] }, { method: "sample", args: ["3"], result: ["2"] }],
        channels: lines("|start", "|win|p1"),
        omniscient: ["|start", "|win|p1"],
        requests: { p1: [], p2: [] },
        state_sha256: HASH("c"),
      },
    ],
    outcome: { winner: "p1", turns: 0 },
  };
}

function tie(): Record<string, unknown> {
  const t = minimal();
  const step = (t.steps as Record<string, unknown>[])[1] as Record<string, unknown>;
  step.actions = [{ kind: "command", command: "forcetie" }];
  step.omniscient = ["|tie"];
  step.channels = lines("|tie");
  t.battle_id = "sample/forced-tie";
  t.outcome = { winner: null, turns: 31 };
  return t;
}

const mutated = (change: (t: Record<string, unknown>) => void): unknown => {
  const t = minimal();
  change(t);
  return t;
};
const step = (t: Record<string, unknown>, i: number): Record<string, unknown> => (t.steps as Record<string, unknown>[])[i] as Record<string, unknown>;
const firstRequest = (t: Record<string, unknown>, seat: "p1" | "p2"): Record<string, unknown> => (step(t, 0).requests as Record<string, Record<string, unknown>[]>)[seat]?.[0] as Record<string, unknown>;

export function battleTraceSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  return {
    valid: { "battle-trace/minimal": minimal(), "battle-trace/forced-tie": tie() },
    invalid: {
      "battle-trace/extra-field": mutated((t) => (t.wall_time_ms = 3)),
      "battle-trace/float-turns": mutated((t) => (t.outcome = { winner: "p1", turns: 1.5 })),
      "battle-trace/number-in-draw": mutated((t) => (step(t, 0).draws = [{ method: "random", args: [16], result: ["3"] }])),
      "battle-trace/unknown-verdict": mutated((t) => (firstRequest(t, "p1").mask = [{ choice: "team 1234", verdict: "hidden" }])),
      "battle-trace/unknown-command": mutated((t) => (step(t, 1).actions = [{ kind: "command", command: "forcewin p1" }])),
      "battle-trace/missing-state-hash": mutated((t) => delete step(t, 1).state_sha256),
      "battle-trace/bad-seed": mutated((t) => (t.seed = "gen5,1234")),
    },
    readerInvalid: {
      "battle-trace/start-with-action": mutated((t) => (step(t, 0).actions = [{ kind: "choice", seat: "p1", choice: "team 1234" }])),
      "battle-trace/step-without-action": mutated((t) => (step(t, 1).actions = [])),
      "battle-trace/request-gap": mutated((t) => (firstRequest(t, "p2").request_seq = 2)),
      "battle-trace/team-hash-mismatch": mutated((t) => (t.team_hashes = { p1: HASH("d"), p2: teamHash(team) })),
      "battle-trace/ends-early": mutated((t) => (step(t, 0).omniscient = ["|teampreview", "|win|p1"])),
      "battle-trace/never-ends": mutated((t) => (step(t, 1).omniscient = ["|start"])),
      "battle-trace/wrong-winner": mutated((t) => (t.outcome = { winner: "p2", turns: 0 })),
    },
  };
}
