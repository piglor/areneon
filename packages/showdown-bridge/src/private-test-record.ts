// The PrivateTestRecord (ADR-015 erratum E1, item 2): the counts and scores of
// one frozen pair on the owner's private test set, written only under a
// private root. The schema checks the shape; the rules the contract profile
// cannot say are checked here and again by the Python reader
// (`areneon_ml.private.private_test_record`): the counts add up, the pair-score
// histogram agrees with wins, ties and losses, matches are sorted and unique
// and each is named for its opponent and mode, and the source fields agree
// with the source kind. A message states a rule and never repeats a value.

import { createHash } from "node:crypto";
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/private-test-record.schema.json" with { type: "json" };
import type { PrivateTestRecord } from "@areneon/schema/types/v1/private-test-record.ts";
import { type Json, jcs } from "./canonical-json.ts";

export type { PrivateTestRecord };
export type PrivateTestMatch = PrivateTestRecord["matches"][number];

/** The label every display of a private test carries. */
export const PRIVATE_TEST_LABEL = "private test set; not the Training gate; not validation";

/** The scripted opponents a match is named for, by policy-id name. */
const BASELINE_NAMES = new Set(["random-legal", "actor-rule", "damage-aware"]);

export class PrivateTestRecordError extends Error {
  override name = "PrivateTestRecordError";
}

const fail = (message: string): never => {
  throw new PrivateTestRecordError(message);
};
const valid = ajvFor("contract").compile(schema);

/** SHA-256 of a record's JCS bytes. */
export const privateTestRecordHash = (record: PrivateTestRecord): string => createHash("sha256").update(jcs(record as unknown as Json)).digest("hex");

/** The opponent's name in a match id: the baseline's name, or `pair` for another frozen pair. */
export function opponentName(match: Pick<PrivateTestMatch, "opponent_policy_id">): string {
  const id = match.opponent_policy_id;
  return id.startsWith("policy:") ? "pair" : id.slice(0, id.indexOf("@"));
}

/**
 * Whether a pair with private ancestry may be tested on a source: the hold-out records of a snapshot (they never
 * trained it) or the validation split of a public dataset (no private run trains on it), never the staged directory
 * (it has no hold-out flag) and never the whole of a snapshot.
 */
export const privateTestAllowed = (source_kind: string, subset: string): boolean => (source_kind === "snapshot" && subset === "holdout") || source_kind === "dataset";

/** Read a PrivateTestRecord: the schema, then the arithmetic, order and source rules. */
export function readPrivateTestRecord(value: unknown): PrivateTestRecord {
  if (!valid(value)) throw new PrivateTestRecordError(`invalid PrivateTestRecord at ${(valid.errors ?? []).map((e) => e.instancePath || "/").join(", ")}`);
  const record = structuredClone(value) as unknown as PrivateTestRecord;
  const { test_set: set, plan, matches } = record;
  if ((set.source_kind === "snapshot") !== (set.snapshot_hash !== undefined)) fail("test_set.snapshot_hash must be present exactly for the snapshot source");
  if ((set.source_kind === "dataset") !== (set.dataset_hash !== undefined)) fail("test_set.dataset_hash must be present exactly for the dataset source");
  if (set.subset === "holdout" && set.source_kind !== "snapshot") fail("the holdout subset exists only in a snapshot");
  if (set.pairs * 2 > set.team_count) fail("test_set.pairs must not exceed half the teams");
  if (plan.games_per_match !== set.pairs * 2) fail("plan.games_per_match must be twice test_set.pairs");
  if (record.candidate_data_class === "private" && !privateTestAllowed(set.source_kind, set.subset)) fail("a pair with private ancestry is tested on hold-out teams of a snapshot or on a public dataset's teams only");
  const ids = matches.map((m) => m.match_id);
  if (ids.some((id, i) => i > 0 && id <= (ids[i - 1] as string))) fail("matches must be sorted by match_id, each once");
  for (const m of matches) {
    const name = opponentName(m);
    if (m.match_id !== `${name}-${m.mode}`) fail("a match is named <opponent>-<mode>");
    if (name !== "pair" && !BASELINE_NAMES.has(name)) fail("an opponent that is not a pair must be a known scripted baseline");
    if ((name === "pair") !== (m.opponent_checkpoints !== undefined)) fail("opponent_checkpoints are present exactly for a pair opponent");
    if (m.games !== plan.games_per_match || m.pairs !== set.pairs) fail("every match plays every pair from both seats");
    if (m.wins + m.ties + m.losses !== m.games) fail("wins + ties + losses must equal games");
    if (m.turn_cap_ties > m.ties) fail("turn_cap_ties must not exceed ties");
    const counts = m.pair_score_counts;
    if (counts.reduce((a, b) => a + b, 0) !== m.pairs) fail("pair_score_counts must sum to the pairs");
    if (counts.reduce((a, c, k) => a + c * k, 0) !== 2 * m.wins + m.ties) fail("pair_score_counts must agree with wins and ties in half-games");
    const keys = m.forfeits.map((f) => `${f.side}\u0000${f.cause}`);
    if (keys.some((key, i) => i > 0 && key <= (keys[i - 1] as string))) fail("forfeits must be sorted by side then cause, each pair once");
    const by = (side: string): number => m.forfeits.filter((f) => f.side === side).reduce((a, f) => a + f.count, 0);
    if (by("candidate") > m.losses || by("opponent") > m.wins) fail("forfeits cannot exceed the losses or wins they decided");
  }
  return record;
}
