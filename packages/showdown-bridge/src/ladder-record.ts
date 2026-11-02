// The LadderRecord (ADR-021 sections 4, 6, 8): what the population ladder decided in one generation and
// why. The schema checks the shape; the rules the contract profile cannot say are checked here and again by
// the Python reader (`areneon_ml.loop.ladder`): sorted and unique lists, table counts that agree with their
// sums, lower at most upper, the reference anchor at its fixed value (and every anchor and frozen snapshot
// strength unchanged from the previous record, when it is given), at most one copy, and the roots counted
// from the members. No strength or bound is computed here: the record only holds them.

import { ajvFor } from "@areneon/schema/ajv";
import ladderRecordSchema from "@areneon/schema/contracts/v1/ladder-record.schema.json" with { type: "json" };
import type { LadderRecord } from "@areneon/schema/types/v1/ladder-record.ts";
import { type Json, jcs } from "./canonical-json.ts";
import { nodeSha256 } from "./player-channel.ts";
import type { TrainingRunConfig } from "./training-run-config.ts";
import { runConfigHash } from "./training-run-config.ts";

export type { LadderRecord };

/** The anchor whose strength fixes the scale, and the value it has by definition (ADR-021 section 4). */
export const LADDER_REFERENCE_ANCHOR = "actor-rule@v1";
export const LADDER_REFERENCE_STRENGTH = 1_000_000_000;
/** A pair score is a multiple of this many ppm (the mean of two games of 0, 500000 or 1000000). */
const SCORE_STEP_PPM = 250_000;

export class LadderRecordError extends Error {
  override name = "LadderRecordError";
}

const validRecord = ajvFor("contract").compile(ladderRecordSchema);

type Member = LadderRecord["members"][number];
type Settings = Member["settings"];

const fail = (message: string): never => {
  throw new LadderRecordError(message);
};

/** Ids that are compared as ASCII code units: the same order Python's string comparison gives. */
function strictlyAscending(keys: readonly string[]): boolean {
  return keys.every((key, index) => index === 0 || key > (keys[index - 1] as string));
}

function checkSettings(settings: Settings): void {
  const { ladder_ppm, self_play_ppm, past_pool_ppm } = settings.shares;
  if (ladder_ppm + self_play_ppm + past_pool_ppm !== 1_000_000) fail("a member's opponent shares must add up to 1000000");
}

function distinctRoots(roots: readonly string[]): number {
  return new Set(roots).size;
}

/** The least floor(score_sum_ppm / pairs) over a member's anchor cells; null when it has none. */
function anchorMinimum(record: LadderRecord, member: string): number | null {
  const scores = record.cells.filter((cell) => cell.member === member && cell.opponent.kind === "anchor").map((cell) => Math.floor(cell.score_sum_ppm / cell.pairs));
  return scores.length === 0 ? null : Math.min(...scores);
}

function checkSelections(record: LadderRecord, memberIds: ReadonlySet<string>): void {
  const [first, ...rest] = record.selections;
  if (first === undefined || first.kind === "health") fail("the first selection must be the copy test's outcome (copy or none)");
  const healths = rest.map((entry) => (entry.kind === "health" ? entry : fail("at most one copy: every selection after the first must be a health replacement")));
  for (const entry of healths) if (!memberIds.has(entry.member)) fail("a health replacement names a member the record does not have");
  if (!strictlyAscending(healths.map((entry) => entry.member))) fail("health replacements must be sorted by member, each once");
  if (first?.kind !== "copy") return;
  const { loser, donor } = first;
  if (loser === donor) fail("a copy's loser must differ from its donor");
  if (!memberIds.has(loser) || !memberIds.has(donor)) fail("a copy names a member the record does not have");
  if (healths.some((entry) => entry.member === loser || entry.member === donor)) fail("a health replacement cannot name a copy's loser or donor");
  if (first.lower_bound_ppm <= 500_000) fail("a copy needs a lower bound above 500000");
  checkSettings(first.settings);
  const byId = new Map(record.members.map((member) => [member.member, member]));
  const loserMember = byId.get(loser) as Member;
  const donorMember = byId.get(donor) as Member;
  if (donorMember.strength.strength < loserMember.strength.strength) fail("a copy's donor cannot be weaker than its loser");
  const cell = record.cells.find((candidate) => candidate.member === donor && candidate.opponent.kind === "member" && candidate.opponent.id === loser);
  if (cell === undefined || cell.pairs !== first.pairs) fail("a copy's pairs must equal the donor-against-loser cell's pairs");
  const donorMin = anchorMinimum(record, donor);
  const loserMin = anchorMinimum(record, loser);
  if (donorMin === null || loserMin === null) fail("a copy needs anchor cells for its donor and its loser");
  if (donorMin !== first.donor_anchor_min_ppm || loserMin !== first.loser_anchor_min_ppm) fail("a copy's anchor minima must be the least anchor pair scores of its donor and loser");
  if ((donorMin as number) < (loserMin as number)) fail("a copy's donor cannot have a lower anchor minimum than its loser");
  const before = distinctRoots(record.members.map((member) => member.root));
  const after = distinctRoots(record.members.map((member) => (member.member === loser ? donorMember.root : member.root)));
  if (first.roots_before !== before || first.roots_after !== after) fail("a copy's roots must be counted from the members' roots");
}

function checkCells(record: LadderRecord, memberIds: ReadonlySet<string>, snapshotIds: ReadonlySet<string>, anchorIds: ReadonlySet<string>): void {
  const keys = record.cells.map((cell) => `${cell.member}\u0000${cell.opponent.kind}\u0000${cell.opponent.id}`);
  if (!strictlyAscending(keys)) fail("cells must be sorted by member, opponent kind and opponent id, each once");
  for (const cell of record.cells) {
    if (!memberIds.has(cell.member)) fail("a cell names a member the record does not have");
    const { kind, id } = cell.opponent;
    const known = kind === "member" ? memberIds : kind === "snapshot" ? snapshotIds : anchorIds;
    if (!known.has(id)) fail("a cell's opponent is not in the record");
    if (kind === "member" && id === cell.member) fail("a member does not play itself in the table");
    if (cell.scores.reduce((sum, count) => sum + count, 0) !== cell.pairs) fail("a cell's pairs must equal the sum of its scores");
    const weighted = cell.scores.reduce((sum, count, index) => sum + count * index, 0);
    if (weighted * SCORE_STEP_PPM !== cell.score_sum_ppm) fail("a cell's score_sum_ppm must be 250000 times its weighted score counts");
  }
}

/** Read a record: the schema, then the rules the profile cannot say; `previous` (generation - 1 of the same run) adds the cross-generation rules. */
export function readLadderRecord(value: unknown, previous?: LadderRecord): LadderRecord {
  if (!validRecord(value)) throw new LadderRecordError(`invalid LadderRecord: ${JSON.stringify(validRecord.errors)}`);
  const record = structuredClone(value) as unknown as LadderRecord;
  const { window } = record;
  if (window.last_generation !== record.generation) fail("the window must end at the record's generation");
  if (window.first_generation !== Math.max(1, record.generation - window.size + 1)) fail("the window must start size generations before its end, but not before generation 1");

  if (!strictlyAscending(record.anchors.map((anchor) => anchor.anchor_id))) fail("anchors must be sorted by anchor_id, each once");
  const reference = record.anchors.find((anchor) => anchor.anchor_id === LADDER_REFERENCE_ANCHOR);
  if (reference?.strength !== LADDER_REFERENCE_STRENGTH) fail("the reference anchor must be present at exactly 1000000000");

  if (!strictlyAscending(record.members.map((member) => member.member))) fail("members must be sorted by member, each once");
  for (const member of record.members) {
    if (member.strength.lower > member.strength.upper) fail("a strength's lower bound cannot exceed its upper bound");
    checkSettings(member.settings);
  }
  if (!strictlyAscending(record.snapshots.map((snapshot) => snapshot.snapshot_id))) fail("snapshots must be sorted by snapshot_id, each once");
  const memberIds = new Set(record.members.map((member) => member.member));
  for (const snapshot of record.snapshots) {
    if (snapshot.snapshot_id !== `${snapshot.member}@g${snapshot.generation}`) fail("a snapshot's id must be its member and generation");
    if (!memberIds.has(snapshot.member)) fail("a snapshot names a member the record does not have");
    if (snapshot.strength.lower > snapshot.strength.upper) fail("a strength's lower bound cannot exceed its upper bound");
  }
  checkCells(record, memberIds, new Set(record.snapshots.map((snapshot) => snapshot.snapshot_id)), new Set(record.anchors.map((anchor) => anchor.anchor_id)));
  const m = record.members.length;
  if (record.cyclic_triples > (m * (m - 1) * (m - 2)) / 6) fail("cyclic_triples cannot exceed the number of member triples");
  checkSelections(record, memberIds);
  if (previous !== undefined) checkAgainstPrevious(record, previous);
  return record;
}

/** The cross-generation rules: the same run and config, the next generation, anchors and frozen snapshot strengths unchanged. */
function checkAgainstPrevious(record: LadderRecord, previous: LadderRecord): void {
  if (previous.run_id !== record.run_id || previous.run_config_hash !== record.run_config_hash) fail("the previous record belongs to another run or config");
  if (previous.generation + 1 !== record.generation) fail("the previous record must be the record of the generation before");
  const anchorKey = (anchors: LadderRecord["anchors"]): string => anchors.map((anchor) => `${anchor.anchor_id}=${anchor.strength}`).join(",");
  if (anchorKey(previous.anchors) !== anchorKey(record.anchors)) fail("anchor strengths are frozen for the run: they differ from the previous record's");
  const now = new Map(record.snapshots.map((snapshot) => [snapshot.snapshot_id, snapshot.strength]));
  for (const snapshot of previous.snapshots) {
    const kept = now.get(snapshot.snapshot_id);
    if (kept !== undefined && kept.strength !== snapshot.strength.strength) fail("a snapshot's strength is frozen at entry: it differs from the previous record's");
  }
}

/** The hash a later record names this one by: SHA-256 of its JCS bytes. */
export const ladderRecordHash = (record: LadderRecord): Promise<string> => nodeSha256(new TextEncoder().encode(jcs(record as unknown as Json)));

/** Check that a record belongs to `config`: its run key and its hash. */
export async function checkLadderRecord(record: LadderRecord, config: TrainingRunConfig): Promise<void> {
  if (record.run_id !== config.seeds.run_key) fail(`run_id ${record.run_id} is not the config's run key`);
  if (record.run_config_hash !== (await runConfigHash(config))) fail("run_config_hash is not the config's hash");
}
