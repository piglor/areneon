// The LadderRecord's share of the sample corpus (ADR-008 section 2, ADR-021 section 8).
// Placeholders that pin a shape: strengths and scores of nothing real, hashes that are single repeated digits.

type Cases = Record<string, unknown>;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Rec = Record<string, any>;
const mutated = (value: unknown, change: (copy: Rec) => void): Rec => {
  const copy = structuredClone(value) as Rec;
  change(copy);
  return copy;
};

const strength = (value: number): Rec => ({ strength: value, lower: Math.floor((value * 9) / 10), upper: Math.floor((value * 11) / 10) });
const settings = (): Rec => ({
  learning_rate_micro: 300,
  entropy_coef_micro: 10_000,
  epochs: 4,
  clip_micro: 200_000,
  shares: { ladder_ppm: 500_000, self_play_ppm: 250_000, past_pool_ppm: 250_000 },
});

/** A cell from its five pair-score counts (0, 250000, 500000, 750000, 1000000 ppm). */
function cell(member: string, kind: string, id: string, scores: number[]): Rec {
  const pairs = scores.reduce((sum, count) => sum + count, 0);
  const weighted = scores.reduce((sum, count, index) => sum + count * index, 0);
  return { member, opponent: { kind, id }, scores, pairs, score_sum_ppm: weighted * 250_000 };
}

const MEMBERS = ["m0", "m1", "m2", "m3"];

/** Every member against every other member and the anchors (and the two snapshots), 96 pairs against another member. */
function table(): Rec[] {
  const cells: Rec[] = [];
  for (const member of MEMBERS) {
    const rank = MEMBERS.indexOf(member);
    for (const other of MEMBERS) {
      if (other === member) continue;
      const stronger = rank > MEMBERS.indexOf(other);
      cells.push(cell(member, "member", other, stronger ? [4, 10, 22, 26, 34] : [34, 26, 22, 10, 4]));
    }
    cells.push(cell(member, "anchor", "actor-rule@v1", [2, 6, 14, 30, 44]));
    cells.push(cell(member, "anchor", "random-legal@v1", [0, 1, 5, 20, 70]));
    cells.push(cell(member, "snapshot", "m0@g0", [3, 9, 24, 36, 24]));
  }
  return cells;
}
const keyOf = (c: Rec): string => `${c.member}\u0000${c.opponent.kind}\u0000${c.opponent.id}`;

function anchorMin(cells: Rec[], member: string): number {
  return Math.min(...cells.filter((c) => c.member === member && c.opponent.kind === "anchor").map((c) => Math.floor(c.score_sum_ppm / c.pairs)));
}

export function ladderRecordSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const cells = table().sort((a, b) => (keyOf(a) < keyOf(b) ? -1 : 1));
  const base: Rec = {
    schema_version: 1,
    data_class: "synthetic",
    run_id: "tiny-run",
    run_config_hash: "a".repeat(64),
    generation: 5,
    window: { size: 3, first_generation: 3, last_generation: 5 },
    anchors: [
      { anchor_id: "actor-rule@v1", strength: 1_000_000_000 },
      { anchor_id: "random-legal@v1", strength: 380_000_000 },
    ],
    members: MEMBERS.map((member, index) => ({ member, root: member, strength: strength(900_000_000 + index * 150_000_000), settings: settings() })),
    snapshots: [{ snapshot_id: "m0@g0", member: "m0", generation: 0, strength: strength(700_000_000) }],
    cells,
    cyclic_triples: 0,
    selections: [{ kind: "none", failed: "lower_bound" }],
  };
  const copy: Rec = {
    kind: "copy",
    loser: "m0",
    donor: "m3",
    pairs: 96,
    margin_ppm: 50_000,
    lower_bound_ppm: 560_000,
    donor_anchor_min_ppm: anchorMin(cells, "m3"),
    loser_anchor_min_ppm: anchorMin(cells, "m0"),
    roots_before: 4,
    roots_after: 3,
    learner_seed_index: 7,
    settings: { ...settings(), learning_rate_micro: 375, target_kl_micro: 15_000 },
  };
  const withCopy = (change: (c: Rec) => void = () => {}): Rec => mutated(base, (r) => (r.selections = [mutated(copy, change)]));
  const health = { kind: "health", member: "m1", cause: "approx_kl" };
  const w = (change: (r: Rec) => void): Rec => mutated(base, change);

  const valid: Cases = {
    "ladder-record/none": base,
    "ladder-record/copy": withCopy(),
    "ladder-record/health": w((r) => (r.selections = [{ kind: "none", failed: "no_ready_member" }, health])),
    "ladder-record/copy-and-health": w((r) => (r.selections = [mutated(copy, () => {}), { kind: "health", member: "m1", cause: "non_finite_loss" }, { kind: "health", member: "m2", cause: "forfeits" }])),
    "ladder-record/first-generation": w((r) => {
      r.generation = 1;
      r.window = { size: 3, first_generation: 1, last_generation: 1 };
      r.snapshots = [];
      r.cells = cells.filter((c) => c.opponent.kind !== "snapshot");
    }),
    "ladder-record/single-member": w((r) => {
      r.members = [r.members[0]];
      r.snapshots = [];
      r.cells = cells.filter((c) => c.member === "m0" && c.opponent.kind === "anchor");
      r.selections = [{ kind: "none", failed: "no_ready_member" }];
    }),
    "ladder-record/cyclic": w((r) => (r.cyclic_triples = 2)),
  };

  const invalid: Cases = {
    "ladder-record/extra-field": w((r) => (r.notes = "x")),
    "ladder-record/other-data-class": w((r) => (r.data_class = "public")),
    "ladder-record/zero-generation": w((r) => (r.generation = 0)),
    "ladder-record/fractional-strength": w((r) => (r.members[0].strength.strength = 1.5)),
    "ladder-record/zero-strength": w((r) => (r.anchors[1].strength = 0)),
    "ladder-record/four-scores": w((r) => (r.cells[0].scores = [1, 2, 3, 4])),
    "ladder-record/negative-score": w((r) => (r.cells[0].scores[0] = -1)),
    "ladder-record/unknown-opponent-kind": w((r) => (r.cells[0].opponent.kind = "exploiter")),
    "ladder-record/unknown-selection-kind": w((r) => (r.selections = [{ kind: "reset", member: "m0" }])),
    "ladder-record/unknown-failed-condition": w((r) => (r.selections = [{ kind: "none", failed: "tired" }])),
    "ladder-record/unknown-health-cause": w((r) => (r.selections = [{ kind: "none", failed: "lower_bound" }, { ...health, cause: "boredom" }])),
    "ladder-record/no-selections": w((r) => (r.selections = [])),
    "ladder-record/bad-member-id": w((r) => (r.members[0].member = "member-0")),
    "ladder-record/no-window": w((r) => delete r.window),
    "ladder-record/no-members": w((r) => (r.members = [])),
    "ladder-record/copy-without-settings": withCopy((c) => delete c.settings),
    "ladder-record/shares-field-missing": w((r) => delete r.members[0].settings.shares.past_pool_ppm),
    "ladder-record/copy-margin-over-half": withCopy((c) => (c.margin_ppm = 500_001)),
    "ladder-record/copy-without-margin": withCopy((c) => delete c.margin_ppm),
    "ladder-record/oversize-pairs": w((r) => {
      r.cells[0].pairs = 1_000_000_001;
      r.cells[0].scores = [1_000_000_001, 0, 0, 0, 0];
    }),
    "ladder-record/next-version": w((r) => (r.schema_version = 2)),
  };

  const readerInvalid: Cases = {
    "ladder-record/window-end-not-generation": w((r) => (r.window.last_generation = 4)),
    "ladder-record/window-start-wrong": w((r) => (r.window.first_generation = 2)),
    "ladder-record/anchors-unsorted": w((r) => (r.anchors = [...r.anchors].reverse())),
    "ladder-record/reference-anchor-missing": w((r) => ((r.anchors = [r.anchors[1]]), (r.cells = r.cells.filter((c: Rec) => c.opponent.id !== "actor-rule@v1")))),
    "ladder-record/reference-anchor-moved": w((r) => (r.anchors[0].strength = 999_999_999)),
    "ladder-record/members-unsorted": w((r) => (r.members = [...r.members].reverse())),
    "ladder-record/lower-above-upper": w((r) => (r.members[1].strength.lower = r.members[1].strength.upper + 1)),
    "ladder-record/snapshot-lower-above-upper": w((r) => (r.snapshots[0].strength.lower = r.snapshots[0].strength.upper + 1)),
    "ladder-record/shares-do-not-sum": w((r) => (r.members[2].settings.shares.self_play_ppm = 200_000)),
    "ladder-record/snapshots-unsorted": w((r) => {
      r.snapshots = [{ snapshot_id: "m1@g0", member: "m1", generation: 0, strength: strength(650_000_000) }, r.snapshots[0]];
      r.cells = r.cells.filter((c: Rec) => c.opponent.kind !== "snapshot");
    }),
    "ladder-record/snapshot-id-mismatch": w((r) => (r.snapshots[0].generation = 5)),
    "ladder-record/snapshot-unknown-member": w((r) => ((r.snapshots[0].member = "m9"), (r.snapshots[0].snapshot_id = "m9@g0"))),
    "ladder-record/cells-unsorted": w((r) => (r.cells = [r.cells[1], r.cells[0], ...r.cells.slice(2)])),
    "ladder-record/duplicate-cell": w((r) => (r.cells = [r.cells[0], r.cells[0], ...r.cells.slice(1)])),
    "ladder-record/cell-unknown-member": w((r) => (r.cells[r.cells.length - 1].member = "m7")),
    "ladder-record/cell-unknown-opponent": w((r) => (r.cells[4].opponent.id = "m9")),
    "ladder-record/cell-plays-itself": w((r) => (r.cells[2] = cell("m0", "member", "m0", [1, 1, 1, 1, 1]))),
    "ladder-record/cell-pairs-not-sum": w((r) => (r.cells[0].pairs += 1)),
    "ladder-record/cell-score-sum-wrong": w((r) => (r.cells[0].score_sum_ppm += 250_000)),
    "ladder-record/cell-score-sum-off-step": w((r) => (r.cells[0].score_sum_ppm += 1)),
    "ladder-record/cyclic-more-than-triples": w((r) => (r.cyclic_triples = 5)),
    "ladder-record/first-selection-health": w((r) => (r.selections = [health])),
    "ladder-record/two-copies": w((r) => (r.selections = [mutated(copy, () => {}), mutated(copy, (c) => ((c.loser = "m1"), (c.pairs = 96)))])),
    "ladder-record/copy-after-health": w((r) => (r.selections = [{ kind: "none", failed: "lower_bound" }, mutated(copy, () => {})])),
    "ladder-record/health-unsorted": w((r) => (r.selections = [{ kind: "none", failed: "lower_bound" }, { ...health, member: "m2" }, health])),
    "ladder-record/health-duplicate": w((r) => (r.selections = [{ kind: "none", failed: "lower_bound" }, health, health])),
    "ladder-record/health-unknown-member": w((r) => (r.selections = [{ kind: "none", failed: "lower_bound" }, { ...health, member: "m8" }])),
    "ladder-record/health-names-loser": w((r) => (r.selections = [mutated(copy, () => {}), { ...health, member: "m0" }])),
    "ladder-record/copy-loser-is-donor": withCopy((c) => (c.loser = "m3")),
    "ladder-record/copy-unknown-member": withCopy((c) => (c.donor = "m6")),
    "ladder-record/copy-lower-bound-too-low": withCopy((c) => (c.lower_bound_ppm = 550_000)),
    "ladder-record/copy-donor-not-top": w((r) => (r.selections = [mutated(copy, (c) => ((c.loser = "m3"), (c.donor = "m0"), (c.pairs = 96)))])),
    "ladder-record/copy-pairs-not-cell": withCopy((c) => (c.pairs = 95)),
    "ladder-record/copy-anchor-minimum-wrong": withCopy((c) => (c.donor_anchor_min_ppm += 1)),
    "ladder-record/copy-donor-anchor-below-loser": w((r) => {
      r.cells = r.cells.map((c: Rec) => (c.member === "m3" && c.opponent.id === "actor-rule@v1" ? cell("m3", "anchor", "actor-rule@v1", [30, 30, 20, 10, 6]) : c));
      r.selections = [mutated(copy, (c) => {
        c.donor_anchor_min_ppm = anchorMin(r.cells, "m3");
      })];
    }),
    "ladder-record/copy-roots-before-wrong": withCopy((c) => (c.roots_before = 3)),
    "ladder-record/copy-roots-after-wrong": withCopy((c) => (c.roots_after = 4)),
    "ladder-record/member-missing-anchor-cell": w((r) => (r.cells = r.cells.filter((c: Rec) => !(c.member === "m0" && c.opponent.id === "random-legal@v1")))),
    "ladder-record/snapshot-not-before-generation": w((r) => {
      r.snapshots[0] = { ...r.snapshots[0], snapshot_id: "m0@g5", generation: 5 };
      for (const c of r.cells) if (c.opponent.id === "m0@g0") c.opponent.id = "m0@g5";
    }),
    "ladder-record/copy-shares-do-not-sum": withCopy((c) => (c.settings.shares.self_play_ppm = 200_000)),
  };
  return { valid, invalid, readerInvalid };
}
