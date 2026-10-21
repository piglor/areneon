// The private team record's share of the sample corpus (ADR-008 §2, ADR-015
// sections 2 and 5). Synthetic only: the pastes are the committed synthetic
// team corpus (`fixtures/teams/`).
//
// A PrivateTeamRecord sample says `data_class: private`, so each such file is
// listed in `PRIVATE_SAMPLE_MARKERS` (scripts/check_data_class.py), which also
// requires its paste to be a corpus paste and its shape to be exactly a
// record. The guard therefore cannot hold an invalid sample that carries the
// private label with a broken field, and an `invalid` sample here says
// `data_class: synthetic` instead (itself a violation of the contract's
// const). A test flips the label back and requires the sample to stay
// invalid, so each one is refused for its own fault.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Engine } from "./engine.ts";
import { TEAMS_DIR } from "./golden.ts";
import { canonicalTeamHash, recordHash } from "./team-records.ts";
import { teamHash } from "./team-class.ts";
import { importPaste } from "./teams.ts";

type Cases = Record<string, unknown>;

/** The corpus pastes the record samples carry. */
export const PRIVATE_SAMPLE_PASTES = ["legal-synthetic.txt", "legal-opponent.txt"] as const;

/** A copy of `value` with `change` applied: one mutation per sample. */
function mutated<T>(value: T, change: (copy: Record<string, unknown>) => void): Record<string, unknown> {
  const copy = structuredClone(value) as unknown as Record<string, unknown>;
  change(copy);
  return copy;
}

const hexOf = (c: string): string => c.repeat(64);
/** The record samples' `valid` and `reader-invalid` cases carry the private label; `invalid` ones the synthetic one. */
export function privateSamples(engine: Engine): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const pastes = PRIVATE_SAMPLE_PASTES.map((file) => readFileSync(join(TEAMS_DIR, file), "utf8")) as [string, string];
  const hashesOf = (paste: string): { team_hash: string; canonical_team_hash: string } => {
    const sets = importPaste(engine, paste);
    return { team_hash: teamHash(sets), canonical_team_hash: canonicalTeamHash(sets) };
  };
  const first = {
    schema_version: 1,
    data_class: "private",
    team_id: "00112233445566778899aabbccddeeff",
    revision: 1,
    supersedes_record_hash: null,
    ...hashesOf(pastes[0]),
    paste: pastes[0],
    status: "active",
    tags: [],
    labels: [],
    holdout: false,
  };
  // A hold-out team (ADR-015 section 2): the flag follows team_id alone, so the same value on every revision.
  const holdout = { ...first, team_id: "ffeeddccbbaa99887766554433221100", ...hashesOf(pastes[1]), paste: pastes[1], holdout: true };
  const edited = {
    ...first,
    revision: 2,
    supersedes_record_hash: recordHash(first),
    ...hashesOf(pastes[1]),
    paste: pastes[1],
    tags: ["synthetic", "vgc"],
    labels: [{ label_id: "archetype", value: "offense" }],
  };
  const retired = { ...edited, revision: 3, supersedes_record_hash: recordHash(edited), status: "retired" };
  const withRecord = (change: (r: Record<string, unknown>) => void): Record<string, unknown> => mutated(edited, change);
  const synthetic = (change: (r: Record<string, unknown>) => void): Record<string, unknown> =>
    mutated(edited, (r) => {
      r.data_class = "synthetic";
      change(r);
    });

  return {
    valid: {
      "private-team-record/first": first,
      "private-team-record/edited": edited,
      "private-team-record/retired": retired,
      "private-team-record/holdout": holdout,
    },
    invalid: {
      "private-team-record/synthetic-class": synthetic(() => undefined),
      "private-team-record/extra-field": synthetic((r) => (r.rating = 1500)),
      "private-team-record/next-version": synthetic((r) => (r.schema_version = 2)),
      "private-team-record/no-team-id": synthetic((r) => delete r.team_id),
      "private-team-record/short-team-id": synthetic((r) => (r.team_id = "0".repeat(31))),
      "private-team-record/uppercase-team-id": synthetic((r) => (r.team_id = "A".repeat(32))),
      "private-team-record/revision-zero": synthetic((r) => (r.revision = 0)),
      "private-team-record/revision-over-max": synthetic((r) => (r.revision = 2 ** 31)),
      "private-team-record/fractional-revision": synthetic((r) => (r.revision = 1.5)),
      "private-team-record/no-supersedes": synthetic((r) => delete r.supersedes_record_hash),
      "private-team-record/short-supersedes": synthetic((r) => (r.supersedes_record_hash = "abc")),
      "private-team-record/uppercase-team-hash": synthetic((r) => (r.team_hash = hexOf("A"))),
      "private-team-record/empty-paste": synthetic((r) => (r.paste = "")),
      "private-team-record/paste-too-long": synthetic((r) => (r.paste = "x".repeat(8193))),
      "private-team-record/no-holdout": synthetic((r) => delete r.holdout),
      "private-team-record/holdout-not-boolean": synthetic((r) => (r.holdout = "yes")),
      "private-team-record/unknown-status": synthetic((r) => (r.status = "archived")),
      "private-team-record/too-many-tags": synthetic((r) => (r.tags = Array.from({ length: 17 }, (_, i) => `tag-${String.fromCharCode(97 + i)}`))),
      "private-team-record/uppercase-tag": synthetic((r) => (r.tags = ["Vgc"])),
      "private-team-record/long-tag": synthetic((r) => (r.tags = ["a".repeat(33)])),
      "private-team-record/too-many-labels": synthetic((r) => (r.labels = Array.from({ length: 65 }, (_, i) => ({ label_id: `l${i}`, value: "yes" })))),
      "private-team-record/label-extra-field": synthetic((r) => (r.labels = [{ label_id: "archetype", value: "offense", weight: 1 }])),
    },
    readerInvalid: {
      "private-team-record/team-hash-mismatch": withRecord((r) => (r.team_hash = hexOf("a"))),
      "private-team-record/canonical-hash-mismatch": withRecord((r) => (r.canonical_team_hash = hexOf("a"))),
      "private-team-record/first-revision-supersedes": mutated(first, (r) => (r.supersedes_record_hash = hexOf("a"))),
      "private-team-record/later-revision-no-predecessor": withRecord((r) => (r.supersedes_record_hash = null)),
      "private-team-record/unsorted-tags": withRecord((r) => (r.tags = ["vgc", "synthetic"])),
      "private-team-record/duplicate-tags": withRecord((r) => (r.tags = ["vgc", "vgc"])),
      "private-team-record/duplicate-label-ids": withRecord((r) => (r.labels = [{ label_id: "archetype", value: "offense" }, { label_id: "archetype", value: "defense" }])),
    },
  };
}
