/* eslint-disable @typescript-eslint/no-explicit-any */
// The BackendAdmission share of the sample corpus (ADR-008 section 2, ADR-030 groundwork G1): ledgers built by hand by the
// admission rules (the harness's own `buildLedger` lives in `@areneon/differential`, which the bridge does not import; its tests
// read these files back with both readers). Reader-invalid ledgers pass the schema and break one rule each.
import { createHash } from "node:crypto";
import { type Json, jcs } from "./canonical-json.ts";
import { PINNED_SHOWDOWN_COMMIT } from "./engine.ts";

type Cases = Record<string, unknown>;

const BACKEND = { kind: "damage-kernel", backend_format: 1, backend_commit: "4".repeat(40) };
const freshSeed = (backend: { backend_commit: string; backend_format: number }): string =>
  createHash("sha256").update(jcs({ backend_commit: backend.backend_commit, backend_format: backend.backend_format, purpose: "differential-fresh/1" } as Json)).digest("hex");
const FRESH_ID = `fresh-${freshSeed(BACKEND).slice(0, 16)}`;

const kind = (n: number) => ({ kind: "kind:damage", co_exercised: n, co_equal: n });
const count = (corpus: string, present: number, exercised: number, equal: number, differing: number) => ({ corpus, present, exercised, equal, differing });

/** A ledger: a regression corpus that found one difference, the fresh corpus that admits three entities and lists one pair. */
function admitted(): Record<string, unknown> {
  return {
    schema_version: 1,
    data_class: "synthetic",
    rule: "admission/1",
    level: "mechanics-query",
    backend: { ...BACKEND },
    showdown_commit: PINNED_SHOWDOWN_COMMIT,
    corpora: [
      { id: "committed-random", kind: "regression", seed: "a".repeat(64), pool_hash: "a".repeat(64), content_hash: "1".repeat(64), tally_hash: "2".repeat(64), items: 300, distinct_items: 300, count_mismatches: 0 },
      { id: FRESH_ID, kind: "fresh", seed: freshSeed(BACKEND), pool_hash: "9".repeat(64), content_hash: "3".repeat(64), tally_hash: "5".repeat(64), items: 200_000, distinct_items: 200_000, count_mismatches: 0 },
    ],
    entities: [
      { id: "ability:roughskin", admitted: false, counts: [count("committed-random", 5, 5, 4, 1), count(FRESH_ID, 25, 25, 25, 0)], kinds: [kind(25)] },
      { id: "kind:damage", admitted: true, counts: [count("committed-random", 250, 250, 250, 0), count(FRESH_ID, 40, 40, 40, 0)], kinds: [] },
      { id: "move:earthquake", admitted: true, counts: [count(FRESH_ID, 30, 30, 30, 0)], kinds: [kind(30)] },
      { id: "move:protect", admitted: false, counts: [count(FRESH_ID, 30, 0, 0, 0)], kinds: [kind(0)] },
      { id: "species:garchomp", admitted: true, counts: [count("committed-random", 300, 120, 120, 0), count(FRESH_ID, 22, 21, 21, 0)], kinds: [kind(21)] },
      { id: "item:choiceband", admitted: false, counts: [count("committed-random", 300, 300, 300, 0)], kinds: [] },
    ].sort((a, b) => (a.id < b.id ? -1 : 1)),
    pairs: [
      { a: "kind:damage", b: "move:earthquake", co_present: 30, co_exercised: 30, co_equal: 30, admitted: true },
      { a: "kind:damage", b: "species:garchomp", co_present: 22, co_exercised: 3, co_equal: 3, admitted: false },
    ],
    revoked: [{ id: "ability:roughskin", corpus: "committed-random", position: 17 }],
  };
}

/** The ledger a pin bump leaves: no evidence, nothing admitted. */
const empty = (): Record<string, unknown> => ({ ...admitted(), corpora: [], entities: [], pairs: [], revoked: [] });

/** A shadow sample revoked an entity the fresh corpus had admitted. */
function shadowRevoked(): Record<string, unknown> {
  const l = admitted() as any;
  l.corpora.push({ id: "shadow-g000012", kind: "shadow", seed: "b".repeat(64), pool_hash: "b".repeat(64), content_hash: "6".repeat(64), tally_hash: "7".repeat(64), items: 256, distinct_items: 256, count_mismatches: 0 });
  const quake = l.entities.find((e: any) => e.id === "move:earthquake");
  quake.admitted = false;
  quake.counts.push(count("shadow-g000012", 3, 3, 2, 1));
  l.revoked.push({ id: "move:earthquake", corpus: "shadow-g000012", position: 200 });
  l.pairs = l.pairs.filter((p: any) => p.b !== "move:earthquake");
  return l;
}

function change(base: () => Record<string, unknown>, fn: (copy: any) => void): Record<string, unknown> {
  const copy = base();
  fn(copy);
  return copy;
}

const entity = (copy: any, id: string): any => copy.entities.find((e: any) => e.id === id);

export function backendAdmissionSamples(): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  return {
    valid: {
      "backend-admission/admitted": admitted(),
      "backend-admission/empty-after-pin-bump": empty(),
      "backend-admission/shadow-revoked": shadowRevoked(),
    },
    invalid: {
      "backend-admission/extra-field": change(admitted, (c) => (c.note = "x")),
      "backend-admission/fractional-count": change(admitted, (c) => (entity(c, "kind:damage").counts[0].present = 1.5)),
      "backend-admission/unknown-corpus-kind": change(admitted, (c) => (c.corpora[0].kind = "debug")),
      "backend-admission/short-commit": change(admitted, (c) => (c.backend.backend_commit = "4".repeat(39))),
      "backend-admission/next-rule": change(admitted, (c) => (c.rule = "admission/2")),
      "backend-admission/missing-revoked": change(admitted, (c) => delete c.revoked),
      "backend-admission/bad-entity-id": change(admitted, (c) => (entity(c, "move:protect").id = "Protect")),
      "backend-admission/unknown-class": change(admitted, (c) => (c.data_class = "public")),
      "backend-admission/missing-tally-hash": change(admitted, (c) => delete c.corpora[1].tally_hash),
      "backend-admission/missing-kinds": change(admitted, (c) => delete entity(c, "move:earthquake").kinds),
    },
    readerInvalid: {
      "backend-admission/regression-evidence": change(admitted, (c) => (entity(c, "item:choiceband").admitted = true)),
      "backend-admission/nineteen-exercising": change(admitted, (c) => {
        const quake = entity(c, "move:earthquake");
        quake.counts[0] = count(FRESH_ID, 30, 19, 19, 0);
        quake.kinds = [kind(19)];
      }),
      "backend-admission/wrong-fresh-seed": change(admitted, (c) => (c.corpora[1].seed = "c".repeat(64))),
      "backend-admission/two-fresh": change(admitted, (c) => (c.corpora[0].kind = "fresh")),
      "backend-admission/other-pin": change(admitted, (c) => (c.showdown_commit = "0".repeat(40))),
      "backend-admission/unsorted-entities": change(admitted, (c) => c.entities.reverse()),
      "backend-admission/missing-revocation": change(admitted, (c) => (c.revoked = [])),
      "backend-admission/counts-not-nested": change(admitted, (c) => (entity(c, "move:protect").counts[0].equal = 1)),
      "backend-admission/revocation-past-corpus": change(admitted, (c) => (c.revoked[0].position = 300)),
      "backend-admission/pair-flag": change(admitted, (c) => (c.pairs[1].admitted = true)),
      "backend-admission/pair-not-admitted-entity": change(admitted, (c) => c.pairs.splice(1, 0, { a: "kind:damage", b: "move:protect", co_present: 30, co_exercised: 0, co_equal: 0, admitted: false })),
      "backend-admission/fresh-below-minimum": change(admitted, (c) => {
        c.corpora[1].items = 199_999;
        c.corpora[1].distinct_items = 199_999;
      }),
      "backend-admission/fresh-repeated-items": change(admitted, (c) => (c.corpora[1].distinct_items = 100)),
      "backend-admission/count-mismatch": change(admitted, (c) => (c.corpora[0].count_mismatches = 1)),
      "backend-admission/kinds-not-query-kind": change(admitted, (c) => (entity(c, "move:earthquake").kinds[0].kind = "move:protect")),
      "backend-admission/kinds-on-query-kind": change(admitted, (c) => entity(c, "kind:damage").kinds.push(kind(0))),
      "backend-admission/rare-pair": change(admitted, (c) => {
        c.corpora[1].items = 400_000;
        c.corpora[1].distinct_items = 400_000;
      }),
    },
  };
}
