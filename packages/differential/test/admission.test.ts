// The admission arithmetic (ADR-030 rules 2, 4, 6 and 7; ADR-028 section 5; ADR-026 slice F's N_EX edge cases): 20 against 19
// exercising items, equal counts, a pair below 5, revocation, a ledger of another pin, regression evidence refused for
// admission, the fresh seed, the router.
import { describe, expect, it } from "vitest";
import {
  AdmissionError,
  admissionHash,
  admissionRouter,
  admits,
  type BackendAdmission,
  buildLedger,
  freshCorpus,
  freshCorpusId,
  freshSeed,
  frequent,
  MIN_FRESH_ITEMS,
  N_EXERCISED,
  PAIR_ODDS,
  readBackendAdmission,
  verifyLedger,
  withCorpus,
} from "../src/admission.ts";
import { sha256Jcs } from "../src/hash.ts";
import { addItem, type CorpusTally, newTally } from "../src/tally.ts";
import { BACKEND, corpus, FRESH, PIN, POOL, REGRESSION, tallyOf } from "./support.ts";

const ledgerOf = (...tallies: CorpusTally[]): BackendAdmission => buildLedger({ level: "mechanics-query", backend: BACKEND, showdown_commit: PIN, tallies });
const entity = (ledger: BackendAdmission, id: string) => ledger.entities.find((e) => e.id === id);
const read = (value: unknown) => readBackendAdmission(value, { showdown_commit: PIN });
const doctor = (ledger: BackendAdmission, change: (copy: BackendAdmission) => void): BackendAdmission => {
  const copy = structuredClone(ledger);
  change(copy);
  return copy;
};

describe("the fresh seed", () => {
  it("is SHA-256 over the JCS of the frozen commit, the format and the purpose", () => {
    expect(freshSeed(BACKEND)).toBe(sha256Jcs({ backend_commit: BACKEND.backend_commit, backend_format: 3, purpose: "differential-fresh/1" }));
    expect(freshSeed({ ...BACKEND, backend_format: 4 })).not.toBe(freshSeed(BACKEND));
    expect(freshCorpusId(BACKEND)).toBe(`fresh-${freshSeed(BACKEND).slice(0, 16)}`);
  });
  it("is the only thing a fresh corpus's draw is given", () => {
    const seen: string[] = [];
    const corpus = freshCorpus(BACKEND, POOL, (seed) => {
      seen.push(seed);
      return [1, 2];
    });
    expect(seen).toEqual([freshSeed(BACKEND)]);
    expect(corpus).toEqual({ spec: { id: freshCorpusId(BACKEND), kind: "fresh", seed: freshSeed(BACKEND), pool: POOL }, items: [1, 2] });
  });
});

describe("entity admission", () => {
  it("admits at 20 equal exercising fresh items and not at 19", () => {
    const ledger = ledgerOf(tallyOf(FRESH, [{ count: N_EXERCISED, entities: ["move:a", "kind:damage"] }, { count: N_EXERCISED - 1, entities: ["move:b", "kind:damage"] }]));
    expect(entity(ledger, "move:a")?.admitted).toBe(true);
    expect(entity(ledger, "move:b")?.admitted).toBe(false);
    expect(entity(ledger, "kind:damage")?.counts).toEqual([{ corpus: FRESH.id, present: 39, exercised: 39, equal: 39, differing: 0 }]);
  });
  it("needs exercising items, not presence", () => {
    const ledger = ledgerOf(tallyOf(FRESH, [{ count: 30, entities: ["move:a", "item:x"], reads: ["move:a"] }]));
    expect(entity(ledger, "move:a")?.admitted).toBe(true);
    expect(entity(ledger, "item:x")).toMatchObject({ admitted: false, counts: [{ present: 30, exercised: 0 }] });
  });
  it("revokes every entity of an item with a difference, at the first corpus that found one", () => {
    const fresh = tallyOf(FRESH, [{ count: 40, entities: ["move:a", "ability:b"], differing: [25] }]);
    const regression = tallyOf(REGRESSION, [{ count: 3, entities: ["move:a"], differing: [2] }]);
    const ledger = ledgerOf(fresh, regression);
    expect(entity(ledger, "move:a")?.admitted).toBe(false);
    expect(entity(ledger, "ability:b")?.admitted).toBe(false);
    // "committed" sorts before the fresh corpus's id, so it is the first corpus that found a difference in move:a.
    expect(ledger.revoked).toEqual([
      { id: "ability:b", corpus: FRESH.id, position: 25 },
      { id: "move:a", corpus: REGRESSION.id, position: 2 },
    ]);
  });
  it("never admits from regression evidence, however much of it", () => {
    const ledger = ledgerOf(tallyOf(REGRESSION, [{ count: 500, entities: ["move:a"] }]));
    expect(entity(ledger, "move:a")?.admitted).toBe(false);
    const forged = doctor(ledger, (l) => void ((l.entities[0] as { admitted: boolean }).admitted = true));
    expect(() => read(forged)).toThrow("admission evidence must come from the fresh corpus");
  });
  it("needs every exercising fresh item equal", () => {
    expect(admits({ exercised: 20, equal: 20 }, 0, false, true)).toBe(true);
    expect(admits({ exercised: 20, equal: 19 }, 0, false, true)).toBe(false);
    expect(admits({ exercised: 20, equal: 20 }, 1, false, true)).toBe(false);
    expect(admits({ exercised: 20, equal: 20 }, 0, true, true)).toBe(false);
    expect(admits({ exercised: 20, equal: 20 }, 0, false, false)).toBe(false);
    expect(admits(undefined, 0, false, true)).toBe(false);
  });
});

describe("pairs", () => {
  it("lists frequent pairs of admitted entities and admits them at 5 co-exercised items, not 4", () => {
    const fresh = tallyOf(FRESH, [
      { count: 20, entities: ["move:a"] },
      { count: 20, entities: ["move:b"] },
      { count: 20, entities: ["move:c"] },
      { count: 20, entities: ["move:a", "move:b"] },
      { count: 4, entities: ["move:a", "move:c"] },
      { count: 16, entities: ["move:a", "move:c"], reads: ["move:a"] },
      { count: 21, entities: ["move:b", "move:c"], reads: ["move:b"] },
    ]);
    const ledger = ledgerOf(fresh);
    expect(ledger.pairs).toEqual([
      { a: "move:a", b: "move:b", co_present: 20, co_exercised: 20, co_equal: 20, admitted: true },
      { a: "move:a", b: "move:c", co_present: 20, co_exercised: 4, co_equal: 4, admitted: false },
      { a: "move:b", b: "move:c", co_present: 21, co_exercised: 0, co_equal: 0, admitted: false },
    ]);
    // A query names exactly one kind: these name none, so none is routed to the backend.
    expect(admissionRouter(ledger).route(["move:b", "move:a"])).toEqual({ kinds: 0 });
  });
  it("leaves out a pair present in fewer than 1 in 10^4 fresh items", () => {
    expect(frequent(1, 10_000)).toBe(true);
    expect(frequent(1, 10_001)).toBe(false);
    expect(frequent(0, 0)).toBe(false);
    const fresh = tallyOf(FRESH, [{ count: 20, entities: ["move:a"] }, { count: 19, entities: ["move:b"] }, { count: 19, entities: ["move:a", "move:b"] }]);
    const ledger = ledgerOf(fresh);
    expect(ledger.pairs).toEqual([]);
  });
  it("routes nothing without a ledger", () => {
    expect(admissionRouter(null).route(["kind:damage"])).toEqual({ entity: "kind:damage" });
    expect(admissionRouter(null).route([])).toEqual({ kinds: 0 });
  });
});

describe("withCorpus (a shadow sample or another regression run)", () => {
  const base = ledgerOf(tallyOf(FRESH, [{ count: 20, entities: ["move:a", "move:b"] }, { count: 20, entities: ["move:c"] }]));
  it("revokes what the new corpus finds a difference in and drops its pairs", () => {
    expect(base.pairs.map((p) => `${p.a}|${p.b}`)).toEqual(["move:a|move:b"]);
    const shadow = tallyOf(corpus("shadow-g1", "shadow", "d".repeat(64)), [{ count: 2, entities: ["move:a"], differing: [1] }, { count: 1, entities: ["move:c"] }]);
    const next = withCorpus(base, shadow);
    expect(entity(next, "move:a")).toMatchObject({ admitted: false, counts: [{ corpus: FRESH.id }, { corpus: "shadow-g1", differing: 1 }] });
    expect(entity(next, "move:b")?.admitted).toBe(true);
    expect(entity(next, "move:c")?.admitted).toBe(true);
    expect(next.revoked).toEqual([{ id: "move:a", corpus: "shadow-g1", position: 1 }]);
    expect(next.pairs).toEqual([]);
    expect(admissionHash(next)).not.toBe(admissionHash(base));
  });
  it("moves a revocation to an earlier corpus that found a difference, and keeps a later one's", () => {
    const later = withCorpus(base, tallyOf(corpus("zz-shadow", "shadow", "d".repeat(64)), [{ count: 1, entities: ["move:c"], differing: [0] }]));
    expect(later.revoked).toEqual([{ id: "move:c", corpus: "zz-shadow", position: 0 }]);
    const earlier = withCorpus(later, tallyOf(corpus("aa-regression", "regression", "e".repeat(64)), [{ count: 2, entities: ["move:c"], differing: [1] }]));
    expect(earlier.revoked).toEqual([{ id: "move:c", corpus: "aa-regression", position: 1 }]);
    const after = withCorpus(earlier, tallyOf(corpus("zzz", "regression", "e".repeat(64)), [{ count: 1, entities: ["move:c"], differing: [0] }]));
    expect(after.revoked).toEqual(earlier.revoked);
  });
  it("adds an entity only the new corpus holds", () => {
    const next = withCorpus(base, tallyOf(REGRESSION, [{ count: 2, entities: ["item:new"] }]));
    expect(entity(next, "item:new")).toEqual({ id: "item:new", admitted: false, counts: [{ corpus: REGRESSION.id, present: 2, exercised: 2, equal: 2, differing: 0 }], kinds: [] });
  });
  it("refuses a fresh corpus and a corpus it already holds", () => {
    expect(() => withCorpus(base, newTally(FRESH))).toThrow("never added later");
    expect(() => withCorpus(base, newTally({ ...FRESH, kind: "regression" }))).toThrow("already holds");
  });
});

describe("the reader", () => {
  const ledger = ledgerOf(tallyOf(FRESH, [{ count: 20, entities: ["move:a", "move:b"], differing: [] }, { count: 3, entities: ["move:c"], differing: [1] }]), tallyOf(REGRESSION, [{ count: 2, entities: ["move:a"] }]));
  it("reads what buildLedger wrote and hashes it", () => {
    expect(read(ledger)).toEqual(ledger);
    expect(admissionHash(ledger)).toBe(sha256Jcs(ledger as never));
  });
  it("refuses a ledger of another pinned engine", () => {
    expect(() => readBackendAdmission(ledger, { showdown_commit: "f".repeat(40) })).toThrow("another pinned engine");
  });
  it.each<[string, (l: BackendAdmission) => void, string]>([
    ["schema", (l) => void ((l as { rule: string }).rule = "admission/2"), "invalid BackendAdmission"],
    ["corpora order", (l) => l.corpora.reverse(), "corpora must be sorted"],
    ["two fresh", (l) => void ((l.corpora[0] as { kind: string }).kind = "fresh"), "at most one fresh"],
    ["fresh seed", (l) => void ((l.corpora[1] as { seed: string }).seed = "0".repeat(64)), "fresh seed"],
    ["entities order", (l) => l.entities.reverse(), "entities must be sorted"],
    ["revocations order", (l) => l.revoked.push({ ...(l.revoked[0] as BackendAdmission["revoked"][number]) }), "revocations must be sorted"],
    ["unknown corpus", (l) => void ((entityOf(l, "move:b").counts[0] as { corpus: string }).corpus = "nope"), "known corpora"],
    ["counts order", (l) => entityOf(l, "move:a").counts.reverse(), "known corpora"],
    ["present 0", (l) => void ((entityOf(l, "move:b").counts[0] as { present: number }).present = 0), "present in at least one"],
    ["present over items", (l) => void ((entityOf(l, "move:b").counts[0] as { present: number }).present = 200_001), "present in at least one"],
    ["nesting", (l) => void ((entityOf(l, "move:b").counts[0] as { equal: number }).equal = 21), "must nest"],
    ["unequal not differing", (l) => void ((entityOf(l, "move:b").counts[0] as { equal: number }).equal = 19), "unequal exercising item"],
    ["missing revocation", (l) => l.revoked.splice(0, 1), "revoked exactly when"],
    ["extra revocation", (l) => l.revoked.unshift({ id: "move:a", corpus: FRESH.id, position: 0 }), "revoked exactly when"],
    ["admitted flag", (l) => void (entityOf(l, "move:b").admitted = false), "admitted flag"],
    ["revocation corpus", (l) => void ((l.revoked[0] as { position: number }).position = 200_000), "within its corpus"],
  ])("refuses a doctored ledger: %s", (_, change, message) => {
    expect(() => read(doctor(ledger, change))).toThrow(message);
    expect(() => read(doctor(ledger, change))).toThrow(AdmissionError);
  });
  it("refuses a revocation of an unknown entity and doctored pairs", () => {
    const withPairs = ledgerOf(tallyOf(FRESH, [{ count: 20, entities: ["move:a", "move:b"] }, { count: 20, entities: ["move:c"] }]));
    expect(withPairs.pairs).toHaveLength(1);
    expect(() => read(doctor(withPairs, (l) => l.revoked.push({ id: "move:zz", corpus: FRESH.id, position: 0 })))).toThrow("known entity");
    const pair = (l: BackendAdmission) => l.pairs[0] as BackendAdmission["pairs"][number];
    expect(() => read(doctor(withPairs, (l) => void ([pair(l).a, pair(l).b] = [pair(l).b, pair(l).a])))).toThrow("a below b");
    expect(() => read(doctor(withPairs, (l) => l.pairs.push({ ...pair(l) })))).toThrow("sorted and unique");
    expect(() => read(doctor(withPairs, (l) => l.pairs.push({ ...pair(l), a: "move:a", b: "move:zz" })))).toThrow("two admitted entities");
    expect(() => read(doctor(withPairs, (l) => void (pair(l).co_equal = 21)))).toThrow("must nest");
    expect(() => read(doctor(withPairs, (l) => void (pair(l).co_present = 0)))).toThrow("must nest");
    expect(() => read(doctor(withPairs, (l) => void (pair(l).admitted = false)))).toThrow("pair's admitted flag");
    const sparse = doctor(withPairs, (l) => void Object.assign(l.corpora[0] as object, { items: 300_000, distinct_items: 300_000 }));
    expect(() => read(sparse)).toThrow("1 in 10^4");
  });
});

function entityOf(ledger: BackendAdmission, id: string) {
  return ledger.entities.find((e) => e.id === id) as BackendAdmission["entities"][number];
}

describe("the evidence rules of the review (PR #413)", () => {
  const good = () => ledgerOf(tallyOf(FRESH, [{ count: 25, entities: ["kind:damage", "move:a"] }]));
  it("the per-level minimum is at least three times the pair odds", () => {
    for (const min of Object.values(MIN_FRESH_ITEMS)) expect(min).toBeGreaterThanOrEqual(3 * PAIR_ODDS);
    expect(MIN_FRESH_ITEMS["mechanics-query"]).toBe(200_000);
  });
  it("lists per query kind the fresh items that exercised an entity, and routes by them", () => {
    const ledger = good();
    expect(entity(ledger, "move:a")?.kinds).toEqual([{ kind: "kind:damage", co_exercised: 25, co_equal: 25 }]);
    expect(entity(ledger, "kind:damage")?.kinds).toEqual([]);
    expect(admissionRouter(ledger).route(["kind:damage", "move:a"])).toBeNull();
  });
  it("refuses an admitted entity a small fresh corpus or a count mismatch cannot support", () => {
    expect(() => read(doctor(good(), (l) => void Object.assign(l.corpora[0] as object, { items: 199_999, distinct_items: 199_999 })))).toThrow("minimum size");
    expect(() => read(doctor(good(), (l) => void ((l.corpora[0] as { count_mismatches: number }).count_mismatches = 1)))).toThrow("count mismatch");
  });
  it("refuses kinds that are unsorted, not query kinds, on a query kind, or above the fresh counts", () => {
    const two = ledgerOf(tallyOf(FRESH, [{ count: 25, entities: ["kind:damage", "move:a"] }, { count: 25, entities: ["kind:stats", "move:a"] }]));
    expect(() => read(doctor(two, (l) => entityOf(l, "move:a").kinds.reverse()))).toThrow("sorted and unique");
    expect(() => read(doctor(two, (l) => void ((entityOf(l, "move:a").kinds[0] as { kind: string }).kind = "move:a")))).toThrow("sorted and unique");
    expect(() => read(doctor(two, (l) => entityOf(l, "kind:damage").kinds.push({ kind: "kind:stats", co_exercised: 0, co_equal: 0 })))).toThrow("only an entity the fresh corpus holds");
    expect(() => read(doctor(two, (l) => void ((entityOf(l, "move:a").kinds[0] as { co_exercised: number }).co_exercised = 51)))).toThrow("must nest within");
  });
  it("verifyLedger holds a ledger to its tallies: a dropped pair or a doctored corpus is refused", () => {
    const tallies = [tallyOf(FRESH, [{ count: 25, entities: ["move:a", "move:b"] }])];
    const ledger = ledgerOf(...tallies);
    expect(ledger.pairs).toHaveLength(1);
    expect(() => verifyLedger(ledger, tallies)).not.toThrow();
    expect(() => verifyLedger(doctor(ledger, (l) => void (l.pairs = [])), tallies)).toThrow("not the one its corpora give");
    expect(() => verifyLedger(doctor(ledger, (l) => void ((l.corpora[0] as { content_hash: string }).content_hash = "0".repeat(64))), tallies)).toThrow("content or tally");
    expect(() => verifyLedger(ledger, [tallyOf(FRESH, [{ count: 26, entities: ["move:a", "move:b"] }])])).toThrow("content or tally");
  });
  it("withCorpus equals buildLedger over all the tallies", () => {
    const fresh = tallyOf(FRESH, [{ count: 25, entities: ["move:a", "move:b"] }]);
    const shadow = tallyOf(corpus("shadow-g2", "shadow", "d".repeat(64)), [{ count: 3, entities: ["move:a"], differing: [2] }]);
    expect(admissionHash(withCorpus(ledgerOf(fresh), shadow))).toBe(admissionHash(ledgerOf(fresh, shadow)));
  });
  it("routes only by a ledger the reader returned, which is deeply frozen", () => {
    expect(() => admissionRouter(structuredClone(good()))).toThrow("readBackendAdmission");
    const ledger = good();
    expect(Object.isFrozen(ledger) && Object.isFrozen(ledger.entities[0]) && Object.isFrozen(ledger.entities[0]?.counts[0])).toBe(true);
    expect(() => (ledger.entities[0] as { admitted: boolean }).admitted = false).toThrow(TypeError);
  });
  it("routes a query of exactly one kind whose entities and pairs are admitted with it", () => {
    const ledger = ledgerOf(
      tallyOf(FRESH, [
        { count: 25, entities: ["kind:damage", "move:a", "move:b"] },
        { count: 20, entities: ["kind:damage", "move:a", "move:c"], reads: ["kind:damage", "move:a"] },
        { count: 25, entities: ["kind:damage", "move:c"] },
        { count: 25, entities: ["kind:stats", "species:x"] },
      ]),
    );
    const router = admissionRouter(ledger);
    expect(router.route(["kind:damage", "move:a", "move:b"])).toBeNull();
    expect(router.route(["kind:damage", "move:a", "move:c"])).toEqual({ pair: ["move:a", "move:c"] });
    expect(router.route(["kind:damage", "kind:stats", "move:a"])).toEqual({ kinds: 2 });
    expect(router.route(["kind:damage", "move:z"])).toEqual({ entity: "move:z" });
    expect(router.route(["kind:stats", "move:a"])).toEqual({ pair: ["kind:stats", "move:a"] });
  });
  it("refuses a repeated item in a fresh corpus and a ledger whose fresh corpus repeats items", () => {
    const twice = newTally(FRESH);
    addItem(twice, { position: 0, key: "q", entities: [], exercised: [], difference: null });
    expect(() => addItem(twice, { position: 1, key: "q", entities: [], exercised: [], difference: null })).toThrow("each item once");
    const regression = newTally(REGRESSION);
    addItem(regression, { position: 0, key: "q", entities: [], exercised: [], difference: null });
    addItem(regression, { position: 1, key: "q", entities: [], exercised: [], difference: null });
    expect(regression.seen.size).toBe(1);
    expect(() => read(doctor(good(), (l) => void ((l.corpora[0] as { distinct_items: number }).distinct_items = 5)))).toThrow("each item once");
  });
});
