// The admission ledger (`BackendAdmission`, ADR-030 common rules 2, 4, 6 and 7, step 3 gate 3; ADR-028 section 5): which entities
// one backend version may answer for training games, derived from corpus tallies by one arithmetic, read back by a reader that
// recomputes it.
//
// The rules, in one place:
// - Evidence: admission needs ONE fresh corpus of at least `MIN_FRESH_ITEMS[level]` items (200,000 at both levels: ADR-030 step 3
//   gate 3 and ADR-028 section 5; at least 3 x `PAIR_ODDS`, so a pair at the listing threshold is expected three times) and no
//   corpus whose backend ever returned another number of outcomes than inputs (a count mismatch). Without that, nothing is admitted.
// - An entity is admitted iff that evidence holds, the fresh corpus holds at least `N_EXERCISED` items that exercise it, all of
//   them byte-equal, no item of any corpus that holds it has a difference, and it is not revoked. Regression (debugging) and
//   shadow corpora are counted and reported, never admission evidence (rule 7, PokaiEngine's warning).
// - The fresh corpus's seed is `freshSeed(backend)`, SHA-256 over the JCS of {backend_commit, backend_format, purpose}: it
//   cannot exist before the backend version is frozen. Every corpus also names its draw pool, a hash of its content (the sorted
//   item keys) and a hash of its tally, so `verifyLedger` can hold a ledger to the corpora it claims (a dropped pair is caught).
// - Entities are not scoped by query kind, so every entity row lists, per query kind, how many fresh items exercised it with that
//   kind: a query is routed to the backend only if each of its entities was co-exercised with the query's kind at least `N_PAIR`
//   times, all equal. Any other unseen kind x entity pair routes to the oracle.
// - A pair of admitted entities present together in at least 1 in `PAIR_ODDS` fresh items is listed; it is admitted iff at
//   least `N_PAIR` fresh items exercise both, all equal. A listed pair that is not admitted routes to the oracle.
// - Any difference revokes every entity of its item (the first such item, in corpus order, is recorded).
// - A ledger is bound to one `showdown_commit`: the reader refuses another pin, so after a pin bump every query routes to the
//   engine (rule 6).
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/backend-admission.schema.json" with { type: "json" };
import type { BackendAdmission, Corpus, Entity, EntityCount, Pair, Revoked } from "@areneon/schema/types/v1/backend-admission.ts";
import type { Json } from "./canonical.ts";
import { sha256Jcs } from "./hash.ts";
import { contentHash, type CorpusKind, type CorpusTally, tallyHash } from "./tally.ts";

export type { BackendAdmission };

/** Exercising items an entity needs on the fresh corpus (ADR-026 slice F's `N_EX`). */
export const N_EXERCISED = 20;
/** Co-exercised items a frequent pair, and an entity with a query kind, need (ADR-028 section 5). */
export const N_PAIR = 5;
/** A pair is frequent when present in at least 1 in this many fresh items (probability 10^-4). */
export const PAIR_ODDS = 10_000;
/** The least fresh corpus that admits anything, per level. */
export const MIN_FRESH_ITEMS: Readonly<Record<BackendAdmission["level"], number>> = { "mechanics-query": 200_000, "battle-step": 200_000 };
export const ADMISSION_RULE = "admission/1";
export const FRESH_PURPOSE = "differential-fresh/1";

export class AdmissionError extends Error {
  override name = "AdmissionError";
}

export interface BackendIdentity {
  kind: string;
  backend_format: number;
  backend_commit: string;
}

/** The seed of a backend version's fresh admission corpus: SHA-256 over the JCS of its frozen commit and format. */
export const freshSeed = (backend: Pick<BackendIdentity, "backend_commit" | "backend_format">): string =>
  sha256Jcs({ backend_commit: backend.backend_commit, backend_format: backend.backend_format, purpose: FRESH_PURPOSE });

/** The id of a backend version's fresh corpus. */
export const freshCorpusId = (backend: Pick<BackendIdentity, "backend_commit" | "backend_format">): string => `fresh-${freshSeed(backend).slice(0, 16)}`;

/**
 * The fresh admission corpus of a backend version: `draw` is called with the version's fresh seed and nothing else, so the corpus
 * is a function of the frozen backend and of what `draw` samples from, the pool named by `pool` (the generator pool, never a
 * debugging corpus).
 */
export function freshCorpus<T>(backend: Pick<BackendIdentity, "backend_commit" | "backend_format">, pool: string, draw: (seed: string) => T[]): { spec: { id: string; kind: "fresh"; seed: string; pool: string }; items: T[] } {
  const seed = freshSeed(backend);
  return { spec: { id: freshCorpusId(backend), kind: "fresh", seed, pool }, items: draw(seed) };
}

const kindOf = (corpora: readonly Corpus[], id: string): CorpusKind | undefined => corpora.find((c) => c.id === id)?.kind;

/** Whether a ledger's corpora can admit at all: one fresh corpus of the level's minimum size and no count mismatch anywhere. */
export function evidenceHolds(ledger: Pick<BackendAdmission, "level" | "corpora">): boolean {
  const fresh = ledger.corpora.find((c) => c.kind === "fresh");
  return fresh !== undefined && fresh.items >= MIN_FRESH_ITEMS[ledger.level] && ledger.corpora.every((c) => c.count_mismatches === 0);
}

function totals(ledger: Pick<BackendAdmission, "corpora">, entity: Pick<Entity, "counts">): { fresh: EntityCount | undefined; differing: number } {
  let fresh: EntityCount | undefined;
  let differing = 0;
  for (const count of entity.counts) {
    differing += count.differing;
    if (kindOf(ledger.corpora, count.corpus) === "fresh") fresh = count;
  }
  return { fresh, differing };
}

/** Whether counts admit an entity: the ledger's evidence holds, enough equal exercising fresh items, no difference anywhere. */
export function admits(fresh: Pick<EntityCount, "exercised" | "equal"> | undefined, differing: number, revoked: boolean, evidence: boolean): boolean {
  return evidence && fresh !== undefined && fresh.exercised >= N_EXERCISED && fresh.equal === fresh.exercised && differing === 0 && !revoked;
}

/** Whether co-exercised counts admit a pair (or an entity with a query kind). */
export const admitsPair = (pair: { co_exercised: number; co_equal: number }): boolean => pair.co_exercised >= N_PAIR && pair.co_equal === pair.co_exercised;

/** Whether a pair present in `co_present` of `items` fresh items is frequent enough to be listed (integers: no division). */
export const frequent = (co_present: number, items: number): boolean => co_present > 0 && co_present * PAIR_ODDS >= items;

// Ids are unique in every list sorted here, so no comparison is ever equal.
const byId = <T extends { id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : 1);
const isKind = (id: string): boolean => id.startsWith("kind:");

export interface LedgerInput {
  level: BackendAdmission["level"];
  backend: BackendIdentity;
  showdown_commit: string;
  tallies: readonly CorpusTally[];
}

const corpusOf = (t: CorpusTally): Corpus => ({
  id: t.spec.id,
  kind: t.spec.kind,
  seed: t.spec.seed,
  pool_hash: t.spec.pool,
  content_hash: contentHash(t),
  tally_hash: tallyHash(t),
  items: t.items,
  count_mismatches: t.count_mismatches,
});

/** Per entity, per query kind, the fresh items that exercised it with that kind (from the fresh tally's pairs, read once). */
function kindsTable(fresh: CorpusTally | undefined): Map<string, Entity["kinds"]> {
  const table = new Map<string, Entity["kinds"]>();
  for (const [key, row] of fresh?.pairs ?? []) {
    const [a, b] = key.split("|") as [string, string];
    if (isKind(a) === isKind(b)) continue;
    const [kind, id] = isKind(a) ? [a, b] : [b, a];
    const rows = table.get(id) ?? [];
    rows.push({ kind, co_exercised: row.co_exercised, co_equal: row.co_equal });
    table.set(id, rows);
  }
  for (const rows of table.values()) rows.sort((x, y) => (x.kind < y.kind ? -1 : 1));
  return table;
}

/** The ledger of `tallies`: corpora, per-entity counts and kinds, admission, revocations and the frequent pairs, all by the rules above. */
export function buildLedger(input: LedgerInput): BackendAdmission {
  const corpora = input.tallies.map(corpusOf).sort(byId);
  const evidence = evidenceHolds({ level: input.level, corpora });
  const tallies = [...input.tallies].sort((a, b) => byId(a.spec, b.spec));
  const fresh = tallies.find((t) => t.spec.kind === "fresh");
  const ids = [...new Set(tallies.flatMap((t) => [...t.entities.keys()]))].sort();
  const kinds = kindsTable(fresh);
  const revoked: Revoked[] = [];
  const entities: Entity[] = ids.map((id) => {
    const counts = tallies.flatMap((t) => {
      const row = t.entities.get(id);
      return row === undefined ? [] : [{ corpus: t.spec.id, present: row.present, exercised: row.exercised, equal: row.equal, differing: row.differing }];
    }) as Entity["counts"];
    const first = tallies.find((t) => (t.entities.get(id)?.differing ?? 0) > 0);
    if (first !== undefined) revoked.push({ id, corpus: first.spec.id, position: first.entities.get(id)?.first_difference as number });
    const sum = totals({ corpora }, { counts });
    return { id, admitted: admits(sum.fresh, sum.differing, first !== undefined, evidence), counts, kinds: kinds.get(id) ?? [] };
  });
  const admitted = new Set(entities.filter((e) => e.admitted).map((e) => e.id));
  const pairs: Pair[] = [];
  if (fresh !== undefined) {
    for (const [key, row] of fresh.pairs) {
      const [a, b] = key.split("|") as [string, string];
      if (!admitted.has(a) || !admitted.has(b) || !frequent(row.co_present, fresh.items)) continue;
      pairs.push({ a, b, co_present: row.co_present, co_exercised: row.co_exercised, co_equal: row.co_equal, admitted: admitsPair(row) });
    }
  }
  // By a, then b: a NUL separator sorts below every id character, so the joined keys order as the pairs do.
  const pairOrder = (p: Pair): string => `${p.a}\u0000${p.b}`;
  pairs.sort((x, y) => (pairOrder(x) < pairOrder(y) ? -1 : 1));
  const ledger: BackendAdmission = {
    schema_version: 1,
    data_class: "synthetic",
    rule: ADMISSION_RULE,
    level: input.level,
    backend: { ...input.backend },
    showdown_commit: input.showdown_commit,
    corpora,
    entities,
    pairs,
    revoked,
  };
  return readBackendAdmission(ledger, { showdown_commit: input.showdown_commit });
}

/**
 * The ledger after one more non-fresh corpus (a regression run, or a generation's shadow sample): counts merged, any entity the
 * new corpus found a difference in revoked, admission and the pairs recomputed (admission only ever falls; a count mismatch in
 * the new corpus ends every admission). Equal to `buildLedger` over all the tallies.
 */
export function withCorpus(ledger: BackendAdmission, tally: CorpusTally): BackendAdmission {
  if (tally.spec.kind === "fresh") throw new AdmissionError("a fresh corpus is part of the ledger it admits from, never added later");
  if (ledger.corpora.some((c) => c.id === tally.spec.id)) throw new AdmissionError("the ledger already holds that corpus");
  const corpora = [...ledger.corpora, corpusOf(tally)].sort(byId);
  const evidence = evidenceHolds({ level: ledger.level, corpora });
  const order = new Map(corpora.map((c, i) => [c.id, i]));
  const rank = (corpus: string): number => order.get(corpus) as number;
  const revokedBy = new Map(ledger.revoked.map((r) => [r.id, r]));
  const ids = [...new Set([...ledger.entities.map((e) => e.id), ...tally.entities.keys()])].sort();
  const old = new Map(ledger.entities.map((e) => [e.id, e]));
  const entities: Entity[] = ids.map((id) => {
    const row = tally.entities.get(id);
    const added = row === undefined ? [] : [{ corpus: tally.spec.id, present: row.present, exercised: row.exercised, equal: row.equal, differing: row.differing }];
    const counts = [...(old.get(id)?.counts ?? []), ...added].sort((a, b) => rank(a.corpus) - rank(b.corpus)) as Entity["counts"];
    if (row !== undefined && row.differing > 0) {
      const before = revokedBy.get(id);
      if (before === undefined || rank(tally.spec.id) < rank(before.corpus)) revokedBy.set(id, { id, corpus: tally.spec.id, position: row.first_difference as number });
    }
    const sum = totals({ corpora }, { counts });
    return { id, admitted: admits(sum.fresh, sum.differing, revokedBy.has(id), evidence), counts, kinds: old.get(id)?.kinds ?? [] };
  });
  const admitted = new Set(entities.filter((e) => e.admitted).map((e) => e.id));
  const next: BackendAdmission = {
    ...ledger,
    corpora,
    entities,
    pairs: ledger.pairs.filter((p) => admitted.has(p.a) && admitted.has(p.b)),
    revoked: [...revokedBy.values()].sort(byId),
  };
  return readBackendAdmission(next, { showdown_commit: ledger.showdown_commit });
}

/** `admission_hash`: SHA-256 of the ledger's JCS bytes (the hash `generation.backends` names). */
export const admissionHash = (ledger: BackendAdmission): string => sha256Jcs(ledger as unknown as Json);

/**
 * Hold a ledger to the corpora it claims: every corpus's content and tally hash, and the whole ledger, must be exactly what
 * `buildLedger` gives for `tallies`. A hand-edited ledger (a pair or a count dropped, a corpus swapped) is refused.
 */
export function verifyLedger(ledger: BackendAdmission, tallies: readonly CorpusTally[]): void {
  const rebuilt = buildLedger({ level: ledger.level, backend: ledger.backend, showdown_commit: ledger.showdown_commit, tallies });
  for (const corpus of ledger.corpora) {
    const found = rebuilt.corpora.find((c) => c.id === corpus.id);
    if (found === undefined || found.content_hash !== corpus.content_hash || found.tally_hash !== corpus.tally_hash) throw new AdmissionError("a corpus's content or tally is not the one the ledger names");
  }
  if (admissionHash(rebuilt) !== admissionHash(ledger)) throw new AdmissionError("the ledger is not the one its corpora give (a pair, count or flag differs)");
}

const validate = ajvFor("contract").compile(schema);

const sortedUnique = (values: readonly string[]): boolean => values.every((v, i) => i === 0 || (values[i - 1] as string) < v);

/** Ledgers that went through the reader: the only ones `admissionRouter` takes. */
const read = new WeakSet<BackendAdmission>();

/**
 * Read a ledger: the schema, then every rule the profile cannot say, recomputed. `showdown_commit` is the pin of the reading
 * checkout; a ledger of another pin is refused, never reused.
 */
export function readBackendAdmission(value: unknown, expected: { showdown_commit: string }): BackendAdmission {
  if (!validate(value)) throw new AdmissionError(`invalid BackendAdmission: ${JSON.stringify(validate.errors)}`);
  const ledger = structuredClone(value) as unknown as BackendAdmission;
  const fail = (message: string): never => {
    throw new AdmissionError(message);
  };
  if (ledger.showdown_commit !== expected.showdown_commit) fail("the ledger is for another pinned engine: every query routes to the engine");
  if (!sortedUnique(ledger.corpora.map((c) => c.id))) fail("corpora must be sorted by id and unique");
  const fresh = ledger.corpora.filter((c) => c.kind === "fresh");
  if (fresh.length > 1) fail("a ledger holds at most one fresh corpus");
  const items = new Map(ledger.corpora.map((c) => [c.id, c.items]));
  if (fresh[0] !== undefined && fresh[0].seed !== freshSeed(ledger.backend)) fail("the fresh corpus's seed is not the backend version's fresh seed");
  if (!sortedUnique(ledger.entities.map((e) => e.id))) fail("entities must be sorted by id and unique");
  const known = new Set(ledger.entities.map((e) => e.id));
  const order = new Map(ledger.corpora.map((c, i) => [c.id, i]));
  const revoked = new Map(ledger.revoked.map((r) => [r.id, r]));
  if (!sortedUnique(ledger.revoked.map((r) => r.id))) fail("revocations must be sorted by id and unique");
  const evidence = evidenceHolds(ledger);
  for (const entity of ledger.entities) {
    const ranks = entity.counts.map((c) => order.get(c.corpus) ?? -1);
    if (ranks.some((rank, i) => rank < 0 || (i > 0 && rank <= (ranks[i - 1] as number)))) fail("an entity's counts must name known corpora, in corpus order, once each");
    let firstDiffering: string | undefined;
    for (const c of entity.counts) {
      if (c.present < 1 || c.present > (items.get(c.corpus) as number)) fail("an entity is present in at least one and at most every item of a corpus");
      if (c.exercised > c.present || c.equal > c.exercised || c.differing > c.present) fail("an entity's counts must nest: equal <= exercised <= present, differing <= present");
      if (c.exercised - c.equal > c.differing) fail("an unequal exercising item is a differing item");
      if (c.differing > 0) firstDiffering ??= c.corpus;
    }
    const r = revoked.get(entity.id);
    if (r?.corpus !== firstDiffering) fail("an entity is revoked exactly when a difference named it, at the first corpus that found one");
    const sum = totals(ledger, entity);
    const kinds = entity.kinds.map((k) => k.kind);
    if (!sortedUnique(kinds) || kinds.some((k) => !isKind(k) || !known.has(k))) fail("an entity's kinds name query-kind entities of the ledger, sorted and unique");
    if ((isKind(entity.id) || sum.fresh === undefined) && kinds.length > 0) fail("only an entity the fresh corpus holds, and no query kind, lists kinds");
    if (entity.kinds.some((k) => k.co_equal > k.co_exercised || k.co_exercised > (sum.fresh?.exercised ?? 0))) fail("an entity's kind counts must nest within its fresh exercising items");
    const ok = admits(sum.fresh, sum.differing, r !== undefined, evidence);
    if (entity.admitted !== ok) {
      const otherwise = admits(sum.fresh, sum.differing, r !== undefined, true);
      const asIfFresh = entity.counts.some((c) => kindOf(ledger.corpora, c.corpus) !== "fresh" && c.exercised >= N_EXERCISED && c.equal === c.exercised);
      if (entity.admitted && otherwise) fail("nothing is admitted without a fresh corpus of the level's minimum size and with any count mismatch");
      fail(entity.admitted && asIfFresh && sum.differing === 0 && r === undefined ? "admission evidence must come from the fresh corpus: regression and shadow counts never admit" : "an entity's admitted flag must equal the recomputed admission");
    }
  }
  for (const r of ledger.revoked) {
    if (!known.has(r.id) || !order.has(r.corpus)) fail("a revocation names a known entity and corpus");
    if (r.position >= (items.get(r.corpus) as number)) fail("a revocation's position lies within its corpus");
  }
  const admitted = new Set(ledger.entities.filter((e) => e.admitted).map((e) => e.id));
  const freshItems = fresh[0]?.items ?? 0;
  ledger.pairs.forEach((p, i) => {
    const prev = ledger.pairs[i - 1];
    if (!(p.a < p.b) || (prev !== undefined && (prev.a > p.a || (prev.a === p.a && prev.b >= p.b)))) fail("pairs must have a below b and be sorted and unique");
    if (!admitted.has(p.a) || !admitted.has(p.b)) fail("a pair lists two admitted entities");
    if (p.co_exercised > p.co_present || p.co_equal > p.co_exercised || p.co_present > freshItems) fail("a pair's counts must nest within the fresh corpus");
    if (!frequent(p.co_present, freshItems)) fail("a listed pair is present in at least 1 in 10^4 fresh items");
    if (p.admitted !== admitsPair(p)) fail("a pair's admitted flag must equal the recomputed admission");
  });
  read.add(ledger);
  return ledger;
}

/** Why a query was routed to the oracle (counted per entity in run metrics), or null when the backend may answer it. */
export type Refusal = { entity: string } | { pair: [string, string] } | null;

/**
 * The routing rule of a ledger the reader returned: the backend answers a query only if every entity is admitted, each was
 * co-exercised at least `N_PAIR` times, all equal, with each query kind the query names, and no listed unadmitted pair is in it.
 */
export function admissionRouter(ledger: BackendAdmission | null): { route(entities: readonly string[]): Refusal } {
  if (ledger !== null && !read.has(ledger)) throw new AdmissionError("route only by a ledger readBackendAdmission returned");
  const admitted = new Set(ledger?.entities.filter((e) => e.admitted).map((e) => e.id) ?? []);
  const withKind = new Set(ledger?.entities.flatMap((e) => e.kinds.filter(admitsPair).map((k) => `${k.kind}|${e.id}`)) ?? []);
  const blocked = new Set((ledger?.pairs ?? []).filter((p) => !p.admitted).map((p) => `${p.a}|${p.b}`));
  return {
    route(entities) {
      const sorted = [...new Set(entities)].sort();
      for (const id of sorted) if (!admitted.has(id)) return { entity: id };
      const kinds = sorted.filter(isKind);
      for (const kind of kinds) for (const id of sorted) if (!isKind(id) && !withKind.has(`${kind}|${id}`)) return { pair: [kind, id] };
      for (let i = 0; i < sorted.length; i++) {
        for (let j = i + 1; j < sorted.length; j++) {
          const a = sorted[i] as string;
          const b = sorted[j] as string;
          if (blocked.has(`${a}|${b}`)) return { pair: [a, b] };
        }
      }
      return null;
    },
  };
}
