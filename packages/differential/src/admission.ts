// The admission ledger (`BackendAdmission`, ADR-030 common rules 2, 4, 6 and 7; ADR-028 section 5): which entities one backend
// version may answer for training games, derived from corpus tallies by one arithmetic, read back by a reader that recomputes it.
//
// The rules, in one place:
// - An entity is admitted iff the FRESH corpus holds at least `N_EXERCISED` items that exercise it, all of them byte-equal,
//   no item of any corpus that holds it has a difference, and it is not revoked. Regression (debugging) and shadow corpora are
//   counted and reported, never admission evidence (rule 7, PokaiEngine's warning).
// - The fresh corpus's seed is `freshSeed(backend)`, SHA-256 over the JCS of {backend_commit, backend_format, purpose}: it
//   cannot exist before the backend version is frozen, so a corpus used to debug that version cannot be passed off as fresh.
// - A pair of admitted entities present together in at least 1 in `PAIR_ODDS` fresh items is listed; it is admitted iff at
//   least `N_PAIR` fresh items exercise both, all equal. A listed pair that is not admitted routes to the oracle.
// - Any difference revokes every entity of its item (the first such item, in corpus order, is recorded).
// - A ledger is bound to one `showdown_commit`: the reader refuses another pin, so after a pin bump every query routes to the
//   engine (rule 6).
import { ajvFor } from "@areneon/schema/ajv";
import schema from "@areneon/schema/contracts/v1/backend-admission.schema.json" with { type: "json" };
import type { BackendAdmission, Entity, EntityCount, Pair, Revoked } from "@areneon/schema/types/v1/backend-admission.ts";
import type { Json } from "./canonical.ts";
import { sha256Jcs } from "./hash.ts";
import type { CorpusKind, CorpusTally } from "./tally.ts";

export type { BackendAdmission };

/** Exercising items an entity needs on the fresh corpus (ADR-026 slice F's `N_EX`). */
export const N_EXERCISED = 20;
/** Co-exercised items a frequent pair needs (ADR-028 section 5). */
export const N_PAIR = 5;
/** A pair is frequent when present in at least 1 in this many fresh items (probability 10^-4). */
export const PAIR_ODDS = 10_000;
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
 * is a function of the frozen backend (and of what `draw` samples from: the generator pool, never a debugging corpus).
 */
export function freshCorpus<T>(backend: Pick<BackendIdentity, "backend_commit" | "backend_format">, draw: (seed: string) => T[]): { spec: { id: string; kind: "fresh"; seed: string }; items: T[] } {
  const seed = freshSeed(backend);
  return { spec: { id: freshCorpusId(backend), kind: "fresh", seed }, items: draw(seed) };
}

interface Totals {
  fresh: EntityCount | undefined;
  differing: number;
}

const freshKind = (corpora: BackendAdmission["corpora"], id: string): CorpusKind | undefined => corpora.find((c) => c.id === id)?.kind;

function totals(ledger: Pick<BackendAdmission, "corpora">, entity: Pick<Entity, "counts">): Totals {
  let fresh: EntityCount | undefined;
  let differing = 0;
  for (const count of entity.counts) {
    differing += count.differing;
    if (freshKind(ledger.corpora, count.corpus) === "fresh") fresh = count;
  }
  return { fresh, differing };
}

/** Whether counts admit an entity: enough equal exercising items on the fresh corpus and no difference anywhere. */
export function admits(fresh: Pick<EntityCount, "exercised" | "equal"> | undefined, differing: number, revoked: boolean): boolean {
  return fresh !== undefined && fresh.exercised >= N_EXERCISED && fresh.equal === fresh.exercised && differing === 0 && !revoked;
}

/** Whether a listed pair is admitted. */
export const admitsPair = (pair: Pick<Pair, "co_exercised" | "co_equal">): boolean => pair.co_exercised >= N_PAIR && pair.co_equal === pair.co_exercised;

/** Whether a pair present in `co_present` of `items` fresh items is frequent enough to be listed (integers: no division). */
export const frequent = (co_present: number, items: number): boolean => co_present > 0 && co_present * PAIR_ODDS >= items;

// Ids are unique in every list sorted here, so no comparison is ever equal.
const byId = <T extends { id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : 1);

export interface LedgerInput {
  level: BackendAdmission["level"];
  backend: BackendIdentity;
  showdown_commit: string;
  tallies: readonly CorpusTally[];
}

/** The ledger of `tallies`: corpora, per-entity counts, admission, revocations and the frequent pairs, all by the rules above. */
export function buildLedger(input: LedgerInput): BackendAdmission {
  const corpora = input.tallies.map((t) => ({ id: t.spec.id, kind: t.spec.kind, seed: t.spec.seed, items: t.items })).sort(byId);
  const tallies = [...input.tallies].sort((a, b) => byId(a.spec, b.spec));
  const ids = [...new Set(tallies.flatMap((t) => [...t.entities.keys()]))].sort();
  const revoked: Revoked[] = [];
  const entities: Entity[] = ids.map((id) => {
    const counts = tallies.flatMap((t) => {
      const row = t.entities.get(id);
      return row === undefined ? [] : [{ corpus: t.spec.id, present: row.present, exercised: row.exercised, equal: row.equal, differing: row.differing }];
    }) as Entity["counts"];
    const first = tallies.find((t) => (t.entities.get(id)?.differing ?? 0) > 0);
    if (first !== undefined) revoked.push({ id, corpus: first.spec.id, position: first.entities.get(id)?.first_difference as number });
    const sum = totals({ corpora }, { counts });
    return { id, admitted: admits(sum.fresh, sum.differing, first !== undefined), counts };
  });
  const admitted = new Set(entities.filter((e) => e.admitted).map((e) => e.id));
  const fresh = tallies.find((t) => t.spec.kind === "fresh");
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
 * new corpus found a difference in revoked, admission and the pairs recomputed (admission only ever falls).
 */
export function withCorpus(ledger: BackendAdmission, tally: CorpusTally): BackendAdmission {
  if (tally.spec.kind === "fresh") throw new AdmissionError("a fresh corpus is part of the ledger it admits from, never added later");
  if (ledger.corpora.some((c) => c.id === tally.spec.id)) throw new AdmissionError("the ledger already holds that corpus");
  const corpora = [...ledger.corpora, { id: tally.spec.id, kind: tally.spec.kind, seed: tally.spec.seed, items: tally.items }].sort(byId);
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
    return { id, admitted: admits(sum.fresh, sum.differing, revokedBy.has(id)), counts };
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

const validate = ajvFor("contract").compile(schema);

const sortedUnique = (values: readonly string[]): boolean => values.every((v, i) => i === 0 || (values[i - 1] as string) < v);

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
  const order = new Map(ledger.corpora.map((c, i) => [c.id, i]));
  const revoked = new Map(ledger.revoked.map((r) => [r.id, r]));
  if (!sortedUnique(ledger.revoked.map((r) => r.id))) fail("revocations must be sorted by id and unique");
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
    const ok = admits(sum.fresh, sum.differing, r !== undefined);
    if (entity.admitted !== ok) {
      const regression = entity.counts.filter((c) => freshKind(ledger.corpora, c.corpus) !== "fresh");
      const asIfFresh = regression.some((c) => c.exercised >= N_EXERCISED && c.equal === c.exercised);
      fail(entity.admitted && asIfFresh && sum.differing === 0 && r === undefined ? "admission evidence must come from the fresh corpus: regression and shadow counts never admit" : "an entity's admitted flag must equal the recomputed admission");
    }
  }
  const known = new Set(ledger.entities.map((e) => e.id));
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
  return ledger;
}

/** Why a query was routed to the oracle (counted per entity in run metrics), or null when the backend may answer it. */
export type Refusal = { entity: string } | { pair: [string, string] } | null;

/** The routing rule of a ledger: the backend answers a query only if every entity is admitted and no unadmitted pair is in it. */
export function admissionRouter(ledger: BackendAdmission | null): { route(entities: readonly string[]): Refusal } {
  const admitted = new Set(ledger?.entities.filter((e) => e.admitted).map((e) => e.id) ?? []);
  const blocked = new Set((ledger?.pairs ?? []).filter((p) => !p.admitted).map((p) => `${p.a}|${p.b}`));
  return {
    route(entities) {
      const sorted = [...new Set(entities)].sort();
      for (const id of sorted) if (!admitted.has(id)) return { entity: id };
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
