// The evaluation register (evaluation protocol I6, ADR-010 O1): an append-only,
// hash-chained JSON Lines file kept in the isolated store. Each line is the
// JCS bytes of one `EvaluationRegisterEntry`; each entry carries the previous
// entry's hash, so changing, dropping or reordering any entry changes every
// later hash. The chain head (`EvaluationRegisterHead`, a hash and a position,
// no secret) is merged to main by pull request, or attested: that is the
// timestamp the owner cannot backdate. A register shorter than a published
// head (rollback) or with another hash at its position (rewrite, fork) is
// refused. The entries' own `recorded_at` is a claim, not a trusted time.
//
// Pure and offline: `verifyRegister` reads text and returns the state or
// throws a RegisterError naming the line. Engine-free.

import { ajvFor } from "@areneon/schema/ajv";
import entrySchema from "@areneon/schema/contracts/v1/evaluation-register-entry.schema.json" with { type: "json" };
import headSchema from "@areneon/schema/contracts/v1/evaluation-register-head.schema.json" with { type: "json" };
import type { EvaluationRegisterEntry } from "@areneon/schema/types/v1/evaluation-register-entry.ts";
import type { EvaluationRegisterHead } from "@areneon/schema/types/v1/evaluation-register-head.ts";
import { jcs, type Json } from "./canonical-json.ts";
import { domainHash, type EvaluationConfig, strictlyAscending } from "./evaluation-config.ts";
import { checkCommitment, checkPlan, commitmentHash, readEvaluationPlan, readSealedSetCommitment, planHash, type EvaluationPlan, type SealedSetCommitment } from "./evaluation-plan.ts";

export type { EvaluationRegisterEntry, EvaluationRegisterHead };

/** The `prev_hash` of the first entry. */
export const GENESIS = "0".repeat(64);

export class RegisterError extends Error {
  override name = "RegisterError";
  /** The 1-based line of the register the problem is on, when there is one. */
  readonly line: number | undefined;
  constructor(message: string, line?: number) {
    super(line === undefined ? message : `line ${line}: ${message}`);
    this.line = line;
  }
}

const validEntry = ajvFor("contract").compile(entrySchema);
const validHead = ajvFor("contract").compile(headSchema);

export const entryHash = (entry: EvaluationRegisterEntry): string => domainHash("evaluation-register-entry", entry);

/** Read one entry: the schema, then the rules inside it that the profile cannot state. */
export function readEntry(value: unknown): EvaluationRegisterEntry {
  if (!validEntry(value)) throw new RegisterError(`invalid EvaluationRegisterEntry: ${JSON.stringify(validEntry.errors)}`);
  const entry = structuredClone(value) as unknown as EvaluationRegisterEntry;
  // A real date, not just the shape of one: 9999-99-99 would lock out every later entry.
  const date = new Date(entry.recorded_at);
  if (Number.isNaN(date.getTime()) || `${date.toISOString().slice(0, 19)}Z` !== entry.recorded_at) {
    throw new RegisterError(`recorded_at ${entry.recorded_at} is not a real UTC date and time`);
  }
  try {
    if (entry.kind === "set_commitment") readSealedSetCommitment(entry.commitment);
    if (entry.kind === "plan_registration") readEvaluationPlan(entry.plan);
  } catch (error) {
    throw new RegisterError((error as Error).message);
  }
  switch (entry.kind) {
    case "run_outcome":
      if (entry.outcome === "reported" && entry.outcome_hash === undefined) throw new RegisterError("a reported run needs its outcome_hash");
      if (entry.outcome !== "reported" && entry.reason === undefined) throw new RegisterError(`a run that is ${entry.outcome} needs its reason`);
      if (entry.outcome !== "reported" && entry.outcome_hash !== undefined) throw new RegisterError(`a run that is ${entry.outcome} has no outcome_hash`);
      break;
    case "incident":
      if (!strictlyAscending(entry.affected_candidates)) throw new RegisterError("affected_candidates must be sorted and unique");
      break;
    default:
      break;
  }
  return entry;
}

/** The state of one sealed set along the chain. */
export interface SetState {
  commitment: SealedSetCommitment;
  /** Registered plans, in order, with the position of their entry. */
  plans: { plan_hash: string; seq: number; candidate: string }[];
  /** Plan hashes whose run has an outcome entry. */
  outcomes: Set<string>;
  /** Plan hashes with a passing L2a re-check since registration. */
  rechecked: Set<string>;
  contaminated: boolean;
  retired: boolean;
  /** The records the commitment names, when the verifier was given them; the budget then comes from the parameters. */
  config: EvaluationConfig | undefined;
  /** A set_retirement entry exists. */
  retirementLogged: boolean;
}

export interface RegisterState {
  entries: EvaluationRegisterEntry[];
  /** Every entry's hash, by position (index = seq - 1). */
  hashes: string[];
  /** The chain head: absent for an empty register. */
  head: EvaluationRegisterHead | null;
  sets: Map<string, SetState>;
  /** The records to check every commitment and plan against; none means the chain alone is checked. */
  configs: readonly EvaluationConfig[] | undefined;
}

const asRegisterError = (check: () => void): void => {
  try {
    check();
  } catch (error) {
    throw new RegisterError((error as Error).message);
  }
};

/** Applies one entry to the state, or throws what rule it breaks. */
function apply(state: RegisterState, entry: EvaluationRegisterEntry): void {
  const setOf = (id: string): SetState => {
    const set = state.sets.get(id);
    if (set === undefined) throw new RegisterError(`set ${id} has no commitment before this entry`);
    return set;
  };
  const planOf = (hash: string): { set: SetState; seq: number } => {
    for (const set of state.sets.values()) {
      const found = set.plans.find((p) => p.plan_hash === hash);
      if (found !== undefined) return { set, seq: found.seq };
    }
    throw new RegisterError(`plan ${hash} is not registered before this entry`);
  };
  switch (entry.kind) {
    case "set_commitment": {
      const c = entry.commitment;
      if (state.sets.has(c.set_id)) throw new RegisterError(`set ${c.set_id} is committed twice`);
      for (const other of state.sets.values()) {
        if (other.commitment.set_commitment === c.set_commitment || other.commitment.key_commitment === c.key_commitment) {
          throw new RegisterError(`set ${c.set_id} reuses the commitment or key commitment of ${other.commitment.set_id}`);
        }
      }
      let config: EvaluationConfig | undefined;
      if (state.configs !== undefined) {
        config = state.configs.find((k) => k.parameters_hash === c.evaluation_parameters_hash && k.rule_hash === c.training_gate_rule_hash);
        if (config === undefined) throw new RegisterError(`set ${c.set_id} names records (${c.evaluation_parameters_hash}, ${c.training_gate_rule_hash}) that were not given to the verifier`);
        const known = config;
        asRegisterError(() => checkCommitment(c, known));
      }
      state.sets.set(c.set_id, { commitment: c, plans: [], outcomes: new Set(), rechecked: new Set(), config, contaminated: false, retired: false, retirementLogged: false });
      return;
    }
    case "plan_registration": {
      const plan: EvaluationPlan = entry.plan;
      const set = setOf(plan.set_id);
      if (set.retired) throw new RegisterError(`set ${plan.set_id} is retired`);
      if (plan.set_commitment_hash !== commitmentHash(set.commitment)) throw new RegisterError(`the plan does not name set ${plan.set_id}'s commitment`);
      const hash = planHash(plan);
      if ([...state.sets.values()].some((s) => s.plans.some((p) => p.plan_hash === hash))) throw new RegisterError(`plan ${hash} is registered twice`);
      if (set.config !== undefined) {
        const known = set.config;
        asRegisterError(() => checkPlan(plan, { config: known, commitment: set.commitment }));
      }
      const budget = set.config?.parameters.b_sealed ?? set.commitment.b_sealed;
      if (set.plans.length >= budget) throw new RegisterError(`set ${plan.set_id} allows ${budget} run(s) and they are all registered (failed runs count)`);
      const candidate = plan.candidate.map((c) => c.checkpoint_id).join("+");
      if (set.plans.some((p) => p.candidate === candidate)) throw new RegisterError(`candidate ${candidate} already has a plan on set ${plan.set_id}`);
      set.plans.push({ plan_hash: hash, seq: entry.seq, candidate });
      return;
    }
    case "l2a_recheck": {
      const set = setOf(entry.set_id);
      if (set.retired) throw new RegisterError(`set ${entry.set_id} is retired`);
      if (entry.plan_hash !== undefined) {
        const { set: owner } = planOf(entry.plan_hash);
        if (owner !== set) throw new RegisterError(`plan ${entry.plan_hash} belongs to set ${owner.commitment.set_id}, not ${entry.set_id}`);
        if (entry.result === "pass") set.rechecked.add(entry.plan_hash);
      }
      // A failed re-check retires the set (I8): its pass no longer counts.
      if (entry.result === "fail") {
        set.contaminated = true;
        set.retired = true;
      }
      return;
    }
    case "run_outcome": {
      const { set } = planOf(entry.plan_hash);
      if (set.outcomes.has(entry.plan_hash)) throw new RegisterError(`plan ${entry.plan_hash} already has an outcome`);
      if ((set.retired || set.contaminated) && entry.outcome !== "aborted") {
        throw new RegisterError(`set ${set.commitment.set_id} is ${set.contaminated ? "contaminated" : "retired"}: only an aborted run may be logged`);
      }
      if (entry.outcome !== "aborted" && !set.rechecked.has(entry.plan_hash)) {
        throw new RegisterError(`plan ${entry.plan_hash} has no passing L2a re-check before its run`);
      }
      set.outcomes.add(entry.plan_hash);
      return;
    }
    case "incident": {
      const set = setOf(entry.set_id);
      set.contaminated = true;
      set.retired = true;
      return;
    }
    case "set_retirement": {
      const set = setOf(entry.set_id);
      if (set.retirementLogged) throw new RegisterError(`set ${entry.set_id} is already retired`);
      if ((entry.reason === "contaminated") !== set.contaminated) {
        throw new RegisterError(`set ${entry.set_id} is ${set.contaminated ? "" : "not "}contaminated, so its retirement reason cannot be ${entry.reason}`);
      }
      if (entry.reason === "budget_spent" && (set.plans.length < (set.config?.parameters.b_sealed ?? set.commitment.b_sealed) || set.outcomes.size < set.plans.length)) {
        throw new RegisterError(`set ${entry.set_id} has budget left or a run without an outcome`);
      }
      set.retired = true;
      set.retirementLogged = true;
      return;
    }
  }
}

/** Verifies the chain and the state rules of a whole register; throws a RegisterError naming the line. */
export function verifyRegister(text: string, configs?: readonly EvaluationConfig[]): RegisterState {
  const state: RegisterState = { entries: [], hashes: [], head: null, sets: new Map(), configs };
  if (text === "") return state;
  if (!text.endsWith("\n")) throw new RegisterError("the register must end with a newline (a torn last line?)");
  const lines = text.slice(0, -1).split("\n");
  for (const [index, line] of lines.entries()) {
    const n = index + 1;
    try {
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new RegisterError("not JSON");
      }
      const entry = readEntry(parsed);
      if (jcs(entry as unknown as Json) !== line) throw new RegisterError("the line is not the entry's JCS bytes");
      if (entry.seq !== n) throw new RegisterError(`seq ${entry.seq} where ${n} is due`);
      const previous = state.hashes[index - 1] ?? GENESIS;
      if (entry.prev_hash !== previous) throw new RegisterError(`prev_hash ${entry.prev_hash} does not follow the previous entry (${previous})`);
      const before = state.entries[index - 1];
      if (before !== undefined && entry.recorded_at < before.recorded_at) throw new RegisterError(`recorded_at ${entry.recorded_at} is before the previous entry's ${before.recorded_at}`);
      apply(state, entry);
      state.entries.push(entry);
      state.hashes.push(entryHash(entry));
    } catch (error) {
      throw new RegisterError((error as Error).message, n);
    }
  }
  state.head = { schema_version: 1, seq: state.entries.length, entry_hash: state.hashes[state.hashes.length - 1] as string };
  return state;
}

/** The fields an operator gives; the register supplies the rest. */
export type EntryDraft = EvaluationRegisterEntry extends infer E ? (E extends unknown ? Omit<E, "schema_version" | "seq" | "prev_hash"> : never) : never;

/** The register text with one entry appended, verified as a whole; the old text is never touched. */
export function appendEntry(text: string, draft: EntryDraft, configs?: readonly EvaluationConfig[]): { text: string; entry: EvaluationRegisterEntry; head: EvaluationRegisterHead } {
  const state = verifyRegister(text, configs);
  const entry = readEntry({ ...draft, schema_version: 1, seq: state.entries.length + 1, prev_hash: state.head?.entry_hash ?? GENESIS });
  const next = `${text}${jcs(entry as unknown as Json)}\n`;
  const after = verifyRegister(next, configs);
  return { text: next, entry, head: after.head as EvaluationRegisterHead };
}

/** Read a published head. */
export function readHead(value: unknown): EvaluationRegisterHead {
  if (!validHead(value)) throw new RegisterError(`invalid EvaluationRegisterHead: ${JSON.stringify(validHead.errors)}`);
  return structuredClone(value) as unknown as EvaluationRegisterHead;
}

export interface HeadCheck {
  /** The highest published position: entries up to it have a trusted timestamp. */
  published_through: number;
  /** Entries after it, not yet published. */
  unpublished: number;
}

/**
 * The register against every published head. Refused: a head position the
 * register does not reach (rollback or truncation), another hash at a head's
 * position (a rewritten history or a fork), the same position published twice.
 */
export function checkHeads(state: RegisterState, heads: readonly unknown[]): HeadCheck {
  let through = 0;
  const seen = new Set<number>();
  for (const value of heads) {
    const head = readHead(value);
    if (seen.has(head.seq)) throw new RegisterError(`two published heads at position ${head.seq}`);
    seen.add(head.seq);
    const actual = state.hashes[head.seq - 1];
    if (actual === undefined) {
      throw new RegisterError(`rollback: a head is published at position ${head.seq} but the register has ${state.entries.length} entries`);
    }
    if (actual !== head.entry_hash) {
      throw new RegisterError(`fork or rewrite: position ${head.seq} holds ${actual}, the published head says ${head.entry_hash}`);
    }
    through = Math.max(through, head.seq);
  }
  return { published_through: through, unpublished: state.entries.length - through };
}

/** I2: the run may start only when the plan's own entry is covered by a published head. */
export function assertPlanPublished(state: RegisterState, check: HeadCheck, hash: string): number {
  for (const set of state.sets.values()) {
    const found = set.plans.find((p) => p.plan_hash === hash);
    if (found !== undefined) {
      if (found.seq > check.published_through) {
        throw new RegisterError(`plan ${hash} is at position ${found.seq} but the published heads reach ${check.published_through}: merge the chain head first`);
      }
      return found.seq;
    }
  }
  throw new RegisterError(`plan ${hash} is not in the register`);
}

/** The file a head at `seq` is kept in, under the heads directory. */
export const headFileName = (seq: number): string => `${String(seq).padStart(10, "0")}.json`;

/**
 * The head files of a pull request against those on the base branch: heads
 * are add-only. A base file may not change or go (that would rewrite a
 * published timestamp), every file is named by its own position, and a new
 * head lies beyond every base head. Files are name -> text.
 */
export function checkHeadsAddOnly(base: ReadonlyMap<string, string>, current: ReadonlyMap<string, string>): void {
  for (const [name, text] of base) {
    const now = current.get(name);
    if (now === undefined) throw new RegisterError(`${name} is a published head and was deleted`);
    if (now !== text) throw new RegisterError(`${name} is a published head and was modified`);
  }
  let top = 0;
  for (const name of base.keys()) top = Math.max(top, readHead(JSON.parse(base.get(name) as string)).seq);
  for (const [name, text] of current) {
    let head: EvaluationRegisterHead;
    try {
      head = readHead(JSON.parse(text));
    } catch (error) {
      throw new RegisterError(`${name}: ${(error as Error).message}`);
    }
    if (name !== headFileName(head.seq)) throw new RegisterError(`${name}: the file name must be the zero-padded position (${headFileName(head.seq)})`);
    if (!base.has(name) && head.seq <= top) throw new RegisterError(`${name}: a new head at position ${head.seq} is not beyond the published ${top}`);
  }
}
