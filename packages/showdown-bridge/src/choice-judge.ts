import { enumerateCandidates } from "./choices.ts";
import type { Engine, SimBattle, SimSide } from "./engine.ts";
import type { DecodedRequest, JsonObject } from "./request.ts";
import { requestHash } from "./request-identity.ts";

/** The pinned engine's answer for one choice string. */
export interface Verdict {
  choice: string;
  legal: boolean;
  /** The engine's error message; empty when legal. */
  error: string;
}

/**
 * Asks the pinned engine about choices for one request (ADR-006 rule 4),
 * returning one verdict per choice in order. `CloneJudge` judges before
 * submission where the engine's `Battle` exists (A1 and CI); the
 * SubmissionJudge, the engine's verdict on a submitted choice, comes with
 * the runner (#714).
 */
export interface ChoiceJudge {
  judge(request_hash: string, choices: readonly string[]): Verdict[];
}

/** The judge's battle is not at the request it was asked about. */
export class JudgeStateError extends Error {
  override name = "JudgeStateError";
}

/**
 * Judges choices on a clone of the battle (`Battle.fromJSON(battle.toJSON())`),
 * never on the live battle: a refused choice makes the engine write an
 * `|error|` and can change the request and Pokémon flags. It checks one side
 * with `choose` + `isChoiceDone` only, never `allChoicesDone`, which would
 * set `cantUndo`.
 */
export class CloneJudge implements ChoiceJudge {
  readonly #engine: Engine;
  readonly #battle: SimBattle;
  readonly #side: number;

  constructor(engine: Engine, battle: SimBattle, player: string) {
    const side = battle.sides.findIndex((s) => s.id === player);
    if (side < 0) throw new JudgeStateError(`battle has no side ${player}`);
    // Under strictChoices a refusal throws instead of returning false.
    if (battle.strictChoices === true) {
      throw new JudgeStateError("CloneJudge cannot judge a strictChoices battle");
    }
    this.#engine = engine;
    this.#battle = battle;
    this.#side = side;
  }

  #clone(): SimSide {
    const clone = this.#engine.sim.Battle.fromJSON(this.#battle.toJSON());
    return clone.sides[this.#side] as SimSide;
  }

  /**
   * One clone per request; after a refusal the rest are judged on a fresh
   * clone, because `emitChoiceError` can rewrite the clone's request.
   * Throws `JudgeStateError` unless the battle is at `request_hash`.
   */
  judge(request_hash: string, choices: readonly string[]): Verdict[] {
    const at = this.#clone();
    const actual = requestHash(JSON.parse(JSON.stringify(at.activeRequest)) as JsonObject);
    if (actual !== request_hash) {
      throw new JudgeStateError(`battle is at request ${actual}, not ${request_hash}`);
    }
    let side: SimSide | undefined = at;
    return choices.map((choice) => {
      side ??= this.#clone();
      side.clearChoice();
      const legal = side.choose(choice) && side.isChoiceDone();
      const error = legal ? "" : side.choice.error || "incomplete choice";
      side.clearChoice();
      // A refusal may have updated the clone's request; judge the rest fresh.
      if (!legal) side = undefined;
      return { choice, legal, error };
    });
  }
}

/** The set a policy chooses from (ADR-006). */
export interface LegalChoiceSet {
  request_seq: number;
  request_hash: string;
  showdown_commit: string;
  /** Not pending; engine-accepted wherever a `ChoiceJudge` exists. */
  choices: string[];
  /**
   * Offered without a verdict: their legality depends on hidden
   * information. The engine decides when one is submitted.
   */
  pending: string[];
}

/** The enumerator proposed a choice the engine refuses: a defect, never retried. */
export class EnumeratorDefectError extends Error {
  override name = "EnumeratorDefectError";
  readonly refused: Verdict[];
  constructor(refused: Verdict[]) {
    const shown = refused
      .slice(0, 3)
      .map((v) => `${v.choice} (${v.error})`)
      .join("; ");
    super(`engine refused ${refused.length} enumerated choice(s): ${shown}`);
    this.refused = refused;
  }
}

export interface LegalChoiceOptions {
  request: DecodedRequest & { request_seq: number };
  showdown_commit: string;
  /**
   * `null` only where no engine state exists (A2, the advisor); there the
   * engine judges on submission (ADR-006 rule 6).
   */
  judge: ChoiceJudge | null;
}

/**
 * Enumerate the request's candidates and have the engine judge every
 * non-pending one. Throws `EnumeratorDefectError` if it refuses any.
 * Pending candidates are never judged here, so no verdict on hidden
 * information can reach a policy.
 */
export function legalChoiceSet({
  request,
  showdown_commit,
  judge,
}: LegalChoiceOptions): LegalChoiceSet {
  const { candidates } = enumerateCandidates(request);
  const choices = candidates.filter((c) => !c.pending).map((c) => c.choice);
  const pending = candidates.filter((c) => c.pending).map((c) => c.choice);
  const verdicts = judge?.judge(request.request_hash, choices) ?? [];
  if (
    judge &&
    (verdicts.length !== choices.length || verdicts.some((v, i) => v.choice !== choices[i]))
  ) {
    throw new JudgeStateError("judge did not return one verdict per choice, in order");
  }
  const refused = verdicts.filter((v) => !v.legal);
  if (refused.length > 0) throw new EnumeratorDefectError(refused);
  return {
    request_seq: request.request_seq,
    request_hash: request.request_hash,
    showdown_commit,
    choices,
    pending,
  };
}
