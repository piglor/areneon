import { enumerateCandidates } from "./choices.ts";
import type { Engine, SimAction, SimBattle, SimSide } from "./engine.ts";
import type { DecodedRequest, JsonObject } from "./request.ts";
import { requestHash } from "./request-identity.ts";
import type { LegalChoiceSet } from "@areneon/schema/types/v1/legal-choice-set.ts";

/** The set a policy chooses from (ADR-006): generated from its contract schema (ADR-008). */
export type { LegalChoiceSet };

/** The pinned engine's answer for one choice string. */
export interface Verdict {
  choice: string;
  legal: boolean;
  /** The engine's error message; empty when legal. */
  error: string;
  /**
   * `CloneJudge` only, when legal: the engine's parsed action, normalised,
   * so two spellings of one action compare equal (the corpus oracle).
   */
  action?: string;
}

/**
 * A canonical key for the engine's parsed actions of one choice: the
 * fields that make two actions differ, with Pokémon as 1-based slots.
 */
function actionKey(side: SimSide, actions: SimAction[]): string {
  const slot = (mon: object | undefined): number => side.pokemon.indexOf(mon as object) + 1;
  return JSON.stringify(
    actions.map((a) => [
      a.choice,
      slot(a.pokemon),
      slot(a.target),
      a.index,
      a.moveid,
      a.targetLoc,
      a.mega,
      a.megax,
      a.megay,
      a.terastallize,
    ]),
  );
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

/**
 * A judge does not match the request it was asked about: its battle is at
 * another request (or none), or its verdicts do not answer the choices.
 */
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

  #live(): SimSide {
    return this.#battle.sides[this.#side] as SimSide;
  }

  #clone(): SimSide {
    const clone = this.#engine.sim.Battle.fromJSON(this.#battle.toJSON());
    const side = clone.sides[this.#side] as SimSide;
    // fromJSON rebuilds `activeRequest` from the Pokémon (sim/state.ts).
    // Unless the engine edited the live request in place, which it marks
    // `update: true` (an `[Unavailable choice]`: `trapped` set, `maybe*`
    // deleted), the rebuilt one must equal it: that is the check that the
    // clone is the battle the player sees.
    const live = JSON.parse(JSON.stringify(this.#live().activeRequest)) as JsonObject;
    const rebuilt = JSON.parse(JSON.stringify(side.activeRequest)) as JsonObject;
    if (live.update !== true && requestHash(rebuilt) !== requestHash(live)) {
      throw new JudgeStateError("the clone does not reproduce the live battle's request");
    }
    // Judge against the request a submission will be checked against.
    side.activeRequest = live;
    return side;
  }

  /**
   * One clone per request. A refusal that reveals hidden information
   * (`[Unavailable choice]`) rewrites the clone's request and flags, so the
   * rest are then judged on a fresh clone. Comparing the request JSON is
   * enough to see that: in the pinned engine a refusal changes state only
   * through `emitChoiceError` -> `updateRequestForPokemon`, which edits the
   * request in place (deleting `maybe*`, marking a move disabled, setting
   * `trapped`) and re-emits it with `update: true`; the Pokémon flags it
   * clears mirror those request keys (sim/side.ts, sim/pokemon.ts). A test
   * checks the batched verdicts against one fresh clone per choice.
   * Throws `JudgeStateError` unless the battle is at `request_hash`.
   */
  judge(request_hash: string, choices: readonly string[]): Verdict[] {
    // Checked on the live side: a battle that has not started cannot be cloned.
    if (this.#live().activeRequest === null) {
      throw new JudgeStateError("side has no active request");
    }
    const at = this.#clone();
    const actual = requestHash(JSON.parse(JSON.stringify(at.activeRequest)) as JsonObject);
    if (actual !== request_hash) {
      throw new JudgeStateError(`battle is at request ${actual}, not ${request_hash}`);
    }
    let side: SimSide | undefined = at;
    let request = JSON.stringify(at.activeRequest);
    return choices.map((choice) => {
      if (side === undefined) {
        side = this.#clone();
        request = JSON.stringify(side.activeRequest);
      }
      side.clearChoice();
      const legal = side.choose(choice) && side.isChoiceDone();
      const error = legal ? "" : side.choice.error || "incomplete choice";
      const verdict: Verdict = { choice, legal, error };
      if (legal) verdict.action = actionKey(side, side.choice.actions);
      side.clearChoice();
      if (!legal && JSON.stringify(side.activeRequest) !== request) side = undefined;
      return verdict;
    });
  }
}

/**
 * The SubmissionJudge (ADR-006 rule 4): the engine's own verdict on a
 * submitted choice, on any runtime. It answers only on refusal, with an
 * `|error|` line; silence means the engine took the choice.
 */
export interface SubmissionVerdict extends Verdict {
  legal: false;
  /**
   * `[Unavailable choice]`: refused because the attempt revealed hidden
   * information. The engine follows it with an updated `|request|`, a new
   * request to answer. `[Invalid choice]` is not revealing: an enumerated
   * non-pending candidate refused that way is an enumerator defect.
   */
  revealed: boolean;
}

/** Classify the text of an `|error|` line the engine sent after `choice`. */
export function judgeSubmission(choice: string, error: string): SubmissionVerdict {
  return { choice, legal: false, error, revealed: error.startsWith("[Unavailable choice]") };
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
   * `null` only where no engine state exists (A2); there the engine judges
   * on submission (ADR-006 rule 6).
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
    schema_version: 1,
    request_seq: request.request_seq,
    request_hash: request.request_hash,
    ...(request.rqid === undefined ? {} : { rqid: request.rqid }),
    showdown_commit,
    choices,
    pending,
  };
}
