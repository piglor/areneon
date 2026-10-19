import { EnumeratorDefectError, type LegalChoiceSet } from "@areneon/showdown-bridge";

/**
 * Where the bot plays. `harness` is every test and harness run: an
 * enumerator defect fails it. `live_match` is a live human match on A2,
 * the only place ADR-005 §1 lets the bot carry on after a defect.
 */
export type RefusalMode = "harness" | "live_match";

/** The bot's own request state when an `|error|` arrives. */
export interface CurrentRequest {
  request_seq: number;
  /** The server's id for it; the bot sends it with every choice. */
  rqid?: number;
  /** The set the bot chose from; `null` for a `wait` request, which asks for nothing. */
  legal: LegalChoiceSet | null;
}

/** A choice the bot sent, and the request it answered. */
export interface Submission {
  request_seq: number;
  choice: string;
}

export interface RefusalInput {
  /** The `|error|` text, without the `|error|` prefix. */
  error: string;
  submission: Submission;
  /** `null` when no request is open (none yet, or withdrawn by `|request|null`). */
  current: CurrentRequest | null;
  mode: RefusalMode;
  /** Live match only: choices for the current request the engine already refused as defects. */
  refused?: readonly string[];
}

export type Refusal =
  /**
   * `[Unavailable choice]` of a pending candidate: the engine revealed
   * hidden state, and the updated request that follows is a new request
   * (ADR-005 §1, ADR-006 rule 5).
   */
  | { kind: "unavailable"; choice: string }
  /**
   * Not a defect, dropped and logged (ADR-005 §1): the choice answered a
   * request that is no longer current (`not_current`), no request was open
   * (`no_request`), or the server or engine itself said so (`refused_as_stale`).
   */
  | { kind: "stale_choice"; choice: string; reason: "not_current" | "no_request" | "refused_as_stale" }
  /**
   * Live match only: the engine refused a current, non-pending candidate.
   * The incident is `error`; the match is marked `defect: true` and the bot
   * sends `next` (ADR-005 §1).
   */
  | { kind: "defect"; error: EnumeratorDefectError; defect: true; next: { choice: string; rqid?: number } };

/** A refusal no rule classifies. It is never retried. */
export class RefusalError extends Error {
  override name = "RefusalError";
}

/**
 * The refusals that say the choice met no open request, so it was never
 * judged as a choice. The server layer sends the first two without asking
 * the sim (server/room-battle.ts `choose`); the sim sends the third when the
 * side has no request (sim/side.ts `choose`), and the server forwards it.
 */
const STALE_REFUSALS: readonly string[] = [
  "[Invalid choice] There's nothing to choose",
  "[Invalid choice] Sorry, too late to make a different move",
  "[Invalid choice] Can't do anything: ",
];

/**
 * Classifies an `|error|` the bot received after a submission (ADR-005 §1,
 * ADR-006 rules 5 and 6), by the bot's own request state first:
 *
 * - a choice for a request that is no longer current, or with no request
 *   open, is a `stale_choice`; so is a server-layer refusal;
 * - `[Unavailable choice]` of a pending candidate is `unavailable`;
 * - a refusal of a current, non-pending candidate is an enumerator defect:
 *   it throws `EnumeratorDefectError` in a harness. In a live match it
 *   returns the next non-pending candidate in canonical order not yet
 *   refused, and finally `default`, each with the current `rqid`.
 *
 * Anything else throws `RefusalError`: a text that is not a choice refusal,
 * a refused pending candidate that revealed nothing, or a refused choice
 * outside the candidate set (`default` included).
 */
export function classifyRefusal(input: RefusalInput): Refusal {
  const { error, submission, current, mode, refused = [] } = input;
  const { choice } = submission;
  if (!error.startsWith("[Invalid choice]") && !error.startsWith("[Unavailable choice]")) {
    throw new RefusalError(`not a choice refusal: ${error}`);
  }
  if (current === null || current.legal === null) return { kind: "stale_choice", choice, reason: "no_request" };
  if (submission.request_seq !== current.request_seq) return { kind: "stale_choice", choice, reason: "not_current" };
  if (STALE_REFUSALS.some((text) => error.startsWith(text))) {
    return { kind: "stale_choice", choice, reason: "refused_as_stale" };
  }
  const legal = current.legal;
  if (legal.request_seq !== current.request_seq) {
    throw new RefusalError(`legal set is for request ${legal.request_seq}, not ${current.request_seq}`);
  }
  if (error.startsWith("[Unavailable choice]") && legal.pending.includes(choice)) {
    return { kind: "unavailable", choice };
  }
  const index = legal.choices.indexOf(choice);
  if (index < 0) throw new RefusalError(`refused ${choice}, which is not a non-pending candidate: ${error}`);
  const defect = new EnumeratorDefectError([{ choice, legal: false, error }]);
  if (mode === "harness") throw defect;
  const tried = new Set([...refused, choice]);
  const nextChoice = legal.choices.slice(index + 1).find((c) => !tried.has(c)) ?? "default";
  const next = current.rqid === undefined ? { choice: nextChoice } : { choice: nextChoice, rqid: current.rqid };
  return { kind: "defect", error: defect, defect: true, next };
}
