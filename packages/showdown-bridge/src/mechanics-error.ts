// The probe's refusal to answer at all (ADR-016 figure 4): a query the
// contract schema accepts but the probe cannot turn into a scenario battle or
// read exactly, with no `unsupported` reason to say so. No result is made.

/** The code of a refusal of a declared history fact (an id the engine lacks, a start handler that refused, a volatile that had run out): the caller may answer `history_unknown`. */
export const DECLARED_FACT = "declared_fact";

export class MechanicsProbeError extends Error {
  override name = "MechanicsProbeError";
  /** A closed word for the kind of refusal a caller may act on (`DECLARED_FACT`), or null. */
  readonly code: string | null;
  constructor(message: string, code: string | null = null) {
    super(message);
    this.code = code;
  }
}
