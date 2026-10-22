// The probe's refusal to answer at all (ADR-016 figure 4): a query the
// contract schema accepts but the probe cannot turn into a scenario battle or
// read exactly, with no `unsupported` reason to say so. No result is made.

export class MechanicsProbeError extends Error {
  override name = "MechanicsProbeError";
}
