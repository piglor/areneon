/* eslint-disable @typescript-eslint/no-explicit-any */
// The PreviewSolveConfig's share of the sample corpus (ADR-008 section 2, ADR-029 section 4): the committed record, a
// variant, schema-invalid mutations and reader-invalid ones that pass the schema and that `readPreviewSolveConfig` refuses.

import { loadPreviewSolveConfig } from "./preview-solve-config.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

export function previewSolveConfigSamples(formatId: string): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base = loadPreviewSolveConfig(formatId);
  const change = (fn: (copy: any) => void) => mutated(base, fn);
  return {
    valid: {
      "preview-solve-config/preview-solve-1": base,
      "preview-solve-config/rollout-plus": change((c) => {
        c.leaf = { kind: "rollout", turns: 2 };
        c.solver = { kind: "regret-matching-plus", iterations: 1024 };
        c.selection_rule = "prior-top-k";
        c.worlds.prior = { kind: "role-sets", label: "belief from the role-set table" };
      }),
      "preview-solve-config/exact-support": change((c) => (c.solver = { kind: "exact-support" })),
    },
    invalid: {
      "preview-solve-config/extra-field": change((c) => (c.note = "x")),
      "preview-solve-config/fractional-k-own": change((c) => (c.k_own = 2.5)),
      "preview-solve-config/zero-k-opp": change((c) => (c.k_opp = 0)),
      "preview-solve-config/unknown-algorithm": change((c) => (c.algorithm = { kind: "preview-solve-2" })),
      "preview-solve-config/unknown-leaf": change((c) => (c.leaf = { kind: "value-head" })),
      "preview-solve-config/rollout-without-turns": change((c) => (c.leaf = { kind: "rollout" })),
      "preview-solve-config/unknown-solver": change((c) => (c.solver = { kind: "fictitious-play", iterations: 10 })),
      "preview-solve-config/unknown-rule": change((c) => (c.selection_rule = "top-k")),
      "preview-solve-config/bad-seed-derivation": change((c) => (c.seed_derivation = "preview-solve/2")),
      "preview-solve-config/short-belief-hash": change((c) => (c.belief_identity_hash = "abc")),
      "preview-solve-config/missing-worlds": change((c) => delete c.worlds),
      "preview-solve-config/next-version": change((c) => (c.schema_version = 2)),
    },
    readerInvalid: {
      "preview-solve-config/min-valid-over-n": change((c) => (c.worlds.min_valid = c.worlds.n + 1)),
      "preview-solve-config/iterative-without-iterations": change((c) => delete c.solver.iterations),
      "preview-solve-config/exact-support-with-iterations": change((c) => (c.solver = { kind: "exact-support", iterations: 10 })),
    },
  };
}
