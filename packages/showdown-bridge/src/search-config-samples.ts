/* eslint-disable @typescript-eslint/no-explicit-any */
// The SearchConfig's share of the sample corpus (ADR-008 section 2, ADR-026 section 5): the committed record, a
// variant, schema-invalid mutations and reader-invalid ones that pass the schema and that `readSearchConfig` refuses.

import { loadSearchConfig } from "./search-config.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

export function searchConfigSamples(formatId: string): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base = loadSearchConfig(formatId);
  const change = (fn: (copy: any) => void) => mutated(base, fn);
  return {
    valid: {
      "search-config/rerank-1": base,
      "search-config/uniform-replies-hp-leaf": change((c) => {
        c.replies = { max_damage_ppm: 0, uniform_draws: 4 };
        c.leaf = { kind: "hp-ko-differential" };
        c.worlds.prior = { kind: "role-sets", label: "belief from the role-set table" };
      }),
    },
    invalid: {
      "search-config/extra-field": change((c) => (c.note = "x")),
      "search-config/fractional-top-k": change((c) => (c.top_k = 4.5)),
      "search-config/zero-top-k": change((c) => (c.top_k = 0)),
      "search-config/unknown-algorithm": change((c) => (c.algorithm = { kind: "rerank-2" })),
      "search-config/unknown-leaf": change((c) => (c.leaf = { kind: "rollout" })),
      "search-config/unknown-hp-rounding": change((c) => (c.worlds.hp_rounding = "round")),
      "search-config/bad-seed-derivation": change((c) => (c.seed_derivation = "search/2")),
      "search-config/short-belief-hash": change((c) => (c.belief_identity_hash = "abc")),
      "search-config/missing-replies": change((c) => delete c.replies),
      "search-config/next-version": change((c) => (c.schema_version = 2)),
    },
    readerInvalid: {
      "search-config/min-worlds-over-n": change((c) => (c.min_valid_worlds = c.worlds.n + 1)),
      "search-config/inexact-uniform-share": change((c) => (c.replies = { max_damage_ppm: 500000, uniform_draws: 3 })),
      "search-config/no-uniform-without-full-weight": change((c) => (c.replies = { max_damage_ppm: 500000, uniform_draws: 0 })),
    },
  };
}
