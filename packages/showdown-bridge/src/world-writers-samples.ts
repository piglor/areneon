/* eslint-disable @typescript-eslint/no-explicit-any */
// The WorldWriters' share of the sample corpus (ADR-008 section 2, ADR-026 Evidence 1): the committed record,
// one with admitted ids, schema-invalid mutations and reader-invalid ones that pass the schema and that
// `readWorldWriters` refuses against the pinned engine.

import type { Engine } from "./engine.ts";
import { loadWorldWriters } from "./world-writers.ts";

type Cases = Record<string, unknown>;

function mutated(value: unknown, change: (copy: any) => void): Record<string, unknown> {
  const copy = structuredClone(value) as Record<string, unknown>;
  change(copy);
  return copy;
}

const row = (copy: any, id: string): any => copy.items.find((item: any) => item.id === id);

export function worldWritersSamples(engine: Engine, formatId: string): { valid: Cases; invalid: Cases; readerInvalid: Cases } {
  const base = loadWorldWriters(engine, formatId);
  const change = (fn: (copy: any) => void) => mutated(base, fn);
  return {
    valid: {
      "world-writers/v1": base,
      "world-writers/admitted": change((c) => {
        c.admitted = [
          { kind: "start_trigger", id: "whiteherb", class: "W" },
          { kind: "volatile", id: "encore", class: "W" },
          { kind: "volatile", id: "taunt", class: "W" },
        ];
      }),
    },
    invalid: {
      "world-writers/extra-field": change((c) => (c.note = "x")),
      "world-writers/unknown-item": change((c) => (row(c, "turn_number").id = "turn_numbers")),
      "world-writers/unknown-class": change((c) => (row(c, "turn_number").class = "Z")),
      "world-writers/fractional-version": change((c) => (c.schema_version = 1.5)),
      "world-writers/missing-admitted": change((c) => delete c.admitted),
      "world-writers/unknown-admitted-kind": change((c) => (c.admitted = [{ kind: "move", id: "taunt", class: "W" }])),
      "world-writers/next-version": change((c) => (c.schema_version = 2)),
    },
    readerInvalid: {
      "world-writers/missing-item": change((c) => c.items.splice(3, 1)),
      "world-writers/unsorted-items": change((c) => c.items.reverse()),
      "world-writers/class-not-offered": change((c) => (row(c, "turn_number").class = "X")),
      "world-writers/condition-on-wrong-item": change((c) => (row(c, "turn_number").condition = "stall")),
      "world-writers/unknown-condition": change((c) => (row(c, "protect_counter").condition = "notacondition")),
      "world-writers/unknown-engine-id": change((c) => (row(c, "substitute").ids = ["notanid"])),
      "world-writers/unsorted-ids": change((c) => (row(c, "locked_moves").ids = ["rollout", "outrage"])),
      "world-writers/id-under-two-rows": change((c) => (row(c, "substitute").ids = ["wish"])),
      "world-writers/bad-line-kind": change((c) => (row(c, "unavailable_mechanics").ids = ["terastallize"])),
      "world-writers/duplicate-admitted": change((c) => (c.admitted = [{ kind: "volatile", id: "taunt", class: "W" }, { kind: "volatile", id: "taunt", class: "E" }])),
      "world-writers/unsorted-admitted": change((c) => (c.admitted = [{ kind: "volatile", id: "taunt", class: "W" }, { kind: "volatile", id: "encore", class: "W" }])),
      "world-writers/admitted-class": change((c) => (c.admitted = [{ kind: "start_trigger", id: "whiteherb", class: "E" }])),
      "world-writers/admitted-unknown-id": change((c) => (c.admitted = [{ kind: "volatile", id: "notacondition", class: "E" }])),
      "world-writers/admitted-refusal-id": change((c) => (c.admitted = [{ kind: "volatile", id: "wish", class: "E" }])),
      "world-writers/admitted-confusion": change((c) => (c.admitted = [{ kind: "volatile", id: "confusion", class: "E" }])),
    },
  };
}
