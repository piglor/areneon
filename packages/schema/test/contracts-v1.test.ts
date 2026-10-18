import { describe, expect, it } from "vitest";
import { ajvFor } from "../src/ajv.ts";
import { readSchemas } from "../src/contracts.ts";

const schemas = new Map(readSchemas().map((f) => [f.name, f.schema as Record<string, unknown>]));
const schema = (name: string): Record<string, unknown> => {
  const found = schemas.get(name);
  if (!found) throw new Error(`no contract schema ${name}`);
  return found;
};
const validator = (name: string) => ajvFor("contract").compile(schema(name));
const properties = (name: string) => schema(name).properties as Record<string, unknown>;

const HASH = "ab".repeat(32);

describe("RequestIdentity", () => {
  const valid = validator("request-identity");

  it("needs request_seq and request_hash; rqid is optional", () => {
    expect(valid({ request_seq: 1, request_hash: HASH })).toBe(true);
    expect(valid({ request_seq: 1, request_hash: HASH, rqid: 3 })).toBe(true);
    expect(valid({ request_hash: HASH })).toBe(false);
  });

  it.each([
    ["request_seq below 1", { request_seq: 0, request_hash: HASH }],
    ["an uppercase hash", { request_seq: 1, request_hash: HASH.toUpperCase() }],
    ["rqid 0", { request_seq: 1, request_hash: HASH, rqid: 0 }],
    ["an unknown field", { request_seq: 1, request_hash: HASH, turn: 2 }],
  ])("refuses %s", (_label, value) => {
    expect(valid(value)).toBe(false);
  });
});

describe("LegalChoiceSet", () => {
  it("defines its identity fields exactly as RequestIdentity does", () => {
    // Contract schemas allow only local $refs, so the fields are repeated;
    // this keeps the two copies one definition.
    for (const field of ["request_seq", "request_hash", "rqid"]) {
      expect(properties("legal-choice-set")[field], field).toEqual(properties("request-identity")[field]);
    }
    // And with the same optionality: required here exactly where RequestIdentity requires it.
    const required = (name: string) => schema(name).required as string[];
    const identityFields = Object.keys(properties("request-identity"));
    expect(required("legal-choice-set").filter((f) => identityFields.includes(f))).toEqual(required("request-identity"));
  });

  it("is a persisted root: schema_version 1, closed", () => {
    const valid = validator("legal-choice-set");
    const set = {
      schema_version: 1,
      request_seq: 2,
      request_hash: HASH,
      showdown_commit: "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b",
      choices: ["move 1 1, move 2 2"],
      pending: [],
    };
    expect(valid(set)).toBe(true);
    expect(valid({ ...set, schema_version: 2 })).toBe(false);
    expect(valid({ ...set, verdicts: [] })).toBe(false);
    expect(valid({ ...set, choices: [""] })).toBe(false);
  });
});

describe("PreviewObservation and BattleObservation", () => {
  it("differ only in identity and kind, so one closure walk serves both", () => {
    const strip = (name: string) => {
      const { $id, title, description, properties, ...rest } = schema(name);
      const { kind, ...others } = properties as Record<string, unknown>;
      return { rest, others, kind };
    };
    const preview = strip("preview-observation");
    const battle = strip("battle-observation");
    expect(battle.rest).toEqual(preview.rest);
    expect(battle.others).toEqual(preview.others);
    expect([preview.kind, battle.kind]).toEqual([{ const: "preview" }, { const: "battle" }]);
  });
});

describe("Recommendation", () => {
  const valid = validator("recommendation");
  const rec = {
    schema_version: 1,
    identity: { request_seq: 2, request_hash: HASH },
    choice: "move 1 1, move 2 2",
    source_policy_id: "first-legal",
    elapsed_ms: 40,
    deadline_outcome: "in_budget",
  };

  it("needs a known deadline outcome and a policy", () => {
    expect(valid(rec)).toBe(true);
    expect(valid({ ...rec, deadline_outcome: "late" })).toBe(false);
    expect(valid({ ...rec, source_policy_id: "" })).toBe(false);
    expect(valid({ ...rec, confidence: 0.9 })).toBe(false);
  });
});

describe("Episode decision deadline", () => {
  // The Deadline definition on its own, with the Episode's $defs for its refs.
  const { $defs } = schema("episode");
  const valid = ajvFor("contract").compile({ $defs, $ref: "#/$defs/Deadline" });
  const record = {
    deadline_outcome: "fallback_timeout",
    elapsed_ms: 20100,
    submitted_after_ms: 20150,
    bank_seconds_after: null,
    server_auto_choose_detected: false,
  };

  it("keeps bank_seconds_after required but nullable, and auto-choose a plain boolean", () => {
    expect(valid(record)).toBe(true);
    expect(valid({ ...record, bank_seconds_after: 400 })).toBe(true);
    expect(valid({ ...record, server_auto_choose_detected: true })).toBe(true);
    const { bank_seconds_after, ...missing } = record;
    expect(valid(missing)).toBe(false);
    expect(valid({ ...record, server_auto_choose_detected: "no" })).toBe(false);
  });

  it("is carried by a decision record, and only in its own shape", () => {
    const decision = ajvFor("contract").compile({ $defs, $ref: "#/$defs/DecisionRecord" });
    const base = { identity: { request_seq: 2, request_hash: HASH }, choice: "move 1 1, move 1 1", source_policy_id: "p" };
    expect(decision(base)).toBe(true);
    expect(decision({ ...base, deadline: record })).toBe(true);
    expect(decision({ ...base, deadline: { ...record, late: true } })).toBe(false);
  });
});

describe("DecisionContext", () => {
  const valid = validator("decision-context");
  const preview = {
    timer_source: "server",
    timer_rules: { starting: 420, grace: 90, add_per_turn: 0, max_per_turn: 55, max_first_turn: 90, tick: 5 },
    turn_seconds_left: 90,
    bank_seconds_left: 420,
    grace_seconds_left: 90,
    safety_margin_ms: 3000,
    submit_within_ms: 60000,
    policy_budget_ms: 59000,
  };

  it("accepts a preview context without an estimate, and a battle context with one", () => {
    expect(valid(preview)).toBe(true);
    expect(valid({ ...preview, grace_seconds_left: 0, remaining_decisions_estimate: 12 })).toBe(true);
  });

  it.each([
    ["a wall-clock timestamp", { ...preview, received_at: 1792335591 }],
    ["an unknown timer source", { ...preview, timer_source: "client" }],
    ["a negative budget", { ...preview, policy_budget_ms: -1 }],
    ["an estimate of 0", { ...preview, remaining_decisions_estimate: 0 }],
    ["incomplete timer rules", { ...preview, timer_rules: { starting: 420 } }],
  ])("refuses %s", (_label, value) => {
    expect(valid(value)).toBe(false);
  });
});
