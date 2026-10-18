import { describe, expect, it } from "vitest";
import { checkProfile, MAX_SAFE } from "../src/profile.ts";

const DRAFT = "https://json-schema.org/draft/2020-12/schema";

/** A conforming contract-profile document exercising every allowed construct. */
function valid(): Record<string, unknown> {
  return {
    $schema: DRAFT,
    $id: "urn:areneon:schema:contracts:v1:sample",
    title: "Sample",
    description: "Every allowed keyword once.",
    type: "object",
    additionalProperties: false,
    required: ["schema_version", "seq", "hash", "observation"],
    properties: {
      schema_version: { const: 1 },
      seq: { type: "integer", minimum: 1, maximum: MAX_SAFE },
      bank: { type: ["integer", "null"], minimum: 0, maximum: 420 },
      hash: { $ref: "#/$defs/Sha256Hex", description: "A digest." },
      side: { enum: ["p1", "p2"] },
      names: { type: "array", items: { type: "string", minLength: 1, maxLength: 18 }, minItems: 1, maxItems: 6 },
      observation: { $ref: "#/$defs/Observation" },
    },
    $defs: {
      Sha256Hex: { type: "string", pattern: "^[0-9a-f]{64}$" },
      Observation: {
        type: "object",
        oneOf: [{ $ref: "#/$defs/Preview" }, { $ref: "#/$defs/Battle" }],
        discriminator: { propertyName: "kind" },
      },
      Preview: {
        type: "object",
        additionalProperties: false,
        required: ["kind"],
        properties: { kind: { const: "preview" } },
      },
      Battle: {
        type: "object",
        additionalProperties: false,
        required: ["kind", "turn"],
        properties: { kind: { const: "battle" }, turn: { type: "integer", minimum: 1, maximum: 1000 } },
      },
    },
  };
}

function messages(schema: unknown, profile: "contract" | "engine-mirror" = "contract"): string[] {
  return checkProfile(schema, profile).map((v) => `${v.at} ${v.message}`);
}

/** `valid()` with one change applied at `path`. */
function mutate(path: string[], value: unknown): Record<string, unknown> {
  const doc = valid();
  let node: Record<string, unknown> = doc;
  for (const key of path.slice(0, -1)) node = node[key] as Record<string, unknown>;
  const last = path.at(-1) as string;
  if (value === undefined) Reflect.deleteProperty(node, last);
  else node[last] = value;
  return doc;
}

describe("checkProfile: contract profile", () => {
  it("accepts a document that uses every allowed construct", () => {
    expect(messages(valid())).toEqual([]);
  });

  it("refuses a non-object document and a wrong $schema", () => {
    expect(messages([])).toEqual([" a schema document must be an object"]);
    expect(messages(mutate(["$schema"], "http://json-schema.org/draft-07/schema#"))).toEqual([
      ` $schema must be ${DRAFT}`,
    ]);
  });

  it.each([
    ["anyOf", { anyOf: [{ type: "string" }] }],
    ["allOf", { allOf: [{ type: "string" }] }],
    ["not", { not: { type: "string" } }],
    ["if", { if: { type: "string" } }],
    ["patternProperties", { type: "object", additionalProperties: false, patternProperties: {} }],
    ["prefixItems", { type: "array", items: { type: "string" }, prefixItems: [] }],
    ["unevaluatedProperties", { type: "object", additionalProperties: false, unevaluatedProperties: false }],
    ["$dynamicRef", { $dynamicRef: "#meta" }],
    ["format", { type: "string", format: "date-time" }],
    ["default", { type: "string", default: "x" }],
  ])("refuses the banned keyword %s", (keyword, sub) => {
    const found = messages(mutate(["properties", "extra"], sub));
    expect(found).toContain(`/properties/extra/${keyword} keyword ${keyword} is not in the contract profile`);
  });

  it("refuses root-only keywords below the root", () => {
    const found = messages(mutate(["properties", "extra"], { $id: "x", type: "string" }));
    expect(found).toEqual(["/properties/extra/$id $id is allowed only at the root"]);
  });

  it("requires additionalProperties: false on every object", () => {
    expect(messages(mutate(["additionalProperties"], undefined))).toEqual([
      " an object must set additionalProperties: false",
    ]);
    expect(messages(mutate(["$defs", "Preview", "additionalProperties"], true))).toEqual([
      "/$defs/Preview an object must set additionalProperties: false",
    ]);
  });

  it("requires every required name to be a property, and required to be a list of names", () => {
    expect(messages(mutate(["required"], ["schema_version", "ghost"]))).toEqual([
      "/required required names ghost, which is not a property",
    ]);
    expect(messages(mutate(["required"], "seq"))).toEqual(["/required required must be an array of property names"]);
    expect(messages(mutate(["properties"], []))).toContain("/properties properties must be an object");
  });

  it("requires both bounds on every integer, within ±(2^53−1)", () => {
    expect(messages(mutate(["properties", "seq"], { type: "integer", minimum: 1 }))).toEqual([
      "/properties/seq an integer must declare maximum",
    ]);
    expect(messages(mutate(["properties", "seq"], { type: "integer", minimum: 1, maximum: 2 ** 53 }))).toEqual([
      "/properties/seq/maximum maximum must be an integer within ±(2^53−1)",
    ]);
    expect(messages(mutate(["properties", "seq"], { type: "integer", minimum: 5, maximum: 4 }))).toEqual([
      "/properties/seq minimum must not exceed maximum",
    ]);
    expect(messages(mutate(["properties", "bank"], { type: ["integer", "null"], maximum: 1 }))).toEqual([
      "/properties/bank an integer must declare minimum",
    ]);
  });

  it("bans non-integer and out-of-range numbers in const and enum", () => {
    expect(messages(mutate(["properties", "side"], { const: 0.5 }))).toEqual([
      "/properties/side/const a numeric value must be an integer within ±(2^53−1) (number is banned)",
    ]);
    expect(messages(mutate(["properties", "side"], { enum: [1, 2 ** 60] }))).toEqual([
      "/properties/side/enum a numeric value must be an integer within ±(2^53−1) (number is banned)",
    ]);
    expect(messages(mutate(["properties", "side"], { enum: [1, 2] }))).toEqual([]);
  });

  it("bans number and unknown type names", () => {
    expect(messages(mutate(["properties", "seq"], { type: "number" }))).toEqual([
      "/properties/seq/type type must name only string, integer, boolean, null, object, array (number is banned)",
    ]);
    expect(messages(mutate(["properties", "seq"], { type: [] }))).toHaveLength(1);
  });

  it('allows a type array only as [T, "null"]', () => {
    expect(messages(mutate(["properties", "side"], { type: ["string", "boolean"] }))).toEqual([
      '/properties/side/type a type array must be [T, "null"] in the contract profile',
    ]);
    expect(messages(mutate(["properties", "side"], { type: ["null", "string"] }))).toHaveLength(1);
  });

  it("requires local $refs that resolve, without siblings", () => {
    expect(messages(mutate(["properties", "hash"], { $ref: "other.schema.json#/x" }))).toEqual([
      '/properties/hash/$ref $ref must point to a local #/$defs entry, got "other.schema.json#/x"',
    ]);
    expect(messages(mutate(["properties", "hash"], { $ref: "#/$defs/Missing" }))).toHaveLength(1);
    expect(messages(mutate(["properties", "hash"], { $ref: "#/$defs/Sha256Hex", minLength: 1 }))).toEqual([
      "/properties/hash $ref takes no sibling keywords (found minLength)",
    ]);
  });

  it("requires arrays to declare items", () => {
    expect(messages(mutate(["properties", "names"], { type: "array", maxItems: 2 }))).toEqual([
      "/properties/names an array must declare items",
    ]);
  });

  it("requires a non-empty enum of unique values", () => {
    expect(messages(mutate(["properties", "side"], { enum: [] }))).toEqual([
      "/properties/side/enum enum must be a non-empty array",
    ]);
    expect(messages(mutate(["properties", "side"], { enum: ["p1", "p1"] }))).toEqual([
      "/properties/side/enum enum values must be unique",
    ]);
  });

  it("refuses $defs and subschemas that are not objects", () => {
    expect(messages(mutate(["$defs"], [true]))).toContain("/$defs $defs must be an object");
    expect(messages(mutate(["properties", "side"], true))).toEqual(["/properties/side a subschema must be an object"]);
  });

  describe("discriminated unions", () => {
    it("requires type object and a kind discriminator", () => {
      expect(messages(mutate(["$defs", "Observation", "type"], undefined))).toEqual([
        '/$defs/Observation a discriminated union must declare "type": "object"',
      ]);
      expect(messages(mutate(["$defs", "Observation", "discriminator"], { propertyName: "type" }))).toEqual([
        '/$defs/Observation/discriminator a union needs discriminator: {"propertyName": "kind"}',
      ]);
    });

    it("requires every branch to be a lone $ref to a kind-tagged object", () => {
      const inline = mutate(["$defs", "Observation", "oneOf"], [{ type: "string" }]);
      expect(messages(inline)).toContain("/$defs/Observation/oneOf/0 every oneOf branch must be a lone $ref");
      const untagged = mutate(["$defs", "Preview", "properties", "kind"], { type: "string" });
      expect(messages(untagged)).toContain(
        '/$defs/Observation/oneOf/0 the branch must require a "kind" property that is a lone string const',
      );
      const doubled = mutate(["$defs", "Preview", "properties", "kind"], { const: "preview", enum: ["preview"] });
      expect(messages(doubled)).toEqual([
        '/$defs/Observation/oneOf/0 the branch must require a "kind" property that is a lone string const',
      ]);
      const unresolved = mutate(["$defs", "Observation", "oneOf"], [{ $ref: "#/$defs/Nope" }]);
      expect(messages(unresolved)).toEqual([
        '/$defs/Observation/oneOf/0/$ref $ref must point to a local #/$defs entry, got "#/$defs/Nope"',
      ]);
      expect(messages(mutate(["$defs", "Observation", "oneOf"], []))).toEqual([
        "/$defs/Observation/oneOf oneOf must be a non-empty array",
      ]);
    });

    it("refuses anything on a union node beyond type, oneOf and discriminator", () => {
      // Without this, the node's own keywords would escape every other check.
      const doc = mutate(["$defs", "Observation", "properties"], {
        kind: { anyOf: [{ type: "string" }] },
        extra: { type: "number", default: 1 },
      });
      ((doc.$defs as Record<string, unknown>).Observation as Record<string, unknown>).additionalProperties = true;
      expect(messages(doc)).toEqual([
        "/$defs/Observation a union node takes only type, oneOf, discriminator, title and description (found properties, additionalProperties)",
      ]);
    });

    it("allows a union at the root, next to the root-only keywords", () => {
      const doc: Record<string, unknown> = {
        ...(valid().$defs as Record<string, Record<string, unknown>>).Observation,
        $schema: DRAFT,
        $id: "urn:areneon:schema:contracts:v1:observation",
        title: "Observation",
        $defs: valid().$defs,
      };
      expect(messages(doc)).toEqual([]);
    });

    it("refuses duplicate tags", () => {
      const duplicate = mutate(["$defs", "Battle", "properties", "kind"], { const: "preview" });
      expect(messages(duplicate)).toEqual(['/$defs/Observation/oneOf/1 duplicate kind tag "preview"']);
    });
  });
});

describe("checkProfile: engine-mirror profile", () => {
  it("additionally allows a union of scalar types", () => {
    const disabled = mutate(["properties", "side"], { type: ["string", "boolean"] });
    expect(messages(disabled, "engine-mirror")).toEqual([]);
  });

  it("still refuses object or array members in a type union", () => {
    const mixed = mutate(["properties", "side"], { type: ["string", "array"], items: { type: "string" } });
    expect(messages(mixed, "engine-mirror")).toEqual([
      '/properties/side/type a type array must be [T, "null"] or scalar types only in the engine-mirror profile',
    ]);
  });

  it("keeps every other contract rule, including closed objects and bounded integers", () => {
    expect(messages(mutate(["additionalProperties"], undefined), "engine-mirror")).toEqual([
      " an object must set additionalProperties: false",
    ]);
    expect(messages(mutate(["properties", "seq"], { type: "integer" }), "engine-mirror")).toHaveLength(2);
  });
});
