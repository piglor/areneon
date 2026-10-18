// The two schema profiles of ADR-008 §1. A profile is a whitelist: every
// keyword it does not name is refused, so a banned construct (anyOf, format,
// default, patternProperties, …) fails without being listed one by one.

export type Profile = "contract" | "engine-mirror";

export interface ProfileViolation {
  /** JSON Pointer to the offending subschema. */
  readonly at: string;
  readonly message: string;
}

/** I-JSON's exact integer range (RFC 7493), the outer bound of every integer. */
export const MAX_SAFE = Number.MAX_SAFE_INTEGER;

const DRAFT = "https://json-schema.org/draft/2020-12/schema";
const ROOT_ONLY = new Set(["$schema", "$id", "$defs"]);
const KEYWORDS = new Set([
  ...ROOT_ONLY,
  "$ref",
  "title",
  "description",
  "type",
  "properties",
  "required",
  "additionalProperties",
  "items",
  "minItems",
  "maxItems",
  "enum",
  "const",
  "minimum",
  "maximum",
  "minLength",
  "maxLength",
  "pattern",
  "oneOf",
  "discriminator",
]);
const SCALARS = new Set(["string", "integer", "boolean", "null"]);
const TYPES = new Set([...SCALARS, "object", "array"]);
const TAG = "kind";

type Schema = Record<string, unknown>;

function isSchema(value: unknown): value is Schema {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pointer(at: string, key: string): string {
  return `${at}/${key.replaceAll("~", "~0").replaceAll("/", "~1")}`;
}

/**
 * Checks one schema document against a profile. Returns every violation,
 * in document order; an empty list means the schema conforms.
 */
export function checkProfile(root: unknown, profile: Profile): ProfileViolation[] {
  const out: ProfileViolation[] = [];
  const fail = (at: string, message: string): void => {
    out.push({ at, message });
  };
  if (!isSchema(root)) {
    fail("", "a schema document must be an object");
    return out;
  }
  if (root.$schema !== DRAFT) fail("", `$schema must be ${DRAFT}`);
  const defs = isSchema(root.$defs) ? root.$defs : {};
  if (root.$defs !== undefined && !isSchema(root.$defs)) fail("/$defs", "$defs must be an object");

  const resolve = (ref: unknown, at: string): Schema | undefined => {
    const name = typeof ref === "string" && ref.startsWith("#/$defs/") ? ref.slice("#/$defs/".length) : undefined;
    const target = name === undefined ? undefined : defs[name];
    if (!isSchema(target)) {
      fail(at, `$ref must point to a local #/$defs entry, got ${JSON.stringify(ref)}`);
      return undefined;
    }
    return target;
  };

  const walk = (schema: unknown, at: string, isRoot: boolean): void => {
    if (!isSchema(schema)) {
      fail(at, "a subschema must be an object");
      return;
    }
    for (const key of Object.keys(schema)) {
      if (!KEYWORDS.has(key)) fail(pointer(at, key), `keyword ${key} is not in the ${profile} profile`);
      else if (ROOT_ONLY.has(key) && !isRoot) fail(pointer(at, key), `${key} is allowed only at the root`);
    }
    if ("$ref" in schema) {
      resolve(schema.$ref, pointer(at, "$ref"));
      const siblings = Object.keys(schema).filter((k) => k !== "$ref" && k !== "description" && k !== "title");
      if (siblings.length > 0) fail(at, `$ref takes no sibling keywords (found ${siblings.join(", ")})`);
      return;
    }
    const types = checkType(schema, at, profile, fail);
    // A union node only declares "type": "object" for Ajv; its branches are
    // the closed objects (a closed union node would accept nothing).
    const isObject = !("oneOf" in schema) && (types.includes("object") || "properties" in schema);
    if (isObject) checkObject(schema, at, fail, walk);
    if (types.includes("array") || "items" in schema) {
      if (schema.items === undefined) fail(at, "an array must declare items");
      else walk(schema.items, pointer(at, "items"), false);
    }
    if (types.includes("integer")) checkBounds(schema, at, fail);
    if ("enum" in schema) {
      const values = schema.enum;
      if (!Array.isArray(values) || values.length === 0) fail(pointer(at, "enum"), "enum must be a non-empty array");
      else if (new Set(values.map((v) => JSON.stringify(v))).size !== values.length) {
        fail(pointer(at, "enum"), "enum values must be unique");
      }
    }
    if ("oneOf" in schema) checkUnion(schema, at, fail, resolve);
  };

  for (const [name, def] of Object.entries(defs)) walk(def, pointer("/$defs", name), false);
  walk(root, "", true);
  return out;
}

function checkType(
  schema: Schema,
  at: string,
  profile: Profile,
  fail: (at: string, message: string) => void,
): string[] {
  const type = schema.type;
  if (type === undefined) return [];
  const list = Array.isArray(type) ? type : [type];
  if (list.length === 0 || !list.every((t) => typeof t === "string" && TYPES.has(t))) {
    fail(pointer(at, "type"), `type must name only ${[...TYPES].join(", ")} (number is banned)`);
    return [];
  }
  const names = list as string[];
  if (Array.isArray(type)) {
    const nullable = names.length === 2 && names[1] === "null" && names[0] !== "null";
    const scalarUnion = profile === "engine-mirror" && names.every((t) => SCALARS.has(t));
    if (!nullable && !scalarUnion) {
      fail(
        pointer(at, "type"),
        profile === "contract"
          ? 'a type array must be [T, "null"] in the contract profile'
          : 'a type array must be [T, "null"] or scalar types only in the engine-mirror profile',
      );
    }
  }
  return names;
}

function checkObject(
  schema: Schema,
  at: string,
  fail: (at: string, message: string) => void,
  walk: (schema: unknown, at: string, isRoot: boolean) => void,
): void {
  if (schema.additionalProperties !== false) fail(at, "an object must set additionalProperties: false");
  const properties = isSchema(schema.properties) ? schema.properties : {};
  if (schema.properties !== undefined && !isSchema(schema.properties)) {
    fail(pointer(at, "properties"), "properties must be an object");
  }
  for (const [name, sub] of Object.entries(properties)) walk(sub, pointer(pointer(at, "properties"), name), false);
  if (schema.required !== undefined) {
    const required = schema.required;
    if (!Array.isArray(required) || !required.every((r) => typeof r === "string")) {
      fail(pointer(at, "required"), "required must be an array of property names");
    } else {
      for (const name of required) {
        if (!(name in properties)) fail(pointer(at, "required"), `required names ${name}, which is not a property`);
      }
    }
  }
}

function checkBounds(schema: Schema, at: string, fail: (at: string, message: string) => void): void {
  for (const key of ["minimum", "maximum"] as const) {
    const bound = schema[key];
    if (bound === undefined) fail(at, `an integer must declare ${key}`);
    else if (typeof bound !== "number" || !Number.isSafeInteger(bound)) {
      fail(pointer(at, key), `${key} must be an integer within ±(2^53−1)`);
    }
  }
}

function checkUnion(
  schema: Schema,
  at: string,
  fail: (at: string, message: string) => void,
  resolve: (ref: unknown, at: string) => Schema | undefined,
): void {
  const branches = schema.oneOf;
  if (!Array.isArray(branches) || branches.length === 0) {
    fail(pointer(at, "oneOf"), "oneOf must be a non-empty array");
    return;
  }
  if (schema.type !== "object") fail(at, 'a discriminated union must declare "type": "object"');
  const discriminator = schema.discriminator;
  if (!isSchema(discriminator) || discriminator.propertyName !== TAG || Object.keys(discriminator).length !== 1) {
    fail(pointer(at, "discriminator"), `a union needs discriminator: {"propertyName": "${TAG}"}`);
  }
  const tags = new Set<unknown>();
  branches.forEach((branch, i) => {
    const where = pointer(pointer(at, "oneOf"), String(i));
    if (!isSchema(branch) || Object.keys(branch).length !== 1 || !("$ref" in branch)) {
      fail(where, "every oneOf branch must be a lone $ref");
      return;
    }
    const target = resolve(branch.$ref, pointer(where, "$ref"));
    if (target === undefined) return;
    const properties = isSchema(target.properties) ? target.properties : {};
    const tag = properties[TAG];
    const required = Array.isArray(target.required) ? target.required : [];
    if (!isSchema(tag) || typeof tag.const !== "string" || !required.includes(TAG)) {
      fail(where, `the branch must require a "${TAG}" property with a string const`);
      return;
    }
    if (tags.has(tag.const)) fail(where, `duplicate ${TAG} tag ${JSON.stringify(tag.const)}`);
    tags.add(tag.const);
  });
}
