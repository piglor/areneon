import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ajvFor } from "../src/ajv.ts";
import {
  checkSchemas,
  ContractsReadError,
  CONTRACTS_DIR,
  declaredProfile,
  driftAgainst,
  GENERATED_TS_DIR,
  generateAll,
  readSchemas,
  runContracts,
  type SchemaFile,
} from "../src/contracts.ts";

const DRAFT = "https://json-schema.org/draft/2020-12/schema";

function identity(id = "urn:areneon:schema:contracts:v1:request-identity"): Record<string, unknown> {
  return {
    $schema: DRAFT,
    $id: id,
    title: "RequestIdentity",
    type: "object",
    additionalProperties: false,
    required: ["request_seq", "request_hash"],
    properties: {
      request_seq: { type: "integer", minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
      request_hash: { type: "string", pattern: "^[0-9a-f]{64}$" },
      rqid: { type: "integer", minimum: 1, maximum: Number.MAX_SAFE_INTEGER },
    },
  };
}

function engineMove(): Record<string, unknown> {
  return {
    $schema: DRAFT,
    $id: "urn:areneon:schema:engine:v1:engine-move",
    title: "EngineMove",
    type: "object",
    additionalProperties: false,
    required: ["move"],
    properties: { move: { type: "string" }, disabled: { type: ["string", "boolean"] } },
  };
}

let root: string;
let contracts: string;
let generated: string;

function put(path: string, schema: unknown): void {
  mkdirSync(join(contracts, path, ".."), { recursive: true });
  writeFileSync(join(contracts, path), JSON.stringify(schema));
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "areneon-contracts-"));
  contracts = join(root, "contracts");
  generated = join(root, "generated");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("readSchemas", () => {
  it("reads v<N>/*.schema.json in version then name order and ignores other files", () => {
    put("v2/b.schema.json", identity());
    put("v10/a.schema.json", identity());
    put("v1/z.schema.json", identity());
    put("v1/notes.md", {});
    put("draft/x.schema.json", identity());
    expect(readSchemas(contracts).map((f) => f.path)).toEqual([
      "v1/z.schema.json",
      "v2/b.schema.json",
      "v10/a.schema.json",
    ]);
  });

  it("treats a missing directory as holding no schemas", () => {
    expect(readSchemas(join(root, "absent"))).toEqual([]);
  });

  it("fails closed on any other read error instead of checking nothing", () => {
    writeFileSync(join(root, "a-file"), "");
    expect(() => readSchemas(join(root, "a-file"))).toThrow(ContractsReadError);
  });

  it("names the file that is not JSON", () => {
    mkdirSync(join(contracts, "v1"), { recursive: true });
    writeFileSync(join(contracts, "v1", "broken.schema.json"), "{");
    expect(() => readSchemas(contracts)).toThrow(/^v1\/broken\.schema\.json: /);
  });
});

describe("declaredProfile", () => {
  const file = (schema: unknown, path = "v1/request-identity.schema.json"): SchemaFile => ({
    path,
    version: Number(path.slice(1, path.indexOf("/"))),
    name: path.slice(path.indexOf("/") + 1, -".schema.json".length),
    schema,
  });

  it("reads the profile from the $id namespace", () => {
    expect(declaredProfile(file(identity()))).toEqual({ profile: "contract" });
    expect(declaredProfile(file(engineMove(), "v1/engine-move.schema.json"))).toEqual({ profile: "engine-mirror" });
  });

  it("refuses an $id outside the scheme or one that does not match its path", () => {
    const error = (schema: unknown, path?: string): string =>
      (declaredProfile(file(schema, path)) as { error: string }).error;
    expect(error({ $id: "urn:areneon:schema:rules:v1:format" })).toMatch(/must be urn:areneon/);
    expect(error(null)).toMatch(/got undefined/);
    expect(error(identity(), "v2/request-identity.schema.json")).toMatch(/does not match its path/);
    expect(error(identity(), "v1/other.schema.json")).toMatch(/does not match its path/);
  });
});

describe("checkSchemas", () => {
  it("accepts conforming contract and engine-mirror schemas", () => {
    put("v1/request-identity.schema.json", identity());
    put("v1/engine-move.schema.json", engineMove());
    expect(checkSchemas(readSchemas(contracts))).toEqual([]);
  });

  it("checks an engine-shaped union against the contract profile when the $id says contracts", () => {
    put("v1/engine-move.schema.json", { ...engineMove(), $id: "urn:areneon:schema:contracts:v1:engine-move" });
    expect(checkSchemas(readSchemas(contracts)).map((f) => f.message)).toEqual([
      '/properties/disabled/type: a type array must be [T, "null"] in the contract profile',
      expect.stringMatching(/^Ajv: strict mode: use allowUnionTypes/),
    ]);
  });

  it("reports profile violations and Ajv compile errors with their file", () => {
    put("v1/request-identity.schema.json", { ...identity(), additionalProperties: true });
    put("v1/bad-id.schema.json", { ...identity(), $id: "nope" });
    const findings = checkSchemas(readSchemas(contracts));
    expect(findings.map((f) => f.path)).toEqual(["v1/bad-id.schema.json", "v1/request-identity.schema.json"]);
    expect(findings[1]?.message).toBe("/: an object must set additionalProperties: false");
  });

  it("requires a PascalCase root title, which names the generated type", () => {
    put("v1/request-identity.schema.json", { ...identity(), title: undefined });
    put("v1/engine-move.schema.json", { ...engineMove(), title: "engine move" });
    expect(checkSchemas(readSchemas(contracts))).toEqual([
      { path: "v1/engine-move.schema.json", message: "the root needs a PascalCase title naming its generated type" },
      { path: "v1/request-identity.schema.json", message: "the root needs a PascalCase title naming its generated type" },
    ]);
  });

  it("surfaces Ajv strict-mode errors the profile walk cannot see", () => {
    // Every keyword is whitelisted, but `minimum` does not apply to a string:
    // only Ajv's strictTypes sees that.
    const schema = identity();
    (schema.properties as Record<string, unknown>).request_hash = { type: "string", minimum: 1 };
    put("v1/request-identity.schema.json", schema);
    expect(checkSchemas(readSchemas(contracts)).map((f) => f.message)).toEqual([
      expect.stringMatching(/^Ajv: strict mode: .*minimum/),
    ]);
  });
});

describe("ajvFor", () => {
  it("allows scalar type unions only on the engine-mirror instance", () => {
    expect(() => ajvFor("engine-mirror").compile(engineMove())).not.toThrow();
    expect(() => ajvFor("contract").compile(engineMove())).toThrow(/union type/);
  });

  it("enforces closed objects and the discriminator at validation time", () => {
    const validate = ajvFor("contract").compile(identity());
    expect(validate({ request_seq: 1, request_hash: "a".repeat(64) })).toBe(true);
    expect(validate({ request_seq: 1, request_hash: "a".repeat(64), extra: 1 })).toBe(false);
    expect(validate({ request_seq: 0, request_hash: "a".repeat(64) })).toBe(false);
  });
});

describe("generation and drift", () => {
  it("generates the same bytes on every run, with a banner naming the source", async () => {
    put("v1/request-identity.schema.json", identity());
    const files = readSchemas(contracts);
    const first = await generateAll(files);
    const second = await generateAll(files);
    expect([...first.keys()]).toEqual(["v1/request-identity.ts"]);
    expect(second).toEqual(first);
    const ts = first.get("v1/request-identity.ts") as string;
    expect(ts).toContain("packages/schema/contracts/v1/request-identity.schema.json. Do not edit.");
    expect(ts).toContain("export interface RequestIdentity {");
    expect(ts).toContain("request_seq: number;");
    expect(ts).toContain("rqid?: number;");
  });

  it("finds missing, changed and orphaned generated files", () => {
    const expected = new Map([
      ["v1/a.ts", "A\n"],
      ["v1/b.ts", "B\n"],
    ]);
    mkdirSync(join(generated, "v1"), { recursive: true });
    writeFileSync(join(generated, "v1", "b.ts"), "stale\n");
    writeFileSync(join(generated, "v1", "c.ts"), "C\n");
    expect(driftAgainst(expected, generated)).toEqual([
      { path: "v1/a.ts", message: "generated file is missing" },
      { path: "v1/b.ts", message: "generated file differs from its schema" },
      { path: "v1/c.ts", message: "generated file has no schema" },
    ]);
    expect(driftAgainst(new Map(), join(root, "absent"))).toEqual([]);
    writeFileSync(join(root, "a-file"), "");
    expect(() => driftAgainst(new Map(), join(root, "a-file"))).toThrow(ContractsReadError);
  });

  it("writes exactly the expected files, then reports no drift", async () => {
    put("v1/request-identity.schema.json", identity());
    mkdirSync(join(generated, "v1"), { recursive: true });
    writeFileSync(join(generated, "v1", "orphan.ts"), "x");
    const options = { contractsDir: contracts, generatedDir: generated };
    expect(await runContracts({ ...options, write: false })).toHaveLength(2);
    expect(await runContracts({ ...options, write: true })).toEqual([]);
    expect(await runContracts({ ...options, write: false })).toEqual([]);
    expect(readFileSync(join(generated, "v1", "request-identity.ts"), "utf8")).toContain("RequestIdentity");
  });

  it("stops before generating when a schema is outside its profile", async () => {
    put("v1/request-identity.schema.json", { ...identity(), additionalProperties: true });
    const findings = await runContracts({ write: true, contractsDir: contracts, generatedDir: generated });
    expect(findings).toHaveLength(1);
    expect(readSchemas(generated)).toEqual([]);
  });

  it("keeps the repository's checked-in generated types in step with its schemas", async () => {
    expect(await runContracts({ write: false, contractsDir: CONTRACTS_DIR, generatedDir: GENERATED_TS_DIR })).toEqual(
      [],
    );
  });
});
