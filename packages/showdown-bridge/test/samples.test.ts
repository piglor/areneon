import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import { beforeAll, describe, expect, it } from "vitest";
import { loadEngine } from "../src/index.ts";
import { contractSamples, SAMPLE_HASHES, SAMPLES_DIR } from "../src/samples.ts";

const SCHEMAS = new URL("../../schema/contracts/v1/", import.meta.url);
const committed = (file: string): unknown => JSON.parse(readFileSync(join(SAMPLES_DIR, file), "utf8"));

let fresh: Record<string, unknown>;
beforeAll(async () => {
  fresh = await contractSamples(loadEngine());
});

describe("the shared contract sample corpus (ADR-008 §2)", () => {
  it("matches a fresh capture: re-run scripts/capture-samples.ts and review the diff", () => {
    const files = readdirSync(join(SAMPLES_DIR, "v1"), { recursive: true, withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => join(e.parentPath, e.name).slice(SAMPLES_DIR.length + 1));
    expect([...files, SAMPLE_HASHES].sort()).toEqual(Object.keys(fresh).sort());
    for (const [file, content] of Object.entries(fresh)) expect(committed(file), file).toEqual(content);
  });

  it("covers every contract, with a valid and an invalid case each", () => {
    const contracts = readdirSync(SCHEMAS).map((f) => f.replace(".schema.json", ""));
    for (const contract of contracts) {
      for (const kind of ["valid", "invalid"]) {
        expect(Object.keys(fresh).some((f) => f.startsWith(`v1/${contract}/${kind}/`)), `${contract} ${kind}`).toBe(true);
      }
    }
  });

  it("is classified by Ajv exactly as its directory says", () => {
    for (const [file, content] of Object.entries(fresh)) {
      if (file === SAMPLE_HASHES) continue;
      const [, contract, kind] = file.split("/") as [string, string, string];
      const schema = JSON.parse(readFileSync(new URL(`${contract}.schema.json`, SCHEMAS), "utf8")) as { $id: string };
      const profile = schema.$id.includes(":engine:") ? "engine-mirror" : "contract";
      expect(ajvFor(profile).validate(schema, content), file).toBe(kind === "valid");
    }
  });
});
