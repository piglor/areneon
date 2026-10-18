// TS runtime checks: Ajv (2020-12) verdicts and the quicktype converter's
// parse verdict + JCS round trip. Prints one JSON line per sample.
// Run: node harness/check-ts.ts <out>
import { readFileSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { jcs } from "../../../packages/showdown-bridge/src/rules.ts";
import { requestHash, type Json } from "../../../packages/showdown-bridge/src/request-identity.ts";

interface Validator { (data: unknown): boolean; errors?: unknown[] | null }
interface AjvLike { compile(schema: object): Validator }
interface QuicktypeConvert {
  toAreneonContracts(json: string): object;
  areneonContractsToJson(value: object): string;
}

const out = process.argv[2];
if (out === undefined) throw new Error("usage: node harness/check-ts.ts <out>");
const spike = join(import.meta.dirname, "..");
const require = createRequire(join(spike, "..", "..", "packages", "showdown-bridge", "package.json"));
const { Ajv2020 } = require("ajv/dist/2020.js") as { Ajv2020: new (o: object) => AjvLike };
const ajvVersion = (require("ajv/package.json") as { version: string }).version;
const schema = JSON.parse(readFileSync(join(spike, "schema", "contracts.schema.json"), "utf8")) as object;
const validate = new Ajv2020({ allErrors: true, strict: true, allowUnionTypes: true }).compile(schema);
const { Convert } = (await import(join(out, "ts", "quicktype.ts"))) as { Convert: QuicktypeConvert };

for (const kind of ["valid", "invalid"]) {
  const dir = join(out, "samples", kind);
  for (const file of readdirSync(dir).sort()) {
    const key = file.split(".")[0] as string;
    const value = JSON.parse(readFileSync(join(dir, file), "utf8")) as Json;
    const catalog = { [key]: value };
    const row: Record<string, unknown> = { lang: "ts", sample: `${kind}/${file}`, ajv: validate(catalog) };
    try {
      const parsed = Convert.toAreneonContracts(JSON.stringify(catalog));
      const back = JSON.parse(Convert.areneonContractsToJson(parsed)) as Json;
      row.quicktype = true;
      row.quicktype_jcs_equal = jcs(back) === jcs(catalog);
    } catch {
      row.quicktype = false;
    }
    console.log(JSON.stringify(row));
  }
}
for (const file of readdirSync(join(out, "samples", "valid")).filter((f) => f.startsWith("request_fixture.")).sort()) {
  const fixture = JSON.parse(readFileSync(join(out, "samples", "valid", file), "utf8")) as {
    request: { [key: string]: Json };
    request_hash: string;
  };
  console.error(JSON.stringify({ lang: "ts", request_hash: file, matches: requestHash(fixture.request) === fixture.request_hash }));
}
console.error(`ajv ${ajvVersion}`);
