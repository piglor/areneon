// Print the repo's jcs() output per line of harness/jcs-vectors.txt.
import { readFileSync } from "node:fs";
import { jcs } from "../../../packages/showdown-bridge/src/rules.ts";
import type { Json } from "../../../packages/showdown-bridge/src/request-identity.ts";

for (const line of readFileSync(process.argv[2] as string, "utf8").split("\n").filter(Boolean)) {
  try {
    console.log(jcs(JSON.parse(line) as Json));
  } catch (error) {
    console.log(`ERROR ${(error as Error).name}`);
  }
}
