// The command line of teams-publish, run by .github/workflows/teams-publish.yml:
//
//   plan    --snapshot DIR                      write DIR/plan.json and DIR/release/teams-<hash>.json
//   verify  --snapshot DIR                      the plan again from the files must equal the plan on disk
//   report  --snapshot DIR --platform NAME --out FILE   one platform's report for the two-platform check
//   compare REPORT_A REPORT_B                   exit 1 unless the two platforms agree
//   upload  --snapshot DIR --bucket NAME [--skip-without-store]
//                                               verify, then put the shards and the manifest in the bucket
//
// `upload` reads the R2 credentials from the environment only (docs/runbooks/teams-r2.md);
// with --skip-without-store and no credential variable set it does nothing and exits 0.
// Exit codes: 0 done, 1 refused, 2 usage.

import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { type ObjectLayout, type ObjectStore, R2ObjectStore, R2Settings } from "@areneon/object-store";
import { TEAMS_LAYOUT } from "./layout.ts";
import { assertPlatformsAgree, parseReport, platformReport } from "./platform-check.ts";
import { PublishError, verifyDir, writePlan } from "./plan.ts";
import { uploadSnapshot } from "./upload.ts";

export interface PublishIo {
  out(line: string): void;
  err(line: string): void;
  env: Record<string, string | undefined>;
  /** The store behind `upload`; the R2 client when omitted. */
  openStore?: ((settings: R2Settings, layout: ObjectLayout) => ObjectStore) | undefined;
}

class UsageError extends Error {
  override name = "UsageError";
}

const OPTIONS = {
  snapshot: { type: "string" },
  platform: { type: "string" },
  out: { type: "string" },
  bucket: { type: "string" },
  "skip-without-store": { type: "boolean" },
} as const;

export async function runPublishCli(argv: readonly string[], io: PublishIo): Promise<number> {
  try {
    const { values, positionals } = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true, strict: true });
    const [mode, ...rest] = positionals;
    const need = (name: "snapshot" | "platform" | "out" | "bucket"): string => {
      const value = values[name];
      if (typeof value !== "string") throw new UsageError(`--${name} is required`);
      return value;
    };
    switch (mode) {
      case "plan": {
        const plan = writePlan(need("snapshot"));
        io.out(`planned ${plan.tag} (${plan.shards.length} shards)`);
        return 0;
      }
      case "verify": {
        const plan = verifyDir(need("snapshot"));
        io.out(`verified ${plan.tag}`);
        return 0;
      }
      case "report": {
        writeFileSync(need("out"), `${JSON.stringify(platformReport(need("snapshot"), need("platform")), null, 2)}\n`);
        return 0;
      }
      case "compare": {
        if (rest.length !== 2) throw new UsageError("compare takes two report files");
        assertPlatformsAgree(...(rest.map((f) => parseReport(readFileSync(f, "utf8"))) as [ReturnType<typeof parseReport>, ReturnType<typeof parseReport>]));
        io.out("the two platforms agree");
        return 0;
      }
      case "upload": {
        const dir = need("snapshot");
        if (values["skip-without-store"] === true && io.env.AWS_ACCESS_KEY_ID === undefined) {
          io.out("no store credentials in the environment: upload skipped");
          return 0;
        }
        const plan = verifyDir(dir);
        const settings = R2Settings.fromEnvironment(io.env, argv, need("bucket"));
        const store = (io.openStore ?? ((s, l) => new R2ObjectStore(s, l)))(settings, TEAMS_LAYOUT);
        const result = await uploadSnapshot(store, dir, plan);
        io.out(`uploaded ${result.uploaded.length}, already present ${result.present.length}`);
        return 0;
      }
      default:
        throw new UsageError("usage: plan|verify|report|compare|upload");
    }
  } catch (error) {
    if (error instanceof UsageError || (error as { code?: string }).code?.startsWith("ERR_PARSE_ARGS")) {
      io.err(`teams-publish: ${(error as Error).message}`);
      return 2;
    }
    if (error instanceof PublishError || (error as Error).name === "ObjectStoreError" || (error as Error).name === "CredentialError") {
      io.err(`teams-publish: ${(error as Error).message}`);
      return 1;
    }
    throw error;
  }
}
