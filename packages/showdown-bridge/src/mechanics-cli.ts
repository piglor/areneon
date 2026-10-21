// `scripts/mechanics.ts QUERY.json [--config FILE]`: read a MechanicsQuery
// (a file; `/dev/stdin` reads standard input) and print the MechanicsResult as one
// canonical JSON line (ADR-016 section 6). The mechanics config defaults to
// config/mechanics/<format_id>.json. The validator's own messages about a
// refused set go to stderr, never into the result. Exit 0: a result was
// printed (computed, unsupported or refused alike, as the result says). Exit 2:
// no result, because the query is not a MechanicsQuery (it has no query_hash),
// the format is not supported, the config is wrong, or the usage is.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { type Json, jcs } from "./canonical-json.ts";
import type { Engine } from "./engine.ts";
import { loadMechanicsConfig, readMechanicsConfig } from "./mechanics-config.ts";
import { probeMechanics } from "./mechanics-probe.ts";
import { readMechanicsQuery } from "./mechanics-query.ts";
import { readMechanicsResult } from "./mechanics-result.ts";

export interface MechanicsCliIo {
  out(text: string): void;
  err(text: string): void;
  /** The text of a file (`/dev/stdin` reads standard input). */
  read(path: string): string;
}

export const nodeMechanicsIo = (): MechanicsCliIo => ({
  out: (text) => void process.stdout.write(text),
  err: (text) => void process.stderr.write(text),
  read: (path) => readFileSync(path, "utf8"),
});

const USAGE = "usage: mechanics.ts QUERY.json [--config FILE]";

/** Run the command; returns the exit code. */
export function runMechanicsCli(argv: string[], io: MechanicsCliIo, engine: () => Engine): number {
  try {
    const { values, positionals } = parseArgs({ args: argv, options: { config: { type: "string" } }, allowPositionals: true });
    if (positionals.length !== 1) throw new Error(USAGE);
    const query = readMechanicsQuery(JSON.parse(io.read(positionals[0] as string)) as unknown);
    const config = values.config === undefined ? loadMechanicsConfig(query.format_id) : readMechanicsConfig(JSON.parse(io.read(values.config)) as unknown);
    const { result, diagnostics } = probeMechanics({ engine: engine(), config }, query);
    for (const line of diagnostics) io.err(`${line}\n`);
    io.out(`${jcs(readMechanicsResult(result) as unknown as Json)}\n`);
    return 0;
  } catch (error) {
    io.err(`mechanics: ${(error as Error).message}\n`);
    return 2;
  }
}
