// Not shipped. The design-deviation part of a usage prior's derivation report (ADR-025 section 3, #1120):
//   node packages/opponent-model/scripts/usage-prior-report.ts PRIOR.json
// Reads the prior and the golden team vocabulary of its format and prints, as JSON, the median and the largest
// deviation of the move model's design from the stated marginals and the ten formes where it is largest.

import { readFileSync } from "node:fs";
import { designDeviation } from "../src/index.ts";
import { readUsagePrior } from "../../showdown-bridge/src/usage-prior.ts";

const file = process.argv[2];
if (file === undefined) {
  console.error("usage: usage-prior-report.ts PRIOR.json");
  process.exit(2);
}
const raw = JSON.parse(readFileSync(file, "utf8")) as { format_id: string };
const vocabulary = JSON.parse(readFileSync(`fixtures/showdown/${raw.format_id}/team-vocabulary.json`, "utf8"));
const prior = readUsagePrior(raw, vocabulary);
const pools = new Map<string, readonly string[]>(vocabulary.roster.map((r: { id: string; moves: string[] }) => [r.id, r.moves]));
const rows = designDeviation(prior, pools).sort((a, b) => b.max_ppm - a.max_ppm || (a.forme < b.forme ? -1 : 1));
const sorted = rows.map((r) => r.max_ppm).sort((a, b) => a - b);
console.log(JSON.stringify({ formes: rows.length, median_max_ppm: sorted[Math.floor(sorted.length / 2)], largest_max_ppm: sorted[sorted.length - 1], top: rows.slice(0, 10) }, null, 1));
