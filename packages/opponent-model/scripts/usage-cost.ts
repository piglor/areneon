// Not shipped (#1241). Times `belief` under the committed Smogon usage prior record on the committed channels and prints one JSON line
// per request with its belief digest (and, with --worlds N, the digest of N sampled worlds), so a change of the computation is compared
// byte for byte before and after:
//   node packages/opponent-model/scripts/usage-cost.ts [--prior FILE] [--every K] [--offset O] [--worlds N] [--only GAME:SEAT:REQUEST] [--out FILE]
// One thread; the timings are a measurement of one machine at one moment. Needs the built engine only with --worlds.

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { BeliefError, beliefDigest, createSampler, worldsDigest } from "../src/index.ts";
import { beliefAt, games, points } from "../test/corpus.ts";
import { percentiles } from "../test/measure.ts";
import { ROOT, realVocabulary } from "../test/real.ts";

const arg = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? (process.argv[i + 1] as string) : fallback;
};
const every = Number(arg("--every", "1"));
const offset = Number(arg("--offset", "0"));
const worlds = Number(arg("--worlds", "0"));
const out = arg("--out", "");
const only = arg("--only", ""); // game:seat:request
const file = arg("--prior", "usage-prior.2026-09-1630.v2.json");
const prior = JSON.parse(readFileSync(join(ROOT, "config", "opponent-model", realVocabulary.format_id, file), "utf8")) as UsagePrior;

let sampler: ReturnType<typeof createSampler> | null = null;
if (worlds > 0) {
  const { createSetLegality } = await import("../../showdown-bridge/src/set-legality.ts");
  const { loadEngine } = await import("../../showdown-bridge/src/index.ts");
  sampler = createSampler({
    legality: createSetLegality({ engine: loadEngine(), vocabulary: realVocabulary }),
    spreads: { other: (_s, _f, below) => ({ nature: (realVocabulary.natures[Number(below(BigInt(realVocabulary.natures.length)))] as { id: string }).id, stat_points: { hp: Number(below(33n)), atk: Number(below(33n)), def: 0, spa: 0, spd: 0, spe: Number(below(2n)) } }) },
  });
}

const lines: string[] = [];
const times: number[] = [];
let k = 0;
for (const game of games()) {
  for (const point of points(game)) {
    k += 1;
    if (only !== "" ? `${point.game}:${point.seat}:${point.request}` !== only : (k - 1) % every !== offset) continue;
    const t0 = process.hrtime.bigint();
    let row: Record<string, unknown>;
    try {
      const b = beliefAt(point, prior);
      const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      times.push(ms);
      row = { game: point.game, seat: point.seat, request: point.request, ms, digest: beliefDigest(b) };
      if (sampler !== null) {
        const stream = Uint8Array.from({ length: 32 }, (_, i) => (k * 131 + i * 7) & 255);
        row.worlds = worldsDigest(sampler.sampleWorlds(b, stream, worlds));
      }
    } catch (e) {
      if (!(e instanceof BeliefError)) throw e;
      row = { game: point.game, seat: point.seat, request: point.request, code: e.code };
    }
    lines.push(JSON.stringify(row));
  }
}
if (out !== "") writeFileSync(out, `${lines.join("\n")}\n`);
console.log(JSON.stringify({ prior: file, every, offset, requests: lines.length, load: process.argv.includes("--load") ? "" : undefined, belief: percentiles(times) }));
