// Not shipped (#1132). Times the damage evidence per decision on a sample of the committed requests: for every K-th request of
// both seats, `belief` without the evidence, the port's construction (the reader over the channel, the hits, the engine setup of
// the first request) and `belief` with the evidence under the committed belief config with damage evidence, on the real probe
// and the real vocabulary. One thread; the figures are a measurement of one machine at one moment, printed with its load.
//   node packages/opponent-model/scripts/damage-cost.ts [--prior FILE] [--every K] [--offset O] [--max-checked N] [--game-cache]
// The default prior is the committed Smogon usage record (named spreads); a request the prior cannot explain (the corpus teams are
// not its teams) is skipped and counted.

import { readFileSync } from "node:fs";
import { cpus, loadavg } from "node:os";
import { join } from "node:path";
import { interpretChannel } from "@areneon/public-state";
import type { UsagePrior } from "@areneon/schema/types/v1/usage-prior.ts";
import { loadDamageAwareConfig } from "../../showdown-bridge/src/damage-aware-config.ts";
import { readBeliefConfig } from "../../showdown-bridge/src/belief-config.ts";
import { createDamageCheck, newDamageCheckCache } from "../../showdown-bridge/src/damage-check.ts";
import type { PokemonSet } from "../../showdown-bridge/src/engine.ts";
import { GOLDEN_FORMAT } from "../../showdown-bridge/src/golden.ts";
import { loadEngine } from "../../showdown-bridge/src/index.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { BeliefError, belief } from "../src/index.ts";
import { games, points } from "../test/corpus.ts";
import { percentiles } from "../test/measure.ts";
import { ROOT, realVocabulary } from "../test/real.ts";

const arg = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? (process.argv[i + 1] as string) : fallback;
};
const every = Number(arg("--every", "4"));
const offset = Number(arg("--offset", "0"));
const file = arg("--prior", "usage-prior.2026-09-1630.v2.json");
const dir = join(ROOT, "config", "opponent-model", GOLDEN_FORMAT);
const prior = JSON.parse(readFileSync(join(dir, file), "utf8")) as UsagePrior;
const base = readBeliefConfig(JSON.parse(readFileSync(join(dir, "belief.damage-v1.json"), "utf8")), realVocabulary);
const rules = base.damage_evidence as NonNullable<typeof base.damage_evidence>;
const maxChecked = Number(arg("--max-checked", String(rules.max_checked)));
const config = { ...base, damage_evidence: { ...rules, max_checked: maxChecked } };

const engine = loadEngine();
const base0 = { engine, vocabulary: realVocabulary, config: loadDamageAwareConfig("v1", realVocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief: config, data_class: "synthetic" as const };
const shared = process.argv.includes("--game-cache");

const plain: number[] = [];
const build: number[] = [];
const withDamage: number[] = [];
const total: number[] = [];
const sum = { requests: 0, skipped: 0, with_events: 0, events: 0, applied: 0, checked: 0, rejected: 0, unsupported: 0, unchecked: 0, contradictions: 0, probe_runs: 0 };
let k = 0;
for (const game of games()) {
  for (const seat of ["p1", "p2"] as const) {
    const lines = game.channels[seat] as string[];
    const run = interpretChannel(seat, lines);
    const frames = run.frames.filter((f) => f.kind === "request");
    // with --game-cache the ports of a game and seat share the probe's answers and the verdicts; every K-th request is measured, so the cache fills only from those
    const context = shared ? { ...base0, cache: newDamageCheckCache() } : base0;
    for (const point of points(game).filter((p) => p.seat === seat)) {
      k += 1;
      if ((k - 1) % every !== offset) continue;
      const frame = frames[point.request];
      if (frame === undefined) continue;
      const input = { prior, vocabulary: realVocabulary, config, opponent: point.state.opponent, flags: point.flags };
      try {
        let t = process.hrtime.bigint();
        belief(input);
        plain.push(Number(process.hrtime.bigint() - t) / 1e6);
        t = process.hrtime.bigint();
        const cut = { frames: run.frames.slice(0, frame.index + 1), flags: run.flags.filter((f) => f.index <= frame.index), requests: run.requests.slice(0, frame.request) };
        const port = createDamageCheck(context, { player: seat, lines: lines.slice(0, frame.index + 1), team: game.teams[seat] as unknown as PokemonSet[], run: cut });
        build.push(Number(process.hrtime.bigint() - t) / 1e6);
        t = process.hrtime.bigint();
        const b = belief({ ...input, damage: port });
        withDamage.push(Number(process.hrtime.bigint() - t) / 1e6);
        total.push((build.at(-1) as number) + (withDamage.at(-1) as number));
        sum.requests += 1;
        if (b.damage.events > 0) sum.with_events += 1;
        for (const key of ["events", "applied", "checked", "rejected", "unsupported", "unchecked", "contradictions"] as const) sum[key] += b.damage[key];
        sum.probe_runs += port?.counts().probe_runs ?? 0;
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        sum.skipped += 1;
      }
    }
  }
}
console.log(JSON.stringify({ game_cache: shared, prior: file, every, offset, max_checked: maxChecked, cpus: cpus().length, load: loadavg().map((x) => Math.round(x * 100) / 100), sum, belief_without_ms: percentiles(plain), port_build_ms: percentiles(build), belief_with_ms: percentiles(withDamage), damage_total_ms: percentiles(total) }));
