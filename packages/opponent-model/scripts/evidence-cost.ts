// Not shipped (#1133). The slice-4 cost gate (ADR-025 section 7): times a decision's belief with and without the damage AND the
// speed-order evidence together on a sample of the committed requests: for every K-th request of both seats, `belief` without
// the evidence, then the ports' construction (the reader over the channel, the hits and pairs, the engine setup of the first
// request) and `belief` with both under the committed belief config with both (`belief.speed-v1.json`), on the real probe and the
// real vocabulary. One thread; the figures are a measurement of one machine at one moment, printed with its load. The gate is a
// p99 per decision of at most 500 ms (ADR-022's value) and a generation within 1.5 times the control's: here the ratio of the
// mean decision with the evidence to the mean without it, which bounds a generation whose decisions are all belief.
//   node packages/opponent-model/scripts/evidence-cost.ts [--prior FILE] [--every K] [--offset O] [--no-cache]
// A request the prior cannot explain (the corpus teams are not its teams) is skipped and counted.

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
import { createSpeedCheck } from "../../showdown-bridge/src/speed-check.ts";
import { BeliefError, belief } from "../src/index.ts";
import { games, points } from "../test/corpus.ts";
import { percentiles } from "../test/measure.ts";
import { ROOT, realVocabulary } from "../test/real.ts";

const arg = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? (process.argv[i + 1] as string) : fallback;
};
const every = Number(arg("--every", "11"));
const offset = Number(arg("--offset", "0"));
const file = arg("--prior", "usage-prior.2026-09-1630.v2.json");
const dir = join(ROOT, "config", "opponent-model", GOLDEN_FORMAT);
const prior = JSON.parse(readFileSync(join(dir, file), "utf8")) as UsagePrior;
const config = readBeliefConfig(JSON.parse(readFileSync(join(dir, "belief.speed-v1.json"), "utf8")), realVocabulary);

const engine = loadEngine();
const base = { engine, vocabulary: realVocabulary, config: loadDamageAwareConfig("v1", realVocabulary), mechanics: loadMechanicsConfig(GOLDEN_FORMAT), belief: config, data_class: "synthetic" as const };
const shared = !process.argv.includes("--no-cache");

const plain: number[] = [];
const build: number[] = [];
const withBoth: number[] = [];
const total: number[] = [];
const sum = { requests: 0, skipped: 0, damage_events: 0, speed_events: 0, speed_applied: 0, speed_checked: 0, speed_rejected: 0, speed_unsupported: 0, speed_unchecked: 0, speed_contradictions: 0, damage_applied: 0, damage_rejected: 0, damage_contradictions: 0, probe_runs: 0 };
let k = 0;
for (const game of games()) {
  for (const seat of ["p1", "p2"] as const) {
    const lines = game.channels[seat] as string[];
    const run = interpretChannel(seat, lines);
    const frames = run.frames.filter((f) => f.kind === "request");
    // with the cache the two ports of a game and seat share the probe's answers and the verdicts, as a consumer's would; only the measured requests fill it
    const context = shared ? { ...base, cache: newDamageCheckCache() } : base;
    for (const point of points(game).filter((p) => p.seat === seat)) {
      k += 1;
      if ((k - 1) % every !== offset) continue;
      const frame = frames[point.request];
      if (frame === undefined) continue;
      const input = { prior, vocabulary: realVocabulary, config, opponent: point.state.opponent, flags: point.flags };
      try {
        console.error(`request ${k} ${game.id} ${seat} ${point.request}: plain`);
        let t = process.hrtime.bigint();
        belief(input);
        plain.push(Number(process.hrtime.bigint() - t) / 1e6);
        t = process.hrtime.bigint();
        const cut = { frames: run.frames.slice(0, frame.index + 1), flags: run.flags.filter((f) => f.index <= frame.index), requests: run.requests.slice(0, frame.request) };
        const at = { player: seat, lines: lines.slice(0, frame.index + 1), team: game.teams[seat] as unknown as PokemonSet[], run: cut };
        const damage = createDamageCheck(context, at);
        const speed = createSpeedCheck(context, at);
        build.push(Number(process.hrtime.bigint() - t) / 1e6);
        t = process.hrtime.bigint();
        console.error(`  ports: damage ${damage?.events.length ?? "-"} events, speed ${speed?.events.length ?? "-"} events`);
        const b = belief({ ...input, damage, speed });
        withBoth.push(Number(process.hrtime.bigint() - t) / 1e6);
        total.push((build.at(-1) as number) + (withBoth.at(-1) as number));
        sum.requests += 1;
        console.error(`${sum.requests} plain ${Math.round(plain.at(-1) as number)} build ${Math.round(build.at(-1) as number)} with ${Math.round(withBoth.at(-1) as number)} ms`);
        sum.damage_events += b.damage.events;
        sum.speed_events += b.speed.events;
        sum.speed_applied += b.speed.applied;
        sum.speed_checked += b.speed.checked;
        sum.speed_rejected += b.speed.rejected;
        sum.speed_unsupported += b.speed.unsupported;
        sum.speed_unchecked += b.speed.unchecked;
        sum.speed_contradictions += b.speed.contradictions;
        sum.damage_applied += b.damage.applied;
        sum.damage_rejected += b.damage.rejected;
        sum.damage_contradictions += b.damage.contradictions;
        sum.probe_runs += (damage?.counts().probe_runs ?? 0) + (speed?.counts().probe_runs ?? 0);
      } catch (e) {
        if (!(e instanceof BeliefError)) throw e;
        sum.skipped += 1;
      }
    }
  }
}
const mean = (xs: number[]): number => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
console.log(JSON.stringify({ cache: shared, prior: file, every, offset, cpus: cpus().length, load: loadavg().map((x) => Math.round(x * 100) / 100), sum, belief_without_ms: percentiles(plain), ports_build_ms: percentiles(build), belief_with_ms: percentiles(withBoth), evidence_total_ms: percentiles(total), mean_ms: { without: Math.round(mean(plain)), with: Math.round(mean(total)) }, ratio_mean_with_over_without: Math.round((100 * mean(total)) / mean(plain)) / 100 }));
