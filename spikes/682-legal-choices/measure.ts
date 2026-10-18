// Spike #682 (throwaway): measure candidate counts, engine agreement and
// validation cost for joint-choice enumeration on seeded M-C battles.
//
//   make showdown && node spikes/682-legal-choices/measure.ts [battles]
//
// Teams are the synthetic golden team with Mega Stones swapped in (CC0,
// no real competitive team). Each side plays uniformly at random among the
// engine-accepted candidates, which explores previews, targets, Mega,
// forced switches and conflicting switches.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { enumerate } from "./enumerate.ts";
import { bruteForceUniverse, judge, type SimBattleClass, type SimSide } from "./oracle.ts";

const FORMAT = "gen9championsvgc2026regmc";
const root = fileURLToPath(new URL("../../", import.meta.url));
const showdownDir = process.env.ARENEON_SHOWDOWN_DIR ?? `${root}vendor/pokemon-showdown`;
const sim = createRequire(import.meta.url)(`${showdownDir}/dist/sim/index.js`) as {
  Battle: SimBattleClass;
  Teams: { pack(team: unknown[]): string };
  TeamValidator: { get(id: string): { validateTeam(team: unknown[]): string[] | null } };
};

interface SetLike {
  species: string;
  item: string;
}
const golden = JSON.parse(
  readFileSync(`${root}fixtures/showdown/${FORMAT}/synthetic-team.json`, "utf8"),
) as SetLike[];
const withItems = (items: Record<string, string>): SetLike[] =>
  golden.map((s) => ({ ...s, item: items[s.species] ?? s.item }));
const teams = {
  p1: withItems({ Charizard: "Charizardite Y", Venusaur: "Venusaurite" }),
  p2: withItems({ Gengar: "Gengarite", Blastoise: "Blastoisinite" }),
};
for (const [p, team] of Object.entries(teams)) {
  const problems = sim.TeamValidator.get(FORMAT).validateTeam(team);
  if (problems) throw new Error(`${p}: ${problems.join("; ")}`);
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Row {
  phase: string;
  enumerated: number;
  accepted: number;
  rejected: number;
  universe: number;
  universeAcceptedKeys: number;
  missing: number;
  cloneMs: number;
  validateMs: number;
  bruteMs: number;
  hiddenInfoFlags: boolean;
  doubleSwitch: boolean;
  mega: boolean;
}
const rows: Row[] = [];
const rejectReasons = new Map<string, number>();
const missingSamples: string[] = [];

const battles = Number(process.argv[2] ?? 20);
for (let b = 0; b < battles; b++) {
  const pick = rng(b + 1);
  const battle = new sim.Battle({ formatid: FORMAT, seed: `sodium,${b.toString(16).padStart(32, "0")}` });
  battle.setPlayer("p1", { name: "p1", team: sim.Teams.pack(teams.p1) });
  battle.setPlayer("p2", { name: "p2", team: sim.Teams.pack(teams.p2) });
  while (!battle.ended && battle.turn < 40) {
    let progressed = false;
    battle.sides.forEach((side: SimSide, i) => {
      if (battle.ended || !side.requestState || side.isChoiceDone()) return;
      const request = JSON.parse(JSON.stringify(side.activeRequest)) as Parameters<typeof enumerate>[0];
      const e = enumerate(request);
      if (e.phase === "wait") return;

      let t0 = performance.now();
      const verdicts = judge(sim.Battle, battle, i, e.candidates);
      const validateMs = performance.now() - t0;
      t0 = performance.now();
      judge(sim.Battle, battle, i, []);
      const cloneMs = performance.now() - t0;

      t0 = performance.now();
      const universe = bruteForceUniverse(side.requestState, side.pokemon.length);
      const brute = judge(sim.Battle, battle, i, universe);
      const bruteMs = performance.now() - t0;

      const accepted = verdicts.filter((v) => v.legal);
      const ourKeys = new Set(accepted.map((v) => v.key));
      const bruteKeys = new Set(brute.filter((v) => v.legal).map((v) => v.key));
      const missing = [...bruteKeys].filter((k) => !ourKeys.has(k));
      for (const v of verdicts.filter((x) => !x.legal)) {
        let reason = v.error.replace(/Your .*?'s /, "Your X's ").replace(/slot \d/, "slot N");
        const scope = e.uncertainty.maybeTrapped ? "maybeTrapped request" : "no maybe* flag";
        reason = `${reason} [${scope}]`;
        rejectReasons.set(reason, (rejectReasons.get(reason) ?? 0) + 1);
      }
      if (missing.length > 0 && missingSamples.length < 10) {
        const example = brute.find((v) => v.key === missing[0]);
        missingSamples.push(`${e.phase}: ${example?.choice} -> ${missing[0]}`);
      }
      rows.push({
        phase: e.phase,
        enumerated: e.candidates.length,
        accepted: accepted.length,
        rejected: verdicts.length - accepted.length,
        universe: universe.length,
        universeAcceptedKeys: bruteKeys.size,
        missing: missing.length,
        cloneMs,
        validateMs,
        bruteMs,
        hiddenInfoFlags: e.uncertainty.maybeTrapped || e.uncertainty.maybeDisabled || e.uncertainty.maybeLocked,
        doubleSwitch: e.candidates.some((c) => /^switch \d, switch \d$/.test(c)),
        mega: e.candidates.some((c) => c.includes(" mega")),
      });

      const legal = accepted.map((v) => v.choice);
      if (legal.length === 0) throw new Error(`no legal candidate for ${side.id} (${e.phase})`);
      const choice = legal[Math.floor(pick() * legal.length)] as string;
      if (!battle.choose(side.id, choice)) throw new Error(`live engine refused ${choice}`);
      progressed = true;
    });
    if (!progressed) break;
  }
}

const pct = (xs: number[], p: number): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))] ?? 0;
};
const summary: Record<string, unknown> = {};
for (const phase of ["preview", "move", "forceSwitch"]) {
  const r = rows.filter((x) => x.phase === phase);
  if (r.length === 0) continue;
  summary[phase] = {
    requests: r.length,
    enumerated_p50_max: [pct(r.map((x) => x.enumerated), 50), Math.max(...r.map((x) => x.enumerated))],
    accepted_p50_max: [pct(r.map((x) => x.accepted), 50), Math.max(...r.map((x) => x.accepted))],
    rejected_total: r.reduce((n, x) => n + x.rejected, 0),
    missing_vs_bruteforce_total: r.reduce((n, x) => n + x.missing, 0),
    universe_strings: r[0]?.universe,
    clone_ms_p50_p95: [pct(r.map((x) => x.cloneMs), 50), pct(r.map((x) => x.cloneMs), 95)],
    validate_all_ms_p50_p95: [pct(r.map((x) => x.validateMs), 50), pct(r.map((x) => x.validateMs), 95)],
    bruteforce_ms_p50_p95: [pct(r.map((x) => x.bruteMs), 50), pct(r.map((x) => x.bruteMs), 95)],
    with_mega: r.filter((x) => x.mega).length,
    with_double_switch: r.filter((x) => x.doubleSwitch).length,
    with_hidden_info_flags: r.filter((x) => x.hiddenInfoFlags).length,
  };
}
const round = (_k: string, v: unknown): unknown => (typeof v === "number" ? Math.round(v * 100) / 100 : v);
console.log(JSON.stringify({ battles, summary, rejectReasons: Object.fromEntries(rejectReasons), missingSamples }, round, 2));
