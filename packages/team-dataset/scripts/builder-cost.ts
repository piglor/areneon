// Per-decision cost of the `probe-damage/2` feature builder with the TypeScript probe and with the native backend (#1534, the
// call-tree figures of ADR-031). Read-only; prints one JSON line. Run once per mode, one process at a time, on a machine whose load
// you state with the result:
//
//   ARENEON_NINJASK_PROBE=... ARENEON_NINJASK_DATA=... node packages/team-dataset/scripts/builder-cost.ts ts|native [BATTLES]
//
// The decisions are every non-wait request of both seats of the first BATTLES (default 10) battles of the random corpus, built by
// `buildFeaturesG` with one memo per game and seat, queries answered one at a time. In `native` mode a query the native layer
// abstains on is answered by the TypeScript probe and that time counts as answering time. `ask_ms` is the time spent answering
// queries, `builder_own_ms` the rest of the decision (reader, assumed sets, query construction, rows). There is no warm-up pass
// and no repeated measurement: the first mode run pays the one-time schema compilation, so figures of two modes are comparable
// only to a rough factor, and the native mode, which runs after the TS mode in the committed figures, benefits from a warm disk cache.
// `node --cpu-prof` on this script gives the call tree.
import { loadavg } from "node:os";
import { performance } from "node:perf_hooks";
import { buildFeaturesG, createFeatureMemo, decodeRequest, driveSync, enumerateCandidates, loadEngine, playerChannel, probeBackend, type PokemonSet } from "@areneon/showdown-bridge";
import { COMMITTED_CORPORA, corpusBattles, remembering } from "../src/differential-mechanics.ts";
import { NATIVE_FEATURE_SET, nativeHarness } from "../src/differential-native.ts";

const FORMAT = "gen9championsvgc2026regmc";
const mode = process.argv[2];
if (mode !== "ts" && mode !== "native") {
  console.error("builder-cost: ts | native [BATTLES]");
  process.exit(2);
}
const battleCount = Number(process.argv[3] ?? 10);
const bin = process.env.ARENEON_NINJASK_PROBE;
const data = process.env.ARENEON_NINJASK_DATA;
if (bin === undefined || data === undefined) {
  console.error("builder-cost: ARENEON_NINJASK_PROBE and ARENEON_NINJASK_DATA are required");
  process.exit(2);
}
const commit = process.env.ARENEON_NINJASK_COMMIT ?? "0".repeat(40);

const engine = loadEngine();
const h = nativeHarness(engine, commit, { bin, dex: `${data}/format/${FORMAT}.json` });
const tsProbe = remembering(probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot }));
type Settled = { ok: true; result: unknown } | { ok: false; error: unknown };
const backend = mode === "native" ? h.native : tsProbe;
const battles = corpusBattles((COMMITTED_CORPORA[0] as { dir: string }).dir, battleCount);

let decisions = 0;
let queries = 0;
let askMs = 0;
let totalMs = 0;
for (const battle of battles) {
  for (const player of ["p1", "p2"] as const) {
    const lines = battle.channels[player].lines;
    const team = battle.teams[player] as PokemonSet[];
    const memo = createFeatureMemo();
    lines.forEach((line: string, i: number) => {
      if (!line.startsWith("|request|")) return;
      const decoded = decodeRequest(line.slice("|request|".length));
      if (decoded.phase === "wait") return;
      const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
      const channel = playerChannel({ schema_version: 1, data_class: "synthetic", player, lines: lines.slice(0, i + 1) });
      const ask = (query: Parameters<typeof backend.run>[0][number]): unknown => {
        const start = performance.now();
        queries += 1;
        let settled = backend.run([query])[0] as Settled;
        if (mode === "native" && !settled.ok && (settled.error as { entity?: string }).entity !== undefined) settled = tsProbe.run([query])[0] as Settled;
        askMs += performance.now() - start;
        if (!settled.ok) throw settled.error;
        return settled.result;
      };
      const begun = performance.now();
      try {
        driveSync(buildFeaturesG(engine, h.configs.sets[NATIVE_FEATURE_SET], channel, team, candidates, memo, NATIVE_FEATURE_SET), ask as never);
        decisions += 1;
      } catch {
        // a request the builder itself refuses (reader_flagged, assumed_set) is no decision
      }
      totalMs += performance.now() - begun;
    });
  }
}
const per = (x: number): number => Math.round((x / decisions) * 100) / 100;
console.log(JSON.stringify({ mode, battles: battleCount, decisions, queries, queries_per_decision: Math.round((queries / decisions) * 10) / 10, per_decision_ms: per(totalMs), ask_ms: per(askMs), builder_own_ms: per(totalMs - askMs), us_per_query_asked: Math.round((askMs / queries) * 1000), load: loadavg()[0] }));
h.native.close();
