// Writes the test cases of the vendored fork's declared-state layer (`tests/declared.rs`, `tests/fixtures/declared/cases.jsonl`): each
// line is a query in the layer's own form (the sets numbered from 0 and the query over them) with the answer the PINNED PROBE gave
// it (an answer, an `unsupported` outcome, or `abstain` for a query the probe refuses by throwing). The generator refuses to write a case the
// native layer answers differently, so the file is a statement of equality at the pin it was made at.
//
//   node packages/team-dataset/scripts/native-probe-cases.ts --out CASES.jsonl [--bin PATH --dex PATH] [--speed N] [--seed S]
import { readFileSync, writeFileSync } from "node:fs";
import { jcs, loadEngine, mechanicsQueryHash, nativeAbstained, nativeCase, type ParsedMechanicsQuery, probeBackend, readMechanicsQuery, type Settled } from "@areneon/showdown-bridge";
import { COMMITTED_CORPORA, corpusBattles, remembering } from "../src/differential-mechanics.ts";
import { builderQueries, moveSweep, nativeHarness, speedSweep } from "../src/differential-native.ts";
import { interactionSweep, specQuery, targetedSpecs } from "../src/differential-kernel.ts";

const FORMAT = "gen9championsvgc2026regmc";
const args = process.argv.slice(2);
const flag = (name: string, fallback?: string): string => {
  const i = args.indexOf(`--${name}`);
  const value = i < 0 ? fallback : args[i + 1];
  if (value === undefined) {
    console.error(`native-probe-cases: --${name} is required`);
    process.exit(2);
  }
  return value;
};
const engine = loadEngine();
const command = { bin: flag("bin", process.env.ARENEON_NINJASK_PROBE), dex: flag("dex", process.env.ARENEON_NINJASK_DATA === undefined ? undefined : `${process.env.ARENEON_NINJASK_DATA}/format/${FORMAT}.json`) };
const h = nativeHarness(engine, "0".repeat(40), command);
const oracle = probeBackend(engine, { config: h.configs.oracle, snapshot: h.configs.snapshot });
const context = { engine, config: h.configs.oracle, snapshot: h.configs.snapshot };

// Every hand-worked damage case of the repository's fixtures, then a stride over each generated corpus (the stride keeps the file small).
const queries: { id: string; query: ParsedMechanicsQuery }[] = [];
const fixture = JSON.parse(readFileSync(new URL(`../../../fixtures/mechanics/${FORMAT}/damage/cases.json`, import.meta.url), "utf8")) as { cases: { id: string; query: unknown }[] };
for (const c of fixture.cases) queries.push({ id: `fixture:damage:${c.id}`, query: readMechanicsQuery(c.query) });
for (const [i, spec] of targetedSpecs().entries()) queries.push({ id: `targeted:${i}`, query: specQuery(engine, spec, FORMAT) });
const stride = <T>(xs: readonly T[], every: number): T[] => xs.filter((_, i) => i % every === 0);
const sweep = interactionSweep(engine, FORMAT).flatMap((spec) => {
  try {
    return [{ id: `interaction:${spec.id}`, query: specQuery(engine, spec, FORMAT) }];
  } catch {
    return [];
  }
});
queries.push(...stride(sweep, 45));
queries.push(...stride(moveSweep(engine, FORMAT), 40).map((q, i) => ({ id: `move-sweep:${i}`, query: q })));
queries.push(...speedSweep(engine, FORMAT, Number(flag("speed", "500")), Number(flag("seed", "20261125"))).map((q, i) => ({ id: `speed-sweep:${i}`, query: q })));
const corpus = COMMITTED_CORPORA[0] as (typeof COMMITTED_CORPORA)[number];
queries.push(...stride(builderQueries(h, corpusBattles(corpus.dir, 3), remembering(oracle)), 6).map((q, i) => ({ id: `builder:${i}`, query: q })));

// Queries earlier runs found a difference on (one per line, `{q}`), kept as regression cases.
for (const file of args.includes("--extra") ? flag("extra").split(",") : []) {
  // A file of queries earlier runs found a difference on: every query of a small file, one in six of a large one.
  const lines = readFileSync(file, "utf8").split("\n").filter(Boolean);
  const stride = lines.length > 40 ? 6 : 1;
  for (const [i, line] of lines.entries()) if (i % stride === 0) queries.push({ id: `found:${file.replace(/.*\//, "")}:${i}`, query: readMechanicsQuery((JSON.parse(line) as { q: unknown }).q) });
}
const lines: string[] = [];
const seen = new Set<string>();
const tally: Record<string, number> = {};
for (const { id, query } of queries) {
  const key = mechanicsQueryHash(query);
  if (seen.has(key)) continue;
  seen.add(key);
  const wire = nativeCase(context, query);
  if (wire === null) continue;
  const o = oracle.run([query])[0] as Settled;
  const n = h.native.run([query])[0] as Settled;
  let expected: unknown;
  if (!o.ok) {
    // The probe refuses (throws): the layer must not answer.
    if (!nativeAbstained(n)) throw new Error(`${id}: the probe throws and the native layer answers`);
    expected = { kind: "abstain" };
  } else {
    const result = o.result as unknown as { kind: string; answer?: unknown; reason?: string; position?: number | null };
    if (nativeAbstained(n)) continue; // the layer leaves it to the probe: not a case of equality
    if (!n.ok || jcs(n.result as never) !== jcs(o.result as never)) throw new Error(`${id}: the native layer differs from the probe`);
    expected = result.kind === "computed" ? { kind: "computed", answer: result.answer } : { kind: "unsupported", reason: result.reason, position: result.position ?? null };
  }
  tally[(expected as { kind: string }).kind] = (tally[(expected as { kind: string }).kind] ?? 0) + 1;
  lines.push(JSON.stringify({ id, sets: wire.sets, query: wire.query, expected }));
}
writeFileSync(flag("out"), `${lines.join("\n")}\n`);
// The mechanics config's KO blockers: the cases' KO counts depend on them, so the fork's test sets the same list.
writeFileSync(flag("out").replace(/[^/]*$/, "blockers.json"), `${JSON.stringify(h.configs.oracle.ko_blockers)}\n`);
console.log(JSON.stringify({ cases: lines.length, ...tally }));
h.native.close();
