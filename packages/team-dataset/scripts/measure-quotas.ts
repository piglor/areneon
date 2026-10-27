// A MEASUREMENT SCRIPT, not part of the tests or CI (ADR-025 erratum E2, #1167): how far the validator's rejections and the label
// quotas move the real team distribution away from the proposal's marginals, which the generator-matched prior states. Reproduces
// the figures in the erratum:
//   node packages/team-dataset/scripts/measure-quotas.ts [--draws 20000] [--pilot] [--dataset DIR] [--moves fakeout,...] [--key HEX64]
// --draws N       N raw proposals from the committed generator config (seeds of plan "quota-measure"), each judged by the pinned
//                 validator exactly as the generator judges it (`Drawer`): the acceptance rate, the Mega-forme share of the
//                 proposed and of the accepted sets, and for the accepted sets the root-mean-square z of every (roster entry,
//                 move) inclusion against the prior's k/n (and the worst one).
// --pilot         the same acceptance rate over the config's `pilot_draws` pilot draws (the stream the dataset plans from).
// --dataset DIR   a generated dataset (records.<split>.jsonl, or the committed teams-<split>.jsonl): for each --moves id (default
//                 every move) the observed count of sets holding it against the prior's expectation (sum over the sets of k/n of the
//                 species' pool) with z; the rms z over all moves; the sets are read as written (the Mega-forme species kept).
// It reads the engine, config and vocabulary through ../test/support.ts. Nothing is written. It takes about a minute for 20,000 draws.
// The 1,747-team figure of the erratum came from a generated dataset that is not committed; the committed fixture dataset gives smaller counts.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { importPaste, type PokemonSet } from "@areneon/showdown-bridge";
import { Drawer } from "../src/draw.ts";
import { Proposal } from "../src/proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { config, engine, format, labelSet, table, vocabulary } from "../test/support.ts";

const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const k = args.indexOf(name);
  return k < 0 ? undefined : args[k + 1];
};
const key = flag("--key") ?? "11".repeat(32);
const draws = Number(flag("--draws") ?? 20000);
const toId = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");
const speciesOf = new Map(table.species.map((s) => [s.name, s.id]));
const entries = new Map(vocabulary.roster.map((r) => [r.id, r]));
const slots = config.moves_per_set as number;
const fixed = (x: number, d = 2): number => Number(x.toFixed(d));

const proposal = new Proposal(vocabulary, table, config);
const drawer = new Drawer({ engine, format, proposal, labelSet, generationKey: key });

// inclusion counts per (roster entry, move) over a list of sets, and their z against k/n
function moveZ(sets: readonly { species: string; moves: readonly string[] }[]): { rms: number; worst: number; cells: number } {
  const perEntry = new Map<string, { n: number; hits: Map<string, number> }>();
  for (const s of sets) {
    const e = perEntry.get(s.species) ?? { n: 0, hits: new Map<string, number>() };
    e.n += 1;
    for (const m of s.moves) e.hits.set(m, (e.hits.get(m) ?? 0) + 1);
    perEntry.set(s.species, e);
  }
  let sum = 0;
  let worst = 0;
  let cells = 0;
  for (const [id, e] of perEntry) {
    const pool = (entries.get(id) as NonNullable<ReturnType<typeof entries.get>>).moves;
    const p = Math.min(slots, pool.length) / pool.length;
    if (p >= 1 || e.n < 30) continue;
    for (const m of pool) {
      const z = ((e.hits.get(m) ?? 0) - e.n * p) / Math.sqrt(e.n * p * (1 - p));
      sum += z * z;
      worst = Math.max(worst, Math.abs(z));
      cells += 1;
    }
  }
  return { rms: fixed(Math.sqrt(sum / cells)), worst: fixed(worst), cells };
}

if (draws > 0) {
  const stream = { config_hash: "d".repeat(64), plan_id: "quota-measure" };
  let accepted = 0;
  const proposedSets: { species: string; moves: string[] }[] = [];
  const acceptedSets: { species: string; moves: string[] }[] = [];
  for (let k = 0; k < draws; k++) {
    const sets = proposal.propose(engineRng(engine, drawSeed(key, stream, k))).map((s) => ({ species: speciesOf.get(s.species) as string, moves: s.moves.map(toId) }));
    proposedSets.push(...sets);
    if (drawer.draw(stream, k).status === "accepted") {
      accepted += 1;
      acceptedSets.push(...sets);
    }
  }
  const mega = (sets: readonly { species: string }[]): number => sets.filter((s) => entries.get(s.species)?.required_item !== undefined).length / sets.length;
  console.log(JSON.stringify({ draws, accepted, accepted_share: fixed(accepted / draws, 4), mega_share_proposed: fixed(mega(proposedSets), 4), mega_share_accepted: fixed(mega(acceptedSets), 4), moves_proposed: moveZ(proposedSets), moves_accepted: moveZ(acceptedSets) }));
}

if (args.includes("--pilot")) {
  let rejected = 0;
  for (let k = 0; k < config.pilot_draws; k++) if (drawer.pilot("c".repeat(64), k).status === "rejected") rejected += 1;
  console.log(JSON.stringify({ pilot_draws: config.pilot_draws, rejected, rejected_share: fixed(rejected / config.pilot_draws, 4) }));
}

const dataset = flag("--dataset");
if (dataset !== undefined) {
  const files = readdirSync(dataset).filter((f) => /^(records\.[a-z]+|teams-[a-z]+)\.jsonl$/.test(f)).sort();
  const sets: { species: string; moves: string[] }[] = [];
  let teams = 0;
  for (const file of files) {
    for (const line of readFileSync(join(dataset, file), "utf8").split("\n")) {
      if (line === "") continue;
      teams += 1;
      for (const set of importPaste(engine, (JSON.parse(line) as { paste: string }).paste) as PokemonSet[]) sets.push({ species: speciesOf.get(set.species) as string, moves: set.moves.map(toId) });
    }
  }
  const expected = new Map<string, number>();
  const observed = new Map<string, number>();
  for (const s of sets) {
    const pool = (entries.get(s.species) as NonNullable<ReturnType<typeof entries.get>>).moves;
    const p = Math.min(slots, pool.length) / pool.length;
    for (const m of pool) expected.set(m, (expected.get(m) ?? 0) + p);
    for (const m of s.moves) observed.set(m, (observed.get(m) ?? 0) + 1);
  }
  const wanted = flag("--moves")?.split(",") ?? [...expected.keys()].sort();
  const rows = wanted.map((m) => {
    const e = expected.get(m) ?? 0;
    // the variance of a sum of independent inclusions is approximated by the expectation's binomial form per set
    const z = (( observed.get(m) ?? 0) - e) / Math.sqrt(Math.max(e * (1 - e / sets.length), 1e-9));
    return { move: m, observed: observed.get(m) ?? 0, expected: fixed(e, 1), z: fixed(z) };
  });
  const all = [...expected.keys()].map((m) => {
    const e = expected.get(m) as number;
    return (((observed.get(m) ?? 0) - e) / Math.sqrt(Math.max(e * (1 - e / sets.length), 1e-9))) ** 2;
  });
  console.log(JSON.stringify({ dataset, teams, sets: sets.length, moves_rms_z: fixed(Math.sqrt(all.reduce((a, b) => a + b, 0) / all.length)), rows }));
}
