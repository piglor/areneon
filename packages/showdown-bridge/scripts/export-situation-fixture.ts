// Export the situational reading of requests of the committed corpora for the Python twin (ADR-010 erratum A11.2, #1366):
//   SHOWDOWN_DIR=<engine checkout> node packages/showdown-bridge/scripts/export-situation-fixture.ts [--threatened N] [--other N]
// Rewrites fixtures/situation/rows.json: per sampled request of the random-play and search-fidelity corpora the reduced request
// (which own Pokemon are active and fainted), the `probe-damage/2` arrays `threats`, `speeds` and `foe_priorities` that
// `candidateFeatures2` built for it, and `classifySlot`'s verdict for every living active slot. `tests/ml/learner/test_situation.py`
// recomputes the verdicts with `learner/situation.py` from the stored rows and requires them equal; `test/situation-fixture.test.ts`
// requires the stored verdicts to equal `classifySlot` on the stored rows. Heavy (one probe-backed feature build per request): run
// at low priority.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { candidateFeatures2 } from "../src/candidate-features-2.ts";
import { FeatureBuildError, type FeatureConfigs } from "../src/candidate-features.ts";
import { enumerateCandidates } from "../src/choices.ts";
import { loadDamageAwareConfig } from "../src/damage-aware-config.ts";
import { loadEngine } from "../src/engine.ts";
import { GOLDEN_DIR, GOLDEN_FORMAT, goldenTeamVocabulary } from "../src/golden.ts";
import { loadRulesSnapshot } from "../src/index.ts";
import { loadMechanicsConfig } from "../src/mechanics-config.ts";
import { playerChannel } from "../src/player-channel.ts";
import { loadRandomCorpus } from "../src/random-corpus.ts";
import { decodeRequest } from "../src/request.ts";
import { FIDELITY_DIR } from "../src/search-fidelity-corpus.ts";
import { classifySlot } from "../src/situation.ts";

const arg = (name: string, fallback: number): number => {
  const at = process.argv.indexOf(name);
  return at < 0 ? fallback : Number(process.argv[at + 1]);
};
const WANT_THREATENED = arg("--threatened", 120);
const WANT_OTHER = arg("--other", 120);
const WANT_NON_MOVE = 6;

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const configs: FeatureConfigs = {
  vocabulary,
  config: loadDamageAwareConfig("v1", vocabulary),
  mechanics: loadMechanicsConfig(GOLDEN_FORMAT),
  snapshot: loadRulesSnapshot(engine.commit),
  data_class: "synthetic",
};

interface Entry {
  id: string;
  request: { teamPreview?: boolean; forceSwitch?: boolean[]; wait?: boolean; active?: Record<string, never>[]; side: { pokemon: { active: boolean; condition: string }[] } };
  rows: { threats: unknown[]; speeds: unknown[]; foe_priorities: unknown[] };
  slots: { position: number; situation: string }[];
  threatened: boolean;
}

const battles = [...loadRandomCorpus().map((b) => ({ b, dir: "random" })), ...loadRandomCorpus(FIDELITY_DIR).map((b) => ({ b, dir: "search-fidelity" }))];
const threatened: Entry[] = [];
const other: Entry[] = [];
const nonMove: Entry[] = [];
let built = 0;
let refused = 0;
for (const { b, dir } of battles) {
  for (const player of ["p1", "p2"] as const) {
    const log = b.channels[player];
    const team = b.teams[player];
    const requests = log.lines.flatMap((line, i) => (line.startsWith("|request|") ? [i] : []));
    for (const [n, i] of requests.entries()) {
      // A stride per battle keeps the sample spread over the corpora; the wanted counts cap it.
      if (n % 2 !== 0 && threatened.length >= WANT_THREATENED / 2) continue;
      const body = JSON.parse((log.lines[i] as string).slice("|request|".length)) as Record<string, unknown> & { side: { pokemon: { active: boolean; condition: string }[] } };
      const decoded = decodeRequest(JSON.stringify(body));
      const id = `${dir}/${b.id}/${player}/${n}`;
      const reduced = {
        ...(body.teamPreview === undefined ? {} : { teamPreview: true }),
        ...(body.forceSwitch === undefined ? {} : { forceSwitch: body.forceSwitch as boolean[] }),
        ...(body.wait === undefined ? {} : { wait: true }),
        ...(body.active === undefined ? {} : { active: (body.active as unknown[]).map(() => ({})) }),
        side: { pokemon: body.side.pokemon.map((p) => ({ active: p.active, condition: p.condition })) },
      } as Entry["request"];
      if (decoded.phase === "preview" || decoded.phase === "wait") {
        if (nonMove.length < WANT_NON_MOVE && n % 3 === 0) nonMove.push({ id, request: reduced, rows: { threats: [], speeds: [], foe_priorities: [] }, slots: [], threatened: false });
        continue;
      }
      const wantMore = decoded.phase === "forceSwitch" ? nonMove.length < WANT_NON_MOVE : threatened.length < WANT_THREATENED || other.length < WANT_OTHER;
      if (!wantMore) continue;
      let features;
      try {
        const candidates = enumerateCandidates(decoded).candidates.map(({ choice, pending }) => ({ choice, pending }));
        const channel = playerChannel({ ...log, lines: log.lines.slice(0, i + 1) });
        features = candidateFeatures2(engine, configs, channel, team, candidates);
        built++;
      } catch (error) {
        if (!(error instanceof FeatureBuildError)) throw error;
        refused++;
        continue;
      }
      const rows = { threats: features.threats, speeds: features.speeds, foe_priorities: features.foe_priorities };
      const slots =
        decoded.phase === "move"
          ? body.side.pokemon.flatMap((p, position) => (p.active && !/ fnt$/.test(p.condition) ? [{ position, situation: classifySlot(rows as never, position).situation }] : []))
          : [];
      const entry: Entry = { id, request: reduced, rows, slots, threatened: slots.some((s) => s.situation === "threatened") };
      if (decoded.phase !== "move") nonMove.push(entry);
      else if (entry.threatened) {
        if (threatened.length < WANT_THREATENED) threatened.push(entry);
      } else if (other.length < WANT_OTHER) other.push(entry);
    }
  }
  if (threatened.length >= WANT_THREATENED && other.length >= WANT_OTHER && nonMove.length >= WANT_NON_MOVE) break;
}
const entries = [...threatened, ...other, ...nonMove];
const file = join(GOLDEN_DIR, "..", "..", "situation", "rows.json");
writeFileSync(file, `${JSON.stringify({ schema_version: 1, data_class: "synthetic", provenance: "classifySlot over probe-damage/2 rows of the committed random-play and search-fidelity corpora (scripts/export-situation-fixture.ts)", format: GOLDEN_FORMAT, entries })}\n`);
console.log(`wrote ${entries.length} entries (${threatened.length} threatened, ${other.length} other, ${nonMove.length} not a move request); built ${built}, refused ${refused}`);
