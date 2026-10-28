// The search's opponent worlds on the committed random-play channels (ADR-026 slice B; ADR-025 section 4):
// both prior arms through the same sampler, N worlds per move request with up to R sampler indices each, the
// builder's checks V1 to V5, and the counts. A report of counts only (evaluation tooling, no gate):
//
//   node packages/team-dataset/scripts/search-worlds.ts [--worlds 8] [--attempts 4] [--min-valid 4] [--every 1] [--no-trapping] [--out FILE]
//
// The belief is conditioned on the trapping flags of the own request (ticket #1127) unless `--no-trapping`.
//
// The decisions are the own move requests of both seats of the committed corpus, with the opponent half of the
// reader's state and its flags as the belief's only input; the real opponent teams are never read here. The
// random-play teams are not the generator's, so the generator-matched belief refuses some requests (counted).
import { writeFileSync } from "node:fs";
import {
  createWorldBuilder,
  createWorldSampler,
  loadEngine,
  loadMechanicsConfig,
  loadRulesSnapshot,
  loadWorldWriters,
  type PriorKind,
  PRIOR_KINDS,
  resolveFormat,
  SEARCH_HP_ROUNDING,
  searchStream,
  type WorldRequest,
} from "@areneon/showdown-bridge";
import { GOLDEN_FORMAT, goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { cases } from "../../showdown-bridge/test/world-builder-support.ts";
import { readerSha256 } from "../../public-state/src/index.ts";
import { createSearchWorlds } from "../src/search-worlds.ts";

const flag = (name: string, fallback: number, least = 1): number => {
  const at = process.argv.indexOf(`--${name}`);
  if (at < 0) return fallback;
  const value = Number(process.argv[at + 1]);
  if (!Number.isInteger(value) || value < least) throw new Error(`--${name} needs an integer of at least ${least}`);
  return value;
};
const settings = { worlds: flag("worlds", 8), attempts: flag("attempts", 4), min_valid: flag("min-valid", 4, 0) };
const every = flag("every", 1);
const trapping = !process.argv.includes("--no-trapping");
const out = process.argv.indexOf("--out") >= 0 ? process.argv[process.argv.indexOf("--out") + 1] : undefined;

const engine = loadEngine(process.env.SHOWDOWN_DIR || undefined);
const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, GOLDEN_FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const scenario_id = checkScenario(engine, loadMechanicsConfig(GOLDEN_FORMAT));
const decisions = cases().filter((_, i) => i % every === 0);

const report: Record<string, unknown> = {};
for (const arm of PRIOR_KINDS as readonly PriorKind[]) {
  const port = createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm });
  const builder = createWorldBuilder({ engine, format_id: GOLDEN_FORMAT, scenario_id, hp_rounding: SEARCH_HP_ROUNDING, writers: loadWorldWriters(engine, GOLDEN_FORMAT) });
  const sampler = createWorldSampler({ engine, vocabulary, format_id: GOLDEN_FORMAT, port, builder, settings, ...(trapping ? { trapping: { scenario_id } } : {}) });
  const began = performance.now();
  decisions.forEach(({ id, input }, i) => {
    const { opponent: _opponent, ...request } = input;
    sampler.sample(request as WorldRequest, searchStream({ seed: `sodium,${"00".repeat(16)}`, game: 0, player: input.state.player, policy_id: `search-worlds:${id}`, request_seq: i + 1 }));
  });
  report[arm] = { ...sampler.counts(), prior_hash: port.prior_hash, belief_config_hash: port.belief_config_hash, builder: builder.counts(), elapsed_ms: Math.round(performance.now() - began) };
}
const text = JSON.stringify({ decisions: decisions.length, settings, report }, null, 2);
if (out === undefined) console.log(text);
else writeFileSync(out, `${text}\n`);
