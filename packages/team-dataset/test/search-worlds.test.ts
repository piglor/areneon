// The search's opponent worlds with the real model (ADR-026 slice B): ADR-025's sampler asked by index
// (`i * R + r`), both prior arms, the world builder's checks, the counts. The committed random-play channels are
// the decisions; their teams are not the generator's, so some decisions are refused by the belief or have fewer
// valid worlds: that is counted, and the counts are what these tests read.
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { PublicSide } from "../../public-state/src/index.ts";
import { belief, createSampler, worldStream } from "@areneon/opponent-model";
import {
  createWorldBuilder,
  createWorldSampler,
  EVIDENCE_CLASSES,
  GENERATOR_MATCHED_LABEL,
  loadMechanicsConfig,
  loadRulesSnapshot,
  loadWorldWriters,
  type PriorKind,
  readBeliefConfig,
  readUsagePrior,
  resolveFormat,
  ROLE_SETS_LABEL,
  SEARCH_HP_ROUNDING,
  searchStream,
  teamVocabularyHash,
  type WorldRequest,
  type WorldSampler,
} from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { cases } from "../../showdown-bridge/test/world-builder-support.ts";
import { createSearchWorlds, SearchWorldsError, searchWorldsFiles } from "../src/search-worlds.ts";
import { generatorSpreadSampler } from "../src/spread-sampler.ts";
import { config, engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const files = searchWorldsFiles(FORMAT);
const settings = { worlds: 8, attempts: 4, min_valid: 4 };
const scratch = mkdtempSync(join(tmpdir(), "search-worlds-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const portOf = (arm: PriorKind, f = files) => createSearchWorlds({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm }, f);
const samplerOf = (arm: PriorKind): WorldSampler =>
  createWorldSampler({
    engine,
    vocabulary,
    format_id: FORMAT,
    port: portOf(arm),
    builder: createWorldBuilder({ engine, format_id: FORMAT, scenario_id: checkScenario(engine, loadMechanicsConfig(FORMAT)), hp_rounding: SEARCH_HP_ROUNDING, writers: loadWorldWriters(engine, FORMAT) }),
    settings,
  });
const requestOf = ({ input }: { input: ReturnType<typeof cases>[number]["input"] }): WorldRequest => {
  const { opponent: _opponent, ...rest } = input;
  return rest;
};
const streamOf = (id: string, seq = 1) => searchStream({ seed: `sodium,${"00".repeat(16)}`, game: 0, player: "p1", policy_id: `test:${id}`, request_seq: seq });

// Every fourth committed move request of both seats (the whole corpus is the bridge's world-builder test).
const decisions = cases().filter((_, i) => i % 4 === 0);

describe("the two prior arms", () => {
  it("carry their labels, hashes and identities; the arms differ in prior and identity, not in belief config", () => {
    const a = portOf("generator-matched");
    const b = portOf("role-sets");
    expect(a.prior_label).toBe(GENERATOR_MATCHED_LABEL);
    expect(a.prior_label).toContain("an upper bound for this evaluation");
    expect(b.prior_label).toBe(ROLE_SETS_LABEL);
    expect(a.prior_kind).toBe("generator-matched");
    expect(b.prior_kind).toBe("role-sets");
    expect(a.prior_format_id).toBe(FORMAT);
    expect(a.prior_hash).not.toBe(b.prior_hash);
    expect(a.identity_hash).not.toBe(b.identity_hash);
    expect(a.belief_config_hash).toBe(b.belief_config_hash);
  });

  it("refuse a prior that is not derived from the committed inputs", () => {
    const dir = join(scratch, "doctored");
    mkdirSync(dir, { recursive: true });
    const model = join(dir, "model");
    cpSync(files.model_dir, model, { recursive: true });
    const edit = (file: string, change: (v: Record<string, unknown>) => void): string => {
      const target = join(model, file);
      const v = JSON.parse(readFileSync(target, "utf8")) as Record<string, unknown>;
      change(v);
      writeFileSync(target, JSON.stringify(v));
      return target;
    };
    const f = { ...files, model_dir: model, generator_prior_file: join(model, "generator-matched.uniform-v1.json"), role_sets_prior_file: join(model, "role-sets.fill-uniform-v1.json") };
    // another generator config than the committed one
    edit("generator-matched.uniform-v1.json", (v) => ((v.source as { generator_config_hash: string }).generator_config_hash = "0".repeat(64)));
    expect(() => portOf("generator-matched", f)).toThrow(SearchWorldsError);
    cpSync(files.model_dir, model, { recursive: true });
    // a role-sets record filled from another prior, and over another table
    edit("role-sets.fill-uniform-v1.json", (v) => ((v.source as { fill_prior_hash: string }).fill_prior_hash = "0".repeat(64)));
    expect(() => portOf("role-sets", f)).toThrow(/filled from another prior/);
    cpSync(files.model_dir, model, { recursive: true });
    edit("role-sets.fill-uniform-v1.json", (v) => ((v.source as { role_set_table_hash: string }).role_set_table_hash = "0".repeat(64)));
    expect(() => portOf("role-sets", f)).toThrow(/role-set table/);
    // a role-sets file that holds the generator-matched record
    cpSync(files.model_dir, model, { recursive: true });
    cpSync(join(model, "generator-matched.uniform-v1.json"), join(model, "role-sets.fill-uniform-v1.json"));
    expect(() => portOf("role-sets", f)).toThrow(/not of kind role-sets/);
    // the generator arm does not read the role-sets record at all
    expect(portOf("generator-matched", f).prior_kind).toBe("generator-matched");
  });
});

describe("the belief's own refusals", () => {
  it("are returned by code, and an error that is not the model's is thrown", () => {
    const port = portOf("generator-matched");
    const c = decisions.find((d) => d.input.flags.length > 0) as (typeof decisions)[number];
    expect(port.open({ opponent: c.input.state.opponent, flags: c.input.flags })).toEqual({ kind: "refused", code: "reader_flagged" });
    expect(() => port.open({ opponent: {} as PublicSide, flags: [] })).toThrow();
  });
});

describe("the draw by index", () => {
  it("is ADR-025's sampleWorlds: draw(stream, j) equals world j of sampleWorlds, redraws included", () => {
    const prior = readUsagePrior(JSON.parse(readFileSync(files.generator_prior_file, "utf8")), vocabulary);
    const beliefConfig = readBeliefConfig(JSON.parse(readFileSync(join(files.model_dir, "belief.v1.json"), "utf8")), vocabulary);
    const view = { ...vocabulary, vocabulary_hash: teamVocabularyHash(vocabulary) };
    const model = createSampler({ legality: { legal: () => true }, spreads: generatorSpreadSampler(vocabulary, config) });
    // the port uses the real legality; compare on a stream where no set is illegal, i.e. redraws are 0 in both, and equal worlds
    const c = decisions.find((d) => d.input.state.opponent.roster.length === 6 && d.input.flags.length === 0) as (typeof decisions)[number];
    const stream = streamOf("equal");
    const direct = belief({ prior, vocabulary: view, config: beliefConfig, opponent: c.input.state.opponent, flags: c.input.flags });
    const sampled = model.sampleWorlds(direct, stream, 12);
    const opened = portOf("generator-matched").open({ opponent: c.input.state.opponent, flags: c.input.flags });
    if (opened.kind !== "source") throw new Error("the belief refused");
    let same = 0;
    for (const w of sampled.worlds) {
      const drawn = opened.source.draw(stream, w.index);
      if (drawn.redraws === w.redraws) {
        expect(JSON.stringify(drawn.world)).toBe(JSON.stringify(w.world));
        same += 1;
      }
    }
    expect(same).toBeGreaterThanOrEqual(10);
    // and the per-index stream is the model's
    expect(worldStream(stream, 5)).toEqual(worldStream(stream, 5));
  });
});

describe.each(["generator-matched", "role-sets"] as const)("worlds of the %s arm on committed channels", (arm) => {
  const sampler = samplerOf(arm);
  const outputs = decisions.map((d, i) => sampler.sample(requestOf(d), streamOf(d.id, i)));

  it("redraws by index, uses no index twice, and counts every decision and world", () => {
    const c = sampler.counts();
    expect(c.decisions).toBe(decisions.length);
    expect(c.prior_kind).toBe(arm);
    const refused = Object.values(c.belief_refusals).reduce((a, b) => a + b, 0);
    expect(c.searched + c.below_min_valid + refused).toBe(c.decisions);
    expect(c.worlds_asked).toBe((c.decisions - refused) * settings.worlds);
    expect(c.worlds_valid + c.worlds_refused).toBe(c.worlds_asked);
    expect(c.valid_histogram.reduce((a, b) => a + b, 0)).toBe(c.decisions - refused);
    expect(c.valid_histogram.reduce((a, n, k) => a + n * k, 0)).toBe(c.worlds_valid);
    expect(c.attempts - c.worlds_asked).toBe(c.redraws);
    for (const out of outputs) {
      if (out.kind !== "worlds") continue;
      expect(out.valid.length >= settings.min_valid).toBe(out.searched);
      expect(new Set(out.valid.map((w) => w.world)).size).toBe(out.valid.length);
      for (const w of out.valid) {
        expect(w.attempt).toBeLessThan(settings.attempts);
        expect(w.index).toBe(w.world * settings.attempts + w.attempt);
      }
    }
    // Coverage by evidence class sums to the decisions, and the class table is the report.
    const rows = Object.entries(c.by_evidence);
    expect(rows.map(([name]) => name)).toEqual([...EVIDENCE_CLASSES]);
    expect(rows.reduce((a, [, r]) => a + r.decisions, 0)).toBe(c.decisions);
    expect(rows.reduce((a, [, r]) => a + r.searched, 0)).toBe(c.searched);
    expect(rows.reduce((a, [, r]) => a + r.worlds_valid, 0)).toBe(c.worlds_valid);
    expect(c.worlds_valid).toBeGreaterThan(0);
    expect(c.searched).toBeGreaterThan(0);
    console.log(`${arm}: ${JSON.stringify({ label: c.prior_label, decisions: c.decisions, searched: c.searched, below_min_valid: c.below_min_valid, belief_refusals: c.belief_refusals, worlds_valid: c.worlds_valid, worlds_refused: c.worlds_refused, redraws: c.redraws, legality_redraws: c.legality_redraws, draw_refusals: c.draw_refusals, refusals: c.refusals, refusal_check_class: c.refusal_check_class, redrawn: c.redrawn, valid_histogram: c.valid_histogram, by_evidence: c.by_evidence })}`);
  }, 300_000);

  it("gives the same worlds on the same stream and other worlds on another", () => {
    const pick = decisions.filter((_, i) => i % 6 === 0).slice(0, 12);
    const run = (salt: string): string[] => {
      const again = samplerOf(arm);
      return pick.map((d, i) => {
        const out = again.sample(requestOf(d), streamOf(d.id + salt, i));
        return JSON.stringify(out.kind === "worlds" ? out.valid.map((w) => [w.index, w.built.opponent_order, w.built.battle.sides.map((s) => (s.pokemon as unknown as { set: unknown }[]).map((p) => JSON.stringify(p.set)))]) : out.code);
      });
    };
    const first = run("");
    expect(run("")).toEqual(first);
    expect(run("-other")).not.toEqual(first);
  }, 300_000);
});

describe("the arms are one sampler over two priors", () => {
  it("send the same decisions through the same builder checks and report the same evidence classes", () => {
    const a = samplerOf("generator-matched");
    const b = samplerOf("role-sets");
    for (const d of decisions.slice(0, 40)) {
      a.sample(requestOf(d), streamOf(d.id));
      b.sample(requestOf(d), streamOf(d.id));
    }
    const classes = (s: WorldSampler) => Object.values(s.counts().by_evidence).map((r) => r.decisions);
    expect(classes(a)).toEqual(classes(b));
  }, 300_000);
});
