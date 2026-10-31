// The opponent sheet block with the real model (ADR-025 erratum E4, ADR-027 slice 3): the composition root, the builder over the
// committed random-play channels, the refusals counted per code, the memo, the golden digest. Hidden-information tests of the
// builder itself (invariance, negative control, canary on engine battles) are in `packages/opponent-model/test/hidden.test.ts`.
import { createHash } from "node:crypto";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { OPPONENT_SHEET_FORMAT, SHEET_K, sheetFeatureSetHash } from "@areneon/opponent-model";
import { createSheetBuilder, loadRulesSnapshot, readOpponentSheet, refusedSharePpm, resolveFormat, teamVocabularyHash } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { readerSha256 } from "../../public-state/src/index.ts";
import { checkScenario } from "../../showdown-bridge/src/mechanics-scenario.ts";
import { loadMechanicsConfig } from "../../showdown-bridge/src/mechanics-config.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { cases } from "../../showdown-bridge/test/world-builder-support.ts";
import { createBeliefFeatures, evidenceKey } from "../src/belief-features.ts";
import { SearchWorldsError, searchWorldsFiles } from "../src/search-worlds.ts";
import { engine, FORMAT } from "./support.ts";

const vocabulary = goldenTeamVocabulary();
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const files = searchWorldsFiles(FORMAT);
const scratch = mkdtempSync(join(tmpdir(), "belief-features-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const portOf = (f = files) => createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest }, f);
/** SHA-256 of the JSON of the blocks of the sampled decisions (source hash left out) under the committed prior and belief config. */
const GOLDEN_BLOCKS = "bd900a72681b70ed867a2b3ce187c3553c7658a6092decca0a18d7339b4a8b45";
const scenario_id = checkScenario(engine, loadMechanicsConfig(FORMAT));
// Every fourth committed move request of both seats (the same sample the search worlds use).
const decisions = cases().filter((_, i) => i % 4 === 0);

describe("the port", () => {
  const port = portOf();

  it("names the belief identity, the committed prior and the feature set", () => {
    expect(port.prior_format_id).toBe(FORMAT);
    expect(port.k).toEqual(SHEET_K);
    expect(port.features_hash).toBe(sheetFeatureSetHash(port.identity_hash, SHEET_K));
    expect(port.features_hash).not.toBe(port.identity_hash);
    expect(OPPONENT_SHEET_FORMAT).toBe("opponent-sheet-builder/1");
  });

  it("a k value, the belief identity or the reader changes the feature set hash", () => {
    expect(createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest, k: { ...SHEET_K, moves: 6 } }, files).features_hash).not.toBe(port.features_hash);
    expect(createBeliefFeatures({ engine, vocabulary, reader_sha256: "0".repeat(64), manifest }, files).features_hash).not.toBe(port.features_hash);
  });

  it("refuses a prior that is not derived from the committed generator config", () => {
    const model = join(scratch, "model");
    mkdirSync(model, { recursive: true });
    cpSync(files.model_dir, model, { recursive: true });
    const target = join(model, "generator-matched.uniform-v1.json");
    const v = JSON.parse(readFileSync(target, "utf8")) as { source: { generator_config_hash: string } };
    v.source.generator_config_hash = "0".repeat(64);
    writeFileSync(target, JSON.stringify(v));
    expect(() => portOf({ ...files, model_dir: model, generator_prior_file: target })).toThrow(SearchWorldsError);
  });
});

describe("the builder over the committed random-play channels", () => {
  const port = portOf();
  const builder = createSheetBuilder({ engine, vocabulary, port, scenario_id });
  const blocks = decisions.map((c) => builder.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: c.input.flags }));

  it("every block is a valid OpponentSheet of the committed feature set, one slot per previewed Pokemon", () => {
    expect(blocks.length).toBeGreaterThan(100);
    blocks.forEach((b, i) => {
      expect(() => readOpponentSheet(b), decisions[i]?.id).not.toThrow();
      expect(b.source.hash).toBe(port.features_hash);
      expect(b.source.kind).toBe("belief");
      expect(b.slots.length).toBe((decisions[i] as (typeof decisions)[number]).input.state.opponent.roster.length);
    });
  });

  it("a flagged request is a refused block with the fixed code, counted; every other block is predicted or revealed", () => {
    const counts = builder.counts();
    const flagged = decisions.filter((c) => c.input.flags.length > 0).length;
    expect(counts.requests).toBe(decisions.length);
    expect(counts.refusals.find((r) => r.code === "reader_flagged")?.count ?? 0).toBe(flagged);
    expect(counts.refused).toBe(counts.refusals.reduce((s, r) => s + r.count, 0));
    expect(refusedSharePpm(counts)).toBe(Math.floor((counts.refused * 1_000_000) / counts.requests));
    for (const b of blocks.filter((x) => x.source.refusal === undefined)) {
      expect(b.slots.every((s) => s.brought_source !== "unknown")).toBe(true);
      expect(b.slots.some((s) => s.brought_source === "revealed")).toBe(true); // the leads are in play at every move request
    }
  });

  it("golden: the digest of every block pins how a value is computed (a change needs an OPPONENT_SHEET_FORMAT bump and a review)", () => {
    // The source hash names the reader file and the pin, which change for other reasons: the values are what is pinned.
    const values = blocks.map((b) => ({ ...b, source: { ...b.source, hash: "-" } }));
    expect(createHash("sha256").update(JSON.stringify(values)).digest("hex")).toBe(GOLDEN_BLOCKS);
  });

  it("is deterministic: a second builder over the same requests gives the same bytes", () => {
    const again = createSheetBuilder({ engine, vocabulary, port: portOf(), scenario_id });
    const second = decisions.map((c) => again.build({ request: c.input.request, team: c.input.team, state: c.input.state, flags: c.input.flags }));
    expect(JSON.stringify(second)).toBe(JSON.stringify(blocks));
  });
});

describe("the memo", () => {
  it("returns what a recomputation returns, and never hands out the stored object", () => {
    const memoed = portOf();
    const fresh = () => portOf();
    const c = decisions.find((d) => d.input.flags.length === 0 && d.input.state.turn > 0) as (typeof decisions)[number];
    const args = { opponent: c.input.state.opponent, flags: c.input.flags };
    const first = memoed.sheet(args);
    const second = memoed.sheet(args);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
    expect(second).not.toBe(first);
    (first.slots[0] as { brought_ppm: number }).brought_ppm = 7; // a caller changing a block does not change the next one
    expect(JSON.stringify(memoed.sheet(args))).toBe(JSON.stringify(second));
    expect(JSON.stringify(fresh().sheet(args))).toBe(JSON.stringify(second));
  });

  it("the key is the evidence projection: equal for equal evidence, different when evidence changes or a line was flagged", () => {
    const c = decisions.find((d) => d.input.flags.length === 0 && d.input.state.turn > 0) as (typeof decisions)[number];
    const o = c.input.state.opponent;
    expect(evidenceKey(o, [])).toBe(evidenceKey(structuredClone(o), []));
    const changed = structuredClone(o);
    const lead = changed.roster[0] as (typeof changed.roster)[number];
    lead.moves_revealed = [...lead.moves_revealed, "protect"];
    expect(evidenceKey(changed, [])).not.toBe(evidenceKey(o, []));
    expect(evidenceKey(o, [{ index: 1 } as never])).not.toBe(evidenceKey(o, []));
    // hit points are no evidence for the belief: the key ignores them
    const hurt = structuredClone(o);
    (hurt.roster[0] as (typeof hurt.roster)[number]).hp_num = 3;
    expect(evidenceKey(hurt, [])).toBe(evidenceKey(o, []));
  });
});

describe("the memo and the trapping check", () => {
  const c = decisions.find((d) => d.input.flags.length === 0 && d.input.state.turn > 0) as (typeof decisions)[number];
  const args = { opponent: c.input.state.opponent, flags: c.input.flags };

  it("a check that observes nothing is ignored by the belief, so it is memoised like no check", () => {
    const port = portOf();
    let asked = 0;
    const silent = { observed: [] as never[], shows: () => (asked += 1, []) };
    const plain = JSON.stringify(portOf().sheet(args));
    expect(JSON.stringify(port.sheet({ ...args, trapping: silent }))).toBe(plain);
    expect(JSON.stringify(port.sheet({ ...args, trapping: silent }))).toBe(plain);
    expect(JSON.stringify(port.sheet(args))).toBe(plain); // the same key as a call with no check
    expect(asked).toBe(0);
  });

  it("a check that observes something is never memoised: it is asked again", () => {
    const port = portOf();
    let asked = 0;
    const informative = { observed: [0 as const], shows: () => (asked += 1, [0 as const]) };
    port.sheet({ ...args, trapping: informative });
    const first = asked;
    expect(first).toBeGreaterThan(0);
    port.sheet({ ...args, trapping: informative });
    expect(asked).toBe(2 * first);
  });
});

describe("the memo's edges and the errors", () => {
  const c = decisions.find((d) => d.input.flags.length === 0 && d.input.state.turn > 0) as (typeof decisions)[number];

  it("keeps at most 256 blocks, oldest first out, and every answer still equals a recomputation", () => {
    const port = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest, arm: "generator-matched" }, files);
    const variant = (n: number) => {
      const o = structuredClone(c.input.state.opponent);
      (o.roster[0] as (typeof o.roster)[number]).moves_revealed = [`move${n}`];
      return o;
    };
    const first = JSON.stringify(port.sheet({ opponent: variant(0), flags: [], trapping: null }));
    for (let n = 1; n <= 256; n++) port.sheet({ opponent: variant(n), flags: [] });
    expect(JSON.stringify(port.sheet({ opponent: variant(0), flags: [] }))).toBe(first); // evicted, recomputed
  });

  it("an error that is not the model's refusal propagates, and nothing is kept", () => {
    const port = portOf();
    const trapping = { observed: [], shows: () => [] };
    expect(() => port.sheet({ opponent: {} as never, flags: [], trapping })).toThrow(TypeError);
  });
});

describe("the vocabulary is the one the prior was derived against", () => {
  it("the port's belief identity binds the vocabulary hash (a pin bump or a changed vocabulary is another identity)", () => {
    expect(teamVocabularyHash(vocabulary)).toMatch(/^[0-9a-f]{64}$/);
    const other = createBeliefFeatures({ engine, vocabulary, reader_sha256: readerSha256(), manifest: { ...manifest, regulation: `${manifest.regulation}-x` } }, files);
    expect(other.identity_hash).not.toBe(portOf().identity_hash);
  });
});
