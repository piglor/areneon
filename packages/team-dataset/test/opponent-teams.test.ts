// The full-team opponent draw with the real model (ADR-029 section 3, ADR-025 erratum E-25.1): `sampleTeam` behind the
// composition root, on the beliefs of seeded generator-team games (the teams the committed `generator-matched` prior is
// derived for, as in `generator-belief.test.ts`). Every team has a set for all six preview slots, holds the Item Clause
// across all six, is legal under the pinned validator with zero redraws, is a function of the stream, and a previewed
// species the prior lacks is a counted refusal.

import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadRulesSnapshot, readTrainingRunConfig, resolveFormat } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { type Game, type Set6, points } from "../../opponent-model/test/corpus.ts";
import { readerSha256 } from "../../public-state/src/index.ts";
import { goldenTeamVocabulary } from "../../showdown-bridge/src/golden.ts";
import { verifiedDatasetHash } from "../src/dataset-hash.ts";
import { DATASETS_DIR } from "../src/fixtures.ts";
import { createSearchWorlds, searchWorldsFiles } from "../src/search-worlds.ts";
import { datasetTeamPool } from "../src/loop-pool.ts";
import { createOpponentTeams } from "../src/opponent-teams.ts";
import { playSweepBattle } from "../src/reducer-sweep.ts";
import { engine, FORMAT } from "./support.ts";

const ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SOURCE = join(DATASETS_DIR(FORMAT), "fixtures-v1");
const raw = JSON.parse(readFileSync(join(ROOT, "fixtures/contracts/v1/training-run-config/valid/tiny.json"), "utf8")) as Record<string, unknown>;
(raw.generation as Record<string, unknown>).excluded_abilities = ["Illusion"];
raw.team_dataset = { path: SOURCE, dataset_hash: verifiedDatasetHash(engine, SOURCE, FORMAT) };
const config = readTrainingRunConfig(raw);
const manifest = resolveFormat(engine, FORMAT, loadRulesSnapshot(engine.commit)).manifest;
const port = () => createOpponentTeams({ engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, arm: "generator-matched" });
const stream = (i: number): Uint8Array => Uint8Array.from({ length: 32 }, (_, k) => (i * 7 + k * 3 + 1) & 255);

describe("createOpponentTeams on generator teams", () => {
  it("draws legal six-slot teams with zero redraws and the Item Clause across all six", { timeout: 300_000 }, async () => {
    const pool = datasetTeamPool(engine, config);
    const teams = port();
    let drawn = 0;
    let previews = 0;
    for (let index = 0; index < 6; index++) {
      const battle = await playSweepBattle(engine, FORMAT, pool, { run_key: "opponent-teams", mix: "random-legal", index, max_turns: 100 });
      expect(battle.failure).toBeUndefined();
      const game: Game = { id: `g${index}`, teams: battle.teams as unknown as Record<"p1" | "p2", Set6[]>, channels: battle.channels };
      const all = points(game);
      for (const [n, point] of all.entries()) {
        const preview = point.request === 0;
        if (!preview && n % 6 !== 0) continue;
        const opened = teams.open({ opponent: point.state.opponent, flags: point.flags });
        if (opened.kind === "refused") throw new Error(`${game.id} ${point.seat} request ${point.request}: ${opened.code}`);
        // A preview belief carries no evidence: 13 teams from each of the twelve gives 156 teams.
        for (let i = 0; i < (preview ? 13 : 1); i++) {
          const out = opened.source.draw(stream(index * 100 + i), i);
          if (out.kind === "refused") throw new Error(`${game.id}: ${out.code}`);
          expect(out.redraws).toBe(0);
          expect(out.team.sets).toHaveLength(6);
          const held = out.team.sets.flatMap((s) => (s.item === null ? [] : [s.item]));
          expect(new Set(held).size).toBe(held.length);
          drawn += 1;
        }
        if (preview) previews += 13;
      }
    }
    expect(previews).toBeGreaterThanOrEqual(150);
    expect(teams.counts()).toMatchObject({ drawn, legality_redraws: 0, belief_refusals: {}, draw_refusals: {} });
  });

  it("is a function of the stream, and a previewed species the prior lacks is a counted refusal", { timeout: 120_000 }, async () => {
    const pool = datasetTeamPool(engine, config);
    const battle = await playSweepBattle(engine, FORMAT, pool, { run_key: "opponent-teams", mix: "random-legal", index: 0, max_turns: 20 });
    const [first] = points({ id: "g0", teams: battle.teams as unknown as Record<"p1" | "p2", Set6[]>, channels: battle.channels });
    if (first === undefined) throw new Error("no request point");
    const a = port().open({ opponent: first.state.opponent, flags: first.flags });
    const b = port().open({ opponent: first.state.opponent, flags: first.flags });
    if (a.kind !== "source" || b.kind !== "source") throw new Error("expected a source");
    expect(a.source.draw(stream(1), 0)).toEqual(b.source.draw(stream(1), 0));
    expect(JSON.stringify(a.source.draw(stream(1), 1))).not.toBe(JSON.stringify(a.source.draw(stream(1), 0)));

    const roster = first.state.opponent.roster.map((m, i) => (i === 0 ? { ...m, species: "notaspecies", base_species: "notaspecies" } : m));
    const lacking = port();
    expect(lacking.open({ opponent: { ...first.state.opponent, roster }, flags: first.flags })).toEqual({ kind: "refused", code: "species_not_in_prior" });
    expect(lacking.counts()).toMatchObject({ opened: 1, belief_refusals: { species_not_in_prior: 1 }, drawn: 0 });

    // A team whose every redraw held an illegal set is a counted refusal; an error that is not the model's is thrown.
    const context = { engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, arm: "generator-matched" } as const;
    const never = createOpponentTeams(context, { legality: { legal: () => false } });
    const opened = never.open({ opponent: first.state.opponent, flags: first.flags });
    if (opened.kind !== "source") throw new Error("expected a source");
    expect(opened.source.draw(stream(1), 0)).toEqual({ kind: "refused", code: "legality_redraws_exhausted" });
    expect(never.counts()).toMatchObject({ drawn: 0, draw_refusals: { legality_redraws_exhausted: 1 } });
    const broken = createOpponentTeams(context, {
      legality: {
        legal: () => {
          throw new Error("not the model's");
        },
      },
    });
    const live = broken.open({ opponent: first.state.opponent, flags: first.flags });
    if (live.kind !== "source") throw new Error("expected a source");
    expect(() => live.source.draw(stream(1), 0)).toThrow("not the model's");
    expect(() => port().open({ opponent: null as never, flags: [] })).toThrow();

    // The input is the reader's opponent half and its flags, nothing else (type test).
    const p = port();
    // @ts-expect-error: the opponent's real team is not an input
    p.open({ opponent: first.state.opponent, flags: [], team: [] });
    // @ts-expect-error: nor a battle record
    p.open({ opponent: first.state.opponent, flags: [], record: {} });
    // @ts-expect-error: nor the seed
    p.open({ opponent: first.state.opponent, flags: [], seed: "sodium,0" });
  });
});

describe("the teams port and the worlds port share one identity", () => {
  const scratch = mkdtempSync(join(tmpdir(), "opponent-teams-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));
  const context = { engine, vocabulary: goldenTeamVocabulary(), reader_sha256: readerSha256(), manifest, arm: "generator-matched" } as const;
  const files = searchWorldsFiles(FORMAT);

  it.each(["belief.v1.json", "belief.damage-v1.json", "belief.speed-v1.json"])("equal identity_hash for %s (evidence checks on or off)", (name) => {
    const model = join(scratch, name);
    cpSync(files.model_dir, model, { recursive: true });
    writeFileSync(join(model, "belief.v1.json"), readFileSync(join(files.model_dir, name)));
    const f = { ...files, model_dir: model, generator_prior_file: join(model, "generator-matched.uniform-v1.json"), role_sets_prior_file: join(model, "role-sets.fill-uniform-v1.json") };
    const teams = createOpponentTeams(context, { files: f });
    const worlds = createSearchWorlds(context, f);
    expect(teams.identity_hash).toBe(worlds.identity_hash);
    expect(teams.prior_hash).toBe(worlds.prior_hash);
    expect(teams.belief_config_hash).toBe(worlds.belief_config_hash);
  });

  it("the three configs give three identities", () => {
    const ids = ["belief.v1.json", "belief.damage-v1.json", "belief.speed-v1.json"].map((name) => {
      const model = join(scratch, `distinct-${name}`);
      cpSync(files.model_dir, model, { recursive: true });
      writeFileSync(join(model, "belief.v1.json"), readFileSync(join(files.model_dir, name)));
      return createOpponentTeams(context, { files: { ...files, model_dir: model } }).identity_hash;
    });
    expect(new Set(ids).size).toBe(3);
  });
});
