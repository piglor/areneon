import { readSearchConfig, SearchConfigError } from "../src/search-config.ts";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ajvFor } from "@areneon/schema/ajv";
import { beforeAll, describe, expect, it } from "vitest";
import { syntheticFixtureHashes } from "../src/golden.ts";
import { EngineDataError, readEngineData, readEngineDataV2 } from "../src/engine-data.ts";
import { contractVersions, EpisodeError, readTeamDatasetManifest, readTeamGeneratorConfig, readTeamLabelSet, readTeamRecord, readTeamSnapshotManifest, readTeamVocabulary, readRoleSetTable, RoleSetError, readMoveClassTable, MoveClassError, TeamRecordError, TeamVocabularyError, loadEngine, PolicyProtocolError, readEvaluationReply, readHandshake, readPolicyQuery, readPolicyReply, readEpisode, readTrainingRunConfig, readCloneRunConfig, CloneRunConfigError, readLeagueRunRecord, LeagueRunRecordError, TrainingRunConfigError, readEvaluationRunManifest, readTrainingExample, readTrainingSet, TrainingLoaderError } from "../src/index.ts";
import { EvaluationConfigError, readEvaluationParameters, readEntry, RegisterError, readEvaluationPlan, readSealedSetCommitment, readTrainingGateRule, readEvaluationConfig, readSealedTeamSetManifest, readRetiredSealedManifest } from "../src/index.ts";
import { contractSamples, SAMPLE_HASHES, SAMPLES_DIR } from "../src/samples.ts";
import { exportEngineData } from "../src/engine-data.ts";
import { GOLDEN_FORMAT } from "../src/golden.ts";

// The Ajv classification below compiles every contract and checks every sample: CPU-bound work that
// grows with each contract. About 1 s locally, 18.6 s on main's CI and over 30 s on a loaded runner
// (PR #219), so the budget is explicit.
const CORPUS_TEST_TIMEOUT_MS = 120_000;

const SCHEMAS = new URL("../../schema/contracts/v1/", import.meta.url);
const committed = (file: string): unknown => JSON.parse(readFileSync(join(SAMPLES_DIR, file), "utf8"));

let fresh: Record<string, unknown>;
beforeAll(async () => {
  fresh = await contractSamples(loadEngine());
});

describe("the shared contract sample corpus (ADR-008 §2)", () => {
  it("matches a fresh capture: re-run scripts/capture-samples.ts and review the diff", () => {
    const files = readdirSync(join(SAMPLES_DIR, "v1"), { recursive: true, withFileTypes: true })
      .filter((e) => e.isFile())
      .map((e) => join(e.parentPath, e.name).slice(SAMPLES_DIR.length + 1));
    expect([...files, SAMPLE_HASHES].sort()).toEqual(Object.keys(fresh).sort());
    for (const [file, content] of Object.entries(fresh)) expect(committed(file), file).toEqual(content);
  });

  it("covers every contract, with a valid and an invalid case each", () => {
    const contracts = readdirSync(SCHEMAS).map((f) => f.replace(".schema.json", ""));
    for (const contract of contracts) {
      for (const kind of ["valid", "invalid"]) {
        expect(Object.keys(fresh).some((f) => f.startsWith(`v1/${contract}/${kind}/`)), `${contract} ${kind}`).toBe(true);
      }
    }
  });

  it("is classified by Ajv as its directory says: only invalid fails; reader-invalid is schema-valid", () => {
    // One parsed schema object per contract: Ajv compiles per object, so parsing it again for every sample recompiled it each time.
    const schemas = new Map<string, { $id: string }>();
    for (const [file, content] of Object.entries(fresh)) {
      if (file === SAMPLE_HASHES) continue;
      const [, contract, kind] = file.split("/") as [string, string, string];
      if (!schemas.has(contract)) schemas.set(contract, JSON.parse(readFileSync(new URL(`${contract}.schema.json`, SCHEMAS), "utf8")) as { $id: string });
      const schema = schemas.get(contract) as { $id: string };
      const profile = schema.$id.includes(":engine:") ? "engine-mirror" : "contract";
      expect(ajvFor(profile).validate(schema, content), file).toBe(kind !== "invalid");
    }
  }, CORPUS_TEST_TIMEOUT_MS);

  // ADR-009 section 4, test 2: schema-valid, yet readEpisode refuses each, for the named rule.
  it.each([
    ["v1/episode/reader-invalid/sparring-synthetic.json", "a sparring Episode is private: it records a battle against a human"],
    [
      "v1/episode/reader-invalid/class-below-opponent.json",
      "the Episode is labelled synthetic, but its own team is synthetic and its opponent's team is private: the class must be private",
    ],
  ])("readEpisode refuses %s: %s", async (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    const known_synthetic = syntheticFixtureHashes(loadEngine());
    await expect(readEpisode(committed(file), { known_synthetic })).rejects.toThrow(EpisodeError);
    await expect(readEpisode(committed(file), { known_synthetic })).rejects.toThrow(message);
  });

  it("holds every reader-invalid sample under its contract's directory, and each is refused above", () => {
    const readerInvalid = Object.keys(fresh).filter((f) => f.includes("/reader-invalid/"));
    expect(readerInvalid.sort()).toEqual([
      "v1/belief-config/reader-invalid/damage-unknown-item.json",
      "v1/belief-config/reader-invalid/damage-unknown-lasting-move.json",
      "v1/belief-config/reader-invalid/damage-unknown-move.json",
      "v1/belief-config/reader-invalid/damage-unsorted-moves.json",
      "v1/belief-config/reader-invalid/duplicate-ability.json",
      "v1/belief-config/reader-invalid/speed-unknown-item.json",
      "v1/belief-config/reader-invalid/speed-unknown-move.json",
      "v1/belief-config/reader-invalid/speed-unsorted-moves.json",
      "v1/belief-config/reader-invalid/unknown-ability.json",
      "v1/belief-config/reader-invalid/unsorted-abilities.json",
      "v1/best-response-record/reader-invalid/a-seat-not-parity.json",
      "v1/best-response-record/reader-invalid/choices-wrong-arms.json",
      "v1/best-response-record/reader-invalid/clamped-not-sum.json",
      "v1/best-response-record/reader-invalid/columns-used-over-columns.json",
      "v1/best-response-record/reader-invalid/complete-with-missing.json",
      "v1/best-response-record/reader-invalid/evaluation-dataset-is-own.json",
      "v1/best-response-record/reader-invalid/forfeits-unsorted.json",
      "v1/best-response-record/reader-invalid/game-rows-short.json",
      "v1/best-response-record/reader-invalid/games-odd.json",
      "v1/best-response-record/reader-invalid/pairs-not-half.json",
      "v1/best-response-record/reader-invalid/rank-over-own.json",
      "v1/best-response-record/reader-invalid/tally-not-rows.json",
      "v1/best-response-record/reader-invalid/worlds-not-plan.json",
      "v1/checkpoint-record/reader-invalid/duplicate-training-sets.json",
      "v1/checkpoint-record/reader-invalid/exploiter-of-self.json",
      "v1/checkpoint-record/reader-invalid/opponent-is-self.json",
      "v1/checkpoint-record/reader-invalid/parent-is-self.json",
      "v1/checkpoint-record/reader-invalid/unsorted-league-opponents.json",
      "v1/clone-run-config/reader-invalid/heads-do-not-divide-width.json",
      "v1/clone-run-config/reader-invalid/picks-exceed-team.json",
      "v1/damage-aware-config-2/reader-invalid/other-format.json",
      "v1/damage-aware-config/reader-invalid/other-format.json",
      "v1/damage-aware-config/reader-invalid/stat-points-over-limit.json",
      "v1/damage-aware-config/reader-invalid/unknown-fixed-item.json",
      "v1/damage-aware-config/reader-invalid/unknown-nature.json",
      "v1/engine-data-2/reader-invalid/duplicate-condition.json",
      "v1/engine-data-2/reader-invalid/unsorted-conditions.json",
      "v1/engine-data/reader-invalid/dangling-type.json",
      "v1/engine-data/reader-invalid/duplicate-species.json",
      "v1/engine-data/reader-invalid/short-damage-row.json",
      "v1/engine-data/reader-invalid/unsorted-species.json",
      "v1/episode/reader-invalid/class-below-opponent.json",
      "v1/episode/reader-invalid/sparring-synthetic.json",
      "v1/evaluation-parameters/reader-invalid/alpha-not-whole-ppm.json",
      "v1/evaluation-parameters/reader-invalid/b-boot-too-small.json",
      "v1/evaluation-parameters/reader-invalid/n-teams-below-floor.json",
      "v1/evaluation-parameters/reader-invalid/n-teams-without-floor.json",
      "v1/evaluation-parameters/reader-invalid/r-cov-too-small.json",
      "v1/evaluation-parameters/reader-invalid/seen-and-held-out.json",
      "v1/evaluation-parameters/reader-invalid/unsorted-baselines.json",
      "v1/evaluation-plan/reader-invalid/baseline-with-lineage.json",
      "v1/evaluation-plan/reader-invalid/candidate-roles-swapped.json",
      "v1/evaluation-plan/reader-invalid/unsorted-opponents.json",
      "v1/evaluation-register-entry/reader-invalid/failed-without-reason.json",
      "v1/evaluation-register-entry/reader-invalid/impossible-date.json",
      "v1/evaluation-register-entry/reader-invalid/reported-without-outcome-hash.json",
      "v1/evaluation-register-entry/reader-invalid/unsorted-affected-candidates.json",
      "v1/evaluation-run-manifest/reader-invalid/duplicate.json",
      "v1/evaluation-run-manifest/reader-invalid/unsorted.json",
      "v1/league-run-record/reader-invalid/baseline-decisions-exceed.json",
      "v1/league-run-record/reader-invalid/baseline-duplicate.json",
      "v1/league-run-record/reader-invalid/baseline-games-disagree.json",
      "v1/league-run-record/reader-invalid/baseline-not-an-opponent.json",
      "v1/league-run-record/reader-invalid/baseline-probe-runs-exceed-queries.json",
      "v1/league-run-record/reader-invalid/baseline-refusals-exceed-queries.json",
      "v1/league-run-record/reader-invalid/baseline-unsupported-exceed-queries.json",
      "v1/league-run-record/reader-invalid/baselines-unsorted.json",
      "v1/league-run-record/reader-invalid/causes-do-not-sum.json",
      "v1/league-run-record/reader-invalid/counts-disagree.json",
      "v1/league-run-record/reader-invalid/duplicate-cause.json",
      "v1/league-run-record/reader-invalid/duplicate-opponent.json",
      "v1/league-run-record/reader-invalid/episodes-not-two-per-game.json",
      "v1/league-run-record/reader-invalid/opponents-do-not-sum.json",
      "v1/league-run-record/reader-invalid/unsorted-causes.json",
      "v1/league-run-record/reader-invalid/unsorted-opponents.json",
      "v1/mechanics-config/reader-invalid/unsorted-ko-non-blockers.json",
      "v1/mechanics-config/reader-invalid/unsorted-removed-rule-ids.json",
      "v1/mechanics-result/reader-invalid/unsorted-gap-ids.json",
      "v1/move-class-table/reader-invalid/duplicate-id.json",
      "v1/move-class-table/reader-invalid/foreign-move.json",
      "v1/move-class-table/reader-invalid/stale-engine-data-hash.json",
      "v1/move-class-table/reader-invalid/status-incomplete.json",
      "v1/move-class-table/reader-invalid/unsorted-class.json",
      "v1/policy-handshake/reader-invalid/duplicate-contracts.json",
      "v1/policy-handshake/reader-invalid/unsorted-checkpoints.json",
      "v1/policy-handshake/reader-invalid/unsorted-contracts.json",
      "v1/policy-query/reader-invalid/duplicate-candidate.json",
      "v1/policy-query/reader-invalid/evaluate-duplicate-candidate.json",
      "v1/policy-query/reader-invalid/features-2-array-under-1.json",
      "v1/policy-query/reader-invalid/features-2-foe-priorities-unsorted.json",
      "v1/policy-query/reader-invalid/features-2-foreign-own.json",
      "v1/policy-query/reader-invalid/features-2-missing-array.json",
      "v1/policy-query/reader-invalid/features-2-missing-foe-priority.json",
      "v1/policy-query/reader-invalid/features-2-missing-own-priority.json",
      "v1/policy-query/reader-invalid/features-2-missing-speed-row.json",
      "v1/policy-query/reader-invalid/features-2-priorities-unsorted.json",
      "v1/policy-query/reader-invalid/features-2-speeds-duplicate.json",
      "v1/policy-query/reader-invalid/features-2-threats-unsorted.json",
      "v1/policy-query/reader-invalid/features-duplicate-row.json",
      "v1/policy-query/reader-invalid/features-missing-row.json",
      "v1/policy-query/reader-invalid/features-unsorted.json",
      "v1/policy-query/reader-invalid/identity-differs.json",
      "v1/policy-query/reader-invalid/temperature-explicit-default.json",
      "v1/private-snapshot-manifest/reader-invalid/active-rows-over-rows.json",
      "v1/private-snapshot-manifest/reader-invalid/both-parent-and-first.json",
      "v1/private-snapshot-manifest/reader-invalid/cell-not-active-sum.json",
      "v1/private-snapshot-manifest/reader-invalid/cells-out-of-order.json",
      "v1/private-snapshot-manifest/reader-invalid/counts-not-shard-rows.json",
      "v1/private-snapshot-manifest/reader-invalid/duplicate-shard.json",
      "v1/private-snapshot-manifest/reader-invalid/neither-parent-nor-first.json",
      "v1/private-snapshot-manifest/reader-invalid/ranges-do-not-tile.json",
      "v1/private-snapshot-manifest/reader-invalid/ranges-out-of-order.json",
      "v1/private-snapshot-manifest/reader-invalid/records-active-not-cells.json",
      "v1/private-snapshot-manifest/reader-invalid/rows-not-range-sum.json",
      "v1/private-snapshot-manifest/reader-invalid/shard-index-not-position.json",
      "v1/private-snapshot-manifest/reader-invalid/shards-out-of-order.json",
      "v1/private-snapshot-manifest/reader-invalid/unlisted-active-cell.json",
      "v1/private-snapshot-manifest/reader-invalid/wrong-logical-hash.json",
      "v1/private-team-record/reader-invalid/canonical-hash-mismatch.json",
      "v1/private-team-record/reader-invalid/duplicate-label-ids.json",
      "v1/private-team-record/reader-invalid/duplicate-tags.json",
      "v1/private-team-record/reader-invalid/first-revision-supersedes.json",
      "v1/private-team-record/reader-invalid/later-revision-no-predecessor.json",
      "v1/private-team-record/reader-invalid/team-hash-mismatch.json",
      "v1/private-team-record/reader-invalid/unsorted-tags.json",
      "v1/private-test-record/reader-invalid/baseline-with-checkpoints.json",
      "v1/private-test-record/reader-invalid/counts-not-games.json",
      "v1/private-test-record/reader-invalid/dataset-hash-on-staging.json",
      "v1/private-test-record/reader-invalid/forfeits-over-losses.json",
      "v1/private-test-record/reader-invalid/forfeits-unsorted.json",
      "v1/private-test-record/reader-invalid/games-not-twice-pairs.json",
      "v1/private-test-record/reader-invalid/histogram-not-pairs.json",
      "v1/private-test-record/reader-invalid/histogram-not-score.json",
      "v1/private-test-record/reader-invalid/holdout-on-staging.json",
      "v1/private-test-record/reader-invalid/match-games-not-plan.json",
      "v1/private-test-record/reader-invalid/match-id-not-opponent-mode.json",
      "v1/private-test-record/reader-invalid/match-twice.json",
      "v1/private-test-record/reader-invalid/matches-unsorted.json",
      "v1/private-test-record/reader-invalid/pair-without-checkpoints.json",
      "v1/private-test-record/reader-invalid/pairs-over-half.json",
      "v1/private-test-record/reader-invalid/private-pair-on-staging.json",
      "v1/private-test-record/reader-invalid/private-pair-on-whole-snapshot.json",
      "v1/private-test-record/reader-invalid/snapshot-hash-on-staging.json",
      "v1/private-test-record/reader-invalid/snapshot-without-hash.json",
      "v1/private-test-record/reader-invalid/turn-cap-over-ties.json",
      "v1/private-test-record/reader-invalid/unknown-baseline.json",
      "v1/progress-evaluation-record-2/reader-invalid/ablation-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/ablation-not-earlier.json",
      "v1/progress-evaluation-record-2/reader-invalid/ablation-vs-anchor.json",
      "v1/progress-evaluation-record-2/reader-invalid/anchor-games-missing.json",
      "v1/progress-evaluation-record-2/reader-invalid/anchor-games-odd.json",
      "v1/progress-evaluation-record-2/reader-invalid/anchor-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/anchor-not-earlier.json",
      "v1/progress-evaluation-record-2/reader-invalid/anchor-plan-without-match.json",
      "v1/progress-evaluation-record-2/reader-invalid/baseline-counts-disagree.json",
      "v1/progress-evaluation-record-2/reader-invalid/baseline-identities-differ.json",
      "v1/progress-evaluation-record-2/reader-invalid/baseline-on-other-match.json",
      "v1/progress-evaluation-record-2/reader-invalid/baseline-probe-runs-exceed-queries.json",
      "v1/progress-evaluation-record-2/reader-invalid/baseline-row-missing.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-counts-on-baseline.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-mode-differs.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-other-identity.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-row-missing.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-searches.json",
      "v1/progress-evaluation-record-2/reader-invalid/control-without-search.json",
      "v1/progress-evaluation-record-2/reader-invalid/counts-disagree.json",
      "v1/progress-evaluation-record-2/reader-invalid/evaluation-dataset-is-own.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-name-without-settings.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-on-anchor.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-on-argmax.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-on-previous.json",
      "v1/progress-evaluation-record-2/reader-invalid/extra-mode-run-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/odd-games.json",
      "v1/progress-evaluation-record-2/reader-invalid/pair-is-game.json",
      "v1/progress-evaluation-record-2/reader-invalid/pairs-used-wrong.json",
      "v1/progress-evaluation-record-2/reader-invalid/plan-label-is-run-id.json",
      "v1/progress-evaluation-record-2/reader-invalid/rows-incomplete.json",
      "v1/progress-evaluation-record-2/reader-invalid/run-games-missing.json",
      "v1/progress-evaluation-record-2/reader-invalid/run-games-odd.json",
      "v1/progress-evaluation-record-2/reader-invalid/run-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/run-plan-without-match.json",
      "v1/progress-evaluation-record-2/reader-invalid/run-same-run-id.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-counts-disagree.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-misnamed.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-mixed-with-plain.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-more-fallbacks-than-decisions.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-on-plain-match.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-row-missing.json",
      "v1/progress-evaluation-record-2/reader-invalid/search-unsorted-table.json",
      "v1/progress-evaluation-record-2/reader-invalid/seat-wrong.json",
      "v1/progress-evaluation-record/reader-invalid/counts-disagree.json",
      "v1/progress-evaluation-record/reader-invalid/forfeits-disagree.json",
      "v1/progress-evaluation-record/reader-invalid/hash-count-wrong.json",
      "v1/progress-evaluation-record/reader-invalid/hashes-unsorted.json",
      "v1/progress-evaluation-record/reader-invalid/pairs-used-wrong.json",
      "v1/progress-evaluation-record/reader-invalid/rows-incomplete.json",
      "v1/progress-evaluation-record/reader-invalid/seat-wrong.json",
      "v1/progress-evaluation-record/reader-invalid/turn-cap-ties-wrong.json",
      "v1/progress-evaluation-record/reader-invalid/unsorted-matches.json",
      "v1/progress-game-record/reader-invalid/baseline-decisions-differ.json",
      "v1/progress-game-record/reader-invalid/baseline-missing.json",
      "v1/progress-game-record/reader-invalid/baseline-seat-wrong.json",
      "v1/progress-game-record/reader-invalid/chosen-not-candidate.json",
      "v1/progress-game-record/reader-invalid/log-hash-wrong.json",
      "v1/progress-game-record/reader-invalid/pair-seed-wrong.json",
      "v1/progress-game-record/reader-invalid/pair-wrong.json",
      "v1/progress-game-record/reader-invalid/plan-label-is-run-id.json",
      "v1/progress-game-record/reader-invalid/plan-label-seed-from-run-id.json",
      "v1/progress-game-record/reader-invalid/plan-label-without-pair.json",
      "v1/progress-game-record/reader-invalid/search-control-other-identity.json",
      "v1/progress-game-record/reader-invalid/search-control-same-seat.json",
      "v1/progress-game-record/reader-invalid/search-control-without-search.json",
      "v1/progress-game-record/reader-invalid/search-decisions-differ.json",
      "v1/progress-game-record/reader-invalid/seed-wrong.json",
      "v1/progress-game-record/reader-invalid/team-hash-wrong.json",
      "v1/retired-sealed-manifest/reader-invalid/seed-is-salt.json",
      "v1/role-set-table/reader-invalid/duplicate-sets.json",
      "v1/role-set-table/reader-invalid/foreign-ability.json",
      "v1/role-set-table/reader-invalid/foreign-move.json",
      "v1/role-set-table/reader-invalid/stale-vocabulary-hash.json",
      "v1/role-set-table/reader-invalid/unknown-species.json",
      "v1/role-set-table/reader-invalid/unsorted-moves.json",
      "v1/role-set-table/reader-invalid/unsorted-sets.json",
      "v1/role-set-table/reader-invalid/unsorted-species.json",
      "v1/scripted-head-to-head-record/reader-invalid/a-seat-not-parity.json",
      "v1/scripted-head-to-head-record/reader-invalid/arm-counts-not-sum.json",
      "v1/scripted-head-to-head-record/reader-invalid/belief-missing-on-v2.json",
      "v1/scripted-head-to-head-record/reader-invalid/belief-on-v1.json",
      "v1/scripted-head-to-head-record/reader-invalid/belief-row-missing.json",
      "v1/scripted-head-to-head-record/reader-invalid/belief-totals-not-sum.json",
      "v1/scripted-head-to-head-record/reader-invalid/counts-missing-on-damage-arm.json",
      "v1/scripted-head-to-head-record/reader-invalid/counts-on-rule-arm.json",
      "v1/scripted-head-to-head-record/reader-invalid/forfeits-unsorted.json",
      "v1/scripted-head-to-head-record/reader-invalid/game-rows-short.json",
      "v1/scripted-head-to-head-record/reader-invalid/games-odd.json",
      "v1/scripted-head-to-head-record/reader-invalid/identity-missing-on-damage-arm.json",
      "v1/scripted-head-to-head-record/reader-invalid/pair-not-half-game.json",
      "v1/scripted-head-to-head-record/reader-invalid/pairs-not-half.json",
      "v1/scripted-head-to-head-record/reader-invalid/previews-over-decisions.json",
      "v1/scripted-head-to-head-record/reader-invalid/probe-runs-over-queries.json",
      "v1/scripted-head-to-head-record/reader-invalid/refusals-over-decisions.json",
      "v1/scripted-head-to-head-record/reader-invalid/refusals-unsorted.json",
      "v1/scripted-head-to-head-record/reader-invalid/result-not-rows.json",
      "v1/scripted-head-to-head-record/reader-invalid/row-counts-missing.json",
      "v1/scripted-head-to-head-record/reader-invalid/same-arm-twice.json",
      "v1/scripted-head-to-head-record/reader-invalid/wins-not-games.json",
      "v1/sealed-set-commitment/reader-invalid/strata-sum-mismatch.json",
      "v1/sealed-set-commitment/reader-invalid/unsorted-strata.json",
      "v1/sealed-team-set-manifest/reader-invalid/quota-sum-mismatch.json",
      "v1/sealed-team-set-manifest/reader-invalid/unsorted-strata.json",
      "v1/search-config/reader-invalid/inexact-uniform-share.json",
      "v1/search-config/reader-invalid/min-worlds-over-n.json",
      "v1/search-config/reader-invalid/no-uniform-without-full-weight.json",
      "v1/team-dataset-manifest/reader-invalid/evaluation-only-near-over-draws.json",
      "v1/team-dataset-manifest/reader-invalid/merged-stratum-missing.json",
      "v1/team-dataset-manifest/reader-invalid/split-cell-is-a-member.json",
      "v1/team-dataset-manifest/reader-invalid/splits-out-of-order.json",
      "v1/team-dataset-manifest/reader-invalid/stratum-not-lowest-member.json",
      "v1/team-dataset-manifest/reader-invalid/teams-not-quota-sum.json",
      "v1/team-dataset-manifest/reader-invalid/unsorted-cells.json",
      "v1/team-generator-config/reader-invalid/components-unsorted.json",
      "v1/team-generator-config/reader-invalid/duplicate-item-weights.json",
      "v1/team-generator-config/reader-invalid/duplicate-kind.json",
      "v1/team-generator-config/reader-invalid/shares-not-1000.json",
      "v1/team-generator-config/reader-invalid/unknown-item.json",
      "v1/team-generator-config/reader-invalid/unsorted-item-weights.json",
      "v1/team-label-set/reader-invalid/default-is-a-value.json",
      "v1/team-label-set/reader-invalid/duplicate-label-ids.json",
      "v1/team-label-set/reader-invalid/duplicate-values.json",
      "v1/team-label-set/reader-invalid/unknown-engine-id.json",
      "v1/team-label-set/reader-invalid/unsorted-ids.json",
      "v1/team-record/reader-invalid/canonical-hash-mismatch.json",
      "v1/team-record/reader-invalid/duplicate-label-ids.json",
      "v1/team-record/reader-invalid/foreign-label-set.json",
      "v1/team-record/reader-invalid/labels-out-of-order.json",
      "v1/team-record/reader-invalid/paste-not-engine-export.json",
      "v1/team-record/reader-invalid/paste-unreproduced-line.json",
      "v1/team-record/reader-invalid/team-hash-mismatch.json",
      "v1/team-record/reader-invalid/value-not-in-label.json",
      "v1/team-snapshot-manifest/reader-invalid/cells-missing.json",
      "v1/team-snapshot-manifest/reader-invalid/cells-not-shard-sum.json",
      "v1/team-snapshot-manifest/reader-invalid/cells-out-of-order.json",
      "v1/team-snapshot-manifest/reader-invalid/duplicate-shard.json",
      "v1/team-snapshot-manifest/reader-invalid/object-path-differs.json",
      "v1/team-snapshot-manifest/reader-invalid/ranges-do-not-tile.json",
      "v1/team-snapshot-manifest/reader-invalid/ranges-out-of-order.json",
      "v1/team-snapshot-manifest/reader-invalid/rows-not-range-sum.json",
      "v1/team-snapshot-manifest/reader-invalid/shards-out-of-order.json",
      "v1/team-snapshot-manifest/reader-invalid/unsorted-dataset-hashes.json",
      "v1/team-snapshot-manifest/reader-invalid/wrong-logical-hash.json",
      "v1/team-vocabulary/reader-invalid/duplicate-moves.json",
      "v1/team-vocabulary/reader-invalid/foreign-ability.json",
      "v1/team-vocabulary/reader-invalid/min-over-max.json",
      "v1/team-vocabulary/reader-invalid/picked-over-max.json",
      "v1/team-vocabulary/reader-invalid/stale-engine-data-hash.json",
      "v1/team-vocabulary/reader-invalid/unknown-species.json",
      "v1/team-vocabulary/reader-invalid/unsorted-items.json",
      "v1/team-vocabulary/reader-invalid/unsorted-natures.json",
      "v1/team-vocabulary/reader-invalid/unsorted-roster.json",
      "v1/training-example/reader-invalid/choice-not-in-candidates.json",
      "v1/training-example/reader-invalid/features-2-array-under-1.json",
      "v1/training-example/reader-invalid/features-2-missing-array.json",
      "v1/training-example/reader-invalid/features-2-speeds-unsorted.json",
      "v1/training-example/reader-invalid/features-unsorted.json",
      "v1/training-gate-rule/reader-invalid/alternative-below-null.json",
      "v1/training-gate-rule/reader-invalid/unsorted-claims.json",
      "v1/training-run-config/reader-invalid/anchor-no-warm-start.json",
      "v1/training-run-config/reader-invalid/duplicate-baseline.json",
      "v1/training-run-config/reader-invalid/heads-do-not-divide-width.json",
      "v1/training-run-config/reader-invalid/picks-exceed-team.json",
      "v1/training-run-config/reader-invalid/play-temperature-default.json",
      "v1/training-run-config/reader-invalid/shared-trunk-warm-start.json",
      "v1/training-run-config/reader-invalid/value-warmup-no-warm-start.json",
      "v1/training-run-config/reader-invalid/warm-start-same-id.json",
      "v1/training-set/reader-invalid/duplicate-manifests.json",
      "v1/training-set/reader-invalid/learner-ready-no-policy.json",
      "v1/training-set/reader-invalid/unsorted-episodes.json",
      "v1/usage-prior/reader-invalid/abilities-sum.json",
      "v1/usage-prior/reader-invalid/duplicate-item.json",
      "v1/usage-prior/reader-invalid/empty-slots-over.json",
      "v1/usage-prior/reader-invalid/fallback-unknown-species.json",
      "v1/usage-prior/reader-invalid/fallback-unsorted.json",
      "v1/usage-prior/reader-invalid/forme-weights-sum.json",
      "v1/usage-prior/reader-invalid/inclusion-over-denominator.json",
      "v1/usage-prior/reader-invalid/items-sum.json",
      "v1/usage-prior/reader-invalid/moves-denominator.json",
      "v1/usage-prior/reader-invalid/moves-sum.json",
      "v1/usage-prior/reader-invalid/other-engine-data.json",
      "v1/usage-prior/reader-invalid/other-format.json",
      "v1/usage-prior/reader-invalid/spread-over-limit.json",
      "v1/usage-prior/reader-invalid/spreads-sum.json",
      "v1/usage-prior/reader-invalid/stale-vocabulary-hash.json",
      "v1/usage-prior/reader-invalid/too-many-items.json",
      "v1/usage-prior/reader-invalid/too-many-moves.json",
      "v1/usage-prior/reader-invalid/too-many-spreads.json",
      "v1/usage-prior/reader-invalid/unknown-ability.json",
      "v1/usage-prior/reader-invalid/unknown-forme.json",
      "v1/usage-prior/reader-invalid/unknown-item.json",
      "v1/usage-prior/reader-invalid/unknown-move.json",
      "v1/usage-prior/reader-invalid/unknown-nature.json",
      "v1/usage-prior/reader-invalid/unknown-species.json",
      "v1/usage-prior/reader-invalid/unsorted-inputs.json",
      "v1/usage-prior/reader-invalid/unsorted-moves.json",
      "v1/usage-prior/reader-invalid/unsorted-species.json",
      "v1/usage-prior/reader-invalid/unsorted-spreads.json",
      "v1/usage-weight-table/reader-invalid/abilities-sum.json",
      "v1/usage-weight-table/reader-invalid/duplicate-item.json",
      "v1/usage-weight-table/reader-invalid/items-sum.json",
      "v1/usage-weight-table/reader-invalid/moves-sum.json",
      "v1/usage-weight-table/reader-invalid/other-engine-data.json",
      "v1/usage-weight-table/reader-invalid/other-format.json",
      "v1/usage-weight-table/reader-invalid/species-root.json",
      "v1/usage-weight-table/reader-invalid/species-sum.json",
      "v1/usage-weight-table/reader-invalid/spread-over-limit.json",
      "v1/usage-weight-table/reader-invalid/spreads-sum.json",
      "v1/usage-weight-table/reader-invalid/stale-vocabulary-hash.json",
      "v1/usage-weight-table/reader-invalid/teammate-root.json",
      "v1/usage-weight-table/reader-invalid/teammates-sum.json",
      "v1/usage-weight-table/reader-invalid/too-many-items.json",
      "v1/usage-weight-table/reader-invalid/too-many-moves.json",
      "v1/usage-weight-table/reader-invalid/too-many-spreads.json",
      "v1/usage-weight-table/reader-invalid/too-many-teammates.json",
      "v1/usage-weight-table/reader-invalid/unknown-ability.json",
      "v1/usage-weight-table/reader-invalid/unknown-item.json",
      "v1/usage-weight-table/reader-invalid/unknown-move.json",
      "v1/usage-weight-table/reader-invalid/unknown-nature.json",
      "v1/usage-weight-table/reader-invalid/unknown-species.json",
      "v1/usage-weight-table/reader-invalid/unknown-teammate.json",
      "v1/usage-weight-table/reader-invalid/unsorted-moves.json",
      "v1/usage-weight-table/reader-invalid/unsorted-sources.json",
      "v1/usage-weight-table/reader-invalid/unsorted-species.json",
      "v1/usage-weight-table/reader-invalid/unsorted-spreads.json",
      "v1/usage-weight-table/reader-invalid/unsorted-teammates.json",
      "v1/world-writers/reader-invalid/admitted-class.json",
      "v1/world-writers/reader-invalid/admitted-confusion.json",
      "v1/world-writers/reader-invalid/admitted-refusal-id.json",
      "v1/world-writers/reader-invalid/admitted-unknown-id.json",
      "v1/world-writers/reader-invalid/bad-line-kind.json",
      "v1/world-writers/reader-invalid/class-not-offered.json",
      "v1/world-writers/reader-invalid/condition-on-wrong-item.json",
      "v1/world-writers/reader-invalid/duplicate-admitted.json",
      "v1/world-writers/reader-invalid/id-under-two-rows.json",
      "v1/world-writers/reader-invalid/missing-item.json",
      "v1/world-writers/reader-invalid/unknown-condition.json",
      "v1/world-writers/reader-invalid/unknown-engine-id.json",
      "v1/world-writers/reader-invalid/unsorted-admitted.json",
      "v1/world-writers/reader-invalid/unsorted-ids.json",
      "v1/world-writers/reader-invalid/unsorted-items.json",
    ]);
  });

  // ADR-009 section 4: the manifests' readers refuse what the profile cannot say.
  it.each([
    ["v1/evaluation-run-manifest/reader-invalid/unsorted.json", readEvaluationRunManifest, "sorted and unique"],
    ["v1/evaluation-run-manifest/reader-invalid/duplicate.json", readEvaluationRunManifest, "sorted and unique"],
    ["v1/training-set/reader-invalid/unsorted-episodes.json", readTrainingSet, "episode_hashes must be sorted and unique"],
    ["v1/training-example/reader-invalid/choice-not-in-candidates.json", readTrainingExample, "is not among its"],
    ["v1/training-example/reader-invalid/features-unsorted.json", readTrainingExample, "sorted and unique"],
    ["v1/training-example/reader-invalid/features-2-array-under-1.json", readTrainingExample, "present exactly under probe-damage/2"],
    ["v1/training-example/reader-invalid/features-2-missing-array.json", readTrainingExample, "present exactly under probe-damage/2"],
    ["v1/training-example/reader-invalid/features-2-speeds-unsorted.json", readTrainingExample, "speeds must be sorted and unique"],
    ["v1/training-set/reader-invalid/learner-ready-no-policy.json", readTrainingSet, "must name its behaviour_policy_id"],
    ["v1/training-set/reader-invalid/duplicate-manifests.json", readTrainingSet, "evaluation_manifest_hashes must be sorted and unique"],
  ])("the reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(TrainingLoaderError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  // ADR-026 section 5: the SearchConfig reader refuses what the profile cannot say.
  it.each([
    ["v1/search-config/reader-invalid/inexact-uniform-share.json", readSearchConfig, "exact millionths"],
    ["v1/search-config/reader-invalid/min-worlds-over-n.json", readSearchConfig, "exceeds the worlds"],
    ["v1/search-config/reader-invalid/no-uniform-without-full-weight.json", readSearchConfig, "whole weight"],
  ])("the search config reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(SearchConfigError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  // ADR-010 section 5: the TrainingRunConfig reader refuses what the profile cannot say.
  it.each([
    ["v1/training-run-config/reader-invalid/heads-do-not-divide-width.json", "must be a multiple of n_heads"],
    ["v1/training-run-config/reader-invalid/picks-exceed-team.json", "cannot exceed max_team"],
    ["v1/training-run-config/reader-invalid/duplicate-baseline.json", "may be listed once"],
    ["v1/training-run-config/reader-invalid/warm-start-same-id.json", "one checkpoint for both roles"],
    ["v1/training-run-config/reader-invalid/value-warmup-no-warm-start.json", "needs warm_start"],
    ["v1/training-run-config/reader-invalid/anchor-no-warm-start.json", "ppo.anchor needs warm_start"],
    ["v1/training-run-config/reader-invalid/play-temperature-default.json", "is the default"],
    ["v1/training-run-config/reader-invalid/shared-trunk-warm-start.json", "shared_trunk cannot be combined"],
  ])("readTrainingRunConfig refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readTrainingRunConfig(committed(file))).toThrow(TrainingRunConfigError);
    expect(() => readTrainingRunConfig(committed(file))).toThrow(message);
  });

  // ADR-019 track 1: the CloneRunConfig reader refuses what the profile cannot say.
  it.each([
    ["v1/clone-run-config/reader-invalid/heads-do-not-divide-width.json", "must be a multiple of n_heads"],
    ["v1/clone-run-config/reader-invalid/picks-exceed-team.json", "cannot exceed max_team"],
  ])("readCloneRunConfig refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readCloneRunConfig(committed(file))).toThrow(CloneRunConfigError);
    expect(() => readCloneRunConfig(committed(file))).toThrow(message);
  });

  // ADR-009 section 3: the LeagueRunRecord reader refuses what the profile cannot say.
  it.each([
    ["v1/league-run-record/reader-invalid/counts-disagree.json", "played + forfeits must equal games"],
    ["v1/league-run-record/reader-invalid/episodes-not-two-per-game.json", "two per played game"],
    ["v1/league-run-record/reader-invalid/unsorted-opponents.json", "sorted by opponent_policy_id"],
    ["v1/league-run-record/reader-invalid/duplicate-opponent.json", "sorted by opponent_policy_id"],
    ["v1/league-run-record/reader-invalid/opponents-do-not-sum.json", "must sum to games"],
  ])("readLeagueRunRecord refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readLeagueRunRecord(committed(file))).toThrow(LeagueRunRecordError);
    expect(() => readLeagueRunRecord(committed(file))).toThrow(message);
  });

  // ADR-010 section 2: the worker protocol's readers refuse what the profile cannot say.
  it.each([
    ["v1/policy-query/reader-invalid/identity-differs.json", readPolicyQuery, "identity differs from its observation"],
    ["v1/policy-query/reader-invalid/temperature-explicit-default.json", readPolicyQuery, "must be omitted"],
    ["v1/policy-query/reader-invalid/duplicate-candidate.json", readPolicyQuery, "is listed twice"],
    ["v1/policy-query/reader-invalid/evaluate-duplicate-candidate.json", readPolicyQuery, "is listed twice"],
    ["v1/policy-query/reader-invalid/features-unsorted.json", readPolicyQuery, "sorted and unique"],
    ["v1/policy-query/reader-invalid/features-duplicate-row.json", readPolicyQuery, "sorted and unique"],
    ["v1/policy-query/reader-invalid/features-2-array-under-1.json", readPolicyQuery, "present exactly under probe-damage/2"],
    ["v1/policy-query/reader-invalid/features-2-missing-array.json", readPolicyQuery, "present exactly under probe-damage/2"],
    ["v1/policy-query/reader-invalid/features-2-threats-unsorted.json", readPolicyQuery, "threats must be sorted and unique"],
    ["v1/policy-query/reader-invalid/features-2-speeds-duplicate.json", readPolicyQuery, "speeds must be sorted and unique"],
    ["v1/policy-query/reader-invalid/features-2-priorities-unsorted.json", readPolicyQuery, "own_priorities must be sorted and unique"],
    ["v1/policy-query/reader-invalid/features-2-foe-priorities-unsorted.json", readPolicyQuery, "foe_priorities must be sorted and unique"],
    ["v1/policy-handshake/reader-invalid/unsorted-contracts.json", readHandshake, "sorted by name"],
    ["v1/policy-handshake/reader-invalid/duplicate-contracts.json", readHandshake, "sorted by name"],
    ["v1/policy-handshake/reader-invalid/unsorted-checkpoints.json", readHandshake, "sorted by role"],
  ])("the policy reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(PolicyProtocolError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  it("the policy readers accept the valid samples, and the hello lists this side's contracts", () => {
    for (const name of ["battle", "preview", "battle-a2", "battle-evaluate", "preview-evaluate", "battle-features-evaluate"]) expect(readPolicyQuery(committed(`v1/policy-query/valid/${name}.json`)).schema_version).toBe(1);
    for (const name of ["recommendation", "unsupported-candidate", "invalid-query"]) {
      expect(readPolicyReply(committed(`v1/policy-reply/valid/${name}.json`)).query_id).toBe(1);
    }
    for (const name of ["evaluation", "value-out-of-unit", "invalid-query"]) {
      expect(readEvaluationReply(committed(`v1/evaluation-reply/valid/${name}.json`)).query_id).toBe(1);
    }
    expect(() => readEvaluationReply({})).toThrow(PolicyProtocolError);
    expect(readHandshake(committed("v1/policy-handshake/valid/hello.json")).message.contracts).toEqual(contractVersions());
    expect(readHandshake(committed("v1/policy-handshake/valid/ready.json")).message.kind).toBe("ready");
    expect(() => readPolicyReply({})).toThrow(PolicyProtocolError);
  });

  // ADR-010 section 3: the engine-data reader refuses what the profile cannot say.
  it.each([
    ["v1/engine-data/reader-invalid/unsorted-species.json", /species: .* repeated or out of order/],
    ["v1/engine-data/reader-invalid/duplicate-species.json", /species: .* repeated or out of order/],
    ["v1/engine-data/reader-invalid/dangling-type.json", /unknown type/],
    ["v1/engine-data/reader-invalid/short-damage-row.json", /damage_taken is not one entry per type/],
  ])("readEngineData refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readEngineData(committed(file))).toThrow(EngineDataError);
    expect(() => readEngineData(committed(file))).toThrow(message);
  });

  it.each([
    ["v1/engine-data-2/reader-invalid/unsorted-conditions.json", /conditions: .* repeated or out of order/],
    ["v1/engine-data-2/reader-invalid/duplicate-condition.json", /conditions: .* repeated or out of order/],
  ])("readEngineDataV2 refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readEngineDataV2(committed(file))).toThrow(EngineDataError);
    expect(() => readEngineDataV2(committed(file))).toThrow(message);
  });

  it("holds a valid engine-data sample the reader accepts", () => {
    expect(readEngineData(committed("v1/engine-data/valid/golden.json")).species).toHaveLength(2);
  });

  // ADR-013 section 4: the team readers refuse what the profile cannot compute or compare.
  describe("the team records", () => {
    const table = exportEngineData(loadEngine(), GOLDEN_FORMAT);
    const slimTable = (): ReturnType<typeof exportEngineData> => {
      const vocabulary = committed("v1/team-vocabulary/valid/golden.json") as { roster: { id: string; moves: string[] }[]; items: string[] };
      const species = new Set(vocabulary.roster.map((r) => r.id));
      const moves = new Set(vocabulary.roster.flatMap((r) => r.moves));
      const kept = table.species.filter((s) => species.has(s.id));
      const abilities = new Set(kept.flatMap((s) => s.abilities));
      return {
        ...table,
        species: kept,
        moves: table.moves.filter((m) => moves.has(m.id)),
        items: table.items.filter((i) => vocabulary.items.includes(i.id)),
        abilities: table.abilities.filter((a) => abilities.has(a.id)),
      };
    };
    const engine = loadEngine();
    const labelSet = (): ReturnType<typeof readTeamLabelSet> => readTeamLabelSet(committed("v1/team-label-set/valid/golden.json"));

    it("accepts every valid sample", () => {
      expect(readTeamVocabulary(committed("v1/team-vocabulary/valid/golden.json"), slimTable()).roster).toHaveLength(2);
      expect(readTeamRecord(engine, committed("v1/team-record/valid/golden.json"), labelSet()).origin.split).toBe("dev");
      expect(readTeamLabelSet(committed("v1/team-label-set/valid/golden.json"), table).labels).toHaveLength(3);
      expect(readTeamGeneratorConfig(committed("v1/team-generator-config/valid/golden.json"), table).iv).toBe(31);
      expect(readTeamDatasetManifest(committed("v1/team-dataset-manifest/valid/golden.json")).splits).toHaveLength(2);
      expect(readTeamDatasetManifest(committed("v1/team-dataset-manifest/valid/merged.json")).merged_cells).toHaveLength(1);
      for (const name of ["first", "extended"]) expect(readTeamSnapshotManifest(committed(`v1/team-snapshot-manifest/valid/${name}.json`)).shards.length).toBeGreaterThan(0);
    });

    it.each([
      ["unsorted-roster", /roster: .* repeated or out of order/],
      ["duplicate-moves", /moves: .* repeated or out of order/],
      ["unsorted-natures", /natures: .* repeated or out of order/],
      ["unsorted-items", /items: .* repeated or out of order/],
      ["picked-over-max", /picked_team_size exceeds max_team_size/],
      ["min-over-max", /min_team_size exceeds max_team_size/],
      ["stale-engine-data-hash", /engine_data_hash is not the hash/],
      ["unknown-species", /is not in the engine-data table/],
      ["foreign-ability", /is not one of its engine-data abilities/],
    ])("readTeamVocabulary refuses team-vocabulary/%s", (name, message) => {
      const file = `v1/team-vocabulary/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamVocabulary(committed(file), slimTable())).toThrow(TeamVocabularyError);
      expect(() => readTeamVocabulary(committed(file), slimTable())).toThrow(message);
    });

    // ADR-020 section 3: the role-set reader refuses what the profile cannot say, against the sample vocabulary.
    it.each([
      ["unsorted-species", /species: .* repeated or out of order/],
      ["duplicate-sets", /set 1 is repeated or out of order/],
      ["unsorted-sets", /set 1 is repeated or out of order/],
      ["unsorted-moves", /moves: .* repeated or out of order/],
      ["unknown-species", /is not in the vocabulary roster/],
      ["foreign-move", /is not in its roster move pool/],
      ["foreign-ability", /is not in its roster abilities/],
      ["stale-vocabulary-hash", /vocabulary_hash is not the hash/],
    ])("readRoleSetTable refuses role-set-table/%s", (name, message) => {
      const file = `v1/role-set-table/reader-invalid/${name}.json`;
      const vocabulary = readTeamVocabulary(committed("v1/team-vocabulary/valid/golden.json"), slimTable());
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readRoleSetTable(committed(file), vocabulary)).toThrow(RoleSetError);
      expect(() => readRoleSetTable(committed(file), vocabulary)).toThrow(message);
    });

    // #1164: the move-class reader refuses what the profile cannot say, against the complete engine-data table.
    it.each([
      ["unsorted-class", /repeated or out of order/],
      ["duplicate-id", /repeated or out of order/],
      ["foreign-move", /is not a move of the engine-data table/],
      ["status-incomplete", /status class is not exactly the Status moves/],
      ["stale-engine-data-hash", /another engine-data table/],
    ])("readMoveClassTable refuses move-class-table/%s", (name, message) => {
      const file = `v1/move-class-table/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readMoveClassTable(committed(file), table)).toThrow(MoveClassError);
      expect(() => readMoveClassTable(committed(file), table)).toThrow(message);
    });

    it("accepts the valid move-class sample, with and without the engine-data table", () => {
      expect(readMoveClassTable(committed("v1/move-class-table/valid/golden.json"), table).classes.protection).toContain("protect");
      expect(readMoveClassTable(committed("v1/move-class-table/valid/golden.json")).classes.status.length).toBeGreaterThan(0);
    });

    it("accepts the valid role-set sample, with and without its vocabulary", () => {
      const vocabulary = readTeamVocabulary(committed("v1/team-vocabulary/valid/golden.json"), slimTable());
      expect(readRoleSetTable(committed("v1/role-set-table/valid/golden.json"), vocabulary).species).toHaveLength(2);
      expect(readRoleSetTable(committed("v1/role-set-table/valid/golden.json")).species).toHaveLength(2);
    });

    it.each([
      ["team-hash-mismatch", /team_hash is not the hash/],
      ["canonical-hash-mismatch", /canonical_team_hash is not the hash/],
      ["paste-not-engine-export", /not the engine's own export/],
      ["paste-unreproduced-line", /refused by the pinned engine/],
      ["duplicate-label-ids", /label ids: a value is repeated/],
      ["labels-out-of-order", /labels\[0\] is fake_out, but the label set's is speed_control/],
      ["value-not-in-label", /sideways is not one of its values/],
      ["foreign-label-set", /label_set_hash is not the hash/],
    ])("readTeamRecord refuses team-record/%s", (name, message) => {
      const file = `v1/team-record/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamRecord(engine, committed(file), labelSet())).toThrow(TeamRecordError);
      expect(() => readTeamRecord(engine, committed(file), labelSet())).toThrow(message);
    });

    it.each([
      ["duplicate-label-ids", /label ids: a value is repeated/],
      ["duplicate-values", /values: a value is repeated/],
      ["default-is-a-value", /the default yes is also a listed value/],
      ["unsorted-ids", /has_move ids: .* repeated or out of order/],
      ["unknown-engine-id", /nosuchmove is not in the engine-data table/],
    ])("readTeamLabelSet refuses team-label-set/%s", (name, message) => {
      const file = `v1/team-label-set/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamLabelSet(committed(file), table)).toThrow(TeamRecordError);
      expect(() => readTeamLabelSet(committed(file), table)).toThrow(message);
    });

    it.each([
      ["unsorted-item-weights", /item_weights: .* repeated or out of order/],
      ["duplicate-item-weights", /item_weights: .* repeated or out of order/],
      ["unknown-item", /nosuchitem is not in the engine-data table/],
      ["shares-not-1000", /shares add up to 900, not 1000/],
      ["components-unsorted", /proposal components: .* repeated or out of order/],
      ["duplicate-kind", /proposal components: .* repeated or out of order/],
    ])("readTeamGeneratorConfig refuses team-generator-config/%s", (name, message) => {
      const file = `v1/team-generator-config/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamGeneratorConfig(committed(file), table)).toThrow(TeamRecordError);
      expect(() => readTeamGeneratorConfig(committed(file), table)).toThrow(message);
    });

    it.each([
      ["splits-out-of-order", /splits are repeated or not in the order/],
      ["merged-stratum-missing", /merged stratum 3 has no cell/],
      ["stratum-not-lowest-member", /is not its lowest member/],
      ["split-cell-is-a-member", /is a member of merged stratum 0/],
      ["teams-not-quota-sum", /is not the sum of its quotas/],
      ["evaluation-only-near-over-draws", /more near-reference drops than draws/],
      ["unsorted-cells", /cell .* is repeated or out of order/],
    ])("readTeamDatasetManifest refuses team-dataset-manifest/%s", (name, message) => {
      const file = `v1/team-dataset-manifest/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamDatasetManifest(committed(file))).toThrow(TeamRecordError);
      expect(() => readTeamDatasetManifest(committed(file))).toThrow(message);
    });

    it.each([
      ["unsorted-dataset-hashes", /dataset_hashes: .* repeated or out of order/],
      ["wrong-logical-hash", /logical_snapshot_hash is not recomputed/],
      ["object-path-differs", /object_key is not pub\/v1\/shards/],
      ["ranges-do-not-tile", /cell ranges do not tile the shard/],
      ["ranges-out-of-order", /repeated or out of stratum order/],
      ["rows-not-range-sum", /is not the sum of its cell ranges/],
      ["cells-not-shard-sum", /rows is not the shards' sum/],
      ["cells-missing", /do not list exactly the cells/],
      ["cells-out-of-order", /cells: .* repeated or out of order/],
      ["shards-out-of-order", /shards: .* repeated or out of order/],
      ["duplicate-shard", /shard logical hashes: a value is repeated/],
    ])("readTeamSnapshotManifest refuses team-snapshot-manifest/%s", (name, message) => {
      const file = `v1/team-snapshot-manifest/reader-invalid/${name}.json`;
      expect(Object.keys(fresh)).toContain(file);
      expect(() => readTeamSnapshotManifest(committed(file))).toThrow(TeamRecordError);
      expect(() => readTeamSnapshotManifest(committed(file))).toThrow(message);
    });
  });

  // ADR-010 sections 6, 7 and O1: the evaluation readers refuse what the profile cannot say.
  it.each([
    ["v1/evaluation-parameters/reader-invalid/alpha-not-whole-ppm.json", readEvaluationParameters, "not a whole number of ppm"],
    ["v1/evaluation-parameters/reader-invalid/b-boot-too-small.json", readEvaluationParameters, "must exceed"],
    ["v1/evaluation-parameters/reader-invalid/r-cov-too-small.json", readEvaluationParameters, "at least 1 / alpha'"],
    ["v1/evaluation-parameters/reader-invalid/unsorted-baselines.json", readEvaluationParameters, "sorted and unique"],
    ["v1/evaluation-parameters/reader-invalid/seen-and-held-out.json", readEvaluationParameters, "both seen and held out"],
    ["v1/evaluation-parameters/reader-invalid/n-teams-below-floor.json", readEvaluationParameters, "below its floor"],
    ["v1/evaluation-parameters/reader-invalid/n-teams-without-floor.json", readEvaluationParameters, "n_teams is set"],
    ["v1/training-gate-rule/reader-invalid/unsorted-claims.json", readTrainingGateRule, "sorted by claim_id"],
    ["v1/training-gate-rule/reader-invalid/alternative-below-null.json", readTrainingGateRule, "must lie above"],
    ["v1/sealed-set-commitment/reader-invalid/strata-sum-mismatch.json", readSealedSetCommitment, "hold 32 teams"],
    ["v1/sealed-team-set-manifest/reader-invalid/unsorted-strata.json", readSealedTeamSetManifest, "sorted by stratum"],
    ["v1/sealed-team-set-manifest/reader-invalid/quota-sum-mismatch.json", readSealedTeamSetManifest, "sum to 32 but team_count is 33"],
    ["v1/retired-sealed-manifest/reader-invalid/seed-is-salt.json", readRetiredSealedManifest, "key and the salt must differ"],
    ["v1/sealed-set-commitment/reader-invalid/unsorted-strata.json", readSealedSetCommitment, "sorted by label"],
    ["v1/evaluation-plan/reader-invalid/candidate-roles-swapped.json", readEvaluationPlan, "battle checkpoint, then the preview"],
    ["v1/evaluation-plan/reader-invalid/unsorted-opponents.json", readEvaluationPlan, "sorted by opponent_id"],
    ["v1/evaluation-plan/reader-invalid/baseline-with-lineage.json", readEvaluationPlan, "no lineage"],
  ])("the evaluation reader refuses %s", (file, reader, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => reader(committed(file))).toThrow(EvaluationConfigError);
    expect(() => reader(committed(file))).toThrow(message);
  });

  it.each([
    ["v1/evaluation-register-entry/reader-invalid/reported-without-outcome-hash.json", "needs its outcome_hash"],
    ["v1/evaluation-register-entry/reader-invalid/failed-without-reason.json", "needs its reason"],
    ["v1/evaluation-register-entry/reader-invalid/unsorted-affected-candidates.json", "sorted and unique"],
    ["v1/evaluation-register-entry/reader-invalid/impossible-date.json", "not a real UTC date"],
  ])("readEntry refuses %s", (file, message) => {
    expect(Object.keys(fresh)).toContain(file);
    expect(() => readEntry(committed(file))).toThrow(RegisterError);
    expect(() => readEntry(committed(file))).toThrow(message);
  });

  it("the valid evaluation samples read, and the provisional parameters pair with the rule", () => {
    const rule = committed("v1/training-gate-rule/valid/v1.json");
    expect(readEvaluationConfig(committed("v1/evaluation-parameters/valid/provisional.json"), rule).parameters.max_turns).toBe(300);
    expect(readEvaluationConfig(committed("v1/evaluation-parameters/valid/complete.json"), rule).parameters.n_teams).toBe(32);
    expect(readEvaluationPlan(committed("v1/evaluation-plan/valid/registered.json")).set_id).toBe("set-1");
    expect(readSealedSetCommitment(committed("v1/sealed-set-commitment/valid/four-strata.json")).team_count).toBe(32);
    expect(readSealedTeamSetManifest(committed("v1/sealed-team-set-manifest/valid/four-strata.json")).team_count).toBe(32);
    expect(readRetiredSealedManifest(committed("v1/retired-sealed-manifest/valid/revealed.json")).set_id).toBe("set-1");
    for (const kind of ["set-commitment", "plan-registration", "l2a-recheck", "run-outcome", "set-retirement", "incident"]) {
      expect(readEntry(committed(`v1/evaluation-register-entry/valid/${kind}.json`)).schema_version).toBe(1);
    }
  });
});
