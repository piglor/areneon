export { actorRule, NoCandidateError } from "./actor-rule.ts";
export {
  randomBelow,
  randomLegal,
  randomLegalActor,
  type RandomLegalState,
  RandomStateError,
} from "./random-legal.ts";
export { pendingFirstActor } from "./random-corpus.ts";
export {
  type Candidate,
  type CandidateSet,
  enumerateCandidates,
  EnumerationError,
} from "./choices.ts";
export {
  type ChoiceJudge,
  CloneJudge,
  EnumeratorDefectError,
  judgeSubmission,
  JudgeStateError,
  type LegalChoiceOptions,
  type LegalChoiceSet,
  legalChoiceSet,
  type SubmissionVerdict,
  type Verdict,
} from "./choice-judge.ts";
export {
  ENGINE_DATA_FILE,
  ENGINE_DATA_SUMMARY,
  type EngineData,
  EngineDataError,
  engineDataHash,
  engineDataSummary,
  exportEngineData,
  readEngineData,
  readEngineDataAny,
} from "./engine-data.ts";
export {
  DEFAULT_SHOWDOWN_DIR,
  type Engine,
  EngineError,
  loadEngine,
  PINNED_SHOWDOWN_COMMIT,
  type PokemonSet,
  type SimBattle,
  type SimSide,
  type StatTable,
  verifyShowdownCheckout,
} from "./engine.ts";
export {
  type BattleObservation,
  buildObservation,
  type Observation,
  ObservationError,
  type ObservationInput,
  type PlayerChannel,
  type PreviewObservation,
  type Sha256,
} from "./observation.ts";
export { assertOwnTeamStats, OwnTeamStatsError } from "./own-team-stats.ts";
export { continueChannel, nodeSha256, observe, playerChannel, recordedChannel } from "./player-channel.ts";
export { answersRequest, type Collection, type CollectionInput, type Episode, episodeHash, type Recommendation, rightsFor } from "./episode.ts";
export {
  type EvaluationRunInput,
  type EvaluationRunManifest,
  evaluationRunManifest,
  loadTrainingSet,
  readEvaluationRunManifest,
  readTrainingExample,
  readTrainingSet,
  type StoredEpisode,
  type StoredTrainingInput,
  type TrainingExample,
  type TrainingInput,
  TrainingLoaderError,
  type TrainingSet,
  type TrainingStream,
  trainingExample,
  writeStoredTrainingStream,
  writeTrainingStream,
} from "./training-loader.ts";
export { interimPoolSource, type PoolSource, runGenerationCli, UsageError } from "./generation-cli.ts";
export { DAMAGE_AWARE_V2_POLICY_ID, DAMAGE_AWARE_V3_POLICY_ID, type DamageAwareSetup, damageAwareSetup, damageAwareSetupV2, type FeatureSetup, featureSetup, GenerationError, type PoolRequest, type TrainingPool as GenerationTeamPool, playGeneration } from "./league-generation.ts";
export { readTrainingRunConfig, runConfigHash, type TrainingRunConfig, TrainingRunConfigError } from "./training-run-config.ts";
export { type CloneRunConfig, CloneRunConfigError, cloneRunConfigHash, readCloneRunConfig } from "./clone-run-config.ts";
export { EpisodeError, episodeFromRecord, type EpisodeOptions, readEpisode } from "./episode-io.ts";
export {
  battleClass,
  type BattleClass,
  classOfSets,
  type DataClass,
  higherClass,
  stampTeam,
  type StampedTeam,
  type SyntheticTeamHashes,
  syntheticTeamHashes,
  teamHash,
} from "./team-class.ts";
export { webSha256 } from "./web-sha256.ts";
export {
  type ResolvedFormat,
  resolveFormat,
  UnsupportedFormatError,
  type VersionManifest,
} from "./formats.ts";
export { type ChannelCheck, checkChannel } from "./rules-channel.ts";
export {
  type ImportChannel,
  type ImportPlan,
  ImportPlanError,
  type ImportSnapshot,
  type ImportStep,
  type ImportStoredSnapshot,
  type KnownSnapshots,
  planImport,
  type ReleaseAsset,
  type ReleaseSource,
} from "./rules-import-plan.ts";
export {
  knownChannels,
  loadChannel,
  loadChannels,
  loadRulesSnapshot,
  parseChannel,
  RULES_SCHEMA_DIR,
} from "./rules-load.ts";
export {
  buildSnapshot,
  type ChannelFile,
  DEFAULT_RULES_DIR,
  type FormatRecord,
  formatRulesHash,
  formatRulesRecord,
  type FormatRulesRecord,
  type GapEvidence,
  type GapRecord,
  jcs,
  jcsBytes,
  type RegulationRecord,
  RulesError,
  rulesSnapshotHash,
  type RulesSnapshot,
  snapshotBytes,
} from "./rules.ts";
export {
  type DecodedRequest,
  decodeRequest,
  type JsonObject,
  type ParsedRequest,
  parseRequest,
  RequestDecodeError,
  type RequestPhase,
} from "./request.ts";
export { hashRequest, requestIdentityBytes, requestIdentityText } from "./request-bytes.ts";
export { canonicalJson, type Json, requestHash } from "./request-identity.ts";
export { serverOnlyLine, type StrippedLines, stripServerLines } from "./server-lines.ts";
export {
  exportPaste,
  importPaste,
  TeamImportError,
  type TeamVerdict,
  validateTeam,
} from "./teams.ts";
export {
  type Actor,
  type ActorContext,
  BattleAbortedError,
  type BattleOptions,
  type BattleRecord,
  FORFEIT_CAUSES,
  type ForfeitCause,
  ForfeitSignal,
  type HiddenVerdicts,
  InvalidTeamError,
  type OmniscientLog,
  type Player,
  type PlayerIdentity,
  PlayersOptionError,
  replayActor,
  type RequestRecord,
  runBattle,
  validatePlayers,
} from "./runner.ts";
export {
  BaselineFailureError,
  type GameOutcome,
  type GuardOptions,
  guardActor,
  type IncidentClass,
  type LatencyBudget,
  type PlayedGame,
  playGame,
  PolicyWorkerError,
  UnsupportedCandidateError,
} from "./evaluation-harness.ts";
export { type PolicyActorOptions, policyActor, type PolicyGameOptions, policyGame, type PolicySeat, RANDOM_PREVIEW_POLICY, randomPreviewSeed, type SamplerInput, sampleU } from "./policy-actor.ts";
export {
  contractVersions,
  type EvaluateQueryInput,
  type Evaluation,
  type EvaluationReply,
  encodeFrame,
  FULL_TEMPERATURE_PPM,
  type PlaySettings,
  type PolicyHandshake,
  type PolicyQuery,
  type PolicyQueryInput,
  PolicyProtocolError,
  policyId,
  type PolicyReply,
  readEvaluationReply,
  readHandshake,
  readPolicyQuery,
  readPolicyReply,
} from "./policy-protocol.ts";
export {
  type EvaluationClient,
  type PolicyClient,
  PolicyHandshakeError,
  PolicyReplyError,
  PolicyWorkerClient,
  type PolicyWorkerOptions,
} from "./policy-worker.ts";
export {
  canonicalTeamHash,
  type DatasetManifestContext,
  labelCellCount,
  logicalSnapshotHash,
  readTeamDatasetManifest,
  readTeamGeneratorConfig,
  readTeamLabelSet,
  readTeamRecord,
  readTeamSnapshotManifest,
  recordHash,
  shardLogicalHash,
  snapshotHash,
  type TeamDatasetManifest,
  type TeamGeneratorConfig,
  type TeamLabelSet,
  type TeamRecord,
  TeamRecordError,
  type TeamSnapshotManifest,
} from "./team-records.ts";
export {
  checkPrivateRecordSet,
  type PrivateTeamRecord,
  PrivateRecordError,
  readPrivateTeamRecord,
} from "./private-records.ts";
export {
  HOLDOUT_PPM,
  HOLDOUT_RULE,
  holdoutFlag,
  PrivateSnapshotError,
  type PrivateSnapshotManifest,
  privateLogicalSnapshotHash,
  privateSnapshotHash,
  readPrivateSnapshotManifest,
} from "./private-snapshot.ts";
export { type UsagePrior, UsagePriorError, readUsagePrior, usagePriorHash } from "./usage-prior.ts";
export { type UsageWeightTable, UsageWeightTableError, readUsageWeightTable, rootedWeight, usageTableHash } from "./usage-weight-table.ts";
export {
  ADMITTED_KINDS,
  STATE_CLASSES,
  STATE_ITEMS,
  STATE_ITEM_IDS,
  type StateClass,
  type StateItemId,
  type WorldWriters,
  WorldWritersError,
  loadWorldWriters,
  readWorldWriters,
  worldWritersHash,
} from "./world-writers.ts";
export { LEAF_FORMAT, loadSearchConfig, readSearchConfig, type SearchConfig, SearchConfigError, searchConfigHash, searchIdentity, searchIdentityHash, type SearchIdentityInputs, SEARCH_POLICY_LABEL } from "./search-config.ts";
export { deriveRoleSetsPrior, ROLE_SETS_PRIOR_SCRIPT, RoleSetsPriorError, type RoleSetsPriorInput } from "./role-sets-prior.ts";
export { deriveGeneratorPrior, GENERATOR_PRIOR_SCRIPT, GeneratorPriorError, type GeneratorPriorInput, generatorConfigHash, megaHolderAbilities, megaPreviewSpecies } from "./generator-prior.ts";
export { DAMAGE_CHECK_FORMAT } from "./damage-check.ts";
export { SPEED_CHECK_FORMAT, evidenceCheckFormats } from "./speed-check.ts";
export { type BeliefConfig, BeliefConfigError, beliefConfigHash, readBeliefConfig } from "./belief-config.ts";
export {
  exportMoveClasses,
  MOVE_CLASS_NAMES,
  MOVE_CLASSES_FILE,
  MOVE_CLASSES_SUMMARY,
  MoveClassError,
  moveClassSummary,
  moveClassTableHash,
  readMoveClassTable,
  type MoveClassTable,
} from "./move-classes.ts";
export {
  exportRoleSets,
  ROLE_SETS_FILE,
  ROLE_SETS_SUMMARY,
  readRoleSetTable,
  type RoleSetTable,
  RoleSetError,
  roleSetSourcePath,
  roleSetSummary,
  roleSetTableHash,
} from "./role-sets.ts";
export {
  exportTeamVocabulary,
  readTeamVocabulary,
  TEAM_VOCABULARY_FILE,
  TEAM_VOCABULARY_SUMMARY,
  type TeamVocabulary,
  TeamVocabularyError,
  teamVocabularyHash,
  teamVocabularySummary,
} from "./team-vocabulary.ts";
export {
  alphaPrimePpm,
  checkGateConsistency,
  covTolPpm,
  deriveClaims,
  type EvaluationConfig,
  EvaluationConfigError,
  type EvaluationParameters,
  halfWidthOf,
  isComplete,
  domainHash,
  maxHalfWidthPpm,
  normalQuantile,
  PPM,
  readEvaluationConfig,
  readEvaluationParameters,
  readTrainingGateRule,
  requireComplete,
  type TrainingGateRule,
} from "./evaluation-config.ts";
export {
  checkCommitment,
  checkPlan,
  checkReveal,
  commitmentHash,
  type EvaluationPlan,
  generateSecret,
  keyCommitment,
  planHash,
  readEvaluationPlan,
  readSealedSetCommitment,
  type SealedSetCommitment,
  setCommitment,
} from "./evaluation-plan.ts";
export {
  readRetiredSealedManifest,
  readSealedTeamSetManifest,
  type RetiredSealedManifest,
  retiredManifestHash,
  type SealedTeamSetManifest,
  sealedManifestHash,
  stratumLabel,
} from "./sealed-records.ts";
export {
  appendEntry,
  assertPlanPublished,
  assertRetirementPublished,
  checkHeads,
  type EvaluationRegisterEntry,
  type EvaluationRegisterHead,
  GENESIS,
  entryHash,
  readEntry,
  readHead,
  RegisterError,
  type RegisterState,
  verifyRegister,
} from "./evaluation-register.ts";
export { readPublishedHeads, runEvaluationCli } from "./evaluation-cli.ts";
export {
  buildLeagueRunRecord,
  checkLeagueRunRecord,
  type LeagueRunFacts,
  type LeagueRunRecord,
  LeagueRunRecordError,
  leagueRunRecordHash,
  readLeagueRunRecord,
} from "./league-run-record.ts";
export { type AssumedSet, type AssumedSetContext, AssumedSetError, assumedHp, buildAssumedSet, buildWorldAssumedSet, WorldMismatchError, type WorldSetLike } from "./damage-aware-assumed-set.ts";
export { type LegalityWorldSet, type SetLegalityPort, createSetLegality } from "./set-legality.ts";
export {
  type AnyDamageAwareConfig,
  DAMAGE_AWARE_CONFIG_DIR,
  type DamageAwareConfig,
  DamageAwareConfigError,
  type DamageAwareConfigV2,
  damageAwareConfigHash,
  loadDamageAwareConfig,
  loadDamageAwareConfigV2,
  readDamageAwareConfig,
  readDamageAwareConfigV2,
} from "./damage-aware-config.ts";
export {
  checkMechanicsConfig,
  loadMechanicsConfig,
  MECHANICS_CONFIG_DIR,
  type MechanicsConfig,
  MechanicsConfigError,
  mechanicsConfigHash,
  readMechanicsConfig,
} from "./mechanics-config.ts";
export { type MechanicsContext, MechanicsProbeError, type ProbeOutcome, probeMechanics } from "./mechanics-probe.ts";
export {
  declaredSets,
  type MechanicsQuery,
  MechanicsQueryError,
  mechanicsQueryHash,
  type ParsedMechanicsQuery,
  readMechanicsQuery,
} from "./mechanics-query.ts";
export {
  buildResult,
  MECHANICS_ADAPTER_FORMAT,
  type MechanicsResult,
  MechanicsResultError,
  readMechanicsResult,
} from "./mechanics-result.ts";
export { checkScenario, ScenarioError, scenarioFormatId } from "./mechanics-scenario.ts";
export {
  type GameStreams,
  LEAGUE_SEED_DERIVATION,
  leagueBattleSeed,
  leagueGameStreams,
  type PolicyStreamLabel,
  policyStreamLabel,
} from "./league-seed.ts";
export {
  DAMAGE_AWARE_POLICY_ID,
  DAMAGE_AWARE_VERSIONS,
  type DamageAwareVersion,
  type ExtraMode,
  damageAwareMatchPrefix,
  damageAwareVersionPolicyId,
  playProgress,
  PROGRESS_BASELINES,
  PROGRESS_MODES,
  ProgressError,
  type ProgressMode,
  type ProgressOptions,
  type ProgressPool,
  buildProgressGameRecord,
  type GameFacts,
  progressPlan,
  type ProgressResult,
  type ProgressTeam,
  writeProgress,
} from "./progress-evaluation.ts";
export { type ExtraPlay, matchSuffix, modeName, parseExtraMode, parseModes, ProgressModeError } from "./progress-modes.ts";
export { EVAL_DATASET_FLAGS, type EvalDataset, readEvalDataset, withEvalDataset } from "./eval-dataset.ts";
export { committedPlanSeeds, type ProgressPoolSource, type ProgressPorts, runProgressCli } from "./progress-cli.ts";
export { controlSeat, searchCountsRecord, searchSeat } from "./search-progress.ts";
export { H2H_FILE, runScriptedH2HCli } from "./scripted-h2h-cli.ts";
export {
  type ScriptedGameRow,
  type ScriptedHeadToHeadRecord,
  ScriptedH2HRecordError,
  scriptedH2HRecordHash,
  SCRIPTED_H2H_LABEL,
  playsOnBelief,
  readScriptedH2HRecord,
  sumBeliefCounts,
  tallyScripted,
} from "./scripted-h2h-record.ts";
export { type ArmGameReport, buildScriptedRecord, playScriptedGames, SCRIPTED_POLICY_IDS, ScriptedH2HError, type ScriptedArm, scriptedArm, type ScriptedArmContext, type ScriptedPlayOptions } from "./scripted-h2h.ts";
export {
  omniscientLogHash,
  progressRecordHash,
  type ProgressEvaluationRecord,
  type ProgressEvaluationRecordV2,
  type ProgressGameRecord,
  ProgressRecordError,
  readProgressEvaluationRecord,
  readProgressEvaluationRecordV2,
  readProgressGameRecord,
} from "./progress-records.ts";
export {
  PROGRESS_PAIRING,
  PROGRESS_SEED_DERIVATION,
  progressGameStreams,
  progressPairs,
  type ProgressStreams,
} from "./progress-seed.ts";
export { parseJointChoice, ChoiceParseError, type SlotPart } from "./damage-aware-choice.ts";
export {
  BELIEF_DEFECTS,
  type BeliefWorldsContext,
  type BeliefWorldsFactory,
  type BeliefWorldsPort,
  refusalCode,
  type SampledWorldsLike,
  WORLDS_PURPOSE,
  type WorldLike,
  type WorldSlotLike,
  worldsStream,
} from "./damage-aware-worlds.ts";
export {
  type BeliefCounts,
  type DamageAwareActor,
  type DamageAwareCounts,
  DamageAwareError,
  type DamageAwareOptions,
  type DamageAwareReport,
  damageAwareActor,
} from "./damage-aware-actor.ts";
export {
  DAMAGE_AWARE_NAME,
  type DamageAwareIdentity,
  damageAwareIdentity,
  damageAwareIdentityHash,
  damageAwarePolicyId,
} from "./damage-aware-identity.ts";
export { actsBefore, canKnockOut, critChancePpm, DAMAGE_VALUE_FORMAT, damageValue, discounted, rollsValue } from "./damage-aware-score.ts";
export {
  BELIEF_FEATURES_FORMAT,
  type BeliefFeatureCounts,
  type BeliefFeatureSetup,
  type CandidateFeatures,
  candidateFeatures,
  createFeatureMemo,
  FEATURE_SET,
  FEATURE_SET_BELIEF,
  FeatureBuildError,
  type FeatureConfigs,
  type FeatureMemo,
  type FeatureRow,
  type FeatureSetId,
  type FeatureSetIdentity,
  featureSetHash,
  featureSetIdentity,
} from "./candidate-features.ts";
export {
  type CandidateFeatures2,
  candidateFeatures2,
  FEATURE_SET_2,
  type FoePriorityRow,
  type OrderReason,
  type OwnPriorityRow,
  type SpeedRow,
  type ThreatRow,
} from "./candidate-features-2.ts";
export { type CandidateFeaturesBelief, candidateFeaturesBelief, FEATURE_WORLDS_PURPOSE, featureWorldsStream, meanRows } from "./candidate-features-belief.ts";
export { type BenchDecision, episodeDecisions, estimateWallTime, progressDecisions, type ProgressGame, runBench, type Skipped } from "./feature-bench.ts";
export { type AnyCandidateFeatures, buildFeatures, configuredSetHash, featureSetOf, isFeatureSetId } from "./feature-build.ts";
export {
  checkTeacherSettings,
  planTeacher,
  playTeacherRun,
  type TeacherEntry,
  type TeacherReport,
  type TeacherResult,
  type TeacherSettings,
  teacherPolicyId,
  writeTeacherRun,
} from "./teacher-generation.ts";
export { runTeacherCli } from "./teacher-cli.ts";
export {
  assertGuardedChild,
  assertOwnerSession,
  OWNER_SESSION_REFUSALS,
  OwnerSessionError,
  type OwnerSession,
  type OwnerSessionCode,
} from "./owner-session.ts";
export {
  assertPrivateRoot,
  PRIVATE_ROOT_REFUSALS,
  PrivateRootError,
  privateRootFile,
  type PrivateRootCode,
  writePrivateFile,
} from "./private-root.ts";
export { PRIVATE_TEST_REFUSALS, PrivateTestRefusal, type PrivateTestRefusalCode, type PrivateTestSeams, type PrivateTestSources, privateTestTeams, runPrivateTestCli, stagedTeams, type TestTeams } from "./private-test-cli.ts";
export { opponentName, PRIVATE_TEST_LABEL, type PrivateTestMatch, type PrivateTestRecord, PrivateTestRecordError, privateTestAllowed, privateTestRecordHash, readPrivateTestRecord } from "./private-test-record.ts";
export { checkSalt, PRIVATE_TEST_PAIRING, PRIVATE_TEST_SEED_DERIVATION, privateTestCommitment, privateTestPairs, privateTestStreams } from "./private-test-seed.ts";
export { pairsAvailable, type PrivateTestOptions, type PrivateTestResult, type PrivateTestTeam, playPrivateTest } from "./private-test.ts";
export { privateDataRoot } from "./private-team-source.ts";
export {
  createWorldBuilder,
  type Resample,
  type ScenarioWorld,
  type WorldBuilder,
  type WorldCheck,
  type WorldContext,
  type WorldCounts,
  type WorldInput,
  type WorldOpponent,
  type WorldOpponentMember,
  type WorldPokemonRef,
  type WorldRefusal,
  WORLD_BUILDER_FORMAT,
  WORLD_CHECKS,
  WORLD_REFUSAL_CODES,
  type WorldRefusalCode,
  type WorldResult,
} from "./world-builder.ts";
export { type CheckFailure, type ConditionSnapshot, type Remainder, firstDifference, plainRequest, v2Candidates, v2Request, v4Conditions, v4Durations } from "./world-checks.ts";
export { type RedrawCounts, type Redrawer, createRedrawer } from "./world-redraw.ts";
export {
  type DecisionWorlds,
  EVIDENCE_CLASSES,
  type EvidenceClass,
  evidenceClass,
  GENERATOR_MATCHED_LABEL,
  mapWorld,
  PRIOR_KINDS,
  type PriorKind,
  type RefusedWorld,
  ROLE_SETS_LABEL,
  SEARCH_HP_ROUNDING,
  SEARCH_PURPOSE,
  searchStream,
  type ValidWorld,
  createWorldSampler,
  type WorldRequest,
  type WorldSampler,
  type WorldSamplerContext,
  type WorldSamplerCounts,
  type WorldSamplerSettings,
  type WorldSamplePort,
  type WorldSource,
  type WorldSourceResult,
  WorldMappingError,
} from "./world-sampler.ts";
export { TRAPPING_CHECK_FORMAT, createTrappingCheck, type TrapCandidate, type TrapLevel, type TrappingCheckContext, type TrappingCheckInput, type TrappingCheckPort } from "./trapping-check.ts";
export { createReplyModel, REPLY_MODEL_FORMAT, ReplyModelError, type ReplyModel } from "./reply-model.ts";
export {
  DROP_REASONS,
  type DropReason,
  engineSeed,
  FALLBACK_REASONS,
  replyIndex,
  SearchError,
  searchActor,
  type SearchActor,
  type SearchActorOptions,
  type SearchCounts,
  type SearchDecision,
  type SearchInput,
  searchPolicy,
  type SearchReport,
  searchSetup,
  searchSkipReason,
  type SearchSkipReason,
  type SearchSetup,
  type SearchSetupInput,
} from "./search-actor.ts";
export { hpKoDifferential, outcomeScore } from "./search-leaf.ts";
export { playerView } from "./search-turn.ts";
export { goldenWorldFactsData, readWorldFactsData, type FactsTable } from "./world-facts-data.ts";
export { legalReplies } from "./reply-model.ts";
export {
  evaluateHypothetical,
  HYPOTHETICAL_BATTLE_FORMAT,
  HYPOTHETICAL_ENGINE_PURPOSE,
  HypotheticalError,
  hypotheticalCandidates,
  hypotheticalEngineSeed,
  hypotheticalRequest,
  type HypotheticalCandidate,
  type HypotheticalRequest,
  startHypothetical,
  type HypotheticalChannel,
  type HypotheticalObservation,
  type HypotheticalPreview,
  type HypotheticalStartInput,
  type HypotheticalTurn,
  type PreviewSolveInput,
  MAX_ROLLOUT_STEPS,
  type RolloutChoose,
} from "./hypothetical-preview.ts";
export {
  BACK_DRAW_PURPOSE,
  backDraw,
  buildPayoffTable,
  PAYOFF_LEAF_FORMAT,
  PayoffError,
  selectOwn,
  strategiesOf,
  type OwnRule,
  type PayoffClient,
  type PayoffConfig,
  type PayoffCounts,
  type PayoffInput,
  type PayoffLeaf,
  type PayoffTable,
  type PayoffWorld,
  type Strategy as PreviewStrategy,
} from "./preview-payoff.ts";
