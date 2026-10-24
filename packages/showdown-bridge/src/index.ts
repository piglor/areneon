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
export { type DamageAwareSetup, damageAwareSetup, GenerationError, type PoolRequest, type TrainingPool as GenerationTeamPool, playGeneration } from "./league-generation.ts";
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
  encodeFrame,
  type PolicyHandshake,
  type PolicyQuery,
  type PolicyQueryInput,
  PolicyProtocolError,
  type PolicyReply,
  readHandshake,
  readPolicyQuery,
  readPolicyReply,
} from "./policy-protocol.ts";
export {
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
export { runEvaluationCli } from "./evaluation-cli.ts";
export {
  buildLeagueRunRecord,
  checkLeagueRunRecord,
  type LeagueRunFacts,
  type LeagueRunRecord,
  LeagueRunRecordError,
  leagueRunRecordHash,
  readLeagueRunRecord,
} from "./league-run-record.ts";
export { type AssumedSet, type AssumedSetContext, AssumedSetError, assumedHp, buildAssumedSet } from "./damage-aware-assumed-set.ts";
export {
  DAMAGE_AWARE_CONFIG_DIR,
  type DamageAwareConfig,
  DamageAwareConfigError,
  damageAwareConfigHash,
  loadDamageAwareConfig,
  readDamageAwareConfig,
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
export { committedPlanSeeds, type ProgressPoolSource, runProgressCli } from "./progress-cli.ts";
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
export { actsBefore, canKnockOut, critChancePpm, damageValue, discounted, rollsValue } from "./damage-aware-score.ts";
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
