// The battle-step level of the differential harness (ADR-030 groundwork G2, ADR-028 section 5): the port, the trace reader, the
// request-by-request comparison, the entity and exercising rule, the declared interaction classes and the corpus run. A separate
// entry point (`@areneon/differential/battle`), so level 1's self-differential report, which hashes every file the core's main
// entry reaches, does not depend on it. Engine-free like the rest of the package.
export { SEATS } from "./battle-backend.ts";
export type { Action, BattleBackend, BattleStarted, BattleStartInput, BattleStepInput, BattleStepOutput, Draw, MaskEntry, PokemonSet, Seat, StepRequest } from "./battle-backend.ts";
export { endStep, readBattleTrace, TraceError } from "./battle-trace.ts";
export type { BattleTrace, Step } from "./battle-trace.ts";
export { BattleHarnessError, compareBattles, compareStep } from "./battle-compare.ts";
export type { BattleComparison, BattleDifference, BattleDifferenceKind, StepListener } from "./battle-compare.ts";
export { stepFacts, teamEntities, toId } from "./battle-level.ts";
export { classesHash, ClassesError, classifyStep, readInteractionClasses } from "./battle-classes.ts";
export type { DrawSignal, InteractionClass, InteractionClasses, LineSignal, Signal } from "./battle-classes.ts";
export { addBattles, newBattleCorpus } from "./battle-corpus.ts";
export type { BattleCorpus, BattleCorpusDifference } from "./battle-corpus.ts";
export { battleCorpusSummary, classCounts } from "./battle-report.ts";
export type { BattleClassesCount } from "./battle-report.ts";
