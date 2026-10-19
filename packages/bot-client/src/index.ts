export { A2Channel, type A2ChannelBuild, A2ChannelError, type A2ChannelOptions } from "./a2-channel.ts";
export {
  A2_BUDGET_CONFIG,
  battleBudget,
  type Budget,
  type BudgetConfig,
  estimateRemainingDecisions,
  previewBudget,
  type RemainingCounts,
  shadowCharge,
  shadowRequest,
  type ShadowTimer,
  startShadowTimer,
  type TimerRules,
  type TimerSnapshot,
} from "./deadline.ts";
export { assertLocal, LocalUrlError } from "./local-url.ts";
export { type MemoryTransport, memoryTransport, type MemoryTransportOptions } from "./memory-transport.ts";
export { NORMAL_CLOSURE, type Transport, TransportClosedError } from "./transport.ts";
export { BINARY_FRAME_CLOSE, ConnectError, connectWebSocket, type WebSocketOptions } from "./websocket-transport.ts";
export {
  type Frame,
  GLOBAL_ROOM,
  ProtocolReader,
  type RoomEvent,
  type RoomRequestState,
  type SequencedRequest,
  splitFrame,
} from "./protocol.ts";
export {
  classifyRefusal,
  type CurrentRequest,
  type Refusal,
  RefusalError,
  type RefusalInput,
  type RefusalMode,
  type Submission,
} from "./refusal.ts";
export {
  type Challenge,
  type ChallengeDecision,
  ChallengeError,
  type ChallengeFilterOptions,
  type ChallengeRefusal,
  decideChallenge,
  parseChallenge,
  timerOn,
  toId,
  userName,
} from "./challenge.ts";
export {
  OtsDecline,
  OtsDeclineError,
  type OtsDeclineOptions,
  type OtsFailure,
  type OtsStatus,
} from "./ots.ts";
export {
  type A2BattleRecord,
  type A2BotOptions,
  type A2SessionResult,
  type BattleResult,
  type DecisionRecord,
  runA2Bot,
  SessionError,
  type SessionLog,
} from "./session.ts";
export {
  type AutoChooseDetection,
  DEFAULT_SNAPSHOT_WAIT_MS,
  DeadlineWatchdog,
  type RequestPhase,
  type WatchdogFire,
  type WatchdogOptions,
  type WatchdogReason,
} from "./watchdog.ts";
