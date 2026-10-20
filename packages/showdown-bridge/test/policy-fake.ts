// Shared by policy-worker.test.ts and fake-policy-worker.ts: what the fake worker claims to be.
export const ENCODER_VERSION = "fake-1";
export const ENGINE_DATA_HASH = "e".repeat(64);
export const CHECKPOINTS: [{ role: "battle"; checkpoint_id: string }, { role: "preview"; checkpoint_id: string }] = [
  { role: "battle", checkpoint_id: "b".repeat(64) },
  { role: "preview", checkpoint_id: "d".repeat(64) },
];
export const MAX_LINE_BYTES = 4_000_000;
