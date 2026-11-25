// The one error the throughput harness's callers cause: a flag or an input it cannot use (exit code 2).
export class UsageError extends Error {
  override name = "UsageError";
}
