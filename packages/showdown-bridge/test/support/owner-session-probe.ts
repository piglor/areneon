// A probe for the owner-session guard tests: runs the real guard in a child
// with the stdio the test chose, and writes the outcome to a file (stdout may
// be a terminal or a pipe, so it is not the channel). Argument 1: "top" or
// "child"; argument 2: the file to write.
import { writeFileSync } from "node:fs";
import { assertGuardedChild, assertOwnerSession, OwnerSessionError } from "../../src/owner-session.ts";

const [mode, out] = process.argv.slice(2) as [string, string];
let result: { ok: true } | { ok: false; code: string; message: string };
try {
  if (mode === "child") assertGuardedChild();
  else assertOwnerSession();
  result = { ok: true };
} catch (error) {
  result =
    error instanceof OwnerSessionError
      ? { ok: false, code: error.code, message: error.message }
      : { ok: false, code: "other", message: String(error) };
}
writeFileSync(out, JSON.stringify(result));
