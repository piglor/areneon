// The Python child of the projection gate (#1464): a child that dies, or a frame it cannot answer, fails the ask at once
// instead of hanging until the test's timeout (the corpus gate's is an hour).
import { describe, expect, it } from "vitest";
import { Child, skipProjection } from "./support/projection-corpus.ts";

describe.skipIf(skipProjection)("the projection child", () => {
  it("rejects a pending ask when the child exits, and every later one", async () => {
    const child = new Child(["-c", "import sys; sys.stdin.readline(); sys.exit(3)"]);
    await expect(child.ask({ kind: "view" })).rejects.toThrow(/exited \(code 3/);
    await expect(child.ask({ kind: "view" })).rejects.toThrow(/exited/);
  });

  it("rejects an ask the child answers with an error frame, naming the frame and the exception", async () => {
    const script = 'import sys, json\nfor line in sys.stdin:\n  print(json.dumps({"id": "x/1/p1", "status": "error", "code": "KeyError", "detail": "state"}), flush=True)\n';
    const child = new Child(["-c", script]);
    await expect(child.ask({ kind: "view" })).rejects.toThrow(/x\/1\/p1: KeyError: state/);
    child.close();
  });
});
