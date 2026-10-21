// Removing a scratch directory that held team text (ADR-015 section 3). The
// removal must never replace the error that made the caller stop, and must say
// nothing about what was in the directory: it answers only whether the
// directory is gone. When the first attempt fails, a best-effort chmod of the
// leftovers (a read-only directory or file stops `rm`) is tried once more.

import { chmodSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

export type Remover = (path: string, options: { recursive: true; force: true }) => void;

function reopen(path: string): void {
  try {
    chmodSync(path, 0o700);
    for (const entry of readdirSync(path, { withFileTypes: true })) {
      const child = join(path, entry.name);
      if (entry.isDirectory()) reopen(child);
      else chmodSync(child, 0o600);
    }
  } catch {
    // best effort: the second removal says whether it was enough
  }
}

/** Remove `path` and everything under it; true when it is gone. Never throws, never reports why. */
export function removeTree(path: string, remove: Remover = rmSync): boolean {
  try {
    remove(path, { recursive: true, force: true });
    return true;
  } catch {
    reopen(path);
  }
  try {
    remove(path, { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}
