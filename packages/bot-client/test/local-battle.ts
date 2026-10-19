import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRuntimeCopy, launch, type LocalServer } from "@areneon/local-server";
import { fileURLToPath } from "node:url";
import { importPaste, loadEngine, type PokemonSet } from "@areneon/showdown-bridge";

/** Helpers for the tests that run the pinned server on 127.0.0.1 (ADR-005 §4). */

export const FORMAT = "gen9championsvgc2026regmc";

export const TEAMS_DIR = fileURLToPath(new URL(`../../../fixtures/teams/${FORMAT}`, import.meta.url));

/** A synthetic CC0 team from fixtures/teams (see its manifest). */
export function fixtureTeam(file: "legal-synthetic.txt" | "legal-opponent.txt"): PokemonSet[] {
  return importPaste(loadEngine(), readFileSync(join(TEAMS_DIR, file), "utf8"));
}

/** A port nothing listens on right now, chosen by the OS on 127.0.0.1. */
export function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.once("error", fail);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      probe.close(() => done(port));
    });
  });
}

export function tempDir(label: string): { dir: string; remove: () => void } {
  const dir = mkdtempSync(join(tmpdir(), `areneon-bot-client-${label}-`));
  return { dir, remove: () => rmSync(dir, { recursive: true, force: true }) };
}

/** The launched pinned server on a free port, from a runtime copy in its own temp dir. */
export async function startLocalServer(): Promise<{ server: LocalServer; ws: string; stop: () => Promise<void> }> {
  const root = tempDir("runtime");
  try {
    const copy = createRuntimeCopy({ runtimeRoot: root.dir });
    const server = await launch({ dir: copy.dir, port: await freePort() });
    const ws = `ws://127.0.0.1:${server.session.port}/showdown/websocket`;
    const stop = async (): Promise<void> => {
      try {
        await server.stop();
      } finally {
        root.remove();
      }
    };
    return { server, ws, stop };
  } catch (error) {
    root.remove();
    throw error;
  }
}

/** Whether any process of the process group `pgid` is alive. */
export function groupAlive(pgid: number): boolean {
  try {
    process.kill(-pgid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}
