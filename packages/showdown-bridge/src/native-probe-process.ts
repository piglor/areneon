// The native probe as a child process behind a SYNCHRONOUS request/response channel (#1534). The builders and the batch seam call
// the probe synchronously (`ProbeRunner`, `BatchRunner`), so the channel is two named pipes the parent reads and writes with
// non-blocking `fs` calls: a request is written whole, the response is polled (a short spin, then short sleeps), so a native
// answer that takes microseconds costs microseconds and a hung child ends in a timeout, not a hung game process.
//
// Why a process and not an addon: a panic in the engine (the vendored tree has a few `unwrap`s) ends the child, never the game
// process, and the backend routes everything to the pinned probe; the child holds no Node state; and nothing in the supply chain
// needs a native Node module. The frames are `[u32 little-endian length][JSON]`.
//
// Linux and macOS (`mkfifo`). The module names no path of the vendored fork: the caller passes the binary and the format export.
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { closeSync, constants, mkdtempSync, openSync, readSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export interface NativeProbeCommand {
  /** The `ninjask-probe` binary. */
  bin: string;
  /** The format export the pinned Showdown made (`make ninjask` writes them), read by the binary. */
  dex: string;
}

export class NativeProbeError extends Error {
  override name = "NativeProbeError";
}

const sleeper = new Int32Array(new SharedArrayBuffer(4));
const sleep = (ms: number): void => void Atomics.wait(sleeper, 0, 0, ms);

/** Default time a response may take (a batch of tens of thousands of queries is under a second). */
const RESPONSE_TIMEOUT_MS = 120_000;
const START_TIMEOUT_MS = 30_000;

export class NativeProbeProcess {
  readonly #dir: string;
  #child: ChildProcess | null = null;
  #fdIn = -1;
  #fdOut = -1;
  #dead: string | null = null;
  readonly #timeout: number;

  constructor(command: NativeProbeCommand, options: { timeout_ms?: number | undefined; start_ms?: number | undefined } = {}) {
    this.#timeout = options.timeout_ms ?? RESPONSE_TIMEOUT_MS;
    this.#dir = mkdtempSync(join(tmpdir(), "areneon-native-"));
    const inPath = join(this.#dir, "in");
    const outPath = join(this.#dir, "out");
    try {
      execFileSync("mkfifo", [inPath, outPath]);
      const child = spawn(command.bin, ["--dex", command.dex, "--in", inPath, "--out", outPath], { stdio: ["ignore", "ignore", "inherit"] });
      this.#child = child;
      child.on("error", (error) => void (this.#dead = `the native probe could not start (${error.message})`));
      child.on("exit", (code, signal) => void (this.#dead = `the native probe exited (${String(signal)} ${String(code)})`));
      child.unref();
      const deadline = Date.now() + (options.start_ms ?? START_TIMEOUT_MS);
      // The child opens its input first: our write end opens (non-blocking) only once its read end exists.
      for (;;) {
        try {
          this.#fdIn = openSync(inPath, constants.O_WRONLY | constants.O_NONBLOCK);
          break;
        } catch {
          // ENXIO: the child has not opened its end yet; any other failure to open ends in the same timeout.
          if (Date.now() > deadline) throw new NativeProbeError("the native probe did not start in time");
          sleep(1);
        }
      }
      // A non-blocking read end reads "end of file" until the child has opened its write end, so the first read end is a
      // blocking one (the open returns once the child, which opens its output right after its input, is connected); a
      // second, non-blocking read end of the same pipe is the one used, and the first is closed.
      const connected = openSync(outPath, constants.O_RDONLY);
      this.#fdOut = openSync(outPath, constants.O_RDONLY | constants.O_NONBLOCK);
      closeSync(connected);
    } catch (error) {
      this.close();
      throw error;
    }
  }

  get alive(): boolean {
    return this.#dead === null && this.#fdIn >= 0;
  }

  /** One frame out, one frame in. A dead or silent child is a `NativeProbeError` and the process is closed. */
  request(payload: Uint8Array): Buffer {
    if (!this.alive) throw new NativeProbeError(this.#dead ?? "the native probe is closed");
    try {
      const head = Buffer.alloc(4);
      head.writeUInt32LE(payload.length, 0);
      this.#writeAll(head);
      this.#writeAll(payload);
      const len = this.#readExactly(4).readUInt32LE(0);
      return this.#readExactly(len);
    } catch (error) {
      this.#dead ??= (error as Error).message;
      this.close();
      throw error;
    }
  }

  #writeAll(data: Uint8Array): void {
    let at = 0;
    const deadline = Date.now() + this.#timeout;
    while (at < data.length) {
      try {
        at += writeSync(this.#fdIn, data, at, data.length - at);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code === "EAGAIN") {
          if (Date.now() > deadline) throw new NativeProbeError("the native probe did not read its request in time");
          sleep(0.05);
          continue;
        }
        throw new NativeProbeError(`the native probe's pipe broke (${String(code)})`);
      }
    }
  }

  #readExactly(n: number): Buffer {
    const out = Buffer.alloc(n);
    let at = 0;
    let spins = 0;
    const deadline = Date.now() + this.#timeout;
    while (at < n) {
      let got: number;
      try {
        got = readSync(this.#fdOut, out, at, n - at, null);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (code !== "EAGAIN") throw new NativeProbeError(`the native probe's pipe broke (${String(code)})`);
        // Nothing yet: spin briefly (a native answer is microseconds), then sleep in short steps; the child may be dead.
        if (++spins > 400) {
          if (Date.now() > deadline) throw new NativeProbeError("the native probe did not answer in time");
          sleep(Math.min(1, spins / 4000));
        }
        continue;
      }
      if (got === 0) {
        // EOF with a writer gone: the child ended.
        throw new NativeProbeError("the native probe closed its output");
      }
      at += got;
      spins = 0;
    }
    return out;
  }

  /** Close the pipes, stop the child and remove the pipes' directory; safe to call twice. */
  close(): void {
    for (const fd of [this.#fdIn, this.#fdOut]) {
      if (fd >= 0) {
        try {
          closeSync(fd);
        } catch {
          // already closed
        }
      }
    }
    this.#fdIn = -1;
    this.#fdOut = -1;
    try {
      this.#child?.kill("SIGTERM");
    } catch {
      // already gone
    }
    try {
      rmSync(this.#dir, { recursive: true, force: true });
    } catch {
      // best effort
    }
  }
}
