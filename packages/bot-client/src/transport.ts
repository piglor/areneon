/**
 * The bot client's port to a Showdown server connection (ADR-005 §1): one
 * text frame in, one protocol line out. A frame is the server's raw message,
 * which may hold several protocol lines; splitting it is the caller's job.
 */
export interface Transport {
  /** Sends one protocol line. Throws once the transport is closed. */
  send(line: string): void;
  /** Registers a frame listener. Frames that arrived before the first listener are delivered to it. */
  onFrame(cb: (frame: string) => void): void;
  /** Registers a close listener; it gets the close code once. A listener registered after the close gets it at once. */
  onClose(cb: (code: number) => void): void;
  /** Closes the connection; a no-op once closed. */
  close(): void;
}

/** A send on a transport that is closed or not yet open. */
export class TransportClosedError extends Error {
  override name = "TransportClosedError";
}

/** Normal closure (RFC 6455 §7.4.1): the code a client-side close() reports. */
export const NORMAL_CLOSURE = 1000;

/**
 * The listener bookkeeping both transports share: frames are queued until the
 * first frame listener, and the close code is delivered exactly once to every
 * close listener, including one registered after the close.
 *
 * Every frame reaches every listener, in arrival order. A frame that arrives
 * during delivery (a listener's send made the peer answer at once) is only
 * queued, and goes out after the current one. A listener that throws does not
 * stop delivery: once the queue is empty, the first error is rethrown, or an
 * AggregateError if there were several.
 */
export class Listeners {
  readonly #frameListeners: ((frame: string) => void)[] = [];
  readonly #closeListeners: ((code: number) => void)[] = [];
  readonly #pending: string[] = [];
  #closeCode: number | undefined;
  #draining = false;

  get closed(): boolean {
    return this.#closeCode !== undefined;
  }

  onFrame(cb: (frame: string) => void): void {
    this.#frameListeners.push(cb);
    this.#drain();
  }

  onClose(cb: (code: number) => void): void {
    if (this.#closeCode === undefined) this.#closeListeners.push(cb);
    else cb(this.#closeCode);
  }

  frame(frame: string): void {
    this.#pending.push(frame);
    this.#drain();
  }

  #drain(): void {
    if (this.#draining || this.#frameListeners.length === 0) return;
    this.#draining = true;
    const errors: unknown[] = [];
    for (let frame = this.#pending.shift(); frame !== undefined; frame = this.#pending.shift()) {
      // A listener added during delivery starts with the next frame.
      for (const cb of [...this.#frameListeners]) {
        try {
          cb(frame);
        } catch (error) {
          errors.push(error);
        }
      }
    }
    this.#draining = false;
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) throw new AggregateError(errors, `${errors.length} frame listeners threw`);
  }

  close(code: number): void {
    if (this.#closeCode !== undefined) return;
    this.#closeCode = code;
    for (const cb of this.#closeListeners.splice(0)) cb(code);
  }
}
