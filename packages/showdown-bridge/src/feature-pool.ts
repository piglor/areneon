// A pool of Node worker threads that run the candidate feature builder (ADR-022 section 4, mitigation 3, #1030).
// `run(tasks)` returns the results in the order of `tasks`, whatever order the threads finish in, so the rows are
// the bytes `candidateFeatures` gives and never depend on scheduling or on the pool size (a test: sizes 0, 1 and N
// give identical rows). A game's tasks go to the thread that saw the game first, so that thread's per-game probe
// memo serves the later decisions; the memo changes the time of a build, never its rows.
// Size 0 runs the service on the calling thread (the single-thread measurement and the tests' reference).
// A worker is a `data:` module that imports `feature-service.ts` and only wires messages to `answerMessage`,
// as the private shard decode does, so no worker script sits outside the coverage gate.
import { Worker } from "node:worker_threads";
import { createFeatureService, type FeatureResult, type FeatureServiceExtras, type FeatureService, type FeatureServiceSpec, type FeatureTask, type WorkerAnswer } from "./feature-service.ts";

export class FeaturePoolError extends Error {
  override name = "FeaturePoolError";
}

const WORKER_SOURCE = (module: string): URL =>
  new URL(
    "data:text/javascript," +
      encodeURIComponent(
        `import { parentPort, workerData } from "node:worker_threads";` +
          `const { answerMessage, createFeatureService } = await import(${JSON.stringify(module)});` +
          `const service = createFeatureService(workerData.spec);` +
          `parentPort.on("message", (message) => parentPort.postMessage(answerMessage(service, message)));`,
      ),
  );

interface Pending {
  resolve: (result: FeatureResult) => void;
  reject: (error: Error) => void;
}

export class FeaturePool {
  readonly size: number;
  readonly #workers: Worker[] = [];
  readonly #load: number[] = [];
  readonly #owner = new Map<string, number>();
  readonly #pending = new Map<number, Pending>();
  readonly #inline: FeatureService | null;
  #next = 0;
  #failure: FeaturePoolError | null = null;

  /** `extras` (the belief port) cannot cross to a worker thread, so a pool with extras runs on the calling thread (size 0). */
  constructor(size: number, spec: FeatureServiceSpec, extras?: FeatureServiceExtras) {
    if (!Number.isInteger(size) || size < 0) throw new FeaturePoolError("the pool size is not a non-negative integer");
    if (extras !== undefined && size !== 0) throw new FeaturePoolError("the belief port stays on the calling thread: only size 0 takes it");
    this.size = size;
    this.#inline = size === 0 ? createFeatureService(spec, undefined, extras) : null;
    for (let k = 0; k < size; k++) {
      const worker = new Worker(WORKER_SOURCE(new URL("./feature-service.ts", import.meta.url).href), { workerData: { spec } });
      worker.on("message", (answer: WorkerAnswer) => this.#settle(k, answer));
      worker.on("error", () => this.#fail("a feature worker thread failed"));
      worker.on("exit", () => this.#fail("a feature worker thread ended"));
      this.#workers.push(worker);
      this.#load.push(0);
    }
  }

  #settle(worker: number, answer: WorkerAnswer): void {
    const pending = this.#pending.get(answer.id);
    this.#pending.delete(answer.id);
    this.#load[worker] = (this.#load[worker] as number) - 1;
    if (pending === undefined) return;
    if ("defect" in answer) pending.reject(new FeaturePoolError(answer.defect));
    else pending.resolve(answer.result);
  }

  #fail(message: string): void {
    this.#failure ??= new FeaturePoolError(message);
    for (const pending of this.#pending.values()) pending.reject(this.#failure);
    this.#pending.clear();
  }

  #route(game: string): number {
    const known = this.#owner.get(game);
    if (known !== undefined) return known;
    const least = this.#load.indexOf(Math.min(...this.#load));
    this.#owner.set(game, least);
    return least;
  }

  /** The results of `tasks`, in the order of `tasks`. */
  async run(tasks: readonly FeatureTask[]): Promise<FeatureResult[]> {
    if (this.#failure !== null) throw this.#failure;
    if (this.#inline !== null) {
      const service = this.#inline;
      return tasks.map((task) => service.run(task));
    }
    return Promise.all(
      tasks.map(
        (task) =>
          new Promise<FeatureResult>((resolve, reject) => {
            const id = this.#next++;
            const worker = this.#route(task.game);
            this.#pending.set(id, { resolve, reject });
            this.#load[worker] = (this.#load[worker] as number) + 1;
            (this.#workers[worker] as Worker).postMessage({ id, task });
            if (task.final === true) this.#owner.delete(task.game);
          }),
      ),
    );
  }

  /** Stop every thread. Safe to call twice. */
  async close(): Promise<void> {
    this.#failure ??= new FeaturePoolError("the pool is closed");
    await Promise.all(this.#workers.map((worker) => worker.terminate()));
  }
}
