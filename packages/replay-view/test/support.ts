import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";
import vm from "node:vm";
import { INTERPRETER, ROOT, loadInterpreter } from "../../public-state/test/support.ts";
import type { Api } from "../../public-state/test/support.ts";

export { FORMAT, FORMAT_DIR, ROOT, fixtures, fullDex, loadInterpreter, readJson } from "../../public-state/test/support.ts";
export type { Api, Fixture, Frame, Interpreted } from "../../public-state/test/support.ts";

const require = createRequire(import.meta.url);

export const WEB = join(ROOT, "packages", "replay-view", "web");
export { INTERPRETER };

export interface ViewApi {
  init(doc: any, root: any, data: any, options?: any): any;
  boot(doc: any, win: any): any;
  make(doc: any, tag: string, attrs: Record<string, string | number> | null, kids?: unknown[], svg?: boolean): any;
  allowUrlAttribute(tag: string, name: string): void;
  forbidUrlAttributes(): void;
  align(master: string[], other: string[]): number[];
  createBoard(doc: any, env: any, perspective: string): any;
  decisionPanel(doc: any, request: any, decision: any, labels: Record<string, string>): any;
  parseChoice(choice: string): { kind: string; index: number; target: string; extra: string; raw: string }[];
}

/** The interpreter and the view, `require`d (the view finds the interpreter as a global, as in the page). */
export function loadView(): { replay: Api; view: ViewApi } {
  const replay = loadInterpreter();
  (globalThis as Record<string, unknown>).ArReplay = replay;
  return { replay, view: require(join(WEB, "view.cjs")) as ViewApi };
}

/** The page's own load: the two files joined into one script and run in a fresh context. */
export function loadPageScript(globals: Record<string, unknown> = {}): { replay: Api; view: ViewApi } {
  const source = [INTERPRETER, join(WEB, "view.cjs")].map((f) => readFileSync(f, "utf8")).join("\n");
  const context = vm.createContext({ ...globals });
  vm.runInContext(source, context);
  return vm.runInContext("({ replay: ArReplay, view: ArView })", context) as { replay: Api; view: ViewApi };
}
