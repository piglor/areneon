// The declared interaction classes (ADR-028 section 5, "Interactions (2)"): a reviewed data file
// (config/differential/battle-interaction-classes.v1.json) and the classifier that tags each step of a trace with the classes it
// exercises. The file holds every word that names game content; this module only reads it. A class matches a step when every
// group of its `all` list has a signal the step shows: a draw of a method, or an omniscient line of a type (optionally holding a
// text, or a `[from]` tag that starts with a text).
import type { Json } from "./canonical.ts";
import { sha256Jcs } from "./hash.ts";
import type { Step } from "@areneon/schema/types/v1/battle-trace.ts";

export interface LineSignal {
  line: string;
  contains?: string;
  from?: string;
}

export interface DrawSignal {
  draw: string;
}

export type Signal = LineSignal | DrawSignal;

export interface InteractionClass {
  id: string;
  description: string;
  all: Signal[][];
}

export interface InteractionClasses {
  schema_version: 1;
  data_class: "synthetic";
  classes: InteractionClass[];
}

export class ClassesError extends Error {
  override name = "ClassesError";
}

const ID = /^[a-z][a-z0-9_]*$/;
const isString = (v: unknown): v is string => typeof v === "string" && v !== "";
const keys = (v: object): string[] => Object.keys(v).sort();
const closed = (v: unknown, allowed: readonly string[], required: readonly string[]): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v) && keys(v).every((k) => allowed.includes(k)) && required.every((k) => k in v);

function readSignal(v: unknown): Signal {
  if (closed(v, ["draw"], ["draw"]) && isString(v.draw)) return { draw: v.draw };
  if (closed(v, ["line", "contains", "from"], ["line"]) && isString(v.line) && (v.contains === undefined || isString(v.contains)) && (v.from === undefined || typeof v.from === "string")) return v as unknown as LineSignal;
  throw new ClassesError("a signal is {draw} or {line, contains?, from?} (from may be empty: any [from] tag)");
}

/** Read the reviewed classes file: closed objects, ids unique and sorted, every class with at least one non-empty group. */
export function readInteractionClasses(value: unknown): InteractionClasses {
  if (!closed(value, ["schema_version", "data_class", "classes"], ["schema_version", "data_class", "classes"]) || value.schema_version !== 1 || value.data_class !== "synthetic" || !Array.isArray(value.classes)) {
    throw new ClassesError("an interaction classes file is {schema_version: 1, data_class: synthetic, classes}");
  }
  const classes = value.classes.map((c: unknown): InteractionClass => {
    if (!closed(c, ["id", "description", "all"], ["id", "description", "all"]) || !isString(c.id) || !ID.test(c.id) || !isString(c.description) || !Array.isArray(c.all) || c.all.length === 0) {
      throw new ClassesError("a class is {id, description, all} with a lower-case id and at least one group");
    }
    const all = c.all.map((group: unknown) => {
      if (!Array.isArray(group) || group.length === 0) throw new ClassesError("a group holds at least one signal");
      return group.map(readSignal);
    });
    return { id: c.id, description: c.description, all };
  });
  const ids = classes.map((c) => c.id);
  if (ids.some((id, i) => i > 0 && id <= (ids[i - 1] as string))) throw new ClassesError("class ids are unique and sorted");
  return { schema_version: 1, data_class: "synthetic", classes };
}

/** SHA-256 over the JCS of the classes file: the classifier's identity in a report. */
export const classesHash = (classes: InteractionClasses): string => sha256Jcs(classes as unknown as Json);

const hasTag = (line: string, prefix: string): boolean => line.split("|").some((field) => field.startsWith(`[from] ${prefix}`));

function shows(step: Pick<Step, "omniscient" | "draws">, signal: Signal): boolean {
  if ("draw" in signal) return step.draws.some((d) => d.method === signal.draw);
  return step.omniscient.some((line) => line.slice(1).split("|")[0] === signal.line && (signal.contains === undefined || line.includes(signal.contains)) && (signal.from === undefined || hasTag(line, signal.from)));
}

/** The ids of the classes a step exercises, in file order. */
export function classifyStep(classes: InteractionClasses, step: Pick<Step, "omniscient" | "draws">): string[] {
  return classes.classes.filter((c) => c.all.every((group) => group.some((signal) => shows(step, signal)))).map((c) => c.id);
}
