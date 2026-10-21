import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Engine, importPaste, loadEngine, resolveFormat } from "@areneon/showdown-bridge";
import { describe, expect, it } from "vitest";
import { ImportPlanError } from "../src/import-error.ts";
import { type ImportPlanInput, planImport } from "../src/import-plan.ts";
import type { StagedFile } from "../src/staged-file.ts";

// Paste canary tests (ADR-015 section 3, "No leaks"): marker strings placed in a
// paste, a file name, a tag, a header value, a comment line, a team_id, a
// --retire value, a label and an id source must appear in no refusal and in no
// summary, nor may the species, moves, items, abilities or nicknames of the team.

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const TEAMS = join(ROOT, "fixtures", "teams", "gen9championsvgc2026regmc");
const engine: Engine = loadEngine();
const format = resolveFormat(engine, "gen9championsvgc2026regmc");
const pool = readFileSync(join(TEAMS, "datasets", "fixtures-v1", "teams-dev.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => (JSON.parse(line) as { paste: string }).paste);
const [A, B] = [pool[0] as string, pool[1] as string];
const illegal = readFileSync(join(TEAMS, "illegal-item-clause.txt"), "utf8");

const NICK = "CanaryNick7";
const NAME = "canary-file-name-7.txt";
const TAG = "canarytag7";
const HEADER = "CanaryHeader7";
const COMMENT = "CanaryComment7";
const RETIRE = "CanaryRetire7";
const LABEL = "CanaryLabel7";
const ID_SRC = "CanaryIdSource7";
const HEXMARK = "c0ffee7c0ffee7c0ffee7c0ffee7c0f";
const MARKERS = [NICK, NAME, TAG, HEADER, COMMENT, RETIRE, LABEL, ID_SRC, HEXMARK];

const nicked = (paste: string, nick = NICK): string => paste.replace(/^([^@\n(]*?)( @ )/, `${nick} ($1)$2`);
const hex = (n: number): string => n.toString(16).padStart(32, "0");
const ports = {
  engine,
  format,
  labels: () => [{ label_id: "demo", value: "x" }],
  holdout: () => false,
  newTeamId: (() => {
    let n = 1;
    return () => hex(n++);
  })(),
};
const input = (staged: StagedFile[], over: Partial<ImportPlanInput> = {}): ImportPlanInput => ({ ...ports, staged, parent: null, retire: [], ...over });
const file = (name: string, text: string): StagedFile => ({ name, text });

// Everything team-shaped the pastes below carry.
const needles = (...pastes: string[]): string[] => {
  const out = new Set<string>(MARKERS);
  for (const paste of pastes) {
    for (const set of importPaste(engine, paste)) {
      for (const v of [set.species, set.item, set.ability, ...set.moves]) if (v.length >= 4) out.add(v);
    }
  }
  return [...out];
};
// All the text an error exposes.
const surface = (error: unknown): string => {
  const e = error as ImportPlanError;
  return [String(error), e.message, e.name, e.stack ?? "", JSON.stringify(e), JSON.stringify({ code: e.code, position: e.position })].join("\n");
};
const leaks = (text: string, found: string[]): string[] => found.filter((n) => text.includes(n));
const thrown = (run: () => unknown): unknown => {
  try {
    run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a refusal");
};

const parentPlan = planImport(input([file("a", A), file("b", B)]));
const parent = parentPlan.records;
const idA = parent[0]?.team_id as string;
const hdr = (id: string, tags = ""): string => `#! team_id=${id}${tags ? ` tags=${tags}` : ""}\n`;

const cases: [string, ImportPlanInput][] = [
  ["bad tag in the header", input([file(NAME, `#! tags=${TAG.toUpperCase()}\n${nicked(A)}`)])],
  ["unknown header key and value", input([file(NAME, `#! note=${HEADER}\n${nicked(A)}`)])],
  ["a comment line", input([file(NAME, `${nicked(A)}\n# ${COMMENT}`)])],
  ["a paste the engine cannot read", input([file(NAME, `${NICK} ${HEADER}\nAbility: ${TAG}`)])],
  ["an illegal team", input([file(NAME, `#! tags=${TAG}\n${nicked(illegal)}`)])],
  ["an unknown team_id", input([file(NAME, `#! team_id=${HEXMARK}ff tags=${TAG}\n${nicked(A)}`)], { parent })],
  ["a repeated team_id", input([file(NAME, hdr(idA, TAG) + nicked(B)), file(`${NAME}2`, hdr(idA) + nicked(B))], { parent })],
  ["a duplicate of an active team", input([file(NAME, `#! tags=${TAG}\n${A}`)], { parent })],
  ["a revision onto another active team", input([file(NAME, hdr(idA, TAG) + B)], { parent })],
  ["a malformed --retire value", input([], { parent, retire: [RETIRE] })],
  ["an unknown --retire team_id", input([], { parent, retire: [`${HEXMARK}ff`] })],
  ["a repeated --retire value", input([], { parent, retire: [idA, idA] })],
  ["retire and stage together", input([file(NAME, hdr(idA, TAG) + nicked(B))], { parent, retire: [idA] })],
  ["an invalid label", input([file(NAME, `#! tags=${TAG}\n${nicked(A)}`)], { labels: () => [{ label_id: `${LABEL}!`, value: "x" }] })],
  ["a bad id source", input([file(NAME, `#! tags=${TAG}\n${nicked(A)}`)], { newTeamId: () => ID_SRC })],
  ["a tampered parent", input([file(NAME, nicked(B))], { parent: parent.map((r) => ({ ...r, tags: [TAG], paste: nicked(r.paste), team_hash: "0".repeat(64) })) })],
  ["an empty result", input([])],
];

describe("paste canaries: refusals", () => {
  it.each(cases)("%s: no marker and no team content in the refusal", (_name, run) => {
    const error = thrown(() => planImport(run));
    expect(error).toBeInstanceOf(ImportPlanError);
    expect(leaks(surface(error), needles(A, B, illegal))).toEqual([]);
  });

  it("covers every refusal code the planner has", () => {
    const codes = new Set(cases.map(([, run]) => (thrown(() => planImport(run)) as ImportPlanError).code));
    // The codes of the cases above; the rest (record rules) are covered by import-plan.test.ts with no free text either.
    expect(codes.size).toBeGreaterThanOrEqual(12);
  });

  it("negative control: the scan does fail on an error that echoes a marker, a team name or a species", () => {
    const species = importPaste(engine, A)[0]?.species as string;
    for (const echo of [NICK, NAME, TAG, HEADER, species]) {
      const leaky = new ImportPlanError("team_illegal");
      leaky.message += ` (${echo})`;
      expect(leaks(surface(leaky), needles(A, B, illegal)), echo).not.toEqual([]);
    }
    // And the markers really are in the inputs the cases refuse.
    expect(nicked(A)).toContain(NICK);
    expect(importPaste(engine, nicked(A))[0]?.name).toBe(NICK);
  });
});

describe("paste canaries: a successful plan", () => {
  it("returns four counts and nothing else in the summary", () => {
    const out = planImport(input([file(NAME, `#! tags=${TAG}\n${nicked(A)}`), file(`${NAME}2`, nicked(B, "CanaryNick8"))]));
    expect(Object.keys(out.summary).sort()).toEqual(["new", "retired", "revised", "unchanged"]);
    expect(Object.values(out.summary).every((n) => Number.isInteger(n))).toBe(true);
    expect(leaks(JSON.stringify(out.summary), needles(A, B))).toEqual([]);
    // The records are the data and do hold the paste, as the negative control needs.
    expect(JSON.stringify(out.records)).toContain(NICK);
  });
});
