import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readTeamVocabulary, type TeamLabelSet } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { checkTeamConfigs, labelProblems, TEAMS_CONFIG_DIR } from "../src/check-labels.ts";
import { CONFIG_DIR, engine, FORMAT, labelSet, table, VOCABULARY_FILE, vocabulary } from "./support.ts";

const scratch = mkdtempSync(join(tmpdir(), "check-labels-"));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
// the golden vocabulary stands in for the 9-second export (team-vocabulary.test.ts holds it equal to a fresh one)
const golden = (_engine: unknown, _format: string, data: typeof table) => readTeamVocabulary(JSON.parse(readFileSync(VOCABULARY_FILE, "utf8")), data);

describe("check-team-labels (ADR-013 section 3)", () => {
  it("accepts the committed label sets and generator configs", () => {
    expect(TEAMS_CONFIG_DIR.endsWith("config/teams")).toBe(true);
    expect(checkTeamConfigs(engine, TEAMS_CONFIG_DIR, golden)).toEqual([]);
  });

  it("finds no problem in the committed label set against the vocabulary", () => {
    expect(labelProblems(labelSet, vocabulary)).toEqual([]);
  });

  it("finds a label value that can never hold, and a label space that is too big", () => {
    const dead: TeamLabelSet = {
      ...labelSet,
      labels: [
        { label_id: "ghost", values: [{ value: "yes", when: { kind: "has_move", moves: ["nosuchmove"] } }], default: "no" },
        { label_id: "other", values: [{ value: "yes", when: { kind: "all_of", of: [{ kind: "has_species", species: ["nosuchmon"] }, { kind: "has_item", items: ["nosuchitem"] }, { kind: "has_ability", abilities: ["nosuchability"] }] } }], default: "no" },
      ],
    };
    const problems = labelProblems(dead, vocabulary);
    expect(problems).toHaveLength(4);
    expect(problems[0]).toMatch(/label ghost, value yes: no id of its has_move is in the vocabulary/);
    expect(problems.join("\n")).toMatch(/has_species.*has_item.*has_ability/s);
    const wide: TeamLabelSet = { ...labelSet, labels: Array.from({ length: 17 }, (_, k) => ({ label_id: `l${k}`, values: [{ value: "y", when: { kind: "has_move" as const, moves: ["protect"] } }], default: "n" })) as unknown as TeamLabelSet["labels"] };
    expect(labelProblems(wide, vocabulary)[0]).toMatch(/more than 65536/);
  });

  it("reports a typo in an id, a missing file and a config the schema refuses, by file", () => {
    const root = join(scratch, "teams");
    const dir = join(root, FORMAT);
    mkdirSync(dir, { recursive: true });
    const labels = JSON.parse(readFileSync(join(CONFIG_DIR, "labels.json"), "utf8")) as TeamLabelSet;
    const typo = structuredClone(labels);
    ((typo.labels[3] as TeamLabelSet["labels"][number]).values[0] as { when: { moves: string[] } }).when.moves = ["fakeoutt"];
    writeFileSync(join(dir, "labels.json"), JSON.stringify(typo));
    writeFileSync(join(dir, "generator.json"), JSON.stringify({ schema_version: 1 }));
    const problems = checkTeamConfigs(engine, root, golden);
    expect(problems).toHaveLength(2);
    expect(problems[0]).toMatch(new RegExp(`${FORMAT}/labels.json: .*fakeoutt is not in the engine-data table`));
    expect(problems[1]).toMatch(new RegExp(`${FORMAT}/generator.json: invalid team generator config`));
    rmSync(join(dir, "generator.json"));
    expect(checkTeamConfigs(engine, root, golden)).toContain(`${FORMAT}/generator.json is missing`);
    cpSync(join(CONFIG_DIR, "labels.json"), join(dir, "labels.json"));
    expect(checkTeamConfigs(engine, root, golden)).toEqual([`${FORMAT}/generator.json is missing`]);
  });
});
