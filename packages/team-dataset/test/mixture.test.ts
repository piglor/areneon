// The mixture proposal (ADR-020 section 2, slice 2): the component is the first draw of a team's own stream,
// component R draws from the role-set table only, a config without `proposal` is the uniform path unchanged,
// and generation is byte-identical on rerun. Generation tests are small: the validator is slow.

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { recordHash, type TeamGeneratorConfig } from "@areneon/showdown-bridge";
import { afterAll, describe, expect, it } from "vitest";
import { KnownHashes, referencePoolHash } from "../src/dedupe.ts";
import { generate } from "../src/generate.ts";
import { Proposal, ProposalError } from "../src/proposal.ts";
import { type RoleSetsSource, roleSetsSource } from "../src/role-set-proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { ASSIGNMENT_KEY, engine, format, FORMAT, KEY, light, small, table, vocabulary } from "./support.ts";

const ROOT = join(import.meta.dirname, "..", "..", "..");
const roleTable = JSON.parse(readFileSync(join(ROOT, "fixtures", "showdown", FORMAT, "role-sets.json"), "utf8"));
const source = roleSetsSource(engine, vocabulary, table, roleTable);
const mixed = (rShare: number, hash: string = source.hash): TeamGeneratorConfig =>
  small({
    config_id: "mixture-test",
    proposal: {
      kind: "mixture",
      components: [
        { kind: "role_sets", role_set_table_hash: hash, share_permille: rShare },
        { kind: "uniform", share_permille: 1000 - rShare },
      ],
    },
  });
const stream = { config_hash: "d".repeat(64), plan_id: "mixture-test" };
const id = (text: string): string => text.toLowerCase().replace(/[^a-z0-9]/g, "");

describe("the committed example config", () => {
  it("is the mixture of uniform 300 and role_sets 700 over the committed table", () => {
    const committed = JSON.parse(readFileSync(join(ROOT, "config", "teams", FORMAT, "generator.mixture-r.json"), "utf8")) as TeamGeneratorConfig;
    expect(committed.proposal?.components).toEqual([
      { kind: "role_sets", role_set_table_hash: source.hash, share_permille: 700 },
      { kind: "uniform", share_permille: 300 },
    ]);
  });
});

describe("the component pick", () => {
  const proposal = new Proposal(vocabulary, table, mixed(700), source);
  const componentOf = (k: number) => proposal.component(engineRng(engine, drawSeed(KEY, stream, k)));

  it("is recomputable from the draw's seed alone and is close to 700 per mille in a large sample", () => {
    const n = 20_000;
    let r = 0;
    for (let k = 0; k < n; k++) if (componentOf(k) === "role_sets") r++;
    const sd = Math.sqrt(n * 0.7 * 0.3);
    expect(Math.abs(r - n * 0.7)).toBeLessThanOrEqual(5.5 * sd);
    // The same seed, the same component.
    expect([0, 1, 2, 3].map(componentOf)).toEqual([0, 1, 2, 3].map(componentOf));
  });

  it("is the first draw: proposing from a fresh stream picks the component the pick says", () => {
    for (let k = 0; k < 30; k++) {
      const team = proposal.propose(engineRng(engine, drawSeed(KEY, stream, k)));
      expect(team).toHaveLength(vocabulary.rules.max_team_size);
    }
  });

  it("draws nothing for the component when no mixture is configured", () => {
    const plain = new Proposal(vocabulary, table, small());
    const rng = { below: (): number => { throw new Error("drawn"); } };
    expect(plain.component(rng)).toBe("uniform");
  });
});

describe("component R uses the role sets only", () => {
  it("gives every member a role's moves and abilities (a Mega entry as its base forme holding the stone)", () => {
    const proposal = new Proposal(vocabulary, table, mixed(1000), source);
    const names = new Map(table.species.map((s) => [s.name, s.id]));
    let megas = 0;
    for (let k = 0; k < 300; k++) {
      const team = proposal.propose(engineRng(engine, drawSeed(KEY, stream, k)));
      // Two table entries may show as one species (a base forme and its Mega forme); the validator rejects such a team, as it does for the uniform proposal.
      for (const set of team) {
        const species = names.get(set.species) as string;
        // The set is some table entry's role: the entry is the species itself, or a Mega forme that shows as it and whose stone the set holds.
        const entries = source.table.species.filter((e) => (source.shows_as.get(e.id) ?? e.id) === species && (e.id === species || vocabulary.roster.find((r) => r.id === e.id)?.required_item === id(set.item)));
        const roles = entries.flatMap((e) => e.sets).filter((role) => role.abilities.includes(id(set.ability)) && set.moves.every((m) => role.moves.includes(id(m))));
        expect(roles.length, `${set.species}: no role lists ${set.ability} and ${set.moves.join(", ")}`).toBeGreaterThan(0);
        if (entries.some((e) => e.id !== species)) megas++;
        expect(new Set(set.moves).size).toBe(set.moves.length);
      }
    }
    expect(megas).toBeGreaterThan(0);
  });

  it("keeps the uniform item, nature and Stat Point draws", () => {
    const proposal = new Proposal(vocabulary, table, mixed(1000), source);
    const team = proposal.propose(engineRng(engine, drawSeed(KEY, stream, 5)));
    for (const set of team) {
      expect(Object.values(set.evs).reduce((a, b) => a + b, 0)).toBe(vocabulary.rules.ev_limit);
      expect(vocabulary.natures.map((n) => n.name)).toContain(set.nature);
      const stone = vocabulary.roster.find((r) => r.required_item !== undefined && id(set.item) === r.required_item);
      if (stone !== undefined) expect(id(set.item)).toBe(stone.required_item);
    }
  });
});

describe("a config without proposal is the uniform path", () => {
  it("proposes the same teams with or without a role-set table to hand", () => {
    const a = new Proposal(vocabulary, table, small());
    const b = new Proposal(vocabulary, table, small(), source);
    for (let k = 0; k < 20; k++) expect(b.propose(engineRng(engine, drawSeed(KEY, stream, k)))).toEqual(a.propose(engineRng(engine, drawSeed(KEY, stream, k))));
  });

});

describe("refusals of the proposal", () => {
  it("refuses a role_sets component with no table", () => {
    expect(() => new Proposal(vocabulary, table, mixed(700))).toThrow(ProposalError);
    expect(() => new Proposal(vocabulary, table, mixed(700))).toThrow(/no role-set table/);
  });

  it("refuses a table whose hash is not the one the config names", () => {
    expect(() => new Proposal(vocabulary, table, mixed(700, "0".repeat(64)), source)).toThrow(/hash mismatch/);
    const other: RoleSetsSource = { ...source, hash: "1".repeat(64) };
    expect(() => new Proposal(vocabulary, table, mixed(700), other)).toThrow(/hash mismatch/);
  });

  it("refuses a table that does not read against the vocabulary", () => {
    expect(() => roleSetsSource(engine, vocabulary, table, { ...roleTable, vocabulary_hash: "0".repeat(64) })).toThrow(/vocabulary_hash/);
  });
});

describe("a small mixture generation", () => {
  const scratch = mkdtempSync(join(tmpdir(), "mixture-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));
  const cfg = mixed(700);
  const run = (n: number) => {
    mkdirSync(join(scratch, `log-${n}`));
    return generate({
      engine, format, table, vocabulary, labelSet: light, config: cfg, roleSets: source,
      generationKey: KEY, assignmentKey: ASSIGNMENT_KEY, datasetId: "mixture-1", planId: "plan-1",
      sizes: { dev: 2, validation: 2, training: 5 }, generatorCommit: "a".repeat(40),
      reference: { hash: referencePoolHash([]), known: new KnownHashes([]) }, logDir: join(scratch, `log-${n}`),
    });
  };

  it("is byte-identical on rerun and records the mixture in its config hash", () => {
    const a = run(0);
    const b = run(1);
    expect(JSON.stringify(a.manifest)).toBe(JSON.stringify(b.manifest));
    expect(JSON.stringify(a.records)).toBe(JSON.stringify(b.records));
    expect(a.pilot.config_hash).toBe(recordHash(cfg));
    expect(recordHash(cfg)).not.toBe(recordHash(small()));
  }, 300_000); // each run draws a few hundred teams through the pinned validator
});
