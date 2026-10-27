import { createHash } from "node:crypto";
import { jcs } from "@areneon/showdown-bridge";
import { expect, it } from "vitest";
import { Proposal } from "../src/proposal.ts";
import { drawSeed, engineRng } from "../src/stream.ts";
import { config, engine, KEY, table, vocabulary } from "./support.ts";

// The generator-matched prior (config/opponent-model/<format>/generator-matched.uniform-v1.json) names a
// `generator_commit`: the commit of this proposal's draw rules. This digest is the behaviour of those
// rules, 200 teams from fixed seeds under the committed config, computed at that commit
// (b7cda7af2f1744ad4934d38fb379b1d3cfc3ffc2, before `drawStatPoints` was lifted out of the proposal).
// It fails on ANY change of what the proposal draws; a deliberate change needs a new commit, a new digest
// here, and a re-derived prior (node packages/showdown-bridge/scripts/export-generator-prior.ts --commit ...).
const GENERATOR_COMMIT = "b7cda7af2f1744ad4934d38fb379b1d3cfc3ffc2";
const DIGEST = "36c1988fa0462043ea4c301490a2dbdfd87681ef756852d493b6e72fdc63d4e4";

it("the proposal still draws what the committed generator-matched prior says it draws", () => {
  const proposal = new Proposal(vocabulary, table, config);
  const teams = Array.from({ length: 200 }, (_, k) => proposal.propose(engineRng(engine, drawSeed(KEY, { config_hash: "c".repeat(64), plan_id: "golden" }, k))));
  expect(createHash("sha256").update(jcs(teams as never)).digest("hex")).toBe(DIGEST);
  expect(GENERATOR_COMMIT).toMatch(/^[0-9a-f]{40}$/);
});
