// Validate config/teams/<format>/{labels,generator}.json against the pinned
// engine (ADR-013 section 3). Part of `make ts`.
import { loadEngine } from "@areneon/showdown-bridge";
import { checkTeamConfigs } from "../src/check-labels.ts";

const problems = checkTeamConfigs(loadEngine());
for (const problem of problems) console.error(problem);
if (problems.length > 0) process.exit(1);
console.log("team label sets and generator configs: ok");
