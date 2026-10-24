// The Node child of the owner's private test command (ADR-015 erratum E1, item 2).
// Do not run it by hand: `python -m areneon_ml.loop private-test` holds the
// terminal guard, the private output root and the verified run directory, and
// calls this with the arguments below. It prints one JSON line (the
// PrivateTestRecord and the wall-clock per match) and writes no file.
import { runPrivateTestCli } from "@areneon/showdown-bridge";
import { datasetValidationPool } from "@areneon/team-dataset";
import { snapshotTeams } from "../src/test-source.ts";

process.exitCode = await runPrivateTestCli(
  process.argv.slice(2),
  (line) => process.stdout.write(`${line}\n`),
  { snapshot: snapshotTeams, dataset: ({ engine, config }) => datasetValidationPool(engine, config) },
  (line) => process.stderr.write(`${line}\n`),
);
