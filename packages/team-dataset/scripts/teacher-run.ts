// The teacher run of ADR-019 track 1 (#965): `damage-aware@v1` plays seeded A1 games on the training
// split of a generated team dataset (or the interim fixture pool without `--dataset`) and the
// directory it writes holds the Episodes, the learner-ready TrainingSet and the teacher's identity.
//   node packages/team-dataset/scripts/teacher-run.ts --run-key NAME --format gen9championsvgc2026regmc --games N --out DIR [--dataset DIR --dataset-hash HEX] …
import { interimPoolSource, runTeacherCli, UsageError } from "@areneon/showdown-bridge";
import { datasetPoolSource } from "../src/loop-pool.ts";

try {
  process.exitCode = await runTeacherCli(process.argv.slice(2), console.log, datasetPoolSource(interimPoolSource), process.argv[1]);
} catch (error) {
  console.error(`teacher-run: ${(error as Error).message}`);
  process.exitCode = error instanceof UsageError ? 2 : 1;
}
