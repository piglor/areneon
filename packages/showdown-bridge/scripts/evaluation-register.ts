// The offline evaluation-register CLI (ADR-010 O1): see src/evaluation-cli.ts.
//   node packages/showdown-bridge/scripts/evaluation-register.ts verify --register R --heads evaluation/register-heads
import { runEvaluationCli } from "../src/evaluation-cli.ts";

const result = runEvaluationCli(process.argv.slice(2));
process.stdout.write(result.out);
process.stderr.write(result.err);
process.exit(result.code);
