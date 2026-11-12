// teams-publish (ADR-013 section 6), run by .github/workflows/teams-publish.yml.
// See src/publish/command.ts for the commands; credentials come from the
// environment only (docs/runbooks/teams-r2.md).
import { runPublishCli } from "../src/publish/command.ts";

process.exitCode = await runPublishCli(process.argv.slice(2), { out: (l) => console.log(l), err: (l) => console.error(l), env: process.env });
