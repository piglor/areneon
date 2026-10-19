// The rules importer's entry point (ADR-007 §6), the container's ENTRYPOINT:
// import every published, attested rules Release of piglor/areneon into the
// rules store.
//
// Environment (never argv, never logged):
//   DATABASE_URL  the `importer` role's connection URL. Production uses
//                 `sslmode=verify-full` (with `sslrootcert=<CA file>` when the
//                 server's CA is not a public one), so the server's
//                 certificate and host name are checked.
//   GH_TOKEN      a read-only GitHub token for gh (Releases and attestations).
//   RULES_IMPORT_STATEMENT_TIMEOUT_MS, RULES_IMPORT_LOCK_TIMEOUT_MS,
//   RULES_IMPORT_IDLE_IN_TRANSACTION_TIMEOUT_MS
//                 optional session timeouts in ms (default 300000, 300000,
//                 60000); exceeding one rolls the import back (exit 1).
//
// Prints one JSON line; exit 0 when every step was inserted or already
// present, 1 on any refusal, 2 on a configuration error.

import { GhAttestationVerifier } from "./attestation.ts";
import { GitHubReleaseSource } from "./github-releases.ts";
import { runImportCommand } from "./import-command.ts";

process.exitCode = await runImportCommand({
  env: process.env,
  source: new GitHubReleaseSource({ verifier: new GhAttestationVerifier() }),
});
