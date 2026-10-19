# Rules release fixtures

Recorded inputs for the `packages/rules-importer` tests and its end-to-end import (`make importer-e2e`), where `test/fake-gh.ts` replays them as gh's answers. CC0 under `fixtures/LICENSE`. The files are kept byte-exact (`-text` in `.gitattributes`) because the tests hash them.

| File | Source |
|---|---|
| `rules-snapshot-7c618d78….json` | GitHub release `rules-snapshot-7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8` of piglor/areneon, downloaded 2026-10-19 |
| `rules-channel-stable-b226c3bf….json` | GitHub release `rules-channel-stable-b226c3bfe889f8cf775933b0a964a44ef6d69913` of piglor/areneon, downloaded 2026-10-19 |
| `gh-verify-rules-publish.json` | `gh attestation verify <asset> --repo piglor/areneon --signer-workflow piglor/areneon/.github/workflows/rules-publish.yml --source-ref refs/heads/main --format json` (gh 2.74.0, 2026-10-19). Both assets gave the same output because one attestation names both. Trimmed to the certificate fields, the statement type, the subjects and the predicate type; the Sigstore bundle is dropped. |
| `gh-verify-unsigned.stderr` | gh's stderr (exit 1) for the same command on a file that no attestation names |
| `gh-api-releases.json` | `gh api --paginate --slurp 'repos/piglor/areneon/releases?per_page=100'` (gh 2.74.0, 2026-10-19), trimmed to each Release's `tag_name`, `draft`, `prerelease` and its assets' `name`, `state` and `digest` |
| `gh-api-tag-refs.json` | `gh api repos/piglor/areneon/git/ref/tags/<tag>` for both tags (same day), trimmed to `ref` and `object.sha` / `object.type`; both are lightweight tags on commits |

Trimmed from each `gh attestation verify` result: `attestation` (the whole Sigstore bundle), `verificationResult.mediaType`, `verificationResult.verifiedTimestamps`, `verificationResult.verifiedIdentity`, `statement.predicate`, and every certificate field except `issuer`, `subjectAlternativeName`, `githubWorkflowRepository`, `githubWorkflowRef`, `githubWorkflowSHA`, `buildSignerURI`, `sourceRepositoryURI`, `sourceRepositoryRef`, `sourceRepositoryDigest`, `runnerEnvironment` and `runInvocationURI`.
