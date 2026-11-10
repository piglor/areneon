# Runbook: publishing public team datasets (ADR-013 section 6)

This runbook is for the project owner. It sets up what the `teams-publish` workflow (`.github/workflows/teams-publish.yml`) needs to publish a **public, synthetic** team dataset snapshot: an R2 bucket with a duration lock, a bucket-scoped CI token, a GitHub environment and its secrets. The owner performs every step with a Cloudflare dashboard and a GitHub repository-admin login. The repository creates no bucket, token or environment, and no workflow runs on a credential until section 5 is done.

Labels: **Owner: verify on Cloudflare** marks a statement about the Cloudflare dashboard or R2 that was not observable while writing this runbook; do not rely on it before you have seen it. **Owner: decide** marks a choice that is the owner's.

| Section | Covers |
|---|---|
| 1 | What is published, and what is not claimed |
| 2 | Create the public-data bucket (default jurisdiction) |
| 3 | Lock `pub/v1/` for 365 days and abort incomplete multipart uploads |
| 4 | Create the CI token |
| 5 | Create the GitHub environment, its secrets and variables |
| 6 | Run the workflow |
| 7 | Rotation |
| 8 | GitHub limits the workflow relies on |
| 9 | Owner checklist |

## 1. What is published, and what is not claimed

Only data that is already public: synthetic team datasets (`data_class: synthetic`, generated with a **public** key). Never a sealed evaluation set, never anything owner-held; the workflow takes a dataset directory of the repository and refuses what the snapshot reader refuses. The bucket holds content-addressed Parquet shards under `pub/v1/shards/<sha256>` and each snapshot's manifest under `pub/v1/snapshots/<snapshot_hash>/snapshot.json`. The same manifest, attested, is the asset of the GitHub Release `teams-<snapshot_hash>`.

The bucket has **no public URL**: readers use credentials or the Release, never an open bucket. (Owner: decide whether a custom public domain is wanted later; it is not part of this slice.)

The Parquet file bytes depend on the writer's version and settings, so they are not compared across platforms; the logical hashes are (ADR-013 section 6). The two-platform check compares them between an x86-64 and an arm64 runner.

## 2. Create the bucket

In the Cloudflare dashboard, R2, create a bucket (any name; the workflow reads it from a repository variable, section 5) in the **default jurisdiction**, as the owner decided. Leave public access off (no `r2.dev` URL, no custom domain). **Owner: verify on Cloudflare** that the bucket shows as not publicly accessible.

## 3. Lock `pub/v1/` and abort incomplete multipart uploads

- Add a bucket lock rule with the prefix `pub/v1/` and a **duration of 365 days** (objects cannot be deleted or overwritten for that time). **Owner: verify on Cloudflare** the rule's exact name and that it applies to prefixes.
- Add a lifecycle rule that **aborts incomplete multipart uploads** (for example after 1 day), so an interrupted upload leaves nothing behind. **Owner: verify on Cloudflare** the rule's location and units.

The workflow uploads with `If-None-Match: *` and never overwrites or deletes; the lock is the second guard.

## 4. Create the CI token

Create an R2 API token with **object read and write**, scoped to **this bucket only**, with an **expiry** (**Owner: decide**; a short one such as 90 days, renewed per section 7). The secret access key is shown once: put it into the GitHub secret of section 5 straight away. Never put the token, its key id, its secret or the account endpoint into a file of the repository, an issue, a log or a command line; the upload command reads them from the environment only and refuses them on its command line.

## 5. GitHub environment, secrets and variables

1. Create the environment `teams-public` (Settings, Environments) with **required reviewers** (the owner) and restrict deployment to the `main` branch.
2. In that environment add the secrets `TEAMS_R2_ACCESS_KEY_ID`, `TEAMS_R2_SECRET_ACCESS_KEY` and `TEAMS_R2_ENDPOINT_URL` (`https://<account id>.r2.cloudflarestorage.com`).
3. Add the repository variable `TEAMS_R2_BUCKET` (the bucket name; not a secret).
4. Last, add the repository variable `TEAMS_PUBLISH_ENABLED` with the value `true`. **Until this variable exists the publish job is skipped**, and without the environment's secrets and bucket variable it verifies the plan and then does nothing (no attestation, Release or upload), so a half-finished setup cannot publish.

## 6. Run the workflow

Actions, `teams-publish`, Run workflow (on `main`) with the dataset directory and label set of the repository (defaults: the committed `fixtures-v1` dev and validation fixtures), a snapshot id, a **public** 64-hex shard key and the splits. The jobs:

1. `build` (x86-64 and arm64): writes the snapshot with the pinned DuckDB tool (`--require-pinned`), reads the shards back against the dataset byte for byte and writes a report of the logical hashes.
2. `check`: the two reports must agree (the two-platform check), then plans the publication (SHA-256 of every shard and of the manifest).
3. `publish` (after your approval): verifies the plan again, attests the manifest, creates the Release `teams-<snapshot_hash>` and uploads the shards and then the manifest. Rerunning is safe: an existing identical object counts as present; different bytes under a key are refused.

Above the size bound (ADR-013 section 2) the check compares 16 HMAC-chosen ranges of the draw stream instead of the whole pipeline: `checkPlan` and `comparePlatforms` (`packages/team-dataset/src/publish/`) choose and compare them; the workflow runs the full-pipeline comparison, which is all a fixture-sized dataset needs. Wiring the range digests into the workflow comes with the first dataset above the bound.

## 7. Rotation

Before the token expires create a new one (section 4), replace the three environment secrets, run the workflow once on a fixture dataset and delete the old token in the Cloudflare dashboard.

## 8. GitHub limits the workflow relies on

Quoted from the GitHub documentation at the time of writing (**Owner: verify** against the current docs before relying on them): a matrix may have up to 256 jobs per workflow run; a job on a GitHub-hosted runner is limited to 6 hours; an artifact can be kept up to 90 days (the workflow keeps the plan 90 days, longer than the 30-day environment approval timeout); artifact storage counts against the account's quota. The `ubuntu-24.04-arm` runner is available for public repositories. One generation job of 2^20 draws takes minutes (about 1 ms per draw after the pilot, `CLAUDE.md`), far inside the job limit.

## 9. Owner checklist

- [ ] Bucket created in the default jurisdiction, no public access.
- [ ] 365-day lock on `pub/v1/`, lifecycle rule aborting incomplete multipart uploads.
- [ ] CI token scoped to the bucket, with an expiry.
- [ ] Environment `teams-public` with reviewers; the three secrets; the variables `TEAMS_R2_BUCKET` and `TEAMS_PUBLISH_ENABLED`.
- [ ] A dry run on the fixture dataset; note the Release tag and the snapshot hash on the ticket.
