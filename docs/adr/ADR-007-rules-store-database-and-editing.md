# ADR-007: Rules store, database and editing workflow

**Status:** Accepted (r4, 2026-10-18) | **Wave:** 1 | **Deciders:** core team (owner approves) | **Date:** 2026-10-18

ADR issue: #702 · Epic: #703 (slices #701, #704, #705) · Related: [ADR-002](ADR-002-languages-and-repository-layout.md), [ADR-003](ADR-003-pinned-showdown-reference-simulator.md) · Implementation: #701 (blocked by #702)

Research and both independent reviews ran on Opus sub-agents. The policy models (`gpt-5.6-sol` / `gpt-5.6-terra` for research, `gpt-5.3-codex-spark` / `gpt-5.4-mini` for review) are unavailable in this runtime.

**Revisions:** r1 (2026-10-18) first proposal. r2 (2026-10-18) addresses the Standards and Spec reviews on #702: byte-exact snapshot identity, a behavioural per-format hash in the manifest, an honest tamper-evidence claim, attested import, reviewed channel activation, self-contained replay, the owner deviation stated explicitly. r3 (2026-10-18) addresses the r2 re-review: channel changes publish as their own attested artifact, rollback is bounded by the engine pin, `reason` text leaves the replay hash, migration role and token named. r4 (2026-10-18) lets the channel check accept the PR's own build (so a pin bump can activate in one PR), names the historical builder, orders channel changes by `seq`, requires up-to-date branches on `main` and re-checks the channel on `main` before publishing; adds Convex to the alternatives.

---

## Context

The owner decided on 2026-10-18 that rules must not be hard-coded in code. They must be stored in a database, self-hosted on the owner's Coolify instance.

Today the supported-format registry (`SUPPORTED_FORMATS`) and the CTS policy check (`forceopenteamsheets`) are constants in `packages/showdown-bridge/src/formats.ts` (#670).

**What counts as "rules" here.** Rules here means **Areneon's own decisions and metadata**, not game mechanics:
- **Format registry:** which Showdown format IDs Areneon runs, mapped to a regulation label, season dates and the required game type.
- **Policy:** engine rules that make a format unacceptable, each with a reason. Example: forced Open Team Sheets breaks CTS.
- **Regulation metadata:** official source URLs and verification dates.
- **Gap ledger:** the verification status of each Champions mechanic, currently `docs/showdown-champions-gaps.md`.

**What stays out.** Legality and mechanics stay in the pinned engine. That covers Stat Point caps, IVs, roster, learnsets, items and clauses. ADR-003 makes the engine the only legality authority, and copying those rules into a database would create a second one. Engine facts such as the number of active slots are read from the engine (the `active` array of a request), never stored.

## Decision drivers

1. **Reproducibility.** Every fixture, episode and benchmark must replay from immutable, content-hashed rules, with no live service.
2. **Offline CI.** Required checks make no live database or network calls.
3. **Self-hosted on Coolify,** with low operations cost for one maintainer, backups included.
4. **Open contribution.** Outside contributors can propose rule changes, and the maintainer reviews them through the existing protected-`main` path: PR, required `ci`, CodeQL and DCO.
5. **Validation before publish.** A rule change is checked against the pinned engine before it can become a snapshot.
6. **Clients in TS, Python and Rust** (ADR-002).
7. **Licence.** Permissive and OSI-approved, compatible with Apache-2.0.
8. **Secondary driver.** The same backend should later be able to hold the encrypted private Team Vault (per-owner access control) and episode and checkpoint metadata. Those are separate ADRs.

## Evidence and alternatives

### A. Database (system of record for published snapshots)

| | **PostgreSQL 18** | Dolt | SQLite file / libSQL | Supabase (self-host) | TerminusDB |
|---|---|---|---|---|---|
| Licence | PostgreSQL Licence (BSD/MIT-like) [S1] | Apache-2.0 [S5] | Public domain [S9]; libSQL MIT [S10] | Apache-2.0 [S12] | Apache-2.0 [S14] |
| Current release | 18.6, supported to 2030-11-14 [S2] | 2.5.0; recent releases note breaking changes [S6] | 3.54.1 (2026-10-16) [S9b] | not stated on repo | v12.0.7 [S14] |
| Versioning / immutability | Not built in; append-only tables plus guards (built by us) | Native commits, `AS OF <hash>`, `dolt_diff` [S7][S8] | None in DB; the file hash is the ID | As Postgres | Native commits / time-travel [S14] |
| Coolify | **One-click database (PostgreSQL 16/17/18)** [S23]; private by default [S16]; scheduled backups to S3; restore [S17] | Not in catalogue; custom compose; own `dolt backup` [S18][S19][S20] | Not a one-click DB [S23] | One-click service, but about 11 components, min 4 GB RAM (8 GB recommended) [S18][S13] | Not in catalogue [S18] |
| TS / Python / Rust | All mature (Rust via sqlx) [S15] | Via the MySQL protocol [S5][S15] | Native; libSQL Python client is experimental [S10] | JS and Python official; Rust community-only [S12] | Rust community-only [S14] |
| Future per-owner vault | Row-level security, default-deny, `FORCE ROW LEVEL SECURITY` [S3] | Row-level security not found in the docs reviewed (unverified) | None | RLS, but the service key bypasses it [S13b] | Unknown |
| Main risk | Snapshot discipline is ours to build | Custom ops, no managed backup, small vendor | Binary diffs; no server for the later vault | Operational sprawl, secret handling | Thin ecosystem |

**Convex (self-hosted), considered at the owner's request.** Its backend is licensed FSL-1.1-Apache-2.0: "Competing Use" is restricted, and each release becomes Apache-2.0 two years after it ships [S34]. That is not OSI-approved, which fails driver 7, as Directus does. It is not a Coolify one-click database; it is deployed as a custom Docker Compose [S35][S18]. Its storage is SQLite by default or an external SQL database such as Postgres [S35], so the production setup runs Postgres anyway, with Convex in front of it. Data is reached through Convex's JavaScript/TypeScript functions, not SQL [S36], so there are no SQL triggers or grants for the append-only guarantees and no RLS for the vault. Its strength, reactive real-time sync to clients, is not something this decision needs: snapshots are immutable and read once per battle. Rejected; it could be revisited for a later real-time UI ADR.

The other Coolify one-click engines [S23] were screened out early. MySQL and MariaDB have no row-level security policies for the later vault (project knowledge, not re-verified here). MongoDB is not relational and its server licence is not OSI-approved (not re-verified here). The key-value stores (Redis, KeyDB, Dragonfly) and ClickHouse don't fit a small, durable, relational record store.

### B. Editing and publishing workflow

| | **GitOps: JSON records + JSON Schema** | Repo JSON + signed Release only (no DB) | GitOps: SQL data migrations | Dolt / DoltHub data PRs | Admin UI (Directus / NocoDB) | Hybrid (UI drafts → PR) |
|---|---|---|---|---|---|---|
| Review | One readable diff per record in a GitHub PR | Same | SQL diffs, harder to read for data | Row diffs; owner merges [S21] | Field compare and publish modal [S25] | PR, as in GitOps |
| Required checks / engine validation | Yes: the existing `ci` gate | Yes | Yes | **No PR actions on DoltHub** ("longer-term roadmap (like actions…)") [S21] | No pre-publish engine hook found (unverified) | Yes |
| Contributor access | Any GitHub user, with DCO | Same | Same | Needs a DoltHub account and Dolt knowledge | Needs a login on our server | Mixed |
| Licence | Ajv MIT [S27], `jsonschema` MIT [S28], Rust `jsonschema` MIT [S29] | Same | dbmate MIT [S22] | Apache-2.0 [S5]; DoltLab terms unknown | **Directus MSCL-1.0 (not OSI) [S24]; NocoDB Sustainable Use License, internal/non-commercial only [S26]** | Supabase Studio Apache-2.0 [S12] |
| Ops | Nothing new in slice 1; Postgres in slice 2 | Nothing new | Nothing new | DoltLab self-host, custom compose | One more Coolify service | Two systems |
| Meets "stored in a database" | Yes (slice 2) | **No** | Yes | Yes | Yes | Yes |

### Debate

**For Dolt (the strongest rival).** Dolt is the only option where versioned data is native. It offers commit hashes, `AS OF` queries and row diffs in SQL [S7][S8], so the database itself is the history.

**Against Dolt.**
- Every property we need from it is already provided by git plus the protected-`main` PR path, and that path is enforced by required checks.
- DoltHub pull requests have no actions or CI [S21], so a rules change would bypass engine validation.
- Dolt is not a Coolify database: it needs custom compose and its own backup path [S18][S20].
- Row-level security for the later vault was not found in its docs (unverified).
- Its recent releases note breaking changes [S6].
- Adopting it means running two version-control systems for one project.

**For "repo JSON + signed Release, no database" (the honest baseline).** In this design the database is a published copy, so the baseline delivers the same replay guarantees with no server at all.

**Against the baseline.** It doesn't meet the owner's requirement that rules are stored in a database. It gives live runtimes (A2, a later API) no queryable store and no path to the vault. Slice 1 of this ADR *is* the baseline; slice 2 adds the database on top without changing the authoring path.

**For an admin UI.** Non-developers could edit forms instead of JSON.

**Against an admin UI.**
- Today's editors are the maintainer and GitHub contributors, so this solves no current problem.
- The two mature UIs carry non-OSI licences [S24][S26].
- Edits would skip required CI and engine validation.

**Against SQL migrations for data.** Reviewers read intent ("M-C ends on 2 Dec") far more easily in a JSON record than in an `INSERT` statement. Migrations remain the right tool for the database schema itself.

**Against a SQLite artifact alone.** It satisfies reproducibility, but binary DB files diff poorly in review [S11], it is not the server-side database the owner asked for, and it gives no path to the vault. JSON snapshots also serve as the offline export.

**The case against our choice.** The snapshot discipline (append-only, hashing, attestation) is ours to build and test. RFC 8785 is Informational, not Standards Track [S30], and implementations can disagree across languages. Mitigation: canonicalisation happens once, at build time, in TypeScript. Every consumer then checks SHA-256 over the published bytes and never re-serialises.

## Proposed decision

> **Inference** from the evidence above.

> **Deviation for the owner to confirm.** The owner asked for rules to be stored in a database. This ADR makes the **git repo the source of record** (where rules are authored and reviewed) and **PostgreSQL the runtime store** (where published snapshots are stored and queried). Slice 1 ships with no database; slice 2 adds it within Wave 1. The reason: it is the only arrangement where every rule change passes required CI and the engine check, and where replay never needs a live service. Accepting this ADR accepts that split.

1. **Database: PostgreSQL 18.** Deploy it as a Coolify one-click database [S23] with scheduled S3 backups [S17]. It stores **published rules snapshots** and the channel history. The same instance can later host the vault under RLS [S3], subject to its own ADR.
2. **Editing: GitOps JSON records.**
   - Rules are authored as one JSON file per record under `rules/` in the repo.
   - Each file is validated by a JSON Schema (draft 2020-12) [S27][S28][S29]. Schemas live in `packages/schema/rules/`, in line with ADR-002's rule that contracts are JSON Schema.
   - Changes go through a normal PR, so the required `ci`, CodeQL and DCO checks apply.
   - The closed schema enums plus the pinned engine are the outer admissibility bound. There is no fallback to compiled constants.
3. **Validation before merge (offline CI).** For each record, CI:
   - runs the schema check;
   - resolves every `format_id` against the **pinned engine** (the format exists, its game type matches, and no forbidden rule is present);
   - checks that the snapshot's `showdown_commit` equals the pinned commit;
   - builds the snapshot bytes and both hashes (below);
   - re-runs the golden fixtures.
4. **Identity (two hashes, exact preimages).**
   - **Canonical form.** `jcs(value)` is RFC 8785 [S30], computed **only in TypeScript** in the bridge. The existing `canonicalJson` already sorts keys by UTF-16 code units and prints numbers the ECMAScript way. `jcs()` wraps it, adds the I-JSON checks (it rejects lone surrogates and non-finite numbers), and is tested against the RFC vectors. `canonicalJson` itself is unchanged, so `request_hash` and `ruleset_hash` keep their bytes; the golden fixtures prove it.
   - **`rules_snapshot_hash`** (release identity) = `sha256(bytes)`, where `bytes` = UTF-8 of `jcs({schema_version, showdown_commit, formats, regulations, gaps})`. Those five keys are the whole preimage. `source_commit` and `published_at` are provenance, outside the hash, so the same rules built from two commits get one hash.
   - **`format_rules_hash`** (replay identity) = `sha256(UTF-8 of jcs({schema_version, format_id, regulation, game_type, forbidden_rules}))` for the one format the battle uses, where `forbidden_rules` here is the list of rule IDs only. The schema requires `forbidden_rules` sorted by `rule` and unique, so reordering cannot change a hash. Reasons, season dates, sources and the gap ledger are not in it, so editing documentation never changes a manifest or forces a golden recapture.
   - The version manifest becomes `{showdown_commit, format_id, regulation, ruleset_hash, format_rules_hash}`. `ruleset_hash` stays the engine-derived hash it is today. Episodes and fixtures also record `rules_snapshot_hash` as provenance metadata next to the manifest; it is not part of replay identity. The digest graph is acyclic: snapshot bytes → `rules_snapshot_hash`; format record → `format_rules_hash` → manifest.
   - Behaviour-changing policy (`forbidden_rules`, `game_type`, `regulation`) is therefore bound into identity, not just range-checked.
5. **Channels and activation (reviewed in the repo).**
   - Channels are a closed enum, initially `stable` only.
   - `rules/channels/<channel>.json` is `{channel, seq, hash}`. It names the snapshot hash a channel serves. Activating or rolling back is a PR that changes `hash`, so it passes the same review and `ci`. Every change must raise `seq` by exactly 1, and CI enforces this against the base branch.
   - **CI check for a channel file (offline).** CI computes the allowed hashes:
     - the snapshot built from the PR's own HEAD, so a pin bump or new records can be activated in the same PR;
     - plus, with a full-history checkout, the snapshot at every earlier `main` commit that changed `rules/`, each rebuilt **with that commit's own builder** (the code at that commit, never today's loader or upcasters), so the bytes match what was published.

     CI then requires that the channel's hash is one of the allowed hashes and that its `showdown_commit` equals the current pin.
   - **Merged tree = tested tree.** The `main` ruleset must require branches to be up to date before merging (strict required checks). `main` uses squash merges with linear history, so "earlier `main` commits" means first-parent commits on `main`. As a second guard, `rules-publish` re-runs the channel check on the merged `main` commit and refuses to publish a channel artifact that fails it. Commits from before slice 1 have no builder and contribute nothing. No network call is needed beyond the checkout.
   - **Rollback is bounded by the engine pin.** Naming an older hash works only for snapshots validated against the current pin, because the runtime refuses any other (see Error behaviour). Rolling back across a pin bump means reverting the `rules/` records in a PR, which builds a new snapshot against the current pin.
   - **Which snapshot is which.** Tests and golden fixtures in CI use the snapshot built from the records at HEAD. Replay of an existing fixture or episode uses the format rules record embedded in that artifact (step 7). The channel hash affects only live runtimes.
6. **Publish (post-merge, pull model; the database is never exposed).**
   - A `rules-publish` job on `main` runs in the protected GitHub environment `rules-prod` with a required reviewer [S32]. The repo is public, so environment reviewers are available on the current plan; the maintainer may approve their own run (accepted, below).
   - The job publishes two kinds of attested artifact, each with a GitHub build-provenance attestation [S33]:
     - **Snapshot:** `rules-snapshot-<hash>.json`, the exact bytes. It is published only if no Release for that hash exists yet (idempotent per hash).
     - **Channel change:** `rules-channel-<channel>-<source_commit>.json` = `jcs({schema_version, channel, seq, hash, source_commit})`. It is published for every `main` commit that changes `rules/channels/<channel>.json`, including a rollback to a hash that already has a Release (idempotent per `(channel, source_commit)`).
   - A small TypeScript **importer** on Coolify, which reuses the bridge's code, pulls new releases over outbound HTTPS. Before it inserts anything it checks:
     - the attestation, pinned to repo `piglor/areneon`, workflow `rules-publish.yml` and ref `refs/heads/main` [S33];
     - that `sha256(bytes)` equals the file name's hash (snapshots);
     - that the bytes pass the schema for the artifact's own `schema_version`. Every published schema version is kept in `packages/schema/rules/`;
     - for a channel change, that the named snapshot is already in `rules_snapshot`, otherwise it waits and retries; and that `seq` is greater than the channel's current maximum, otherwise the artifact is stale and is recorded nowhere.
   - Without these checks, anyone who could write to Releases could slip in a file with a self-consistent hash.
   - Each artifact is imported in one transaction, with `ON CONFLICT DO NOTHING`. `rules_release` takes `source_commit` and `seq` from the channel artifact. Order comes from `seq`, not from arrival time, so a delayed `rules-prod` approval cannot make an older change active.
   - Secrets live only in Coolify. The importer needs the database credentials, plus a fine-grained GitHub token limited to read-only access to public repository contents. The token covers attestation lookups (`gh attestation verify` authenticates through `GH_TOKEN`) and avoids anonymous rate limits [S33]. The Postgres port stays private [S16].
7. **Runtime.**
   - Live runtimes (the A2 sparring bot, a later API) read the active `stable` snapshot through a read-only role. They re-hash `body` on read and **pin the snapshot hash when a battle starts**. A channel change affects only battles that start later.
   - **Replay is self-contained.** Every fixture and episode artifact embeds its format rules record next to the manifest, so replay needs neither Postgres nor GitHub. A snapshot can also be rebuilt from `source_commit` in the repo. Releases are a cache, not an authority.
   - Nothing in the battle loop calls the database per step.

```text
 contributor / owner
        │  PR: rules/*.json, rules/channels/stable.json
        ▼
 GitHub (protected main) ── required ci (offline) ─────────────────────┐
        │   schema 2020-12 ✓  pinned-engine check ✓  golden fixtures ✓ │
        │   bytes = jcs(snapshot) ; rules_snapshot_hash = sha256(bytes)│
        │   format_rules_hash per format → version manifest            │
        │ merge                                                        │
        ▼                                                              │
 rules-publish job (env rules-prod, reviewer; idempotent)             │
        │  rules-snapshot-<hash>.json, rules-channel-<ch>-<commit>.json│
        │  each with a build attestation                               │
        ▼                                                              │
 Coolify ── TS importer (pull; verify attestation, sha256, schema;     │
        │                one transaction, idempotent)                  │
        ▼                                                              │
 PostgreSQL 18:  rules_snapshot (append-only, body bytea = exact bytes)│
                 rules_release  (append-only, ordered by seq)          │
        │ read-only role; re-hash body; pin hash at battle start       │
        ▼                                                              │
 A2 sparring bot / later API ── showdown-bridge ──► pinned Showdown engine
                                                     (only legality authority)
 CI / tests / replay ── format rules record embedded in artifact ──────┘
```

### Database contract (initial)

| Table | Columns | Guarantees |
|---|---|---|
| `rules_snapshot` | `hash text PK` (64 hex), `body bytea` (the exact published bytes), `payload jsonb` (a query copy parsed from `body`), `schema_version int`, `showdown_commit text`, `source_commit text`, `published_at timestamptz` | Append-only. `schema_version` and `showdown_commit` are extracted from `body`, and the importer refuses a row whose columns disagree with it. `hash` is always checked against `body`, never against `payload`. |
| `rules_release` | `id bigserial PK`, `channel text` (CHECK in the closed enum), `seq int`, `hash text FK`, `source_commit text`, `activated_at timestamptz`; `UNIQUE (channel, seq)`, `UNIQUE (channel, source_commit)` | Append-only. A channel's active snapshot is its row with the highest `seq`; `id` and `activated_at` are informational. |

- **Ownership and roles.** Both tables are owned by a `NOLOGIN` role `rules_owner`. `importer` has INSERT and SELECT on both tables (SELECT for idempotency) and USAGE on `rules_release_id_seq`. `areneon_reader` has SELECT only. `rules_migrator` is the only login role that runs dbmate. It is a member of `rules_owner` and runs DDL after `SET ROLE rules_owner`, and its credentials are used only by the deploy step. UPDATE, DELETE and TRUNCATE are granted to nobody. A row-level `BEFORE UPDATE OR DELETE` trigger and a statement-level `BEFORE TRUNCATE` trigger raise errors.
- **The guarantee is tamper-evident, not tamper-proof.** The Coolify `postgres` superuser, or the table owner (including `rules_migrator` through `SET ROLE`), can still disable triggers or bypass them through `session_replication_role`. This is accepted. Readers re-hash `body`, the repo plus Releases can rebuild every row, and S3 backups keep history.
- **Schema migrations:** dbmate (MIT, plain SQL) [S22], reviewed in the repo like code. Temporal columns, if ever needed, can use PG18 `WITHOUT OVERLAPS` [S4].

### Snapshot payload (initial)

| Key | Contents |
|---|---|
| `schema_version` | integer, starts at 1 |
| `showdown_commit` | the pinned engine commit the snapshot was validated against |
| `formats` | per format ID: `{regulation, game_type, season: {start, end}, forbidden_rules: [{rule, reason}]}`, with `forbidden_rules` sorted by `rule` and unique. Slice 1 seeds `gen9championsvgc2026regmc` with `forbidden_rules: [{rule: "forceopenteamsheets", reason: "forced OTS breaks CTS"}]` |
| `regulations` | per regulation: `{sources: [{url, verified_on}]}` |
| `gaps` | per mechanic: `{status: verified \| unverified \| unsupported, evidence: [...]}` (slice 3) |

- Unknown keys are rejected (`additionalProperties: false`). Enums are closed.
- `season` is metadata. Battle start never checks it and replay never checks it, so no result depends on the wall clock.
- **Schema versions.** The loader reads every `schema_version` ever published. A bump ships an upcaster from N−1 to N with tests in the same PR. Old snapshots keep their bytes and hashes.

### Error behaviour

| Situation | Behaviour |
|---|---|
| Malformed, unknown-version or hash-mismatched snapshot | Fails closed |
| Attestation missing or from the wrong repo, workflow or ref | The importer refuses; nothing is inserted |
| A format not present in the snapshot | Rejected, as today |
| Snapshot `showdown_commit` ≠ the runtime's pinned engine, or a `format_id` the pinned engine doesn't define | The runtime refuses to start a battle |
| Database unreachable at runtime | The live runtime refuses to start a battle; it never falls back to compiled constants |
| Channel changes during a battle | No effect; the battle keeps its pinned hash |
| Channel artifact names a snapshot not yet imported | The importer retries later; the channel does not move |
| Channel artifact with `seq` ≤ the channel's current maximum | Stale; skipped and logged |

## Consequences and risks

| Risk | Mitigation / acceptance |
|---|---|
| Our snapshot discipline has a bug | Append-only grants plus triggers, readers re-hash `body`, tests; `format_rules_hash` is in every manifest |
| A superuser or the owner edits rows | Accepted as tamper-evident only: re-hash on read; rebuild from the repo; S3 backups |
| JCS output differs across TS, Python and Rust | Only TS canonicalises; every other consumer hashes the stored bytes |
| `jcs()` silently changes existing hashes | `canonicalJson` is unchanged; golden request fixtures and `ruleset_hash` tests must stay byte-identical |
| Release-asset tampering | Attestation verified before import; Releases are a cache, the repo is the authority |
| Coolify Postgres major-version upgrades are undocumented [S16] | Pin the major; rehearse an upgrade on a restored backup before bumping |
| The importer needs scheduled execution on Coolify (unverified) | Rollout slice 2 includes a spike. Fallback: a long-running container that polls |
| A rules change silently alters behaviour | `format_rules_hash` changes, so golden fixtures recapture, exactly like a pin bump |
| A single maintainer approves their own publish | Accepted for the initial phase; "Prevent self-review" stays off. The environment reviewer adds a deliberate second step |
| `canonicalize` makes no formal conformance claim [S31] | Not adopted; `jcs()` is tested against the RFC vectors directly |
| Rollback across an engine pin bump | Not possible by channel; done by reverting the records in a PR (accepted) |
| Two PRs merge with no conflict and leave a channel naming a hash `main` never builds | Up-to-date branches required; `rules-publish` re-checks on `main` and refuses |
| The full-history CI check grows with history | Bounded: only commits that touch `rules/` are rebuilt; revisit if it becomes slow |

A rules change no longer needs a code change. A new regulation becomes a records PR plus a pin bump.

**This ADR would need a new ADR if:**
- non-developers need to edit (revisit the hybrid workflow with an Apache-2.0 UI);
- rules data becomes large or relational enough that file diffs stop being reviewable (revisit Dolt);
- a hosted, non-self-hosted service is proposed;
- the database must become the authoring source of record.

## Migration and rollout

1. **Slice 1, #701 revised, offline only.**
   - Add `rules/` records, `rules/channels/stable.json` and the JSON Schema.
   - Add the TS loader, `jcs()` with RFC vectors, and both hashes.
   - Remove `SUPPORTED_FORMATS` and the hard-coded `forceopenteamsheets` check; the policy now comes from the seeded record.
   - Add `format_rules_hash` to the manifest; embed the format record in fixtures; recapture the golden fixtures; prove `request_hash` unchanged.
   - Update `CONTEXT.md`: Version manifest (5 fields), Rules snapshot, Format rules record, Rules channel.
   - Turn on "require branches to be up to date before merging" in the `main` ruleset (owner action).
2. **Slice 2: database (Wave 1).**
   - Coolify PostgreSQL 18 with S3 backups and a restore drill.
   - dbmate schema, roles and triggers; TS importer with attestation checks; the `rules-publish` workflow with environment `rules-prod`.
   - Spike: Coolify scheduled execution.
3. **Slice 3: gap ledger.** Move `docs/showdown-champions-gaps.md` into records and generate the doc from them.

**Rollback:**
- Slice 1 is a normal revert.
- In slice 2, Postgres holds only published copies. The repo, the Releases and the embedded artifact records are enough to rebuild every row.
- A bad snapshot is rolled back by a channel PR naming an older hash validated against the current pin, or across a pin bump by reverting the records.

## Non-goals

- Storing mechanics or legality, or engine facts such as active slots.
- The Team Vault, episodes and checkpoints (later ADRs).
- An admin UI.
- Any hosted, non-self-hosted service.
- Per-step database calls from the battle loop.
- Tamper-proofing against the database superuser.

## Sources

All accessed 2026-10-18.

| Label | Source | Supports |
|---|---|---|
| S1 | PostgreSQL Licence, https://www.postgresql.org/about/licence/ | Permissive BSD/MIT-like licence |
| S2 | PostgreSQL Versioning Policy, https://www.postgresql.org/support/versioning/ | 18.6 current; support to 2030-11-14 |
| S3 | PostgreSQL 18 docs, Row Security Policies, https://www.postgresql.org/docs/current/ddl-rowsecurity.html | Default-deny RLS; `FORCE ROW LEVEL SECURITY` |
| S4 | PostgreSQL 18 release notes, https://www.postgresql.org/docs/18/release-18.html (2025-09-25) | Temporal `WITHOUT OVERLAPS` / `PERIOD` |
| S5 | dolthub/dolt README, https://github.com/dolthub/dolt | Apache-2.0; MySQL-compatible server; DoltLab self-host |
| S6 | Dolt releases 2.4.0 and 2.5.0, https://github.com/dolthub/dolt/releases | Both release notes declare backward-incompatible changes |
| S7 | Dolt system tables, https://www.dolthub.com/docs/sql-reference/version-control/dolt-system-tables | `dolt_diff`, `dolt_log`, commit hashes |
| S8 | Dolt querying history, https://www.dolthub.com/docs/sql-reference/version-control/querying-history | `AS OF '<hash>'`, `dolt_history_*` |
| S9 / S9b | SQLite copyright https://www.sqlite.org/copyright.html; release history https://www.sqlite.org/changes.html | Public domain; 3.54.1 on 2026-10-16 |
| S10 | tursodatabase/libsql, https://github.com/tursodatabase/libsql | MIT; experimental Python client |
| S11 | sqldiff, https://www.sqlite.org/sqldiff.html | Diff limits: keyless tables, no triggers or views |
| S12 | supabase/supabase, https://github.com/supabase/supabase | Apache-2.0; official JS and Python clients; Rust community-only |
| S13 / S13b | Supabase self-hosting with Docker https://supabase.com/docs/guides/self-hosting/docker; RLS https://supabase.com/docs/guides/database/postgres/row-level-security | About 11 default components; min 4 GB RAM, 8 GB recommended; service role bypasses RLS |
| S14 | terminusdb/terminusdb, https://github.com/terminusdb/terminusdb | Apache-2.0; commits and time-travel; Rust community-only |
| S15 | launchbadge/sqlx, https://github.com/launchbadge/sqlx | Rust client for Postgres, MySQL and SQLite; MIT/Apache-2.0 |
| S16 | Coolify Databases overview, https://coolify.io/docs/databases/ | Databases get no public endpoint by default; no documented major-upgrade procedure |
| S17 | Coolify Backups, https://coolify.io/docs/databases/backups | Scheduled backups, S3 upload, retention, restore |
| S18 | Coolify Services overview and catalogue, https://coolify.io/docs/services/overview, https://coolify.io/docs/services/all | Supabase, Directus and NocoDB are templates; Dolt, DoltLab and TerminusDB are absent |
| S19 | Coolify Docker Compose, https://coolify.io/docs/knowledge-base/docker/compose | Custom compose deployment |
| S20 | Dolt backups, https://www.dolthub.com/docs/sql-reference/server/backups | `dolt backup sync`; copying files is unsafe while running |
| S21 | DoltHub pull requests, https://dolthub.com/docs/concepts/dolthub/prs | Row diffs; owner merges; "longer-term roadmap (like actions…)" |
| S22 | amacneil/dbmate, https://github.com/amacneil/dbmate | MIT; plain SQL migrations |
| S23 | Coolify "Choose a database", https://coolify.io/docs/databases/choose-a-database | One-click engines, including PostgreSQL 18, 17 and 16 |
| S24 | Directus licence, https://github.com/directus/directus/blob/main/license, https://directus.com/mscl | MSCL-1.0, not OSI; "Competing Use" excluded; GPL-3.0 four years after each release |
| S25 | Directus content versioning, https://directus.com/docs/guides/content/content-versioning | Draft versions, compare/publish modal, `directus_revisions` |
| S26 | NocoDB licence, https://github.com/nocodb/nocodb/blob/develop/LICENSE.md | Sustainable Use License: internal, non-commercial or personal use only |
| S27 | Ajv docs https://ajv.js.org/json-schema.html; npm https://www.npmjs.com/package/ajv (8.20.0) | MIT; draft 2020-12 via a separate class |
| S28 | Python `jsonschema` 4.26.0, https://pypi.org/project/jsonschema/ | MIT; full draft 2020-12 |
| S29 | Rust `jsonschema` 0.59.0, https://github.com/Stranger6667/jsonschema | MIT; draft 2020-12 |
| S30 | RFC 8785 JSON Canonicalization Scheme, https://www.rfc-editor.org/rfc/rfc8785 (June 2020) | Informational; ECMAScript number form; UTF-16 key sort; I-JSON input |
| S31 | npm `canonicalize` 5.1.0, https://www.npmjs.com/package/canonicalize | Apache-2.0; README says "as defined by RFC8785" with no conformance claim (considered, not adopted) |
| S32 | GitHub Actions environments, https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments | Required reviewers; secrets released only after protection rules pass; "Prevent self-review" option |
| S33 | GitHub artifact attestations, https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds; `gh attestation verify` manual, https://cli.github.com/manual/gh_attestation_verify | Build-provenance attestations; `--repo`, `--signer-workflow` and `--source-ref` constraints; authentication via `GH_TOKEN` |
| S34 | Convex backend licence, https://github.com/get-convex/convex-backend/blob/main/LICENSE.md | FSL-1.1-Apache-2.0; "Competing Use" restricted; Apache-2.0 on the second anniversary of each release |
| S35 | get-convex/convex-backend README https://github.com/get-convex/convex-backend; self-hosted guide https://github.com/get-convex/convex-backend/blob/main/self-hosted/README.md | Docker (recommended) or binary; "most features of the cloud product"; "By default the Convex backend will store all state in a local SQLite database"; Postgres or MySQL for production |
| S36 | Convex docs, data types, https://docs.convex.dev/database/types | Documents are JavaScript objects; queries written in JS/TS with index filters |
