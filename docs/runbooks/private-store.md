# Runbook: the private team store (ADR-015)

This runbook is for the project owner. It sets up the **private R2 bucket** that will hold the owner's real teams as plain Parquet shards (ADR-015, #889): the bucket, a 90-day lock on `snapshots/`, and the two scoped tokens, and it says how those tokens are handled. The owner performs every step, on the owner's workstation, with a Cloudflare dashboard login. The repository creates no bucket and no token; no agent and no CI job runs the commands below or holds a token.

Labels: **Owner: verify on Cloudflare** marks a statement about the Cloudflare dashboard or R2 that was not observable while writing this runbook (a menu name, a setting, a limit, a behaviour); do not rely on it before you have seen it. Dashboard names below are the ones to look for, not exact click paths. **Owner: verify** alone marks a behaviour of a local tool that needs a real token to see. **Owner: decide** marks a choice that is the owner's.

| Section | Covers | Status |
|---|---|---|
| 1 | What is protected, and what is not claimed | this runbook |
| 2 | Create the private bucket | this runbook |
| 3 | Check that nothing is public | this runbook |
| 4 | Lock `snapshots/` for 90 days | this runbook |
| 5 | Create the two tokens | this runbook |
| 6 | Credential handling | this runbook |
| 7 | Check the lock with a throwaway object | this runbook |
| 8 | Rotation | this runbook |
| 9 | Backup and rollback | this runbook; commands are planned |
| 10 | Owner checklist and journal note | this runbook |

Commands that do not exist yet are **planned**, not available: the import and `export` commands (#887), the snapshot-backed `PrivateTeamSource` (#888), the DuckDB read recipe for inspection (#890) and the first import (#891). This runbook names no flag or command of them beyond what ADR-015 states, and those tickets own their exact form. Until they land, the only thing to do with the bucket is sections 2 to 8.

## 1. What is protected, and what is not claimed

The asset is team builds, not credentials or personal data. ADR-015 accepts server-side encryption at rest and scoped tokens in place of client-side encryption. Facts the design rests on (ADR-015 sources S1, S2, S7, S11, S14, S15):

- R2 encrypts every object at rest, with keys managed by Cloudflare, so Cloudflare can read the objects.
- A bucket is not publicly accessible by default.
- A bucket's jurisdiction is fixed when it is created.
- Object tokens can be scoped to named buckets, and the secret access key is shown once.
- Writes with `If-None-Match: *` are atomic (the import will use them on every object under `snapshots/`), and R2 has no object versioning.
- A bucket lock on a prefix blocks overwrite and deletion for a set duration.

What this does **not** claim:

- **A leaked token exposes the teams.** A read token exposes every team; a write token can also publish a bad snapshot or overwrite `current.json`. The lock keeps earlier snapshots; it does not stop a new one. Readers check objects against the manifest's digests, which catches corruption, not a deliberate new snapshot from a holder of the write token (ADR-015 section 1).
- **The workstation-only rule is an operator rule.** Nothing in the code stops you running a reader elsewhere. Keep both tokens, and any private run, on this workstation (section 6).
- **The lock is not a backup.** It ends after its duration; section 9 covers backups.

## 2. Create the private bucket

Use the dashboard login of the Cloudflare account, never an Admin API token (a token that can edit the bucket's configuration can also edit its lock; keep that power in the dashboard session).

1. In the Cloudflare dashboard, open R2 object storage and create a bucket. Choose a name that is not the public data bucket's and does not say what is inside (the name is not secret, but nothing needs it to be descriptive). Write it down in your password manager.
   *Expected result:* a new bucket exists, separate from the public CC0 data bucket. **Owner: verify on Cloudflare** that the create form is under R2 object storage.
2. **Location / jurisdiction:** leave the default (no jurisdiction chosen, automatic location). The jurisdiction cannot be changed after creation, and ADR-015 fixes the default.
   *Expected result:* the bucket's settings page shows no jurisdiction restriction. **Owner: verify on Cloudflare** how the default is displayed.
3. Do not enable any public access or domain on this bucket (section 3).
   *Expected result:* a bucket with no public URL.

## 3. Check that nothing is public

Do this now, again after the lock (section 4) and after every change of the bucket's settings.

1. Open the bucket's settings and its public access section.
   - **Custom domains:** none listed.
   - **R2.dev subdomain** (the `r2.dev` URL): disabled.
   *Expected result:* no custom domain and no `r2.dev` URL for this bucket. **Owner: verify on Cloudflare** the section and setting names.
2. If an `r2.dev` URL is shown for the bucket, disable it and confirm the setting reads disabled.
3. Optional, from a shell that holds no token: an unauthenticated request to the bucket's S3 endpoint for a key must be refused.
   ```bash
   curl -sS -o /dev/null -w '%{http_code}\n' "https://<account-id>.r2.cloudflarestorage.com/<bucket>/probe"
   ```
   Replace `<account-id>` and `<bucket>` by typing them in; do not save the command. *Expected result:* a 4xx status (no object is served without a signature). **Owner: verify on Cloudflare** the exact status.

## 4. Lock `snapshots/` for 90 days

ADR-015 section 1: the lock stops deletion and overwrite under `snapshots/` for 90 days. Only `current.json` (outside `snapshots/`) is ever overwritten.

1. In the bucket's settings, find the **bucket lock** rules and add one:
   - **Prefix:** `snapshots/` (a prefix, so `current.json` stays writable);
   - **Condition:** age, **90 days** (a duration, not a fixed date and not indefinite).
   *Expected result:* the lock rule list shows one rule for `snapshots/` of 90 days. **Owner: verify on Cloudflare** the setting names and that a duration counts from each object's creation.
2. Note that the lock is editable with a token that can edit bucket configuration; neither of the two tokens in section 5 can (they are object tokens). Do not create an Admin token to "test" anything.
   *Expected result:* you still hold no API token of any kind.
3. **Owner: decide** whether 90 days is long enough; a longer or indefinite lock cannot be shortened later as easily, and ADR-015 settles on 90 days with the local copy (section 9) as the long-term backup.

## 5. Create the two tokens

Create both in the dashboard (the R2 section for managing API tokens), with your dashboard login.

| Token | Permission | Used by |
|---|---|---|
| write token | **Object Read & Write** | the import command only (planned, #887) |
| read token | **Object Read only** | readers: the snapshot-backed source (#888), DuckDB inspection (#890) |

For each:

1. Choose the permission above. **Never** choose an Admin permission or an account-wide scope.
2. **Specify bucket:** apply to the private bucket **only**.
3. **TTL / expiry:** 90 days. If the dashboard offers a shorter maximum or none of this length, take the longest it offers up to 90 days and record it in section 10.
4. Create the token. The dashboard shows the access key id and the secret access key **once**. Copy them straight into the token's env file (section 6) or your password manager. Do not take a screenshot, and do not paste them anywhere else.
   *Expected result:* two tokens exist; each lists the private bucket only, the right permission and an expiry date about 90 days out. **Owner: verify on Cloudflare** the field names, whether the form offers an expiry, and the longest lifetime it allows.
5. Write the creation date and expiry of each token in your password manager's entry (section 8 uses them).

Do not create a token with the bucket-configuration permission, and do not enable the Admin API for this account's work on this bucket.

## 6. Credential handling

Rules, from ADR-015 sections 1 and 7:

- **One env file per token**, each mode `0600`, under gitignored `private-data/` (for example `private-data/secrets/r2-write.env` and `private-data/secrets/r2-read.env`; the directory mode `0700`). The alternative is a hidden terminal prompt (`read -rs`) that fills the variable for one command. Never both in one file, never in the repository, in a dotfile that a shell loads on every start, or in a Coolify or CI secret.
- **Loaded only into the one command that needs it.** The write token's file goes into the import command only; the read token's file into a reader only. A subshell does it without leaving the variables in your shell:
  ```bash
  ( set -a; . private-data/secrets/r2-read.env; set +a; <one command> )
  ```
  The file holds the variables the commands read: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and the account id (the planned commands, #887 and #890, state the exact names; follow them, not this runbook, if they differ). **Owner: verify** each command against the names it documents.
- **Create the file without echoing the secret:**
  ```bash
  umask 077; mkdir -p private-data/secrets; ${EDITOR:-nano} private-data/secrets/r2-read.env
  stat -c '%a %n' private-data/secrets/r2-read.env
  ```
  *Expected result:* `600`. Confirm `git status` shows nothing under `private-data/` (it is gitignored; if it shows up, stop).
- **Never** put a token, an access key id or the account id in: a chat, a ticket or issue, a PR, a commit, a log, an error message, a screenshot, a command-line argument or a shell history line. An argument shows in `ps` and in history; a variable in the environment of one process does not stay in your shell.
- **Never a persistent DuckDB secret** (`CREATE PERSISTENT SECRET`): it is stored unencrypted under `~/.duckdb/stored_secrets`. The planned recipe (#890) uses a temporary secret only.
- **Agents never run a command that holds a token.** Do not start a coding agent session in a shell where the variables are exported, and do not paste an env file's content into one. The planned commands refuse agent sessions and non-terminals as accident prevention, not as a boundary.
- **This workstation only** (the WSL workstation, ADR-015 section 7). The Coolify host, CI, any cloud worker and any other machine never receive a token or a private team. This is an operator rule, not a code guarantee. If you want another host, that is a re-open trigger of ADR-015, not a step here.
- Environment variables are readable by the same user through `/proc` and inherited by child processes; this is accepted under the single-user assumption.

## 7. Check the lock with a throwaway object

This closes the ADR's Unknown: whether an object token's overwrite and delete are refused under the lock (the docs say so; it was not tested). The command line tool below is user-installed and in no `package.json`, lockfile or image of this repository; any S3 client that can set an endpoint works. Commands that need the planned import (#887) are not used.

Preparation: install an S3-compatible client yourself (for example the AWS CLI). The account id goes into the endpoint, which the client reads from the environment so it does not appear on a command line. Add this line to the write token's env file (and no other file):

```text
AWS_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
AWS_DEFAULT_REGION=auto
```

**Owner: verify** that your client version reads `AWS_ENDPOINT_URL` and accepts `auto` as the region for R2; if not, follow Cloudflare's R2 documentation for your client.

The throwaway object stays in the bucket until its lock ends (that is the point of the test): use a key outside the layout the import will write, and put no team content in it.

1. Upload the object with the write token. Use a key like `snapshots/_lock-check/<yyyy-mm-dd>.txt` and a file that says only `lock check`:
   ```bash
   echo 'lock check' > /tmp/lock-check.txt
   ( set -a; . private-data/secrets/r2-write.env; set +a; aws s3api put-object --bucket <bucket> --key snapshots/_lock-check/<yyyy-mm-dd>.txt --body /tmp/lock-check.txt --if-none-match '*' )
   ```
   *Expected result:* success. A repeat with `--if-none-match '*'` is refused (the object exists). **Owner: verify** that your client has `--if-none-match`; if not, skip the repeat.
2. Overwrite it with the write token (no `--if-none-match`):
   ```bash
   echo 'changed' > /tmp/lock-check.txt
   ( set -a; . private-data/secrets/r2-write.env; set +a; aws s3api put-object --bucket <bucket> --key snapshots/_lock-check/<yyyy-mm-dd>.txt --body /tmp/lock-check.txt )
   ```
   *Expected result:* **refused** with an access or locked-object error. Record the error class only (not the text). If it succeeds, the lock does not work as documented: stop, do not import, and put the result on the ticket.
3. Delete it with the write token:
   ```bash
   ( set -a; . private-data/secrets/r2-write.env; set +a; aws s3api delete-object --bucket <bucket> --key snapshots/_lock-check/<yyyy-mm-dd>.txt )
   ```
   *Expected result:* **refused**. If the delete is accepted or silently does nothing, check that the object still exists (next step) before drawing any conclusion.
4. Read it back with the read token, and check that the content is still `lock check`:
   ```bash
   ( set -a; . private-data/secrets/r2-read.env; set +a; AWS_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com aws s3api get-object --bucket <bucket> --key snapshots/_lock-check/<yyyy-mm-dd>.txt /tmp/lock-check-back.txt )
   cat /tmp/lock-check-back.txt
   ```
   *Expected result:* `lock check`. Delete both `/tmp` files afterwards. The object itself stays for 90 days; it is not a team and is harmless.
5. Record the results (section 10): overwrite refused or not, delete refused or not, the dashboard's maximum token lifetime (section 5), and the date.

Also confirm, with the **read** token, that a write is refused (a `put-object` of a different key under another prefix). *Expected result:* refused. **Owner: verify.**

Planned imports and readers must ignore the `_lock-check/` prefix (it is not a snapshot hash); tell #887 if it does not.

## 8. Rotation

Rotate **each token** at its expiry and **immediately on suspicion** (a token seen in a log, chat, screenshot, issue or PR; a shell history line; a machine you do not trust; a lost laptop or backup).

1. Create a new token of the same permission, bucket scope and 90-day expiry (section 5), and write its file (section 6) as a new file.
2. Replace the env file, then delete the expired or suspected token in the dashboard.
   *Expected result:* the old token is gone from the token list; the old file is deleted (`shred -u` is not meaningful on all filesystems; delete the file, and treat a suspected secret as exposed regardless).
3. If the old token was exposed, look at the bucket: the snapshots listed, `current.json`, and the lock rule. Earlier snapshots under the lock cannot have been changed within 90 days. Treat every snapshot younger than the leak as suspect and compare it with the verified local copy (section 9).
4. Put the next expiry date in your calendar; the expiry is also the reminder.

An expired token fails with an access error at the next command; nothing else breaks.

## 9. Backup and rollback

The bucket is the system of record, and the lock protects snapshots for 90 days. After that nothing but your own copies protects them. ADR-015 sections 1 and 3 settle three copies.

1. **Verified local snapshot copy.** The planned import (#887) writes, after it verifies the uploaded objects by downloading them again, the snapshot under `private-data/snapshots/<snapshot_hash>/` (the manifest and its shards). That directory is both a backup and a reader source. Keep it on a local Linux filesystem, mode `0700` for the directories.
2. **Staging is deleted only after the copy verifies.** The staged `.txt` files in `private-data/teams/` are the change set, not the store. Delete them only after the import has reported the local copy verified, and the snapshot's counts match what you staged. The first import is #891; until it is done, the `.txt` files are your only copy of the teams, so do not delete them and do not move them off this workstation.
3. **Recommended, not required: an offline copy.** Encrypt the local snapshot directory with `age -p` (a passphrase) and keep the result off the workstation (for example on a drive you own), updated after each new snapshot:
   ```bash
   tar -C private-data/snapshots -cf - <snapshot_hash> | age -p -o <snapshot_hash>.tar.age
   ```
   `age` autogenerates a passphrase when you leave the prompt empty; store it in your password manager, apart from the archive. `age` is not a dependency of this repository (a test checks that it is in no manifest); install it yourself. **Owner: verify** the command against the installed `age` version. Do not commit or upload the archive anywhere public; a `*.age` file is gitignored.
4. **Rollback with `export`.** The planned `export` command (#887) writes a snapshot's active records back to staged `.txt` files with their headers, from the verified local copy or from the bucket (read token). That is how you recover an earlier state or edit a team: export, change the files, stage and import again as a new snapshot. Do not rely on it before #887 has landed; the command, its flags and the staged file format are specified there and in ADR-015 section 3.
5. **Loss row.** If the bucket is lost and the local copies are, the teams are gone; the only remaining copy is the offline archive. This is why step 3 is recommended.

Snapshots under the lock and the local copies are the whole set. Private teams never go to a public bucket, a Release, an image, a fixture, a commit or a cloud worker (ADR-015 section 6), and never to a chat.

## 10. Owner checklist and journal note

Check each item, then post the note.

- [ ] The bucket exists, separate from the public data bucket, default jurisdiction.
- [ ] No custom domain and no `r2.dev` URL (section 3).
- [ ] A 90-day lock on `snapshots/` (section 4).
- [ ] An Object Read & Write token and an Object Read only token, each scoped to the bucket only, each with an expiry (section 5); neither is an Admin token, and no Admin API token was created.
- [ ] Each token is in its own mode-`0600` file under `private-data/secrets/` or filled from a prompt; `git status` shows nothing from `private-data/` (section 6).
- [ ] The throwaway-object check ran: overwrite refused, delete refused, read-token write refused (section 7).
- [ ] The dashboard's maximum token lifetime is recorded.
- [ ] The expiry dates are in your calendar (section 8).

**Journal note on #889** (by the owner, in this shape; it contains no secret, no account id, no access key id, no token id and no bucket URL):

```text
Done. Private bucket created (default jurisdiction), separate from the public data bucket.
No custom domain, no r2.dev URL (checked <date>).
Bucket lock: snapshots/, 90 days.
Tokens: Object Read & Write (import only) and Object Read only, bucket-scoped, expiry <date>/<date>.
Lock check (<date>): overwrite with the write token: <refused|accepted>; delete: <refused|accepted>; write with the read token: <refused|accepted>.
Dashboard maximum token lifetime: <value>.
Rotation due: <date>.
```

If any line read "accepted", the lock does not work as documented: stop, say so on the ticket, and do not run the first import (#891).
