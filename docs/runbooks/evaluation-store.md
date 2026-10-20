# Runbook: the evaluation store and register (ADR-010 O1, O11)

This runbook is for the project owner. It sets up the **isolated evaluation store**: the place where the sealed test set, its secret seed key, the sealed run artifacts and the evaluation register live, out of reach of every training job (evaluation protocol I1, I6, I7; ADR-010 O1, O11, owner decision Q7). The owner performs every step. The repository provides the checks and the register tool; it creates no user, container or mount, and no agent or CI job runs the commands below.

Labels: **Owner: decide** marks a choice to record on #846. **Owner: verify** marks a behaviour of the host (kernel, Docker, GitHub) that was not observable while writing this runbook; do not rely on it before you have seen it.

| Section | Covers |
|---|---|
| 1 | What is isolated, from whom, and what is not claimed |
| 2 | Create the store user and the directories |
| 3 | Keep training out: users, containers, mounts |
| 4 | Run the evaluation container |
| 5 | The register: commit a set, register a plan, publish the chain head |
| 6 | Verify the isolation (the check script) |
| 7 | Owner checklist |
| 8 | If the seal breaks |
| 9 | Sealed team generation, retirement and reveal (L3 to L5) |

## 1. What is isolated, from whom

Protocol I1 asks for "a separate OS user or container whose files and secret store training jobs cannot read". ADR-010 O11 (owner decision Q7) fixes the concrete store: **a separate OS user plus a container on this machine**. The files are owned by that user and mounted **read-only only into the evaluation container, never into a training container**.

| Kept in the store | Why it must not reach training |
|---|---|
| `sealed/key` and `sealed/salt` | The seed key fixes every sealed seed (D9); the salt opens the set commitment (L3) |
| `sealed/teams/` | The sealed teams themselves |
| `runs/` | Sealed run artifacts: `omniscient_log` and `hidden_verdicts` hold hidden information (I7) |
| `register/register.jsonl` | The append-only evaluation register (I6) |

Not kept in the store, and public by design: the commitments (hashes), the registered plans (they hold no secret, I2), the config records under `config/evaluation/`, and the chain heads under `evaluation/register-heads/`.

**What this does not claim.**
- **Root and the Docker daemon.** Root, and any user who can talk to the Docker daemon (the `docker` group, or a mounted `docker.sock`), can read everything on this machine. The isolation holds only if no training job runs as root, in the `docker` group, or with the socket mounted. The check script flags a mounted socket; it cannot see group membership beyond what you list in section 7.
- **Hardware and the host kernel** are shared. A side channel between processes of one machine is out of scope.
- **One machine, one owner.** The register's tamper evidence comes from the chain head being merged to `main` (section 5), not from the store's permissions.

## 2. Create the store user and the directories

Owner: decide the names. The commands below use `areneon-eval` for the store user and `/srv/areneon-eval` for its home; change them consistently, and give the same values to the check script.

```bash
# 1. A system user with no login shell and no password. (Owner: verify the useradd flags on this distribution.)
sudo useradd --system --user-group --create-home --home-dir /srv/areneon-eval \
  --shell /usr/sbin/nologin areneon-eval

# 2. Nobody but the user may enter its home.
sudo chmod 700 /srv/areneon-eval

# 3. The store layout, created as that user so every file is owned by it.
sudo -u areneon-eval install -d -m 700 \
  /srv/areneon-eval/store \
  /srv/areneon-eval/store/sealed \
  /srv/areneon-eval/store/sealed/teams \
  /srv/areneon-eval/store/runs \
  /srv/areneon-eval/store/register

# 4. The seed key and the salt: 256 random bits each, hex, mode 0600, never overwritten. The tool prints
#    only the SHA-256 (for the key, that is the key_commitment), never the secret. (Section 5 defines $ER.)
sudo -u areneon-eval -H $ER gen-secret --out /srv/areneon-eval/store/sealed/key
sudo -u areneon-eval -H $ER gen-secret --out /srv/areneon-eval/store/sealed/salt
```

Sealed team generation (section 9) makes its own key and salt for each set, in `sealed/<set_id>/`, so these two files are not needed for it.

`openssl rand -hex 32 > file` (with `umask 077`) makes an equivalent secret. Never print the key or the salt, never copy them out of the store, and never put them in a ticket, a chat or a shell history: the register keeps only their hashes (`key_commitment`, and the salt inside `set_commitment`).

The store must stay **outside every git working tree**, and outside this repository's checkout. `private-data/` is not a substitute: it is gitignored but sits in the working tree and in training's reach.

## 3. Keep training out

1. **Run training under a different user.** Training jobs run as a user that is neither `areneon-eval` nor root, and that is not in the `docker` group. (Owner: decide: your normal login user is fine if it is none of these; a dedicated `areneon-train` user is cleaner.)
2. **Training containers** run with `--user <training uid>:<training gid>` and mount only the training data and run directories. They never mount `/srv/areneon-eval`, any path under it, or `/var/run/docker.sock`.
3. **Mode 700 is the backstop.** Even a mistaken mount shows only an empty-looking or permission-denied directory to a uid that is not `areneon-eval`, because the top directory has no group or other bit.
4. **Do not widen the modes.** If a tool wants group access, change the tool.

## 4. Run the evaluation container

The evaluation runner is not part of this change (it comes with the harness tickets). When it exists, start it so that:

- it runs as the store user's uid and gid (`--user "$(id -u areneon-eval):$(id -g areneon-eval)"`);
- the **sealed inputs are read-only**: `-v /srv/areneon-eval/store/sealed:/store/sealed:ro`;
- only this container gets read-write mounts of the outputs: `-v /srv/areneon-eval/store/runs:/store/runs` and `-v /srv/areneon-eval/store/register:/store/register`;
- the root filesystem is read-only (`--read-only`, with a `tmpfs` for scratch) and it has no network (`--network none`), so nothing sealed can leave except through the mounted outputs;
- no Docker socket and no other host path is mounted.

The container name used in the check script (section 6) is whatever you pass as `--name`.

## 5. The register: commit a set, register a plan, publish the chain head

The register is `/srv/areneon-eval/store/register/register.jsonl`: one JCS line per entry, each carrying the previous entry's hash. The tool is offline: `node packages/showdown-bridge/scripts/evaluation-register.ts <command>`. It needs Node 24 and nothing else.

**The code that runs as the store user must not be the training user's to edit.** The key and the salt are within reach of whatever runs as `areneon-eval`. Use a checkout (of a reviewed `main` commit) that is **owned by `areneon-eval` or root and not writable by group or other**, in a directory whose parents are not writable by anyone else, and never your normal working checkout if the training user can write to it. Section 6 checks this with `--code`. Run the tool as one writer at a time: it appends and fsyncs, but takes no lock, so do not run two commands on one register at once.

Define for brevity (a shell on the owner's machine, in that checkout):

```bash
R=/srv/areneon-eval/store/register/register.jsonl
ER="node packages/showdown-bridge/scripts/evaluation-register.ts"
P=<path of the EvaluationParameters version in force>   # config/evaluation/evaluation-parameters.v1.json until a complete version is committed
G=<path of the TrainingGateRule version in force>        # config/evaluation/training-gate-rule.v1.json
HEADS=evaluation/register-heads                           # in a fresh checkout of main (see step 5)
```

Every command below that reads the register also checks every commitment and plan in it against the records: `--parameters`/`--rule` default to the committed v1 files, may be repeated (paired in order) when a register spans versions, and the commitment's hashes must equal the hashes of the files given, so a v1 file edited in place no longer matches. The run budget is the **parameters'** `b_sealed`, not the commitment's own field.

1. **Start the register once**: `$ER init --register $R`. This is the only command that creates the file; every other command refuses a missing register, so a mistyped path cannot start a second, empty chain.
2. **Check the config records.** `$ER config-check --parameters $P --rule $G --complete`. It refuses a provisional record: `n_teams_floor`, `n_teams`, `n_seeds` and `d_team` are measured in Wave 4 and written into a new parameters version before the commitment.
3. **Commit the sealed set (L3)**, before the first training run of a generation. The evaluation job writes a `SealedSetCommitment` JSON file (hashes and counts only; `set_commitment` is `SHA-256(JCS({salt, teams}))` and `key_commitment` the SHA-256 of the key, both computed inside the store) and runs `$ER commit-set --register $R --commitment <file> --parameters $P --rule $G --operator owner`. It refuses a count other than `n_teams`, a stratum below `n_stratum_min`, and hashes of other records. On retirement, `checkReveal` opens it: the hashes must match, the number of teams must be `team_count`, and the strata of the revealed teams must be exactly `stratum_counts`.
4. **Register the plan (I2)**, before the sealed run: `$ER register-plan --register $R --plan <plan.json> --parameters $P --rule $G --operator owner`. It refuses a plan that does not match the records it names. The plan file holds no secret; commit it in the same pull request as the head (step 5).
5. **Publish the chain head.** After every entry that must have a trusted time (always a plan registration, a commitment, an incident, a retirement):
   ```bash
   $ER head --register $R > head.json            # {schema_version, seq, entry_hash}: a hash and a position
   ```
   In a normal checkout (not the store), copy it to `evaluation/register-heads/$(printf '%010d' <seq>).json` on a branch, open a pull request, and have it merged to `main` through the protected-branch process. The merge time recorded by GitHub is the timestamp the owner cannot backdate (O1). Heads are **add-only**: CI (`heads-check --base`) fails a pull request that changes or deletes a head already on the base branch or adds one at or below a published position, and `CODEOWNERS` puts `evaluation/register-heads/` and `config/evaluation/` under the maintainer's review. A GitHub artifact attestation of the head file is an equivalent alternative (as `rules-publish.yml` does for rules). A Redmine journal note is a convenience copy only: journals are admin-editable and are **not** a trusted timestamp.
6. **Before the run starts**, read the heads from **merged `main`**, not from the working tree (an unmerged local file would satisfy a working-tree read):
   ```bash
   git fetch origin main
   $ER verify --register $R --parameters $P --rule $G --heads $HEADS --heads-ref origin/main --published <plan_hash>
   ```
   It exits 1 unless the whole chain verifies against the records, the register is not shorter than any published head, every published head matches the register's hash at its position, and the plan's own entry is covered by a published head. Start the run only after it exits 0.
7. **After the run**, append the outcome with `$ER append --register $R --entry <draft.json> --operator evaluation-job --parameters $P --rule $G --heads $HEADS --heads-ref origin/main` (a draft such as `{"kind":"run_outcome","plan_hash":"…","outcome":"reported","outcome_hash":"…"}`). An `l2a_recheck` or `run_outcome` naming a plan is **refused unless that plan's head is published** on `main`. The passing L2a re-check entry (`{"kind":"l2a_recheck","set_id":"…","plan_hash":"…","result":"pass"}`) must come first; the tool refuses a run without it. Every run, including a failed or aborted one, is logged (I6). Publish the new head the same way.

The tool refuses: a changed, dropped, reordered or forked entry; a register shorter than a published head; more plans than `b_sealed` (a failed run counts); a second plan for the same candidate; a run without a passing re-check; any result on a contaminated or retired set.

## 6. Verify the isolation

The script `scripts/check-evaluation-store.sh` makes read-only checks. It creates, changes and deletes nothing and prints no file content.

**As the store user (or root):**

```bash
sudo -u areneon-eval scripts/check-evaluation-store.sh \
  --store /srv/areneon-eval/store --owner areneon-eval --repo "$PWD" \
  --code <the checkout that runs the register tool> \
  --training-container <name of a running training container> \
  --evaluation-container <name of the evaluation container>
```
*Expected result:* every line `PASS`, last line `evaluation store: all checks passed`, exit 0. It checks: the owner; mode 700 with no group or other bit on anything below; no parent writable by group or other (a sticky bit as on `/tmp` is fine); not inside a git working tree or under the repository; `--code`: owned by the store user or root and not writable by group or other; per training container, **no mount that is, contains or lies under the store** (so a mount of `/srv` or `/` fails too), no Docker socket, a non-root user and not privileged; per evaluation container, `sealed/` mounted read-only. The container checks need Docker access (run as a user who has it) and fail without it.

**As the training user** (the proof that training cannot read the store):

```bash
sudo -u <training user> scripts/check-evaluation-store.sh --store /srv/areneon-eval/store --probe-denied
```
*Expected result:* two `PASS` lines saying `ls` and `cd` are refused with `Permission denied`. A path that does not exist fails (it is not a refusal), so run the owner-mode check first: it proves the store exists. Run it as the training user **on the host where the store exists**: the probe passes only on "Permission denied" and fails on a path that does not exist, so it cannot be run inside a container that has no such path. For training containers use the mount check above (`--training-container`), which shows that nothing containing the store is mounted.

Record the outputs on #846 (hashes of nothing secret are involved). Repeat after any change to users, groups, containers or mounts. The unit tests of the script (`tests/scripts/test_check_evaluation_store.py`) show that it fails on each leak it names; they run in CI against a scratch directory, not against your store.

## 7. Owner checklist

- [ ] `areneon-eval` exists, has no login shell, and `/srv/areneon-eval` has mode 700.
- [ ] `store/` and everything below it is owned by `areneon-eval`, with no group or other bit.
- [ ] The key and the salt were generated inside the store and were never shown anywhere.
- [ ] The store is outside this repository and any other git working tree.
- [ ] The training user is not root, not `areneon-eval`, and not in the `docker` group (`id <training user>`).
- [ ] No training container runs privileged, as root, or mounts `/srv/areneon-eval` or `docker.sock`.
- [ ] The evaluation container runs as the store user, mounts `sealed/` read-only, has no network and no Docker socket.
- [ ] Section 6 passed as the store user and as the training user; outputs recorded on #846.
- [ ] The checkout that runs the register tool is owned by `areneon-eval` or root and not writable by the training user (`--code` passed).
- [ ] The first chain head was merged to `main` by pull request (or attested) before any sealed run; `verify --heads-ref origin/main --published` passed.
- [ ] A scan for sealed teams in the working tree is scheduled before every sealed run and nightly in the store (protocol 7; ADR-010 O13). Owner: decide when; the scan job comes with the harness.

## 8. If the seal breaks

Protocol I8 applies: a sealed key, team or artifact read by a training job or committed; a result used for selection; an unregistered run; an exceeded budget; a run that deviates from its plan; a failed L2a re-check. Append an `incident` entry (the tool then treats the set as contaminated and retired and accepts only an aborted run for it), publish the new head, make a new set under L3 with a new key, and write the cause and the affected candidates to the tracker. Results obtained before the breach are kept but labelled with it.

## 9. Sealed team generation, retirement and reveal (L3 to L5)

The sealed team set is generated inside the store, by the store user, from a 256-bit key of the OS CSPRNG that the tool makes itself and writes only below the store (ADR-013 section 2; protocol L1 to L5). The commands are `node packages/team-dataset/scripts/sealed.ts <command>`, from the same reviewed checkout as the register tool (section 5). Nothing here is run by an agent or by CI against a real store; the tests use a throwaway directory and throwaway keys.

```bash
SE="node packages/team-dataset/scripts/sealed.ts"
S=/srv/areneon-eval/store
```

The tool refuses a store directory that is not absolute, not a real directory, not owned by the invoking user, has any group or other permission bit, lies inside a git working tree, or sits below a parent that group or other can write to (a sticky bit as on `/tmp` is accepted). It creates directories 0700 and files 0600, never overwrites, and refuses to read a store file that has been loosened. No option takes a key or a salt, and nothing it prints carries one: output is hashes and counts.

1. **Generate** (before the first training run of a generation, once the parameters are complete: `n_teams`, `n_teams_floor`, `n_seeds` and `d_team` measured):
   ```bash
   $SE generate --store $S --set-id set-1 --generation 1 \
     --public-manifest <manifest.json of the public dataset whose strata the set shares> \
     --parameters $P --rule $G --config config/teams/<format>/generator.json --labels config/teams/<format>/labels.json --format <format> \
     --known <records.jsonl> --known <records.jsonl> ... --max-attempts 3
   ```
   - **Same generator as the public pool.** The public manifest must be made with the same config, label set, vocabulary and version manifest (the tool checks the hashes). The strata are the public plan's, merged as it merged them: the sealed plan never merges again.
   - **`--known`** lists JSON Lines files whose lines carry the engine's `paste`: every public split (dev, validation and the training split), every released dataset, the training pool and every retired sealed set (their `teams.jsonl`). L1 (equal canonical hash) and L2 (within `d_team` differing fields, inclusive) skip a candidate that matches any of them. Omitting a file weakens L1/L2: list them all.
   - **Over-provision and discard.** The draw budget is the public `d_max` raised by the config's `sealed_margin_permille`. If a bin is still short, the whole generation is dropped (its key, teams and rejections are deleted), one line of hashes and counts goes to `discards.jsonl`, and a new key is drawn, up to `--max-attempts`. If every attempt is dropped nothing is committed and the discard log stays in `sealed/<set_id>/`: use a new set id. Do not repeat the run selectively on the strength of what was rejected (protocol §4.4).
   - It writes `sealed/<set_id>/{key,salt,teams.jsonl,rejections.jsonl,discards.jsonl,manifest.json,commitment.json}` and prints the commitment hashes. `commitment.json` holds hashes and counts only: pass it to the register as in section 5 step 3 (`commit-set`), and publish the head.
2. **Stamping.** The evaluation job builds `SealedTeamHashes` in memory with `loadSealedTeamHashes` (it recomputes the L3 commitment from the salt and the teams first) and unions it with the public hashes. It has no serialisation: never write it out.
3. **Retire (L5)**, when the budget is used or the set is contaminated. Order matters, because the register must name the reveal before it is published:
   ```bash
   $SE reveal-hash --store $S --set-id set-1          # prints {set_id, reveal_hash}: a hash only
   $ER append --register $R --entry <set_retirement draft with that reveal_hash> ...   # section 5; publish the head
   $SE retire --store $S --set-id set-1 --out <new directory> --register $R --parameters $P --rule $G
   ```
   `retire` refuses unless the register logs `set_retirement` for the set, with exactly this reveal's hash and the commitment this store holds. It writes `retired-manifest.json` (the key and the salt), `set-manifest.json`, `commitment.json`, `teams.jsonl`, `rejections.jsonl` and `discards.jsonl` into a **new** directory (0700, files 0600). From then on the set is CC0 and belongs to the training pool: copy the bundle out of the store only now, and add its teams to every later `--known` list.
4. **Verify (anyone, no store needed).**
   ```bash
   $SE verify --dir <bundle> --parameters $P --rule $G --config ... --labels ... --format <format> --known <reference pool files>
   ```
   It opens both commitments, checks the strata with the labeller (`checkReveal`), regenerates the set from the revealed key against the reference pool (the rejection log, with its L1 and L2 skips, must come out byte-equal) and checks every team again for legality, L1 and L2. Exit 1 names the first difference.

**Decisions this tool fixes where the protocol is silent** (recorded on #861; change them only with the protocol): L2 "within `d_team`" is distance at most `d_team`; the distance of two teams is the fewest differing fields over all pairings of their sets, a field being species, item, ability, nature, the move set or the Stat Points of one set; two sealed teams are compared with each other by L1 only; the quotas share `n_teams` over the public strata by their pilot hits (largest remainder, at least `n_stratum_min` each), so the total is exactly `n_teams`. The L1/L2 index is in memory: at the scale of the shard store it needs a later slice.

**Owner: verify** before relying on it: that the store directory checks pass on the real mount (the tool refuses rather than guesses), and that the retirement entry's `reveal_hash` printed by `reveal-hash` is the one you appended.
