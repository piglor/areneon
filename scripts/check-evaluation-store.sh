#!/usr/bin/env bash
# Read-only isolation checks for the evaluation store (ADR-010 O11, protocol
# I1). The owner runs it on the machine that hosts the store; it creates,
# changes and deletes nothing, needs no sudo of its own and prints no file
# content. Two modes:
#
#   check-evaluation-store.sh --store DIR --owner USER [--repo DIR]
#       [--training-container NAME]... [--evaluation-container NAME]
#     Run as the store owner or as root. Checks that DIR is owned by USER,
#     that no group or other bit is set on DIR or on anything under it, that
#     DIR is not inside a git working tree or under --repo, and, with docker,
#     that no training container mounts DIR and that the evaluation container
#     mounts the sealed inputs read-only.
#
#   check-evaluation-store.sh --store DIR --probe-denied
#     Run as the training user (for example `sudo -u <training user> ...`) or
#     inside a training container. Passes only when this user cannot stat,
#     list or enter DIR.
#
# Exit 0 when every check passes, 1 when one fails, 2 on usage.
set -uo pipefail

store="" owner="" repo="" probe=0
training=() evaluation=()
usage() { sed -n '2,21p' "$0" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --store) store="${2:-}"; shift 2 ;;
    --owner) owner="${2:-}"; shift 2 ;;
    --repo) repo="${2:-}"; shift 2 ;;
    --probe-denied) probe=1; shift ;;
    --training-container) training+=("${2:-}"); shift 2 ;;
    --evaluation-container) evaluation+=("${2:-}"); shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$store" ]] || usage

failures=0
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

if ((probe)); then
  # One check per way of looking: each must be refused to this user.
  if stat "$store" >/dev/null 2>&1; then fail "stat $store succeeds for $(id -un)"; else pass "stat $store is refused for $(id -un)"; fi
  if ls "$store" >/dev/null 2>&1; then fail "ls $store succeeds for $(id -un)"; else pass "ls $store is refused for $(id -un)"; fi
  if (cd "$store") >/dev/null 2>&1; then fail "cd $store succeeds for $(id -un)"; else pass "cd $store is refused for $(id -un)"; fi
  if [[ -r "$store" || -x "$store" ]]; then fail "$store is readable or searchable by $(id -un)"; else pass "$store is neither readable nor searchable by $(id -un)"; fi
  ((failures == 0)) && exit 0 || exit 1
fi

[[ -n "$owner" ]] || usage
if [[ ! -d "$store" ]]; then fail "$store is not a directory (or this user cannot see it)"; exit 1; fi

actual_owner="$(stat -c '%U' "$store")"
[[ "$actual_owner" == "$owner" ]] && pass "$store is owned by $owner" || fail "$store is owned by $actual_owner, not $owner"

mode="$(stat -c '%a' "$store")"
[[ "$mode" =~ ^[0-7]00$ ]] && pass "$store has mode $mode (no group or other bits)" || fail "$store has mode $mode: group or other bits are set"

# Anything below the top directory that group or other can reach.
loose="$(find "$store" -mindepth 1 -perm /077 -print -quit 2>/dev/null)"
[[ -z "$loose" ]] && pass "nothing under $store has a group or other bit" || fail "$loose has a group or other bit"

foreign="$(find "$store" -mindepth 1 ! -user "$owner" -print -quit 2>/dev/null)"
[[ -z "$foreign" ]] && pass "everything under $store is owned by $owner" || fail "$foreign is not owned by $owner"

# No parent may be writable by others (a sticky bit, as on /tmp, keeps others
# from replacing what they do not own), or someone could swap the store.
parent="$(dirname "$(readlink -f "$store")")"
open_parents=0
while [[ "$parent" != "/" ]]; do
  pmode="$(stat -c '%a' "$parent")"
  if (((8#$pmode & 8#1002) == 8#2)); then fail "$parent is world-writable (mode $pmode)"; open_parents=1; fi
  parent="$(dirname "$parent")"
done
((open_parents)) || pass "no parent of $store is world-writable"

real="$(readlink -f "$store")"
if git -C "$real" rev-parse --is-inside-work-tree >/dev/null 2>&1; then fail "$store is inside a git working tree"; else pass "$store is outside any git working tree"; fi
if [[ -n "$repo" ]]; then
  repo_real="$(readlink -f "$repo")"
  case "$real/" in
    "$repo_real"/*) fail "$store is under the repository $repo" ;;
    *) pass "$store is not under the repository $repo" ;;
  esac
fi

if ((${#training[@]} + ${#evaluation[@]})); then
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker is not on PATH, so the container checks cannot run"
  else
    for name in "${training[@]}"; do
      sources="$(docker inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$name" 2>/dev/null)" || { fail "docker cannot inspect training container $name"; continue; }
      if grep -Fq -- "$real" <<<"$sources"; then fail "training container $name mounts a path under $real"; else pass "training container $name mounts nothing under $real"; fi
      sock="$(grep -F docker.sock <<<"$sources" || true)"
      [[ -z "$sock" ]] && pass "training container $name has no Docker socket mount" || fail "training container $name mounts the Docker socket (root-equivalent on this machine)"
    done
    for name in "${evaluation[@]}"; do
      rw="$(docker inspect --format '{{range .Mounts}}{{.Source}} {{.RW}}{{"\n"}}{{end}}' "$name" 2>/dev/null)" || { fail "docker cannot inspect evaluation container $name"; continue; }
      sealed="$(grep -F -- "$real/sealed " <<<"$rw" || true)"
      if [[ -z "$sealed" ]]; then fail "evaluation container $name does not mount $real/sealed"
      elif grep -Fq ' true' <<<"$sealed"; then fail "evaluation container $name mounts $real/sealed read-write"
      else pass "evaluation container $name mounts $real/sealed read-only"; fi
    done
  fi
fi

((failures == 0)) && { echo "evaluation store: all checks passed"; exit 0; }
echo "evaluation store: $failures check(s) failed" >&2
exit 1
