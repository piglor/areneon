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
#     mounts the sealed inputs read-only. The container checks need access to
#     the Docker daemon (the owner's docker rights); without it they fail.
#     With --code DIR (the checkout whose code runs as the store user): it must
#     be owned by the store user or root and not writable by group or other.
#
#   check-evaluation-store.sh --store DIR --probe-denied
#     Run as the training user (for example `sudo -u <training user> ...`) or
#     inside a training container. Passes only when listing and entering DIR
#     are refused with "Permission denied" (EACCES): a path that does not exist
#     fails, so a typo cannot pass. Run the owner-mode check first: it proves
#     DIR exists.
#
# Exit 0 when every check passes, 1 when one fails, 2 on usage.
set -uo pipefail

store="" owner="" repo="" code="" probe=0
training=() evaluation=()
usage() { sed -n '2,30p' "$0" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --store) store="${2:-}"; shift 2 ;;
    --owner) owner="${2:-}"; shift 2 ;;
    --repo) repo="${2:-}"; shift 2 ;;
    --code) code="${2:-}"; shift 2 ;;
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
  me="$(id -un)"
  # Each way of looking must be refused with EACCES. ENOENT (a typo, a store
  # that is not there) is not a refusal and fails.
  denied() { # description, then the command
    local what="$1" out
    shift
    if out="$(LC_ALL=C "$@" 2>&1)"; then fail "$what succeeds for $me"
    elif [[ "$out" == *"Permission denied"* ]]; then pass "$what is refused for $me (Permission denied)"
    else fail "$what failed for $me, but not with Permission denied: ${out:-no message}"; fi
  }
  denied "ls $store" ls -- "$store"
  denied "cd $store" bash -c 'cd -- "$1"' _ "$store"
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

# No parent may be writable by group or other (a sticky bit, as on /tmp, keeps
# others from replacing what they do not own), or someone could swap the store.
parents_open() { # a path; prints each parent that group or other can write
  local parent pmode found=0
  parent="$(dirname "$(readlink -f "$1")")"
  while [[ "$parent" != "/" ]]; do
    pmode="$(stat -c '%a' "$parent")"
    if (((8#$pmode & 8#1022) != 0 && (8#$pmode & 8#1000) == 0)); then echo "$parent (mode $pmode)"; found=1; fi
    parent="$(dirname "$parent")"
  done
  return $((1 - found))
}
if open="$(parents_open "$store")"; then fail "a parent of $store is writable by group or other: $open"; else pass "no parent of $store is writable by group or other"; fi

# The code that runs as the store user must not be the training user's to edit.
if [[ -n "$code" ]]; then
  if [[ ! -d "$code" ]]; then fail "$code is not a directory"
  else
    stray="$(find "$code" ! -type l ! -user "$owner" ! -user root -print -quit 2>/dev/null)"
    [[ -z "$stray" ]] && pass "everything under $code is owned by $owner or root" || fail "$stray under $code is owned by neither $owner nor root"
    loose_code="$(find "$code" ! -type l -perm /022 -print -quit 2>/dev/null)"
    [[ -z "$loose_code" ]] && pass "nothing under $code is writable by group or other" || fail "$loose_code under $code is writable by group or other"
    if open="$(parents_open "$code")"; then fail "a parent of $code is writable by group or other: $open"; else pass "no parent of $code is writable by group or other"; fi
  fi
fi

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
      reaches=""
      while IFS= read -r src; do
        [[ -n "$src" ]] || continue
        trimmed="${src%/}" # "/" becomes empty and covers everything
        if [[ -z "$trimmed" || "$real" == "$trimmed" || "$real" == "$trimmed"/* || "$trimmed" == "$real"/* ]]; then reaches="$src"; break; fi
      done <<<"$sources"
      [[ -z "$reaches" ]] && pass "training container $name mounts nothing that contains or lies under $real" || fail "training container $name mounts $reaches, which contains or lies under $real"
      sock="$(grep -F docker.sock <<<"$sources" || true)"
      [[ -z "$sock" ]] && pass "training container $name has no Docker socket mount" || fail "training container $name mounts the Docker socket (root-equivalent on this machine)"
      cuser="$(docker inspect --format '{{.Config.User}}' "$name" 2>/dev/null)" || cuser="?"
      case "$cuser" in
        ""|"?"|root|root:*|0|0:*) fail "training container $name runs as root or an unknown user (${cuser:-default root}); a root bind mount is host-root equivalent" ;;
        *) pass "training container $name runs as $cuser" ;;
      esac
      priv="$(docker inspect --format '{{.HostConfig.Privileged}}' "$name" 2>/dev/null)" || priv="?"
      [[ "$priv" == "false" ]] && pass "training container $name is not privileged" || fail "training container $name is privileged or its setting is unknown ($priv)"
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
