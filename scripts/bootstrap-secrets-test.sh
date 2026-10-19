#!/usr/bin/env bash
# Test for db/bootstrap.sql (#799): no role password reaches the server except
# inside its own CREATE ROLE statement. Runs the bootstrap against a throwaway
# Postgres 18 that logs every statement and every failing one, on a scratch
# Docker network with no published port, then:
#   1. a missing or blank password is refused, with its message and a nonzero
#      exit, and nothing reaches the server log;
#   2. with all three set, the bootstrap succeeds;
#   3. each password appears in the server log exactly once, on its own
#      CREATE ROLE line (the positive control that statements are logged).
# Passwords are random per run and never printed: counts only.
#   scripts/bootstrap-secrets-test.sh
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
# The image db/compose.yaml pins, for both server and psql client.
image="$(sed -n 's/^ *image: *\(postgres:.*\)$/\1/p' "$repo/db/compose.yaml")"
[[ -n "$image" ]] || { echo "bootstrap-secrets-test: no postgres image in db/compose.yaml" >&2; exit 1; }

run_id="areneon-bootstrap-test-$$-$RANDOM"
net="$run_id"
server="$run_id-pg"
cleanup() {
  docker rm -f "$server" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() { printf 'bootstrap-secrets-test: FAIL: %s\n' "$*" >&2; exit 1; }

docker network create "$net" >/dev/null
docker run -d --name "$server" --network "$net" \
  -e POSTGRES_DB=areneon -e POSTGRES_PASSWORD=throwaway-superuser \
  "$image" -c log_statement=all -c log_min_error_statement=debug5 >/dev/null
for _ in $(seq 60); do
  docker exec "$server" pg_isready -U postgres -d areneon -h 127.0.0.1 >/dev/null 2>&1 && break
  sleep 1
done
docker exec "$server" pg_isready -U postgres -d areneon -h 127.0.0.1 >/dev/null || fail "server did not start"

vars=(RULES_MIGRATOR_PASSWORD RULES_IMPORTER_PASSWORD RULES_READER_PASSWORD)
roles=(rules_migrator importer areneon_reader)
declare -A pw
for v in "${vars[@]}"; do
  pw[$v]="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export "$v=${pw[$v]}"
done

# Run the bootstrap in a client container. Variables pass by name (-e VAR),
# so no value is on a command line. Prints the exit status; output goes to $1.
bootstrap() {
  local out="$1"; shift
  local status=0
  docker run --rm -i --network "$net" -e PGPASSWORD=throwaway-superuser "$@" "$image" \
    psql -h "$server" -U postgres -d areneon -v ON_ERROR_STOP=1 -f - \
    < "$repo/db/bootstrap.sql" > "$out" 2>&1 || status=$?
  echo "$status"
}

# Count server-log lines holding a value; the value reaches grep as a
# pattern file (a process substitution), never in its arguments.
server_log() { docker logs "$server" 2>&1; }
count_in_log() { server_log | grep -cFf <(printf '%s\n' "$1") || true; }
count_on_own_line() {
  server_log | grep -F "CREATE ROLE $2 LOGIN PASSWORD" | grep -cFf <(printf '%s\n' "$1") || true
}

out="$(mktemp)"
trap 'cleanup; rm -f "$out"' EXIT

# 1. Each variable missing, then blank, with the other two set.
for v in "${vars[@]}"; do
  for mode in missing blank; do
    env_args=()
    for o in "${vars[@]}"; do
      if [[ "$o" != "$v" ]]; then env_args+=(-e "$o"); fi
    done
    if [[ "$mode" == blank ]]; then
      env_args+=(-e "$v=")
      want="$v is empty"
    else
      want="$v is not set"
    fi
    status="$(bootstrap "$out" "${env_args[@]}")"
    [[ "$status" != 0 ]] || fail "$v $mode: bootstrap exited 0"
    grep -qF "$want" "$out" || fail "$v $mode: no '$want' message"
    printf 'ok: %s %s refused (exit %s)\n' "$v" "$mode" "$status"
  done
done
roles_made="$(docker exec "$server" psql -U postgres -d areneon -tAc \
  "SELECT count(*) FROM pg_roles WHERE rolname IN ('rules_owner','rules_migrator','importer','areneon_reader')")"
[[ "$roles_made" == 0 ]] || fail "a refused bootstrap created $roles_made roles"
for v in "${vars[@]}"; do
  n="$(count_in_log "${pw[$v]}")"
  [[ "$n" == 0 ]] || fail "$v reached the server log $n times during the refused runs"
done
echo "ok: refused runs created no role and sent no password (0 log lines each)"

# 2. All three set: the bootstrap succeeds.
status="$(bootstrap "$out" -e RULES_MIGRATOR_PASSWORD -e RULES_IMPORTER_PASSWORD -e RULES_READER_PASSWORD)"
[[ "$status" == 0 ]] || fail "bootstrap with all passwords set exited $status"
grep -qx COMMIT "$out" || fail "bootstrap did not commit"
echo "ok: bootstrap with all three passwords set exited 0"
# The server log is the container's output; wait until it has the COMMIT.
for _ in $(seq 20); do
  server_log | grep -q 'statement: COMMIT' && break
  sleep 0.5
done

# 3. Each password is in exactly one log line, its own CREATE ROLE.
for i in "${!vars[@]}"; do
  v="${vars[$i]}"
  total="$(count_in_log "${pw[$v]}")"
  own="$(count_on_own_line "${pw[$v]}" "${roles[$i]}")"
  [[ "$total" == 1 && "$own" == 1 ]] ||
    fail "$v: $total server-log lines hold it, $own of them its own CREATE ROLE (want 1 and 1)"
  printf 'ok: %s in %s server-log line, its own CREATE ROLE\n' "$v" "$total"
done
any="$(server_log | grep -cFf <(printf '%s\n' "${pw[@]}") || true)"
[[ "$any" == 3 ]] || fail "$any server-log lines hold a password (want 3: one CREATE ROLE each)"
echo "ok: 3 server-log lines hold a password, none holds two"
echo "bootstrap-secrets-test: passed"
