#!/usr/bin/env bash
# End-to-end rules import (ADR-007 §6, #768) into the local Postgres 18 that
# `make db-up` starts fresh; `make importer-e2e` runs it, locally and in CI.
#
# The production image is checked for its shape. The `e2e` image (the
# production image plus a stand-in gh that replays the recorded GitHub answers
# for fixtures/rules-releases/) runs the unchanged CLI twice: the first run
# commits both artifacts, which a separate session must then see; the second
# must be a no-op. Only the local throwaway importer password is used.
set -euo pipefail

image=${1:?usage: e2e.sh <image name>}
url="postgres://importer:local-importer@127.0.0.1:55432/areneon"
snapshot=7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8
snapshot_commit=35f9c5f04aecbe1dfc38e55d2599b1013f1b43b7
channel_commit=b226c3bfe889f8cf775933b0a964a44ef6d69913

fail() {
  echo "e2e: $*" >&2
  exit 1
}
expect() { # expect <what> <actual> <expected>
  [[ "$2" == "$3" ]] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1"
}
# Read the stored rows from a separate session as the read-only role.
rows() {
  docker compose -f db/compose.yaml exec -T postgres psql -U areneon_reader -d areneon -At -v ON_ERROR_STOP=1 -c \
    "SELECT hash, source_commit FROM rules.rules_snapshot ORDER BY hash;
     SELECT channel, seq, hash, source_commit FROM rules.rules_release ORDER BY channel, seq;"
}
run_import() { # run_import <expected status of both steps>
  local out code=0
  out=$(docker run --rm --network host -e DATABASE_URL="$url" "$image:e2e") || code=$?
  expect "import exit code ($1)" "$code" 0
  expect "import summary ($1)" "$out" \
    "{\"ok\":true,\"steps\":[{\"kind\":\"snapshot\",\"hash\":\"$snapshot\",\"status\":\"$1\"},{\"kind\":\"channel\",\"channel\":\"stable\",\"seq\":5,\"hash\":\"$snapshot\",\"status\":\"$1\"}]}"
}

# The production image: the real gh, no stand-in, no fixtures, not root, and
# it refuses to start without DATABASE_URL.
gh_version=$(docker run --rm --entrypoint gh "$image:local" --version | head -n 1)
[[ "$gh_version" == "gh version "* ]] || fail "production gh: $gh_version"
expect "production gh is on PATH" "$(docker run --rm --entrypoint sh "$image:local" -c 'command -v gh')" /usr/local/bin/gh
expect "production image has no test files" \
  "$(docker run --rm --entrypoint sh "$image:local" -c 'ls /app/fixtures /app/packages/rules-importer/test 2>&1 >/dev/null | wc -l')" 2
expect "production user" "$(docker run --rm --entrypoint id "$image:local" -u)" 1000
code=0
out=$(docker run --rm "$image:local") || code=$?
expect "production exit without DATABASE_URL" "$code" 2
expect "production output without DATABASE_URL" "$out" '{"ok":false,"error":"ConfigError: DATABASE_URL is not set"}'

expect "rows before the import" "$(rows)" ""
run_import inserted
stored="$snapshot|$snapshot_commit
stable|5|$snapshot|$channel_commit"
expect "rows after the import (committed)" "$(rows)" "$stored"
run_import already-present
expect "rows after the re-run (unchanged)" "$(rows)" "$stored"
echo "e2e: rules import committed, re-run was a no-op"
