#!/usr/bin/env bash
# End-to-end rules import (ADR-007 §6, #768) into the local Postgres 18 that
# `make db-up` starts fresh; `make importer-e2e` runs it, locally and in CI.
#
# The production image is checked for its shape. The `e2e` image (the
# production image plus a stand-in gh that replays the recorded GitHub answers
# for fixtures/rules-releases/) runs the unchanged CLI three times: with one
# asset tampered it must refuse and write nothing; then it must commit both
# artifacts, which a separate session must see; then it must be a no-op that
# downloads only the channel asset, checking the stored snapshot by reference.
# Only the local throwaway importer password is used.
set -euo pipefail

image=${1:?usage: e2e.sh <image name>}
# The database `make db-up` starts; override both to run beside it.
db_host=${RULES_DB_HOST:-127.0.0.1:55432}
compose_project=${RULES_DB_COMPOSE_PROJECT:-areneon-rules-db}
url="postgres://importer:local-importer@$db_host/areneon"
snapshot=7c618d78d6f569e875d7e5be4863152599dc8f4207f6e10cfd5f12301a853fe8
snapshot_commit=35f9c5f04aecbe1dfc38e55d2599b1013f1b43b7
channel_commit=b226c3bfe889f8cf775933b0a964a44ef6d69913
channel_asset=rules-channel-stable-$channel_commit.json

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
  docker compose -p "$compose_project" -f db/compose.yaml exec -T postgres psql -U areneon_reader -d areneon -At -v ON_ERROR_STOP=1 -c \
    "SELECT hash, source_commit FROM rules.rules_snapshot ORDER BY hash;
     SELECT channel, seq, hash, source_commit FROM rules.rules_release ORDER BY channel, seq;"
}
# run_import <snapshot step status> <channel step status> <expected downloads>:
# the stand-in gh logs its calls into a fresh directory, and the assets it was
# asked to download (one name per line, sorted) must be <expected downloads>.
run_import() {
  local out code=0 log
  log=$(mktemp -d -p "$scratch")
  chmod 0777 "$log"
  out=$(docker run --rm --network host -e DATABASE_URL="$url" -v "$log:/fake-gh-log" "$image:e2e") || code=$?
  expect "import exit code ($1, $2)" "$code" 0
  expect "import summary ($1, $2)" "$out" \
    "{\"ok\":true,\"steps\":[{\"kind\":\"snapshot\",\"hash\":\"$snapshot\",\"status\":\"$1\"},{\"kind\":\"channel\",\"channel\":\"stable\",\"seq\":5,\"hash\":\"$snapshot\",\"status\":\"$2\"}]}"
  expect "assets downloaded ($1, $2)" "$(awk '$1 == "release" && $2 == "download" { print $7 }' "$log/calls" | sort)" "$3"
}

# The production image: the real gh, no stand-in, no fixtures, not root, and
# it refuses to start without DATABASE_URL.
gh_version=$(docker run --rm --entrypoint gh "$image:local" --version | head -n 1)
[[ "$gh_version" == "gh version "* ]] || fail "production gh: $gh_version"
expect "production gh is on PATH" "$(docker run --rm --entrypoint sh "$image:local" -c 'command -v gh')" /usr/local/bin/gh
docker run --rm --entrypoint sh "$image:local" -c \
  'test ! -e /app/fixtures && test ! -e /app/packages/rules-importer/test && test ! -e /app/packages/rules-importer/testing &&
   ! grep -rq FakeAttestationVerifier /app/packages' ||
  fail "the production image holds test files or the fake verifier"
echo "ok: production image has no test files and no fake verifier"
found=$(docker run --rm --entrypoint sh "$image:local" -c \
  'for tool in curl npm npx corepack pnpm yarn; do command -v "$tool" && exit 1; done; exit 0') ||
  fail "the production image has ${found:-a tool it should not, or the check did not run}"
echo "ok: production image has no curl, npm, npx, corepack, pnpm or yarn"
expect "production user" "$(docker run --rm --entrypoint id "$image:local" -u)" 1000
code=0
out=$(docker run --rm "$image:local") || code=$?
expect "production exit without DATABASE_URL" "$code" 2
expect "production output without DATABASE_URL" "$out" '{"ok":false,"error":"ConfigError: DATABASE_URL is not set"}'

expect "rows before the import" "$(rows)" ""

# A tampered channel asset (one byte appended, so its sha256 is in no
# attestation) is refused, and nothing is written, not even the snapshot
# that verified before it.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
chmod 0755 "$scratch"
snapshot_asset=rules-snapshot-$snapshot.json
cp "fixtures/rules-releases/$channel_asset" "$scratch/$channel_asset"
printf ' ' >>"$scratch/$channel_asset"
chmod 0644 "$scratch/$channel_asset"
code=0
out=$(docker run --rm --network host -e DATABASE_URL="$url" \
  -v "$scratch/$channel_asset:/app/fixtures/rules-releases/$channel_asset:ro" "$image:e2e") || code=$?
expect "tampered import exit code" "$code" 1
refused='^\{"ok":false,"error":"ImportPlanError: could not list the released rules artifacts: [^"]*/'"$channel_asset"': gh attestation verify exited 1: '
[[ "$out" =~ $refused ]] || fail "tampered import output: $out"
echo "ok: tampered import refused by the attestation check"
expect "rows after the tampered import" "$(rows)" ""

run_import inserted inserted "$channel_asset
$snapshot_asset"
stored="$snapshot|$snapshot_commit
stable|5|$snapshot|$channel_commit"
expect "rows after the import (committed)" "$(rows)" "$stored"
# The re-run downloads only the channel asset: the stored snapshot is planned
# by reference and checked against its stored row.
run_import referenced already-present "$channel_asset"
expect "rows after the re-run (unchanged)" "$(rows)" "$stored"
echo "e2e: tampered import refused, rules import committed, re-run was a no-op that downloaded no snapshot"
