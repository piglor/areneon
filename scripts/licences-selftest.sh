#!/usr/bin/env bash
# Regression test for `make licences`: inject an AGPL `file:` dependency into
# a scratch copy of the workspace and require that target to fail naming it.
# The real checkout and its node_modules are never touched. Needs network
# (pnpm resolves the existing dependencies again), so it runs in CI only.
#   scripts/licences-selftest.sh
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# Run a setup step; on failure show its output, so it cannot pass for a gate failure.
step() {
  local out
  if ! out="$("$@" 2>&1)"; then
    printf 'licences-selftest: setup failed: %s\n%s\n' "$*" "$out" >&2
    exit 1
  fi
}

cd "$repo"
while IFS= read -r -d '' file; do
  mkdir -p "$scratch/$(dirname "$file")"
  cp "$file" "$scratch/$file"
done < <(git ls-files -z Makefile package.json pnpm-lock.yaml pnpm-workspace.yaml \
  'packages/*/package.json' scripts/check_licenses.py)
mkdir "$scratch/agpl-dep"
printf '{"name":"agpl-dep","version":"1.0.0","license":"AGPL-3.0-only"}\n' >"$scratch/agpl-dep/package.json"

cd "$scratch"
step pnpm --filter @areneon/showdown-bridge add --prefer-offline --ignore-scripts "file:$scratch/agpl-dep"
if output="$(make --no-print-directory licences 2>&1)"; then
  echo "licences-selftest: make licences accepted an AGPL dependency" >&2
  exit 1
fi
if [[ "$output" != *"agpl-dep@AGPL-3.0-only"* ]]; then
  printf 'licences-selftest: make licences failed for another reason:\n%s\n' "$output" >&2
  exit 1
fi
echo "licences-selftest: make licences rejects an injected AGPL dependency"
