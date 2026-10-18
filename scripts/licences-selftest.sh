#!/usr/bin/env bash
# Regression test for `make licences`: inject an AGPL `file:` dependency into
# a scratch copy of the workspace manifests and require the gate to fail.
# The real checkout, lockfile and node_modules are never touched.
#   scripts/licences-selftest.sh
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

cd "$repo"
git ls-files -z package.json pnpm-lock.yaml pnpm-workspace.yaml 'packages/*/package.json' |
  xargs -0 -I{} cp --parents {} "$scratch"
mkdir "$scratch/agpl-dep"
printf '{"name":"agpl-dep","version":"1.0.0","license":"AGPL-3.0-only"}\n' >"$scratch/agpl-dep/package.json"

cd "$scratch"
pnpm --filter @areneon/showdown-bridge add --offline --ignore-scripts "file:$scratch/agpl-dep" >/dev/null
report="$(pnpm -r licenses list --prod --json)"
if output="$(uv run --no-project python "$repo/scripts/check_licenses.py" <<<"$report" 2>&1)"; then
  echo "licences-selftest: the gate accepted an AGPL dependency" >&2
  exit 1
fi
if [[ "$output" != *"agpl-dep@AGPL-3.0-only"* ]]; then
  echo "licences-selftest: the gate failed for another reason: $output" >&2
  exit 1
fi
echo "licences-selftest: an injected AGPL dependency is rejected"
