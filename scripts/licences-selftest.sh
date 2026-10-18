#!/usr/bin/env bash
# Regression test for `make licences`: inject an AGPL dependency into a
# scratch copy of the workspace, once on the pnpm side and once on the Python
# side, and require that target to fail naming it each time. The real
# checkout, its node_modules and its .venv are never touched. Needs network
# (pnpm and uv resolve the existing dependencies again), so it runs in CI only.
#   scripts/licences-selftest.sh
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
scratch=""
trap 'rm -rf "$scratch"' EXIT

# Run a setup step; on failure show its output, so it cannot pass for a gate failure.
step() {
  local out
  if ! out="$("$@" 2>&1)"; then
    printf 'licences-selftest: setup failed: %s\n%s\n' "$*" "$out" >&2
    exit 1
  fi
}

# A fresh scratch copy of the tracked files `make licences` reads.
fresh_copy() {
  rm -rf "$scratch"
  scratch="$(mktemp -d)"
  cd "$repo"
  while IFS= read -r -d '' file; do
    mkdir -p "$scratch/$(dirname "$file")"
    cp "$file" "$scratch/$file"
  done < <(git ls-files -z Makefile package.json pnpm-lock.yaml pnpm-workspace.yaml \
    'packages/*/package.json' pyproject.toml uv.lock .python-version 'packages/schema/python/**' \
    scripts/check_licenses.py scripts/python_licences.py)
  cd "$scratch"
}

# `make licences` must fail, naming the injected package.
expect_refused() {
  local name="$1" output
  if output="$(make --no-print-directory licences 2>&1)"; then
    echo "licences-selftest: make licences accepted an AGPL dependency ($name)" >&2
    exit 1
  fi
  if [[ "$output" != *"$name@AGPL-3.0-only"* ]]; then
    printf 'licences-selftest: make licences failed for another reason:\n%s\n' "$output" >&2
    exit 1
  fi
}

# pnpm: an AGPL `file:` dependency of the bridge.
fresh_copy
mkdir "$scratch/agpl-dep"
printf '{"name":"agpl-dep","version":"1.0.0","license":"AGPL-3.0-only"}\n' >"$scratch/agpl-dep/package.json"
step pnpm --filter @areneon/showdown-bridge add --prefer-offline --ignore-scripts "file:$scratch/agpl-dep"
expect_refused agpl-dep

# Python: an AGPL path dependency of the shipped contracts package.
fresh_copy
step pnpm install --frozen-lockfile --prefer-offline --ignore-scripts
mkdir -p "$scratch/agpl-py/agpl_py"
touch "$scratch/agpl-py/agpl_py/__init__.py"
printf '[project]\nname = "agpl-py"\nversion = "1.0.0"\nlicense = "AGPL-3.0-only"\n[build-system]\nrequires = ["hatchling==1.32.4"]\nbuild-backend = "hatchling.build"\n' \
  >"$scratch/agpl-py/pyproject.toml"
step uv add --package areneon-contracts "$scratch/agpl-py"
expect_refused agpl-py

echo "licences-selftest: make licences rejects an injected AGPL dependency on both sides"
