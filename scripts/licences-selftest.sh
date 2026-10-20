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
    'ml/**' scripts/__init__.py scripts/check_licenses.py scripts/python_licences.py \
    scripts/licence_exceptions.json scripts/check_cuda_separation.py)
  cd "$scratch"
}

# `make licences` must fail, naming the injected package and the reason.
expect_refused() {
  local name="$1" reason="${2:-$1@AGPL-3.0-only}" output
  if output="$(make --no-print-directory licences 2>&1)"; then
    echo "licences-selftest: make licences accepted an injected dependency ($name)" >&2
    exit 1
  fi
  if [[ "$output" != *"$reason"* ]]; then
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

# CUDA separation (ADR-010 §5.2 C-2): a permissively licensed package named like
# an NVIDIA one, a path dependency of the shipped ml package. The licence check
# passes it; only the CUDA guard can refuse it.
fresh_copy
step pnpm install --frozen-lockfile --prefer-offline --ignore-scripts
mkdir -p "$scratch/nvidia-selftest/nvidia_selftest"
touch "$scratch/nvidia-selftest/nvidia_selftest/__init__.py"
printf '[project]\nname = "nvidia-selftest"\nversion = "1.0.0"\nlicense = "MIT"\n[build-system]\nrequires = ["hatchling==1.32.4"]\nbuild-backend = "hatchling.build"\n' \
  >"$scratch/nvidia-selftest/pyproject.toml"
step uv add --package areneon-ml "$scratch/nvidia-selftest"
expect_refused nvidia-selftest "CUDA separation violated: the closure of areneon-ml contains nvidia-selftest"

echo "licences-selftest: make licences rejects an injected AGPL dependency on both sides and an injected CUDA package"
