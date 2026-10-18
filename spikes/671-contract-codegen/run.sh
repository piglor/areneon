#!/usr/bin/env bash
# Regenerates every output of the #671 codegen spike into dist/ (gitignored)
# and prints the result matrix. Tools are pinned; nothing is added to the
# repo's package.json, pyproject.toml or a root Cargo.toml.
#
#   spikes/671-contract-codegen/run.sh            # from anywhere
#
# Needs: node 24 + pnpm (repo installed), uv, cargo (rustc >= 1.85).
set -euo pipefail

SPIKE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SPIKE/../.." && pwd)"
OUT="$SPIKE/dist"
TOOLS="${CODEGEN_TOOLS:-$OUT/tools}"
SCHEMA="$SPIKE/schema/contracts.schema.json"

JSON2TS="json-schema-to-typescript@16.0.0"
QUICKTYPE="quicktype@26.0.0"
DMCG="datamodel-code-generator==0.85.0"
TYPIFY="cargo-typify@0.8.0"
SCHEMAFY="schemafy-cli@0.1.0"
PY_WITH=(--with pydantic==2.14.1 --with jsonschema==4.26.0 --with fastjsonschema==2.22.2 --with rfc8785==0.1.4)

mkdir -p "$OUT"/{ts,py,rs,results}
cd "$SPIKE"

step() { printf '\n== %s\n' "$*" >&2; }

step "tools"
[ -x "$TOOLS/bin/cargo-typify" ] || cargo install --locked --quiet --root "$TOOLS" "$TYPIFY"
[ -x "$TOOLS/bin/schemafy-cli" ] || cargo install --locked --quiet --root "$TOOLS" "$SCHEMAFY"

gen_ts_json2ts() { pnpm dlx "$JSON2TS" -i "$SCHEMA" -o "$1" --unreachableDefinitions --additionalProperties false --maxItems -1; }
gen_ts_quicktype() { pnpm dlx "$QUICKTYPE" --src-lang schema --lang ts --top-level AreneonContracts -o "$1" "$SCHEMA"; }
DMCG_ARGS=(--input "$SCHEMA" --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel
  --target-python-version 3.12 --use-annotated --field-constraints --use-double-quotes --disable-timestamp
  --use-schema-description --use-field-description --enum-field-as-literal all --use-union-operator
  --use-standard-collections --formatters builtin --strict-types str int bool)
gen_py_dmcg() { uvx --python 3.12 --from "$DMCG" datamodel-codegen "${DMCG_ARGS[@]}" --output "$1"; }
gen_py_dmcg_missing() {
  uvx --python 3.12 --from "$DMCG" datamodel-codegen "${DMCG_ARGS[@]}" \
    --use-missing-sentinel --target-pydantic-version 2.14 --output "$1"
}
gen_py_quicktype() { pnpm dlx "$QUICKTYPE" --src-lang schema --lang python --python-version 3.7 --top-level AreneonContracts -o "$1" "$SCHEMA"; }
gen_rs_typify() { "$TOOLS/bin/cargo-typify" typify "$SCHEMA" --no-builder -o "$1"; }
gen_rs_quicktype() { pnpm dlx "$QUICKTYPE" --src-lang schema --lang rust --visibility public --derive-debug --derive-clone --top-level AreneonContracts -o "$1" "$SCHEMA"; }
gen_rs_schemafy() { "$TOOLS/bin/schemafy-cli" "$SPIKE/schema" > "$1"; }

GENERATORS=(
  "ts/json2ts.ts gen_ts_json2ts" "ts/quicktype.ts gen_ts_quicktype"
  "py/dmcg_models.py gen_py_dmcg" "py/dmcg_missing_models.py gen_py_dmcg_missing" "py/quicktype_models.py gen_py_quicktype"
  "rs/typify_types.rs gen_rs_typify" "rs/quicktype_types.rs gen_rs_quicktype" "rs/schemafy_types.rs gen_rs_schemafy"
)

step "generate (twice; the second run must be byte-identical)"
: > "$OUT/results/determinism.txt"
for entry in "${GENERATORS[@]}"; do
  read -r file fn <<< "$entry"
  # Same file name both times: quicktype writes the output name into a comment.
  mkdir -p "$OUT/again/$(dirname "$file")"
  "$fn" "$OUT/$file" > /dev/null 2>&1
  "$fn" "$OUT/again/$file" > /dev/null 2>&1
  if cmp -s "$OUT/$file" "$OUT/again/$file"; then verdict=identical; else verdict=DIFFERENT; fi
  echo "$file $verdict $(sha256sum < "$OUT/$file" | cut -c1-16)" | tee -a "$OUT/results/determinism.txt"
done
rm -rf "$OUT/again"

step "samples"
node harness/samples.ts "$OUT"

step "compile: tsc --noEmit (strict base config) incl. type-level expectations"
{ (cd "$REPO" && pnpm exec tsc --noEmit -p spikes/671-contract-codegen/tsconfig.json) \
  && echo "tsc: pass" || echo "tsc: FAIL"; } | tee "$OUT/results/compile.txt"

step "compile: mypy --strict"
for f in dmcg_models dmcg_missing_models quicktype_models; do
  if uv run --no-project --python 3.12 --with mypy==2.4.0 --with pydantic==2.14.1 \
      mypy --strict --no-incremental --cache-dir /dev/null "$OUT/py/$f.py" > "$OUT/results/mypy-$f.txt" 2>&1; then
    echo "mypy $f: pass"
  else
    echo "mypy $f: FAIL ($(tail -1 "$OUT/results/mypy-$f.txt"))"
  fi
done | tee -a "$OUT/results/compile.txt"

step "compile: cargo build / clippy -D warnings / schemafy"
rm -rf "$OUT/rs/harness" && mkdir -p "$OUT/rs/harness"
cp -r harness/rust/. "$OUT/rs/harness/"
cp "$OUT/rs/typify_types.rs" "$OUT/rs/quicktype_types.rs" "$OUT/rs/harness/src/"
{ (cd "$OUT/rs/harness" && cargo build --quiet) \
  && echo "cargo build (typify + quicktype): pass" || echo "cargo build: FAIL"; } | tee -a "$OUT/results/compile.txt"
{ (cd "$OUT/rs/harness" && cargo clippy --quiet -- -D warnings > "$OUT/results/clippy.txt" 2>&1) \
  && echo "clippy -D warnings: pass" \
  || echo "clippy -D warnings: FAIL ($(grep -m1 -A1 '^error' "$OUT/results/clippy.txt" | tr '\n' ' '))"; } \
  | tee -a "$OUT/results/compile.txt"
mkdir -p "$OUT/rs/schemafy-check/src"
printf '[package]\nname = "schemafy-check"\nversion = "0.0.0"\nedition = "2024"\npublish = false\n[dependencies]\nserde = { version = "=1.0.229", features = ["derive"] }\n[workspace]\n' \
  > "$OUT/rs/schemafy-check/Cargo.toml"
{ cat "$OUT/rs/schemafy_types.rs"; echo 'fn main() {}'; } > "$OUT/rs/schemafy-check/src/main.rs"
{ (cd "$OUT/rs/schemafy-check" && cargo check --quiet > "$OUT/results/schemafy.txt" 2>&1) \
  && echo "schemafy cargo check: pass" \
  || echo "schemafy cargo check: FAIL ($(grep -c '^error\[' "$OUT/results/schemafy.txt") errors)"; } \
  | tee -a "$OUT/results/compile.txt"

step "runtime: validators, generated parsers, JCS round trip"
node harness/check-ts.ts "$OUT" > "$OUT/results/ts.jsonl" 2> "$OUT/results/ts.err"
uv run --no-project --python 3.12 "${PY_WITH[@]}" python harness/check_py.py "$OUT" > "$OUT/results/py.jsonl" 2> "$OUT/results/py.err"
(cd "$OUT/rs/harness" && SPIKE_DIR="$SPIKE" cargo run --quiet --bin contract-codegen-spike -- "$OUT") \
  > "$OUT/results/rs.jsonl" 2> "$OUT/results/rs.err"
echo "request_hash recomputed from each golden fixture (canonical request without rqid):"
grep -h request_hash "$OUT"/results/{ts,py,rs}.err

step "JCS cross-language vectors"
node harness/jcs-ts.ts harness/jcs-vectors.txt > "$OUT/results/jcs-ts.txt"
uv run --no-project --python 3.12 --with rfc8785==0.1.4 python -I harness/jcs_py.py harness/jcs-vectors.txt > "$OUT/results/jcs-py.txt"
(cd "$OUT/rs/harness" && cargo run --quiet --bin jcs -- "$SPIKE/harness/jcs-vectors.txt") > "$OUT/results/jcs-rs.txt"

step "fastjsonschema 2020-12 keyword probe"
uv run --no-project --python 3.12 "${PY_WITH[@]}" python -I harness/probe_fastjsonschema.py | tee "$OUT/results/fastjsonschema-probe.txt"

step "summary"
python3 harness/summarize.py "$OUT/results"
