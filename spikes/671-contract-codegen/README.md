# Spike #671: contract code generation

Research spike for ADR-008 (cross-language data contracts, Redmine #671). It
answers one question with evidence: **which generator per language turns one
hand-written JSON Schema 2020-12 document into TS, Python and Rust types that
reject what the schema rejects and round-trip byte-identically under RFC 8785
(JCS)?** Lighter spike gates apply; nothing here is imported by `packages/`.

Throwaway: promote the chosen toolchain into `packages/schema` (#671) or delete
this directory.

## Layout

```
schema/contracts.schema.json  the test schema (2020-12): a slice of the planned contracts
harness/samples.ts            builds valid + invalid instances from the golden fixtures
harness/check-ts.ts           Ajv verdicts, quicktype converter round trip, request_hash
harness/types-*.ts            compile-time expectations (@ts-expect-error) per TS generator
harness/check_py.py           jsonschema / fastjsonschema verdicts, pydantic + quicktype round trip
harness/rust/                 harness crate: jsonschema crate verdicts, typify + quicktype round trip
harness/jcs-*.{ts,py}, jcs-vectors.txt   cross-language JCS vectors (Rust: harness/rust/src/bin/jcs.rs)
harness/probe_fastjsonschema.py          does fastjsonschema honour 2020-12-only keywords?
harness/summarize.py          merges the per-language rows into the matrix below
run.sh                        regenerates everything into dist/ (gitignored)
```

The schema covers: closed objects (`additionalProperties: false`) everywhere;
`$defs` + `$ref`; a `const`-tagged `oneOf` (`Observation` = `PreviewObservation |
BattleObservation` on `kind`); an untagged `oneOf` (the engine's request shapes);
enums (`DeadlineOutcome`, `PlayerId`, …); an optional non-nullable field
(`RequestIdentity.rqid`) and a required nullable field
(`DecisionRecord.bank_seconds_after`); integer min/max (`confidence_bp` 0..10000,
`StatTable` 0..255, `request_seq` ≥ 1); `const` values (`schema_version: 1`,
`server_auto_choose_detected: false`); nested bounded arrays; a `["boolean","string"]`
union (`RequestMove.disabled`). The root is a catalog object so every generator
emits every type.

Samples (`dist/samples/`) are derived, not copied: the three golden request
fixtures (`fixtures/showdown/gen9championsvgc2026regmc/*.request.json`) verbatim
as `RequestFixture`, observations built from them, the `mega` corpus team as a
`PrivateTeamSpec`, and 15 single-mutation invalid instances (three are privacy
probes: an `omniscient_log` key in an observation, an opponent `item` in a CTS
preview entry, an unknown key inside the engine-owned request).

## Run

```bash
spikes/671-contract-codegen/run.sh     # ~40 s warm; first run also cargo-installs two CLIs into dist/tools
```

Needs Node 24 + pnpm (repo installed), uv, cargo (tested with rustc 1.94.1).
Generators run through `pnpm dlx`, `uvx` and `cargo install --root dist/tools`;
nothing is added to the repo's `package.json`, `pyproject.toml` or a root `Cargo.toml`.

## Tools (exact versions, 2026-10-18)

| Language | Tool | Version | Licence | Role |
|---|---|---|---|---|
| TS | json-schema-to-typescript | 16.0.0 | MIT | types only |
| TS | quicktype | 26.0.0 | Apache-2.0 | types + runtime converter |
| TS | ajv (`Ajv2020`, repo dep) | 8.20.0 | MIT | validator |
| Python | datamodel-code-generator | 0.85.0 | MIT | pydantic v2 models |
| Python | quicktype | 26.0.0 | Apache-2.0 | dataclasses + from_dict |
| Python | pydantic | 2.14.1 | MIT | runtime for dmcg models |
| Python | jsonschema | 4.26.0 | MIT | validator |
| Python | fastjsonschema | 2.22.2 | BSD-3-Clause | validator |
| Python | rfc8785 | 0.1.4 | Apache-2.0 | JCS |
| Rust | cargo-typify | 0.8.0 | Apache-2.0 | serde types |
| Rust | quicktype | 26.0.0 | Apache-2.0 | serde types |
| Rust | schemafy-cli / schemafy_lib | 0.1.0 / 0.5.2 | MIT | serde types |
| Rust | jsonschema | 0.59.0 | MIT | validator |
| Rust | serde_json_canonicalizer | 0.4.1 | MIT | JCS |

`cargo-typify 0.10.0-alpha.3` (2026-10-14) was also tried: it does not install on
rustc 1.94.1 (`typespace@0.0.1-alpha.4 requires rustc 1.97`).

datamodel-code-generator flags: `--output-model-type pydantic_v2.BaseModel
--target-python-version 3.12 --use-annotated --field-constraints
--use-union-operator --use-standard-collections --enum-field-as-literal all
--formatters builtin --strict-types str int bool --disable-timestamp`; the
`dmcg_missing` variant adds `--use-missing-sentinel --target-pydantic-version 2.14`.

## Results (run of 2026-10-18)

### Determinism and compilation

| Output | Re-run byte-identical | Compiles / type-checks |
|---|---|---|
| json-schema-to-typescript | yes | `tsc --noEmit` with the repo's strict base config: pass; all type-level expectations hold |
| quicktype TS | yes (the header comment embeds the output file name) | tsc pass |
| dmcg (pydantic v2) | yes | `mypy --strict`: pass |
| dmcg + MISSING sentinel | yes | `mypy --strict`: pass |
| quicktype Python | yes | `mypy --strict`: **134 errors** (untyped helpers) |
| typify 0.8.0 | yes | `cargo build`: pass; `cargo clippy -D warnings`: **1 error** (`enum_variant_names` on a generated union) |
| quicktype Rust | yes | `cargo build`: pass |
| schemafy | yes | **does not compile**: it reads draft-04 `definitions` only, ignores `$defs` and emits references to 6 undefined types |

### Runtime verdicts and JCS round trip

`ok` = correct (valid: parsed and the re-serialised value is JCS-identical to
the input; invalid: rejected). `ACCEPTS` = an invalid instance got through.
`RT!=` = parsed, but the round trip changed the bytes.

| sample | Ajv | qt-ts | py-jsonschema | fastjsonschema | dmcg(unset) | dmcg(none) | dmcg(MISSING) | qt-py | rs-jsonschema | typify | qt-rs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| valid/decision_record.bank-known.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/decision_record.bank-null.json | ok | ok | ok | ok | ok | RT!= | ok | ok | ok | ok | RT!= |
| valid/legal_choice_set.turn.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/observation.battle-force-switch.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/observation.battle-move.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/observation.preview.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/private_team_spec.mega-p1.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/recommendation.in-budget.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/request_fixture.force-switch.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/request_fixture.preview.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| valid/request_fixture.turn.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | RT!= |
| invalid/decision_record.auto-choose.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ACCEPTS | ok | ACCEPTS | ACCEPTS |
| invalid/decision_record.nullable-missing.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.omniscient-log.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.opponent-item.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.request-seq-zero.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.request-unknown-key.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.rqid-null.json | ok | ok | ok | ok | ACCEPTS | ACCEPTS | ok | ACCEPTS | ok | ACCEPTS | ACCEPTS |
| invalid/observation.tag-body-mismatch.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/observation.unknown-tag.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok |
| invalid/private_team_spec.seven.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ACCEPTS | ok | ACCEPTS | ACCEPTS |
| invalid/private_team_spec.sps-key.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| invalid/recommendation.confidence-over-max.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ACCEPTS | ok | ACCEPTS | ACCEPTS |
| invalid/recommendation.fractional-int.json | ok | ACCEPTS | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| invalid/recommendation.unknown-enum.json | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| invalid/request_fixture.unknown-key.json | ok | ok | ok | ok | ok | ok | ok | ACCEPTS | ok | ok | ACCEPTS |
| **wrong verdicts** | 0 | 6 | 0 | 0 | 1 | 2 | 0 | 12 | 0 | 5 | 23 |

json-schema-to-typescript emits types only, so it has no runtime column; its
compile-time checks (`harness/types-json2ts.ts`) all hold: narrowing on `kind`
works, an unknown key, a wrong tag, `rqid: null` (under
`exactOptionalPropertyTypes`), a missing required-nullable field and
`server_auto_choose_detected: true` are all compile errors.

What the wrong verdicts are:

- **quicktype (all three languages)** merges `PreviewObservation | BattleObservation`
  into one `Observation` with every branch-only field optional, widens `const 1`
  to `number` and `const false` to `boolean`, and ignores min/max and
  `maxItems`. Python and Rust output also ignore unknown keys (no
  `extra="forbid"`, no `deny_unknown_fields`); quicktype Rust serialises absent
  optionals as `null`, so no valid instance round-trips.
- **typify 0.8.0** emits `deny_unknown_fields` and enforces patterns, `minLength`,
  `minimum: 1` (as `NonZeroU64`) and 0..255 (as `u8`), and keeps optional
  (`skip_serializing_if`) apart from required-nullable (`deserialize_with`). It
  maps every `const` to `serde_json::Value` (so `kind: "spectator"`,
  `schema_version` and `server_auto_choose_detected: true` pass), emits the
  `oneOf` as `#[serde(untagged)]`, and ignores `maximum` other than a type
  width and `maxItems`. `rqid: null` parses as `None` and is dropped on
  re-serialisation.
- **datamodel-code-generator** without the sentinel maps an optional field to
  `int | None = None`, so `rqid: null` is accepted; dumping with
  `exclude_none=True` also drops the required-nullable `bank_seconds_after: null`.
  With `--use-missing-sentinel --target-pydantic-version 2.14` (pydantic's
  `MISSING`, stable since 2.14) all 26 verdicts are correct.
- **fastjsonschema** gets this schema right, but it implements drafts 04/06/07
  only and silently ignores 2020-12-only keywords: `harness/probe_fastjsonschema.py`
  shows it accepting instances that `unevaluatedProperties: false` and
  `dependentRequired` reject.

### Cross-language identity

- `request_hash` recomputed from each golden fixture (SHA-256 of the JCS form of
  the request without `rqid`) matches the recorded hash in TS (repo
  `requestHash`), Python (`rfc8785`) and Rust (`serde_json_canonicalizer`): 9/9.
- JCS vectors (`harness/jcs-vectors.txt`): 6/8 identical across the three
  languages, including key order by UTF-16 code units (`\r`, `1`, `\u0080`, `ö`,
  `€`, `😀`), `-0`, `1e21`, `5e-324`, `1.0` and control/escape characters. The two
  differences are integers beyond 2^53: TS and Rust silently round
  `9007199254740993` to `…992` (and 2^64 to `18446744073709552000`); `rfc8785`
  raises `IntegerDomainError`. The contracts must therefore keep every integer
  within ±(2^53−1) in the schema itself.

### Golden request fixtures

All three `*.request.json` fixtures validate against `RequestFixture` (closed at
every level) in Ajv, python-jsonschema and the Rust `jsonschema` crate, parse in
every non-failing generator, and round-trip JCS-identically in
json-schema-to-typescript/Ajv, dmcg and typify.

### Other findings

- The repo's Ajv configuration (`strict: true`, `rules-load.ts`) refuses
  `"type": ["boolean","string"]` unless `allowUnionTypes: true`; `["integer","null"]`
  is allowed. The contract profile should use `oneOf` for non-null type unions or
  set `allowUnionTypes`.
- Adding the OpenAPI `discriminator: {"propertyName": "kind"}` keyword (Ajv
  option `discriminator: true`; the union schema then needs `"type": "object"`
  under strict mode) makes Ajv reject an unknown tag with a `discriminator`
  error and makes dmcg emit `Field(discriminator="kind")`; typify still emits an
  untagged enum. Checked by hand, not in `run.sh`.
