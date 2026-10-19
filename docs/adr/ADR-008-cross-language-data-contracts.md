# ADR-008: Cross-language data contracts

**Status:** Accepted (revision 3, 2026-10-18) | **Wave:** 1 | **Deciders:** core team | **Date:** 2026-10-18

**Revision history:** r1 (2026-10-18) Proposed. r2 (2026-10-18) answers the plan-stage review in the #717 journal:
- **Engine schemas:** a separate engine-mirror profile for schemas that copy the engine's wire shapes.
- **A2:** PlayerChannel adapter and builder placement.
- **Private data:** a data class, provenance, rights and a storage rule.
- **Licences:** a decision on the transitive Python set.
- **Staleness:** `rqid` is checked when present.
- **Equality:** A1/A2 equality defined (ADR-005).
- **Fixtures:** golden per-player channel fixtures.
- **Field sets:** made consistent.
- **Small fixes** (P3).

r3 (2026-10-18) answers the r2 re-review:
- **Hashing seam:** the hash function is injected, so the builder runs wherever WebCrypto does.
- **`rqid`:** listed as an optional key in the engine-mirror request schema.
- **Private-data guard:** it also catches unlabelled team-shaped JSON.

ADR issue: #717 · Epic: #720 · Implementation: #671 (slices #721–#727) · Related: [ADR-002](ADR-002-languages-and-repository-layout.md), [ADR-005](ADR-005-bot-opponent-runtime.md), [ADR-006](ADR-006-legal-joint-choice-enumeration.md), [ADR-007](ADR-007-rules-store-database-and-editing.md)

Decision evidence: spike `spikes/671-contract-codegen/`, `piglor/areneon#22`. `run.sh` regenerates every tabulated result; the `discriminator` check was run by hand [S29]. Research ran on Claude Opus 5.5 as the announced fallback: the policy model `gpt-5.6-sol` and its listed fallback `gpt-5.6-terra` were unavailable.

---

## Context

#671 must deliver versioned, typed contracts for PrivateTeamSpec, PreviewObservation, BattleObservation, LegalJointChoice (ADR-006's `LegalChoiceSet`), Recommendation and Episode, with actor-specific visibility. Its acceptance criteria:
- golden parsing tests from player-perspective `|request|` fixtures only;
- adversarial tests proving opponent hidden information and omniscient state cannot enter an actor observation;
- a request-state hash for stale invalidation.

**Already decided, not reopened.** [ADR-002](ADR-002-languages-and-repository-layout.md) makes hand-written JSON Schema in `packages/schema` the source of truth, with TS, pydantic and Rust serde types generated from it and CI checking they stay in sync. It rejected pydantic-, zod- and serde-first contracts. The rules records (ADR-007) already use JSON Schema 2020-12 with `$id` `urn:areneon:schema:rules:v1:<kind>`, closed objects, a `schema_version` const and Ajv 2020 in strict mode, and hashes use RFC 8785 JCS [S30].

**The decision question:** within ADR-002, which schema dialect profile, validators and code generators, which versioning and evolution rules, which canonical form, and which mechanism keep the contracts identical across TypeScript, Python and Rust, replayable for years, and structurally unable to carry opponent hidden information into an actor observation?

Facts from the pinned engine (`7244fe8`):
- `getPlayerStreams` builds each player's channel with `extractChannelMessages`. For every `|split|pN` block, player N and the omniscient channel get the SECRET line, and every other player gets the PUBLIC line (an empty one is dropped) [S23][S24].
- Requests are `sideupdate`s to one player. A Doubles request contains only that player's `side`; `ally` exists only in multi battles [S24].
- In the `champions` mod, exact HP is SECRET and the public line carries a percentage [S24].
- On the server path (A2) the same split applies per connection. The socket layer runs `extractChannelMessages` per channel before sending a room's lines. A player's connection gets that player's channel, and a spectator gets channel 0. A request reaches only its own player, with the server's `rqid` added [S31].
- What Areneon itself adds is not redacted by the engine: `omniscient_log`, `hidden_verdicts`, and the live `Battle` given to the `CloneJudge` [S30].

## Decision drivers

1. **One truth, three languages.** No hand-maintained parallel types (ADR-002).
2. **Fail closed.** An unknown field, an unknown union tag or an out-of-range value is rejected on every runtime, never silently dropped (ADR-003 "stops on unknowns").
3. **Replay.** Persisted Episodes stay readable and their hashes stay valid forever.
4. **Identical hashes across languages** (request identity, content hashes).
5. **Privacy by structure.** The repository's hard rule: opponent hidden information never enters an observation, log or checkpoint.
6. **Small, healthy, allowlisted supply chain** (`scripts/check_licenses.py`).

## Evidence and alternatives

### Prototype [S29]

`spikes/671-contract-codegen/` holds a JSON Schema 2020-12 that covers:
- closed objects, `$defs`/`$ref`, `const`-tagged and untagged `oneOf`, enums;
- an optional field next to a required-nullable one, integer ranges, nested arrays.

It is run against 26 samples:
- 11 valid: the three golden `*.request.json` fixtures verbatim, observations built from them, and a team from the choice corpus;
- 15 invalid, each with a single mutation, three of them privacy probes: a smuggled `omniscient_log`, an opponent item in a preview entry, and an unknown request key.

Toolchain: Node 24.11, Python 3.12, rustc 1.94.1.

| Tool | Version | Deterministic | Compiles / type-checks | Unknown fields | Wrong union tag | Wrong verdicts / 26 | Valid samples JCS-identical after round trip |
|---|---|---|---|---|---|---|---|
| Ajv `Ajv2020` (validator) | 8.20.0 | n/a | n/a | rejected | rejected | 0 | n/a |
| json-schema-to-typescript | 16.0.0 | yes | tsc strict passes; all `@ts-expect-error` hold | compile-time | compile-time | types only | n/a |
| quicktype (TS) | 26.0.0 | yes (embeds file name) | passes | rejected | tag/body mismatch accepted | 6 | 11/11 |
| python-jsonschema (validator) | 4.26.0 | n/a | n/a | rejected | rejected | 0 | n/a |
| fastjsonschema (validator) | 2.22.2 | n/a | n/a | rejected | rejected | 0, but silently ignores 2020-12 keywords (probe) | n/a |
| datamodel-code-generator → pydantic, `MISSING` sentinel | 0.85.0 / 2.14.1 | yes | `mypy --strict` passes | rejected | rejected | **0** | 11/11 |
| datamodel-code-generator → pydantic, default | 0.85.0 / 2.14.1 | yes | passes | rejected | rejected | 1 (`rqid: null` accepted) | 11/11 |
| quicktype (Python) | 26.0.0 | yes | `mypy --strict`: 134 errors | **accepted** | mismatch accepted | 12 (all 3 privacy probes) | 11/11 |
| Rust `jsonschema` (validator) | 0.59.0 | n/a | n/a | rejected | rejected | 0 | n/a |
| typify (cargo-typify) | 0.8.0 | yes | builds; 1 clippy lint | rejected | unknown tag accepted (`const` → `Value`) | 5 | 11/11 |
| quicktype (Rust) | 26.0.0 | yes | builds | **accepted** | mismatch accepted | 23 | **0/11** |
| schemafy | 0.5 / 0.6 | yes | **does not compile** (ignores `$defs`) | n/a | n/a | n/a | n/a |

Further results [S29]:
- **Cross-language hashes:** the `request_hash` of all 3 golden fixtures, recomputed in TS, Python and Rust, matches in all 9 cases.
- **JCS test vectors:** 6 of 8 are identical across the three languages. The 2 differences are integers above 2^53: TS and Rust round them silently, while Python raises. RFC 8785 and I-JSON warn about exactly this [S6][S7].
- **Closed request type:** all three golden request fixtures validate against a fully closed request type in all three validators.
- **Ajv strict mode:** the repository's `strict: true` refuses `"type": ["boolean","string"]` unless `allowUnionTypes` is set [S4].

### Dialect

| Option | Assessment | Verdict |
|---|---|---|
| **JSON Schema 2020-12, restricted contract profile** | Expresses closed objects, `$defs`/`$ref`, `const` tags, enums, ranges, patterns, required-nullable vs optional [S1]. The three validators gave identical verdicts on 26/26 samples [S29]. Same dialect as the rules schemas [S30]. | **Selected** |
| JSON Schema draft-07 | Same subset works. It would split the repository across two dialects, and one Ajv instance cannot mix them [S3]. The only gain is fastjsonschema [S9]. | Rejected |
| JSON Type Definition (RFC 8927) | Experimental RFC without IETF consensus. It has no string patterns, no integer ranges and no 64-bit integers [S5], so hash patterns and bounded counters would move into code. It also contradicts ADR-002. | Rejected |

### Validators

| Language | Option | Verdict |
|---|---|---|
| TS | Ajv 8.20 `Ajv2020` (already used) [S3][S30] | **Selected** |
| Python | python-jsonschema 4.26 `Draft202012Validator`: full 2020-12 support [S8] | **Selected** |
| Python | fastjsonschema: drafts 04/06/07 only [S9]; the spike shows it silently accepting instances that 2020-12 keywords reject [S29] | Rejected |
| Rust | `jsonschema` crate 0.59 `draft202012` [S20] | **Selected** |

### Code generators

| Language | Option | Assessment | Verdict |
|---|---|---|---|
| TS | json-schema-to-typescript 16.0.0 | Tagged unions narrow on the tag; `const` literals; `rqid?: number` and `x: number \| null` kept apart; compiles under the strict base config [S29]. Ranges and patterns cannot be expressed in TS [S15], so Ajv stays the runtime gate. | **Selected** |
| TS | quicktype 26.0.0 [S16] | Merges the `oneOf` into one interface and widens `const`; its converter accepts 6 invalid samples [S29] | Rejected |
| Python | datamodel-code-generator 0.85.0 → pydantic 2.14 with `--use-missing-sentinel --target-pydantic-version 2.14` | 0 wrong verdicts; `extra="forbid"`; `Literal` consts; field bounds; `mypy --strict` passes; deterministic; `--check` exists for CI; supports 2020-12 and `$defs` [S10][S11][S12][S29]. The `MISSING` sentinel is stable since pydantic 2.14 [S13]. | **Selected** |
| Python | The same without the sentinel | `rqid: null` accepted [S29] | Rejected |
| Python | quicktype 26.0.0 | Ignores unknown keys, ranges and `const`; fails all three privacy probes [S29] | Rejected |
| Rust | typify 0.8.0 behind the `jsonschema` validator | `deny_unknown_fields` [S19]; newtypes for patterns and lengths; optional and nullable kept apart. But `const` becomes `Value`, `oneOf` becomes `untagged`, and `maximum`/`maxItems` are not enforced [S17][S29]. That gives 0 wrong verdicts with the validator in front. | **Selected, for the first crate** |
| Rust | quicktype 26.0.0 | No `deny_unknown_fields`; no valid sample round-trips [S29] | Rejected |
| Rust | schemafy | Draft-04 only [S18], does not compile [S29], unmaintained since 2021 [S28] | Rejected |

Generate-at-build was rejected. TypeScript runs by type stripping with no build step, so generated `.ts` files must exist in the tree. CI also re-runs old commits' scripts from bare archives, which needs committed sources [S30].

### Versioning, canonical form, privacy

| Topic | Option | Verdict |
|---|---|---|
| Identifiers | `urn:areneon:schema:contracts:v<major>:<name>`: an absolute URI as 2020-12 requires [S1], the same scheme as the rules schemas [S30], with no hosted URL needed | **Selected** |
| Identifiers | `https://` `$id`s: the same semantics, but they imply a hosted, resolvable URL [S1][S2] | Rejected |
| Evolution | Closed objects; a major bump for any change that rejects or reinterprets existing data; read-time upcasters; stored bytes never rewritten (the event-store upcasting pattern [S27]; BACKWARD_TRANSITIVE for persisted data [S25]) | **Selected** |
| Evolution | Open objects for forward compatibility [S25]: unknown fields would pass, which breaks the privacy rule | Rejected |
| Evolution | Avro-style defaults for missing fields [S26]: needs `default`, which the profile bans | Rejected |
| Canonical form | RFC 8785 JCS in all three languages (TS `jcs()`, Python `rfc8785`, Rust `serde_json_canonicalizer`), integers bounded to ±(2^53−1) [S6][S7][S21][S22]. It matched on every golden request hash [S29]. | **Selected** |
| Canonical form | Language-native "sorted keys, no whitespace": not a standard; Python sorts by code point, not UTF-16 units | Rejected |
| Privacy | One builder per observation type, fed only by the player's own channel and request; closed schemas; omniscient artifacts in separate contracts; canary tests | **Selected** |
| Privacy | Schema closure alone: rejects unknown keys, but not hidden information inside allowed fields (e.g. exact opponent HP in a log line) | Insufficient on its own |
| Privacy | Redacting the omniscient log afterwards: the redactor would have to know every secret line type, which the engine already gets right [S24] | Rejected |

## Proposed decision

*Inference from the facts above, for owner review.*

### 1. Dialect: JSON Schema 2020-12, two profiles

All schemas live in `packages/schema/contracts/v1/*.schema.json`. Each one declares its profile, and a meta-schema check for both profiles runs in `make ts` and CI.

**1a. Contract profile.** It governs every Areneon-owned contract: PrivateTeamSpec, RequestIdentity, LegalChoiceSet, DecisionContext, the observations, Recommendation, Episode and PlayerChannelLog. Allowed keywords:
- **Structure:** `$schema`, `$id`, `$defs`, local `$ref`, `title`, `description`, `type` (single, or `[T, "null"]`), `properties`, `required`, `items`, `minItems`, `maxItems`.
- **Objects:** `additionalProperties: false` is mandatory on every object.
- **Values:** `enum`, `const`, `minimum`/`maximum` (mandatory on every integer, within ±(2^53−1)), `minLength`, `maxLength`, `pattern`.
- **Unions:** `oneOf`, only of `$ref`s, each branch with a required `const` tag property `kind`, plus the OpenAPI `discriminator` annotation.

Banned:
- **Types and logic:** `number` in hashed contracts (quantities such as confidence are integer basis points), `anyOf`, `allOf`, `not`, `if`/`then`/`else`.
- **Open-ended or dynamic structure:** `patternProperties`, `prefixItems`, `unevaluated*`, `$dynamicRef`.
- **Other:** `format`, `default`, remote `$ref`.

A discriminated union declares `"type": "object"` itself, which Ajv's strict mode needs [S3][S29].

**1b. Engine-mirror profile.** It governs only schemas that copy a wire shape the pinned engine defines, today `EngineRequest` and its parts (`MoveRequestData`, `PokemonMoveRequestData`, `SideRequestData`). It is the contract profile plus two relaxations, because the engine's own types need them:
- `type` may be an array of scalar types. The engine sends `disabled?: string | boolean` [S24].
- An integer with no engine bound uses the I-JSON bound ±(2^53−1) [S7], e.g. `pp`.

Engine-mirror schemas are compiled by their own Ajv instance with `allowUnionTypes: true` [S4]; contract schemas never are. `EngineRequest` lists `rqid` as an optional integer ≥ 1, because the server path always adds one [S31] and a closed schema would otherwise refuse every A2 request. Closure is unchanged: an unknown engine key fails closed. The pin-bump PR edits these schemas, re-runs the golden fixtures and cites the engine source of each change.

### 2. Validators and generators: validate, then parse, everywhere

The schema validator is the gate on every runtime; generated types are the typed view of data that has already passed it.

| Language | Validator | Generated types | Status |
|---|---|---|---|
| TS | Ajv `Ajv2020({strict: true, allErrors: true, discriminator: true})`; the engine-mirror instance adds `allowUnionTypes` | json-schema-to-typescript 16.0.0 | #671 |
| Python | python-jsonschema `Draft202012Validator`, at every boundary that receives data | datamodel-code-generator 0.85.0 → pydantic ≥ 2.14, with the `MISSING` sentinel, `extra="forbid"`, and `strict=True` so nothing is coerced after validation | #671 |
| Rust | `jsonschema` `draft202012` | typify 0.8.0 | With the first crate (AGENTS.md); it must pass the shared sample corpus |

- **Pinned tools and checked-in output.** Generator versions are pinned in one place, and the generated code is checked in.
- **Drift check.** `make contracts` regenerates into a temporary directory, and CI fails on any difference. All three generators produced byte-identical output on re-runs [S29].
- **Shared sample corpus.** A committed set of valid and invalid instances, including the privacy probes and the JCS test vectors, runs in every language that has contracts.

### 3. Versioning

- **Version field.** Every persisted or cross-process root (PrivateTeamSpec, PreviewObservation, BattleObservation, LegalChoiceSet, Recommendation, Episode) carries `schema_version: {"const": N}`. `$id` is `urn:areneon:schema:contracts:v<N>:<name>`.
- **Within a major version, allowed:**
  - adding an optional property;
  - adding an enum value or union branch, released to all consumers together;
  - widening a range or relaxing a pattern;
  - documentation changes.
- **Breaking, so a new major:** adding a required property, removing or renaming one, narrowing a range or pattern, changing a type or `const`, or changing what a hash covers.
  - A breaking change creates a new `v<N+1>` directory and `$id`.
  - It also adds a pure read-time upcaster `vN → vN+1` in TS. Upcasters chain [S27].
- **Persisted Episodes are immutable.**
  - An Episode is stored in the version it was written with.
  - Its content hash is over those JCS bytes.
  - An upcasted view is derived on read and never stored under the original hash.
- **Two identities.** The version manifest (`showdown_commit`, `format_rules_hash`, …) stays the replay identity; `schema_version` is the data-shape identity. Neither replaces the other.

### 4. Canonical form and request identity

- **Hashing.** Hash = SHA-256 of the UTF-8 RFC 8785 JCS bytes, lowercase hex, as for `rules_snapshot_hash`. Libraries: TS `jcs()`, Python `rfc8785`, Rust `serde_json_canonicalizer`.
- **What gets hashed.** Only validated, parsed values are hashed, never raw received bytes (duplicate keys, out-of-range integers). For a request the order is: decode the JSON, validate it against the closed engine-mirror schema, then compute `request_hash`. Today `decodeRequest` hashes before any schema exists; #671 inserts the validation step.
- **Request identity.**
  - `RequestIdentity {request_seq ≥ 1, request_hash, rqid?}` is shared by the observations, `LegalChoiceSet`, `Recommendation` and the Episode decision record.
  - `request_hash` stays exactly as `decodeRequest` defines it today. `rqid` is optional and never hashed.
- **Stale invalidation.** A consumer drops a Recommendation unless its `request_seq` and `request_hash` both equal the current request's, and, when both carry an `rqid`, the `rqid` too.
- **A1/A2 equality (ADR-005 gate, #678 §4.1).** Two observations of the same seeded battle are equal when their JCS bytes are identical after removing `decision_context` and every `rqid`. EXP-08 and the ADR-005 gate use this definition.
- **Engine request shape.** The engine-owned `|request|` shape is modelled **closed**: a pin bump that adds a key fails the golden tests instead of passing silently.

### 5. Privacy by structure

```text
engine BattleStream --getPlayerStreams--> omniscient --> BattleRecord, hidden_verdicts
                         |                               (own $id; evaluation only; never an actor input)
                         +--> pN channel (engine-split: SECRET for pN, PUBLIC for others)
                                |  + own |request| (own side only)
                                |  + own PrivateTeamSpec, version manifest, DecisionContext
                                v
                 ObservationBuilder (the only constructor, in showdown-bridge)
                                |  validate (closed schema)
                                v
                 Preview/BattleObservation --> policy, Episode
```

**Sources of a `PlayerChannel`.** One type covers both paths. Each path has its own adapter, and every adapter validates its input against `PlayerChannelLog`.
- **A1 (in-process sim):** the player's stream from `getPlayerStreams` [S24].
- **A2 (server path):** the lines of the player's own battle room as the server sends them on that player's connection [S31]. The adapter refuses a room in which the user is not a player. A2 is activated only once ADR-005 is Accepted.

**The builder.**
- There is one builder per observation type, in an engine-free module of `showdown-bridge`: it imports only contracts, the pure JSON canonicalisation and request parsing. It imports neither `engine.ts` nor any Node built-in, so it runs wherever WebCrypto does.
- **Hashing seam.** The module never computes a digest itself. The builder and the request parser take an injected `Sha256` port, `(bytes: Uint8Array) => Promise<string>` returning lowercase hex, and are therefore asynchronous at that boundary.
  - On A1 and A2, the adapter passes Node's `createHash`. `request-identity.ts` keeps importing only Node built-ins, as CLAUDE.md requires for `rules-hash.ts`.
  - Where Node's `createHash` is not available, the adapter passes WebCrypto's `crypto.subtle.digest`.
  - `decodeRequest` is split into a pure parse step plus hashing through the port.
  - A cross-runtime test hashes the shared corpus with both implementations and requires identical digests. Node 24 has `globalThis.crypto.subtle`, so one vitest run covers both.
- Its parameters are a branded `PlayerChannel` (from either adapter above), the player's own request, the player's own PrivateTeamSpec, the version manifest and the DecisionContext.
- It never receives the `Battle`, the omniscient stream, `hidden_verdicts` or the opponent's team.
- `PlayerChannel` and `OmniscientLog` are distinct branded types, so passing one for the other does not compile.
- Areneon writes no redactor of its own: the engine already splits secrets correctly [S23][S24].

**Separate contracts.** Omniscient artifacts get their own schemas and `$id`s. An Episode references only the actor's own observations and decisions.

**Private team data.**
- `PrivateTeamSpec` and `Episode` carry a required `data_class`: `synthetic` (CC0, may be committed) or `private` (a real team or a human's games; never committed).
- Episodes also carry `provenance` (runtime, `behaviour_policy_id`, version manifest) and `rights` (`cc0-synthetic` or `owner-private`), as the glossary's definition of an Episode requires.
- Storage rule: `private` data lives only in the gitignored `private-data/` or the deployment's private store. Files under `fixtures/` must be `synthetic`.
- #671 adds a repo guard, like the PokePaste guard (which does not see JSON teams). It rejects committed JSON that:
  - has `"data_class": "private"`; or
  - outside `fixtures/`, contains a team-shaped object (an object with `species`, `moves` and `evs` keys) without `"data_class": "synthetic"` at its root.

  An unlabelled team inside `fixtures/` is still caught by review and fixture provenance; that residual risk is accepted.

**Adversarial tests (the #671 acceptance criteria):**
1. **Schema probes.** Smuggled fields (`omniscient_log`, an opponent item, unknown request keys) are rejected in every language.
2. **Canary test.** Seeded golden battles are run. Opponent sets carry distinctive synthetic markers in their never-public fields (items and moves chosen to be unique strings within the fixture). Every whole opponent SECRET line and the opponent's own request `side` entries are collected as canaries. None of them may appear in the JCS bytes of any observation of the other player. Canaries are whole lines and unique strings, never bare numbers, which would collide with unrelated digits.
3. **Type test.** The builder does not accept an `OmniscientLog` or a `Battle`.
4. **Golden parsing.** Tests use only player-perspective fixtures:
   - the requests in `fixtures/showdown/<format>/*.request.json`;
   - new golden per-player channel logs `fixtures/showdown/<format>/channels/p1.log` and `p2.log`, captured by the runner from each player's stream as a `PlayerChannelLog`.

   No test derives an observation from `omniscient_log`.

### 6. Contract field notes

- **DecisionContext:** as in `docs/research/timer-deadline-budget.md` §4.2. It is optional on both observations and excluded by name from A1/A2 equality.
- **Recommendation:** adds `elapsed_ms`, `deadline_outcome` and `source_policy_id` (§4.3).
- **Episode decision record:** adds the §4.4 fields. `bank_seconds_after` is required-nullable. `server_auto_choose_detected` is a plain boolean: an Episode that observed an auto-choose must still be writable, and the ADR-005 gate asserts it is `false`.
- **LegalChoiceSet:** keeps its ADR-006 fields and gains `schema_version` and an optional `rqid` (through RequestIdentity), so it is a persisted root like the others. The schema becomes its source, and the TS interface is generated from it.
- **PlayerChannelLog:** the lines of one player's channel, in order, with `|t:|` dropped as in the runner.

### 7. Supply chain

- **TS:** json-schema-to-typescript is dev-only.
- **Python, production:** pydantic, jsonschema and rfc8785. datamodel-code-generator is dev-only.
- **Licences of the direct dependencies:** MIT or Apache-2.0 [S28].
- **Transitive set:** this is decided here.
  - pydantic pulls in `typing-extensions`, licensed `PSF-2.0` (PyPI `license_expression`) [S28]. That is an OSI-approved permissive licence, but its SPDX id differs from the allowlisted `Python-2.0`. #671 adds `PSF-2.0` to the allowlist.
  - `rfc8785` publishes no `license_expression`, only the classifier "OSI Approved :: Apache Software License" [S28]. The Python licence check falls back to the trove classifier through an explicit classifier-to-SPDX map, and fails on anything unmapped.
- **Licence check:** #671 extends the production check to every Python runtime dependency, transitive ones included. Rust gets `cargo deny` with its first crate (ADR-002). Its allowlist will need `Zlib`, `Unicode-3.0` and `MIT-0` for the typify dependency tree, which that ADR decides.

## Consequences and risks

| Risk | Mitigation or acceptance |
|---|---|
| typify does not enforce `const`, `maximum` or `maxItems`, and generates untagged enums | Rust always validates before parsing; the shared corpus covers each gap. Re-evaluate typify 0.10 once the toolchain reaches rustc 1.97 [S29]. |
| TS types cannot express ranges or patterns [S15] | Ajv validates at every boundary; types are the view of valid data. |
| A generator minor release changes its output | Versions are pinned, and the CI drift check makes any change a reviewed diff. |
| datamodel-code-generator releases often [S28] | Pinned; upgrades are deliberate PRs that must pass the drift check and the corpus. |
| Closed objects slow forward compatibility | Accepted for privacy. In-repo consumers regenerate from the same commit. |
| The OpenAPI `discriminator` is not a 2020-12 keyword | Ajv supports it with `discriminator: true` [S3]. Other validators treat it as an annotation [S1], so verdicts don't change. It improves errors, and it gives pydantic a tagged union: untagged unions use smart mode, whose algorithm may change between minor releases [S14][S29]. |
| Validator conformance numbers (Bowtie) weren't obtained | The profile keeps to core keywords the three validators agreed on, and the shared corpus runs in CI. |
| A breaking change to persisted data | Major bump plus upcaster; stored bytes never change. |
| Engine pin bumps can change the request shape | The closed request schema fails the golden tests, and the schema is edited in the pin-bump PR. |
| A2 privacy rests on the server's per-connection split | Verified in the pinned server source [S31]. The adapter refuses rooms where the user is not a player, and EXP-08's A1/A2 equality test covers the path once ADR-005 is Accepted. |
| A private team leaks into a commit | `data_class` is required. The repo guard rejects committed `private` JSON and unlabelled team-shaped JSON outside `fixtures/`. Fixtures must be `synthetic`. The residual risk inside `fixtures/` is accepted and covered by review. |
| Model-policy deviation for the research | Announced. The owner accepts or rejects the ADR on its evidence, which is reproducible from `run.sh`. |

Replay: contracts are pure data with JCS hashes; Episodes keep their stored version; the version manifest pins the engine. Security: no new network surface, and validation fails closed. Licence: MIT and Apache-2.0 only.

## Migration and rollout

1. **Epic and tickets.** After acceptance, create the implementation Epic on Wave 1 and move #671 under it as its first slices:
   - (a) the contract profile, its meta-schema check and `make contracts`;
   - (a′) a dependency-free module holding `canonicalJson` and the I-JSON checks, moved out of `request-identity.ts` (which imports `node:crypto`) and `rules.ts` (which imports `node:fs`), with both re-exporting it. The engine-free builder imports only this module;
   - (b) the `RequestIdentity`, `LegalChoiceSet` and `DecisionContext` schemas, with the TS types generated and replacing the hand-written `LegalChoiceSet` interface;
   - (c) PrivateTeamSpec, the observations and the `ObservationBuilder`, with the privacy tests;
   - (d) Recommendation and Episode;
   - (e) generated Python models, the Python validators and the licence-check extension (PSF-2.0, classifier fallback);
   - (f) the `data_class` repo guard and the golden per-player channel fixtures, which slices (c) and (d) depend on.
2. **Spike.** It stays until slice (a) lands, then it is deleted.
3. **No persisted data exists yet,** so nothing migrates. The first Episode written is v1.

## Non-goals

- Choosing among legal choices, and fixed-size action encoding (ADR-006 non-goals).
- The Rust crate itself, and `cargo deny` (first crate).
- Hosting resolvable `$id` URLs.
- Changing the rules record schemas (ADR-007).

## Sources

All accessed 2026-10-18. Labels as cited above.

| Label | Title | URL | Publisher / version | Date | Claim(s) supported |
|---|---|---|---|---|---|
| S1 | JSON Schema: A Media Type for Describing JSON Documents (Core) | https://json-schema.org/draft/2020-12/json-schema-core | JSON Schema org, draft-bhutton-json-schema-01 (2020-12), Informational I-D | 2022-06-16 | `$schema` is the dialect identifier and MUST be a URI; `$id` MUST resolve to an absolute URI without a fragment and is the base URI; `$defs` holds reusable subschemas and does not affect validation; unknown keywords SHOULD be treated as annotations (§4.3.1, §8.1.1, §8.2.1, §8.2.4) |
| S2 | Understanding JSON Schema: Structuring a complex schema | https://json-schema.org/understanding-json-schema/structuring | JSON Schema org | undated (accessed 2026-10-18) | Absolute `$id` recommended; `$defs` is the standard place for reusable subschemas; compound (bundled) documents; references must stay the same bundled or not; each embedded resource may use its own dialect |
| S3 | Ajv: JSON Schema (drafts, discriminator) | https://ajv.js.org/json-schema.html | Ajv docs (v8) | undated (accessed 2026-10-18) | 2020-12 needs a separate Ajv class; one instance cannot mix 2020-12 with earlier drafts; "Ajv supports all keywords of JSON Schema draft-2020-12"; `discriminator` (from OpenAPI) needs option `discriminator: true`, each `oneOf` branch needs a `const`/`enum` tag property, unique tags, and the tag should be `required` |
| S4 | Ajv: Strict mode | https://ajv.js.org/strict-mode.html | Ajv docs (v8) | undated (accessed 2026-10-18) | Under `strictTypes`, `type` with several types other than `null` is prohibited unless `allowUnionTypes: true`; by default unknown keywords fail compilation |
| S5 | RFC 8927: JSON Type Definition | https://www.rfc-editor.org/rfc/rfc8927 | IETF, **Experimental**, no IETF consensus | 2020 | JTD is "intentionally ... minimal", aimed at code generation; has a `discriminator` form; integers are only 8/16/32-bit; `int64`/`uint64` are not allowed |
| S6 | RFC 8785: JSON Canonicalization Scheme (JCS) | https://www.rfc-editor.org/rfc/rfc8785 | IETF, Informational | 2020-06 | Input MUST be I-JSON; numbers serialised per ECMA-262 §7.1.12.1 (IEEE 754 doubles); object keys sorted by UTF-16 code units, locale-independent; integers SHOULD be within ±9007199254740991; values that do not fit SHOULD/MUST be JSON strings (§3.1, Appendix B, D) |
| S7 | RFC 7493: The I-JSON Message Format | https://www.rfc-editor.org/rfc/rfc7493 | IETF, Proposed Standard | 2015-03 | Receivers cannot be expected to treat integers with absolute value above 9007199254740991 as exact; encode such values as strings; no duplicate member names; no unpaired surrogates |
| S8 | jsonschema documentation | https://python-jsonschema.readthedocs.io/en/stable/ | python-jsonschema 4.26.0 | docs for 4.26.0 (release 2026-01-07, S28) | "Full support for Draft 2020-12, Draft 2019-09, Draft 7, Draft 6, Draft 4 and Draft 3" |
| S9 | Fast JSON schema for Python | https://horejsek.github.io/python-fastjsonschema/ | fastjsonschema docs | undated (accessed 2026-10-18) | "Supported implementations are draft-04, draft-06 and draft-07"; defaults to draft-07; no 2019-09 / 2020-12 |
| S10 | datamodel-code-generator README | https://github.com/koxudaxi/datamodel-code-generator | koxudaxi, MIT | accessed 2026-10-18 | Inputs include JSON Schema, OpenAPI 3, Avro, Protobuf; outputs include pydantic v2 BaseModel/dataclass, TypedDict, msgspec; MIT |
| S11 | datamodel-code-generator: Schema version support | https://datamodel-code-generator.koxudaxi.dev/supported_formats/ | datamodel-code-generator docs | accessed 2026-10-18 | Supports draft-04/06/07, 2019-09, 2020-12; `$defs`; `prefixItems` and `$dynamicRef` (2020-12); `unevaluatedProperties` partial; `$defs` without `$schema` is detected as 2020-12 |
| S12 | `datamodel-codegen --help` (CLI built-in documentation) | local: `uvx --from datamodel-code-generator==0.85.0 datamodel-codegen --help` | datamodel-code-generator 0.85.0 | release 2026-10-16 (S28) | `--check` "Verify generated files are up-to-date without modifying them. Exits with code 1 if differences found ... Useful for CI"; `--use-missing-sentinel` (pydantic `MISSING` with `--target-pydantic-version 2.14`); `--extra-fields forbid`; `--formatters` default will change to `builtin`; dated `--preset`s pin options |
| S13 | Pydantic v2.14 release | https://pydantic.dev/articles/pydantic-v2-14-release | Pydantic | 2026-10-08 | The `MISSING` sentinel, experimental since 2.12, "is now stabilized and part of the main Pydantic API" (`from pydantic import MISSING`) |
| S14 | Pydantic: Unions | https://pydantic.dev/docs/validation/latest/concepts/unions/ | Pydantic docs (latest) | accessed 2026-10-18 | Discriminated unions "are both more performant and more predictable than untagged unions"; smart mode's algorithm "may change between Pydantic minor releases" |
| S15 | json-schema-to-typescript README | https://github.com/bcherny/json-schema-to-typescript | bcherny, MIT | accessed 2026-10-18 | One pipeline for all drafts; 2019-09/2020-12 support mixed (`prefixItems`, `$dynamicRef`, `unevaluated*` etc. ignored); min/max, `pattern`, `oneOf` exclusivity are "not expressible in TypeScript"; options `unreachableDefinitions`, `additionalProperties`, `maxItems` (-1 ignores bounds); compile-time types only |
| S16 | quicktype README | https://github.com/glideapps/quicktype | glideapps, Apache-2.0 (GitHub API) | accessed 2026-10-18 | Generates models and serializers from JSON, JSON Schema, TypeScript, GraphQL for many languages incl. TypeScript, Python, Rust; no JSON Schema draft named |
| S17 | typify README | https://github.com/oxidecomputer/typify | Oxide Computer, Apache-2.0 | accessed 2026-10-18 (repo pushed 2026-10-14) | `oneOf` maps to a Rust enum; `additionalProperties: false` → `#[serde(deny_unknown_fields)]`; non-required properties → `Option<T>`; "Bounded numbers aren't very well handled", generated types "won't enforce those value constraints"; "Typify is a work in progress" |
| S18 | schemafy README | https://github.com/Marwes/schemafy | Marwes, MIT | accessed 2026-10-18 (repo pushed 2023-11-07; last crate 2021-10-06) | Takes "a JSON schema (draft 4)"; "No checking such as min_value are done"; no guarantee it generates any types for a schema |
| S19 | Serde: Container attributes | https://serde.rs/container-attrs.html | serde.rs | accessed 2026-10-18 | `deny_unknown_fields` errors on unknown fields and is not supported with `flatten`; `tag = "..."` is the internally tagged representation; `untagged` gives no informative error when no variant matches |
| S20 | jsonschema crate docs | https://docs.rs/jsonschema/latest/jsonschema/ | Stranger6667, jsonschema 0.59.0, MIT | 2026-10-16 | Drafts 2020-12, 2019-09, 7, 6, 4, each with its own module (`jsonschema::draft202012::new`); "Compliance levels vary across drafts, with newer versions having some unimplemented keywords" |
| S21 | rfc8785.py README | https://github.com/trailofbits/rfc8785.py | Trail of Bits, Apache-2.0 | accessed 2026-10-18 (PyPI 0.1.4, 2024-09-27) | Pure-Python, dependency-free RFC 8785 implementation, "behaviorally comparable" to the reference implementation; returns canonical UTF-8 bytes |
| S22 | serde-json-canonicalizer README | https://github.com/evik42/serde-json-canonicalizer | evik42, MIT | accessed 2026-10-18 (crate 0.4.1, 2026-10-13) | "An RFC 8785 compatible JSON Canonicalization Scheme output for serde_json"; with `arbitrary_precision` numbers are converted to doubles; store unrepresentable values as strings |
| S23 | Pokémon Showdown `sim/SIMULATOR.md` | https://github.com/smogon/pokemon-showdown/blob/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim/SIMULATOR.md | smogon, MIT, pinned commit 7244fe8 | pinned commit | `\|split\|PLAYERID` is followed by a SECRET line "for the specific player or an omniscient observer" and a PUBLIC line "suitable for display to opponents / teammates / spectators" (may be empty); `sideupdate` goes to one player only; choice requests are sent that way (lines 137-155) |
| S24 | Pokémon Showdown sim source at the pin | https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/sim | smogon, MIT, pinned commit 7244fe8 | pinned commit | `battle.ts:33-59` `extractChannelMessages`: channel `-1` (omniscient) gets SECRET, the named player gets SECRET, every other channel gets PUBLIC, an empty PUBLIC line is dropped; `battle.ts:3081` `addSplit`; `battle.ts:1422-1468` `getRequests` puts only `side.getRequestData()` (plus `ally` only when `side.allySide` exists, i.e. multi battles) in a request; `battle-stream.ts:393-465` `getPlayerStreams` routes `update` through `extractChannelMessages` and `sideupdate` to one player; `side.ts:518-523` requests are `sideupdate`; `pokemon.ts:2060-2075` exact HP is SECRET, the `champions` mod shares a percentage; `battle.ts:2004` and `data/rulesets.ts:652,1965` emit Team Preview `\|poke\|` lines with an empty item field |
| S25 | Confluent: Schema Evolution and Compatibility (**secondary, prior art**) | https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html | Confluent Platform 8.3 docs | accessed 2026-10-18 | BACKWARD = new readers read data written with the last schema; FORWARD = old readers read new data; FULL = both; `_TRANSITIVE` checks against all earlier versions; for JSON Schema the allowed changes depend on the content model (`additionalProperties` open vs closed) |
| S26 | Apache Avro 1.12.0 Specification: Schema Resolution (**prior art**) | https://avro.apache.org/docs/1.12.0/specification/ | Apache Avro 1.12.0 | accessed 2026-10-18 | Writer-only fields are ignored; a reader field missing in the writer uses its default, and without a default "an error is signalled"; an unknown enum symbol maps to the reader's enum default if any |
| S27 | Axon Framework Reference 4.10: Event Versioning (**prior art**) | https://docs.axoniq.io/axon-framework-reference/4.10/events/event-versioning/ | AxonIQ, 4.10 | accessed 2026-10-18 | The event store is read and append-only; upcasters turn revision x into x+1 at read time on the serialized form; upcasters chain and order matters; history stays intact |
| S28 | Registry metadata (npm, PyPI JSON API, crates.io API, GitHub API) | `npm view <pkg>`, `https://pypi.org/pypi/<pkg>/json`, `https://crates.io/api/v1/crates/<crate>`, `https://api.github.com/repos/<owner>/<repo>` | registries | queried 2026-10-18 | Versions, licences and release dates of every package named here, including `typing-extensions` 4.16.0 `PSF-2.0` and `rfc8785` 0.1.4 (classifier only) |
| S29 | Spike results | `spikes/671-contract-codegen/README.md` (worktree) | Areneon project measurement, reproducible with `run.sh` | 2026-10-18 | The prototype results tabulated in *Evidence and alternatives* |
| S30 | Areneon repository at `4a98a18` | `packages/schema/rules/v1/*.schema.json`, `packages/showdown-bridge/src/{rules-load,rules,request-identity,choice-judge,runner}.ts`, `docs/research/timer-deadline-budget.md` §4, `scripts/check_licenses.py` | Areneon | 2026-10-18 | Existing schemas declare `$schema` 2020-12 and `$id` `urn:areneon:schema:rules:v1:<kind>` with `additionalProperties: false` and `schema_version` `const`; loader uses `Ajv2020({allErrors, strict: true})` (Ajv 8.20.0); `jcs()` = `canonicalJson` + I-JSON checks; `requestHash` = SHA-256 of `canonicalJson(request minus rqid)`; `LegalChoiceSet {request_seq, request_hash, showdown_commit, choices, pending}`; `runner.ts` gives `CloneJudge` the live `battleStream.battle` (omniscient object) and keeps `omniscient_log`/`hidden_verdicts` as evaluation-only; DecisionContext fields; licence allowlist (MIT, Apache-2.0, BSD-2/3, ISC, 0BSD, CC0-1.0, CC-BY-4.0, BlueOak-1.0.0, Python-2.0, Unlicense) |
| S31 | Pokémon Showdown server source at the pin | https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b/server | smogon, MIT, pinned commit 7244fe8 | pinned commit | `server/sockets.ts:261` runs `extractChannelMessages(message, [0, 1, 2, 3, 4])` before sending a room broadcast, so each connection gets its channel's lines; `server/rooms.ts:1969-1980` channel 0 is the spectator log, 1-4 a player's log, `getLogForUser` gives a player its own channel; `server/room-battle.ts:793-816` a `sideupdate` request gets the server `rqid` and goes only to that player (`player.sendRoom`) |
