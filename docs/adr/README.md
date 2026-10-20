# Architecture Decision Records

This repository covers offline training, evaluation and local bot sparring. The files here are copies of the accepted text of its architecture decisions; the canonical record is the project's ADR tracker.

Notes for reading the copies:

- `#<number>` references (for example `#682`) are issue numbers in the project's tracker, not GitHub issues or pull requests of this repository. `piglor/areneon#<number>` refers to a GitHub pull request here.
- Spike branches and `spikes/` paths cited as evidence live on their own branches and are not on `main`.

| ADR | Title | Status | Tracker version | Summary |
|---|---|---|---|---|
| [ADR-001](ADR-001-open-source-licence.md) | Open-source licence | Accepted 2026-10-17 | 5 | Apache-2.0 code, CC0 synthetic fixtures, DCO sign-off, no CLA. |
| [ADR-002](ADR-002-languages-and-repository-layout.md) | Languages and repository layout | Accepted 2026-10-17 (r3, Rust added 2026-10-18) | 7 | One polyglot monorepo: TypeScript, Python and Rust, JSON Schema contracts, pinned Showdown submodule. |
| [ADR-003](ADR-003-pinned-showdown-reference-simulator.md) | Pinned Showdown reference simulator and format | Accepted 2026-10-18 (r2) | 4 | Upstream Showdown pinned to an exact commit is the only mechanics and legality authority; Bo1 Closed Team Sheet format. |
| [ADR-005](ADR-005-bot-opponent-runtime.md) | Bot opponent runtime | Accepted 2026-10-19 | 16 | A1 in-process sim for training and evaluation; A2 bot client on a local pinned server for human sparring. |
| [ADR-006](ADR-006-legal-joint-choice-enumeration.md) | Legal joint-choice enumeration | Accepted 2026-10-18 (r3) | 5 | A pure enumerator over the player's own request, judged by the engine; hidden-information candidates stay pending. |
| [ADR-007](ADR-007-rules-store-database-and-editing.md) | Rules store: database and editing workflow | Accepted 2026-10-18 (r4) | 4 | Rules authored as reviewed JSON records in git, published as attested snapshots into an append-only PostgreSQL 18 store. |
| [ADR-008](ADR-008-cross-language-data-contracts.md) | Cross-language data contracts | Accepted 2026-10-18 (r3) | 6 | JSON Schema 2020-12 contract profile with generated TS and pydantic types, JCS hashing, privacy by structure. |
