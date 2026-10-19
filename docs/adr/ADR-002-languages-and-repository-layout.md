# ADR-002: Languages and repository layout

**Status:** Accepted (2026-10-17, project owner). Revision 3 amended by the owner on 2026-10-18 to add Rust.  
**Scope of this repository:** offline training, evaluation and local bot sparring.  
**Wave:** 1  
**ADR issue:** #694 · **Epic:** #688 (Wave 1 — Foundation)  
**Implementation:** #669, #684  
**Review:** plan-stage review (conditional approve; open questions resolved below), recorded in the #694 journal

## Decision
A **polyglot monorepo** (`piglor/areneon`) with three first-class languages, plus the pinned upstream Showdown JS:

- **TypeScript** for the Showdown bridge and schema tooling.
- **Python/PyTorch** for ML, training/eval services and repo scripts.
- **Rust** for performance-sensitive, engine-independent components: calculators (native and WASM), rules catalog, rollout/eval orchestration and typed tooling.

Rust must **not** reimplement battle mechanics or legality. The pinned Showdown engine stays the only legality authority (ADR-003), and Rust code that needs legality calls through the bridge.

```
packages/schema/             # JSON Schema contracts (source of truth)
packages/showdown-bridge/    # TS: headless runner, request decoder, legal-choice enumeration beside the sim
crates/                      # Rust: Cargo workspace members (root Cargo.toml)
services/inference-api/      # Python (Wave 2)
ml/                          # Python/PyTorch
vendor/pokemon-showdown/     # git submodule pinned to an exact master SHA, built by script
fixtures/                    # sanitized/synthetic only, CC0
spikes/<ticket>-<name>/      # experiment code with lighter gates; promoted or deleted
private-data/                # gitignored
```

## Resolved sub-decisions
| Topic | Decision |
|---|---|
| Contract source of truth | Hand-written **JSON Schema** in `packages/schema`. TS types, pydantic models and Rust types (serde) are generated from it, and CI checks they stay in sync. |
| Showdown consumption | **Git submodule** pinned to a `master` SHA. Bumped deliberately (on regulation change or monthly) through a PR that re-runs golden fixtures. |
| Python | 3.12 + `uv`; ruff, `mypy --strict`, pytest + hypothesis |
| Node | 24 LTS + pnpm workspaces; TypeScript strict, eslint, vitest |
| Rust | Stable toolchain pinned in `rust-toolchain.toml`; root Cargo workspace over `crates/*`; `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, `cargo llvm-cov` at ≥90% lines, `cargo deny` for licences and advisories |
| Coverage | ≥90% lines for `packages/*`, `crates/*` and `services/*`; `spikes/` exempt |
| Supply chain | gitleaks, osv-scanner, licence allowlist (pnpm, uv and cargo), Dependabot (npm, uv, cargo, github-actions) |
| Hooks | `pre-commit` framework, including a guard rejecting PokePaste-shaped files outside `fixtures/` |
| Conventions | Conventional Commits, `Refs #N`, DCO |

## Introducing a language toolchain
A toolchain's CI job, hooks and Dependabot entry land **with its first real package or crate**, not before. This avoids empty placeholder jobs. Each new crate or package is a 0→1 component and still needs its own component ADR for its boundary.

## Alternatives rejected
- **Multi-repo:** contract drift.
- **Rust simulator or legality core:** duplicates the engine without a measured bottleneck and splits the legality authority. Rust stays engine-independent.
- **Pure TS:** weak ML ecosystem.
- **pydantic-first, zod-first or serde-first contracts:** one language would own the contract.
- **npm/@pkmn/sim:** predates M-C or lacks the `champions` mod.
