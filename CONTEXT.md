# Areneon: Domain Context & Glossary

A shared language for humans and agents working on Areneon. Use these terms exactly; don't drift to synonyms.

---

## What Areneon Is

**Areneon** is an open-source research project: an AI training pipeline for competitive VGC **Doubles** on the pinned Pokémon Showdown engine. Bots learn by automatic self-play and are judged by independent, held-out evaluation; a human can also spar against the bot on a local server. Target format: Pokémon Champions Regulation M-C, bring 6 choose 4, Closed Team Sheet.

---

## Game and format

| Term | Meaning |
|---|---|
| **Format** | A pinned Showdown format ID plus regulation, e.g. `[Gen 9 Champions] VGC 2026 Reg M-C`. Always recorded in the version manifest. |
| **Regulation** | The Champions rule period (e.g. M-C). A manifest parameter, not hard-coded. |
| **Closed Team Sheet (CTS)** | Opponents see only species/details at Team Preview. On Showdown M-C this means Bo1 with Open Team Sheets declined by every actor. |
| **Open Team Sheets (OTS)** | Showdown rule that reveals moves/items/abilities. Opt-in in Bo1, forced in Bo3. Areneon declines it. |
| **Team Preview** | The phase where each player brings 4 of 6 and picks 2 leads. |
| **VGC Timer** | Showdown timer: 90 s preview, 420 s bank, 55 s max per turn. Areneon deadlines derive from it. |
| **Stat Points** | Champions replacement for EVs (max 32 per stat, IVs fixed at 31). |

## Showdown integration

| Term | Meaning |
|---|---|
| **Reference engine** | The upstream `smogon/pokemon-showdown` simulator pinned to an exact commit. The only authority on mechanics and legality. |
| **Version manifest** | `{showdown_commit, format_id, regulation, ruleset_hash, format_rules_hash}` attached to every fixture, episode and benchmark. `ruleset_hash` is derived from the engine; `format_rules_hash` from the format rules record. |
| **Rules records** | Areneon's own rules (which formats it runs, forbidden engine rules with reasons, regulation sources, gap ledger) as JSON files in `rules/`, validated by `packages/schema/rules/` and the pinned engine (ADR-007). Never constants in code. Legality and mechanics are not rules records; they stay in the engine. |
| **Rules snapshot** | The published, content-hashed set of all rules records for one engine commit. `rules_snapshot_hash` = SHA-256 of its RFC 8785 (JCS) bytes. Provenance, not replay identity. |
| **Format rules record** | `{schema_version, format_id, regulation, game_type, forbidden_rules}` for one format: the rules that change how it plays. Embedded in every artifact; its hash is `format_rules_hash`. |
| **Rules channel** | `rules/channels/<channel>.json` (`{channel, seq, hash}`) naming the snapshot live runtimes serve. Changed only by PR, `seq` +1 per change. Only `stable` exists. |
| **Request** | The per-side `|request|` JSON the engine sends a player: what that player may choose now. |
| **`rqid`** | Request ID assigned by the Showdown **server** only. Absent on the in-process sim path; optional in contracts. |
| **`request_seq` / `request_hash`** | Areneon's request identity on both paths: a per-side monotonic counter plus a hash of the canonical request JSON. Used for stale rejection. |
| **Headless runtime (A1)** | In-process sim BattleStream for training, evaluation and fixtures. |
| **Bot client runtime (A2)** | The bot as a player on a local Showdown server, for local human sparring. |

## Team system

| Term | Meaning |
|---|---|
| **Team vocabulary** | What the team generator may draw from, read from the pinned engine for one format: the roster the validator accepts, each species' abilities and move pool, natures, unbanned items and the rule-table values. Hashed (`vocabulary_hash`). Builds on the engine-data table (ADR-013). |
| **TeamRecord** | One public synthetic team: PokePaste in the engine's canonical form, its hashes, origin and labels. `data_class` is always `synthetic`. |
| **`team_hash` / `canonical_team_hash`** | SHA-256 of the JCS bytes of the sets in paste order / sorted by species id (ties by the set's JCS bytes). The canonical hash is the same for every ordering of one team and is the identity for the no-overlap rules. |
| **TeamLabelSet** | Team labels as data: leaf predicates over engine ids, hashed into every record and manifest. |
| **Team snapshot** | A published set of immutable shards with a `TeamSnapshotManifest`. `logical_snapshot_hash` is independent of the file writer; the snapshot hash identifies one physical publication. |

## Observations and decisions

| Term | Meaning |
|---|---|
| **Player-perspective observation** | Everything one player may legitimately know: their own request, their own team, the public battle log. Never the omniscient stream. |
| **PrivateTeamSpec** | A player's or bot's full team (PokePaste-compatible). Private; never committed. |
| **PreviewObservation / BattleObservation** | Versioned player-perspective inputs for the two decision phases. |
| **Legal joint choice** | A complete Doubles choice for both active slots (moves, targets, switches) validated by the reference engine. |
| **Recommendation** | A policy's output for one request: a legal choice, the policy that produced it, elapsed time and the deadline outcome (ADR-008). No confidence field yet. |
| **Episode** | An immutable recorded battle from one actor's perspective with provenance, rights and behaviour-policy ID. |

## Policies and evaluation

| Term | Meaning |
|---|---|
| **Preview Policy** | Learned model choosing 4 + 2 leads. Separately trained and promoted. |
| **Battle Policy** | Learned model choosing legal joint choices each turn. Separately trained and promoted. |
| **Legal heuristic** | A non-learned baseline/fallback and integration fixture. Never described as AI. |
| **Sealed test** | Held-out opponent teams/policies/seeds never used for fitting or tuning. |

## Project records

| Term | Meaning |
|---|---|
| **Wave** | A Notion delivery contract (Wave 1–9) mapped to a Redmine version. |
| **ADR** | Architecture Decision Record on the Redmine wiki under `ADR`; copies of the ADRs are in `docs/adr/`. |
| **EXP-nn** | A planned experiment from the Notion ADR register, tracked as a Redmine issue. |
