# Areneon: Domain Context & Glossary

A shared language for humans and agents working on Areneon. Use these terms exactly; don't drift to synonyms.

---

## What Areneon Is

**Areneon** is an open-source competitive VGC **Doubles** AI sparring partner for Pokémon Showdown. A human can challenge a private bot opponent, and an optional read-only browser advisor shows recommendations for the player's own decisions. Target format: Pokémon Champions Regulation M-C, bring 6 choose 4, Closed Team Sheet.

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
| **Version manifest** | `{showdown_commit, format_id, regulation, ruleset_hash}` attached to every fixture, episode and benchmark. |
| **Request** | The per-side `|request|` JSON the engine sends a player: what that player may choose now. |
| **`rqid`** | Request ID assigned by the Showdown **server** only. Absent on the in-process sim path; optional in contracts. |
| **`request_seq` / `request_hash`** | Areneon's request identity on both paths: a per-side monotonic counter plus a hash of the canonical request JSON. Used for stale rejection. |
| **Headless runtime (A1)** | In-process sim BattleStream for training, evaluation and fixtures. |
| **Bot client runtime (A2)** | The bot as a player on a local/private Showdown server, for human sparring. |
| **Advisor extension** | Read-only Chromium MV3 extension that reads the player's **own** request and shows recommendations. Never submits by default. |

## Observations and decisions

| Term | Meaning |
|---|---|
| **Player-perspective observation** | Everything one player may legitimately know: their own request, their own team, the public battle log. Never the omniscient stream. |
| **PrivateTeamSpec** | A player's or bot's full team (PokePaste-compatible). Private; never committed. |
| **PreviewObservation / BattleObservation** | Versioned player-perspective inputs for the two decision phases. |
| **Legal joint choice** | A complete Doubles choice for both active slots (moves, targets, switches) validated by the reference engine. |
| **Recommendation** | A policy's output: a legal choice plus confidence, elapsed time and traceability. |
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
| **ADR** | Architecture Decision Record on the Redmine wiki under `ADR`. |
| **EXP-nn** | A planned experiment from the Notion ADR register, tracked as a Redmine issue. |
