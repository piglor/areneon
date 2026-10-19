# Data-source and reuse-rights inventory

Research for Redmine #774 (Notion Wave 1 gate item 3). It feeds ADR-009 (#728), which decides the training data sources. The Champions Battle Data API gets its own assessment in #679; this note covers only its terms.

**Scope.** Every source of data or code that the offline training pipeline uses or could use: the pinned engine, the project's own generated data, replays and simulator logs, public datasets, libraries and data packages, and the official Champions publications the rules records cite. For each source the note records what the source's own terms say, and what Areneon does with it.

**Not legal advice.** A verdict here is the project's engineering decision, made under the terms as quoted and on the date accessed. It is not a statement of what the law allows. Where the terms are missing, unclear or contradictory, the verdict is **open question**, and the question is listed for the owner in [§6](#6-open-questions-for-the-owner). The ticket's rule applies throughout: **material without a licence or terms that grant reuse is excluded** until the owner decides otherwise.

**How to read.**

- Every fact carries a source label `[S#]` (see [Sources](#sources)). All sources were accessed on **2026-10-19**.
- Verdicts: **use** (allowed under the stated terms, with the usual notice duties); **use with conditions** (allowed only as described); **do not use**; **open question** (excluded until the owner decides).
- "Training use" means using the data, or features computed from it, as input for fitting a policy. "Redistribution" means committing it to this public repository or publishing it in any other artifact.
- Facts are what a source says or the pinned code does. Anything derived from them is in [§7 Inference](#7-inference).

## Summary

| # | Source | Licence or terms | Training use | Redistribution | Verdict |
|---|---|---|---|---|---|
| D1 | Pinned Pokémon Showdown engine (server, simulator, `data/`) | MIT [S1] | Yes: it produces all self-play data | Yes, with the MIT notice (already in `NOTICE`) | **Use** |
| D2 | Areneon's own generated data (self-play episodes, simulator logs, synthetic teams, fixtures) | CC0-1.0 for fixtures and synthetic data; Apache-2.0 for code [S27] | Yes | Yes (synthetic only) | **Use** |
| D3 | Pokémon Showdown public replays (replay service and its JSON API) | None found; no reuse terms found [S2][S4][S5] | Not until decided | No | **Open question** (excluded by default) |
| D4a | VGC-Bench battle logs (Hugging Face) | Card says MIT; logs scraped from D3 [S21] | Not until decided; wrong regulations and sheets | No | **Open question** |
| D4b | Metamon raw replays | No licence [S22] | No | No | **Do not use** |
| D4c | Metamon parsed replays | CC BY-NC 4.0 [S23] | No | No | **Do not use** (NC clause treated as binding: I5) |
| D4d | PokéChamp dataset | No licence [S24] | No | No | **Do not use** |
| D4e | "Pokémon Showdown Replays" (HolidayOugi) | No licence [S25] | No | No | **Do not use** |
| D5 | Smogon usage statistics (`smogon.com/stats`) | No terms found; a third party calls them public domain [S7][S8][S9] | Not until decided | No | **Open question** |
| D6 | Smogon sets and analyses (`@smogon/sets`, set data in calculators) | Copyrighted by Smogon, permission required; npm `UNLICENSED` [S1][S9][S10] | No | No | **Do not use** |
| D7 | `@smogon/calc` | MIT [S10][S11] | As a code dependency for features, never as a mechanics authority | Yes, with the MIT notice | **Use with conditions** |
| D8 | `@pkmn/data`, `@pkmn/dex`, `@pkmn/mods`, `@pkmn/smogon` (code) | MIT [S10][S12][S9] | As code only; data reached through `@pkmn/smogon` falls under D5/D6 | Yes, with the MIT notice | **Use with conditions** |
| D9 | PokéAPI | Code and data repositories BSD-3-Clause; fair use policy [S13][S14][S15] | Not needed; cross-check only | Not planned (BSD notice if ever) | **Use with conditions** |
| D10 | Champions Calc | AGPL-3.0-or-later [S17] | No | No | **Do not use** |
| D11 | Pokémon Champions Battle Data API | Site terms: personal, educational, competitive analysis; API rules: no mirror, no dump, attribution [S18][S19][S20] | Not until decided | No | **Open question** (assessment in #679) |
| D12 | Official Champions and Play! Pokémon publications | Pokémon terms of use: personal, noncommercial use; no bulk download [S26] | No (facts only, as rules records) | Facts and links only, never copies | **Use with conditions** |

## 1. Rights vocabulary (ADR-008) and derived artifacts

ADR-008 §5 gives every `PrivateTeamSpec` and `Episode` a required `data_class` and every `Episode` a required `rights` [S27]:

| `data_class` | `rights` | Meaning | Storage |
|---|---|---|---|
| `synthetic` | `cc0-synthetic` | Made by Areneon, dedicated CC0; may be committed | `fixtures/`, any store |
| `private` | `owner-private` | A real team or a human's games, owned by the project owner | gitignored `private-data/` or the deployment's private store only |

`packages/showdown-bridge/src/episode.ts` (`rightsFor`) derives `rights` from `data_class`, and `scripts/check_data_class.py` rejects committed `private` JSON [S27]. ADR-001 adds: real replays and other people's teams are never committed, and a code licence grants no rights to replays, scraped data or trained weights [S27].

**Consequence for third-party data.** Neither `rights` value describes data owned by someone else: `cc0-synthetic` claims a CC0 dedication Areneon cannot make for third-party content, and `owner-private` claims the owner holds the rights. A derived artifact from D3, D4 or D11 therefore has **no valid `rights` value today**. If ADR-009 admits any of them, it must add a value through ADR-008's schema versioning (a new `schema_version`), with the storage rule of `private` data. Until then such data cannot be recorded as an `Episode`. Each source section below names the value a derived artifact would get.

**Two schema points for ADR-009.** (1) `data_class` has only `synthetic` and `private`, and its schema description reads "a real team or a human's games" (`packages/schema/contracts/v1/episode.schema.json`, and the same text in the observation and `private-team-spec` schemas). Third-party replays are neither, so ADR-009 may need a new `data_class` as well as a new `rights` value. (2) `rightsFor` returns `cc0-synthetic` for `synthetic` and `owner-private` for everything else, from `data_class` alone, so a new `rights` value needs a code change in `episode.ts` and not only a schema change. Both read from the code and schemas at `main` on 2026-10-19; whether the second follows from the first is I7.

## 2. Engine and project data

### D1. Pinned Pokémon Showdown engine

| Field | Value |
|---|---|
| What | The upstream simulator and server, git submodule `vendor/pokemon-showdown`, pinned at `7244fe8e28fa8bc4d0809ae30d183c608b62bb2b` (ADR-003). Includes the game data under `data/` (species, moves, learnsets, the `champions` mod). |
| Licence | MIT, "Copyright (c) 2011-2026 Guangcong Luo and other contributors" (`LICENSE`); the README says the server is distributed under the MIT License (`README.md:72-77`) [S1]. The only other licence file in the tree is `tools/set-import/sets/LICENSE`, which covers imported sets (D6), not the simulator [S1]. |
| Version, accessed | Pinned commit above (2026-10-16); accessed 2026-10-19 |
| May do | Use as code, run, store, cache, redistribute (MIT). Its outputs (logs, requests) are Areneon's own generated data (D2). |
| Attribution | Keep the MIT notice. `NOTICE` already names Showdown, its copyright and licence, and states the simulator runs unmodified [S27]. |
| Privacy | None in the engine. Pokémon names in `data/` are trademarks; `NOTICE` carries the non-affiliation statement [S27]. |
| Collection rules | None: local submodule. |
| Verdict | **Use.** |
| Reasoning | MIT is on the licence allowlist and compatible with Apache-2.0 (ADR-001) [S27]. It is already the only legality and mechanics authority (ADR-003). |
| Where | `vendor/pokemon-showdown`, `packages/showdown-bridge`, `packages/local-server`. |
| Derived artifact | Outputs of synthetic battles: `synthetic` / `cc0-synthetic` (see D2). |

The `pokemon-showdown` npm package (0.11.11, MIT [S10]) is the same code base; ADR-003 rejected it as the pin because it predates M-C. It needs no separate verdict.

### D2. Areneon's own generated data

| Field | Value |
|---|---|
| What | Self-play episodes and simulator logs from the headless runtime (A1), golden request fixtures and choice corpus, per-player channel logs, synthetic PokePaste teams and their verdict logs. |
| Licence | Fixtures and synthetic data: CC0-1.0 (`fixtures/LICENSE`, `NOTICE`); code: Apache-2.0 (ADR-001) [S27]. |
| Version, accessed | `main` at the branch point of this note; accessed 2026-10-19 |
| May do | Everything: train, store, cache, redistribute. |
| Attribution | None required (CC0). Provenance is still recorded: the version manifest and `provenance` in every `Episode` (ADR-008). |
| Privacy | Synthetic teams contain no real player's team (`fixtures/teams/<format>/manifest.json` records `"provenance": "synthetic: ..."`) [S27]. The `omniscient_log` and `hidden_verdicts` of a run hold both sides' hidden information: they are evaluation artifacts and never policy input (ADR-008 §5). |
| Collection rules | None. |
| Verdict | **Use.** |
| Reasoning | The project created the data and dedicated it to the public domain. |
| Where | `fixtures/`, `packages/showdown-bridge/src/episode.ts`, `episode-io.ts`; planned `ml/` self-play league. |
| Derived artifact | `synthetic` / `cc0-synthetic`. A synthetic team stays `synthetic` only if it is not copied from a real team (ADR-001); see open question Q4 for teams generated from third-party statistics. |

## 3. Replays and public datasets

### D3. Pokémon Showdown public replays

| Field | Value |
|---|---|
| What | Battles that players upload to the Showdown replay service. A JSON API returns a replay (`<id>.json`, also `.log`) and searches recent replays by user and format (`search.json`, 51 results per page, paginated by upload time) [S2]. |
| Licence | **None found.** The Showdown server code is MIT [S1], but that licence covers software, not battle records. The privacy policy has no section on replays or reuse; it says the site may keep text logs of chat in rooms and battles [S4]. The site rules contain no rule on automated access, replays or reuse [S5]. The API document says the APIs carry `Access-Control-Allow-Origin: *` so they can be read programmatically, and that the replay page's HTML must not be scraped [S2]. |
| Version, accessed | `WEB-API.md` at client commit `ab4b77b` (2026-10-18); policy pages undated; accessed 2026-10-19 |
| Content of a replay | A search for `gen9championsvgc2026regmc` on 2026-10-19 returned 51 public entries with the fields `uploadtime, id, format, players, rating, private, password` [S6]. One replay's JSON held `id, format, players, log, uploadtime, views, formatid, rating, private, password`, and no `inputlog` [S6]; the API document says an inputlog is served only for formats with generated teams [S2]. The uploaded log is the battle room's spectator scrollback (`server/rooms.ts:2063-2072`) [S1]; the server also stores the inputlog with the replay (`server/rooms.ts:2099-2109`) [S1]. |
| Privacy | The log names both players (`\|player\|`) and carries room lines: joins and leaves (`\|j\|`, `\|l\|`), timer messages, rating changes (`\|raw\|`) [S6], and room chat as `\|c\|` lines, since chat is added to the same room log (`server/chat.ts:727`, `server/rooms.ts:379-381`) [S1]. A player can ask for a private replay (`!` before their name; `private` 1 = private, 3 = deleted) (`server/replays.ts:32-48`) [S1]. In a Bo1 M-C battle where both players accept Open Team Sheets, the log carries each team sheet as a `\|showteam\|` line (`sim/battle.ts:3220`) [S1]: another player's team. The stored inputlog holds both full teams (hidden information under CTS). |
| Collection rules | No published rate limit; the service sits behind Cloudflare [S6]. If the owner admits D3, Areneon's own rules would be: the JSON API only, never page scraping [S2]; one sequential client with back-off on errors; a fixed set of replay ids per collection run, cached, never refetched; public replays only (`private` = 0); deletion honoured on re-check. |
| Verdict | **Open question.** Excluded by default. |
| Reasoning | No licence or terms grant reuse of replays, so the ticket rule excludes them. The service publishes an API for programmatic reads [S2], which shows access is expected, but access is not a grant to copy, train on or republish. Even if admitted, a replay is a spectator log: it has no `\|request\|` lines [S6], so it cannot become a `PlayerChannelLog` without reconstruction, and the player's own team is not in it. Replays with `\|showteam\|` are not CTS battles and would not match the target format. |
| Where | Nowhere today. If admitted: a collector and a reconstruction step under the planned `ml/` or `services/`, data only in a private store, never in `fixtures/` (ADR-001). |
| Derived artifact | `data_class: private` (a human's games, never committed); **no valid `rights` value** (see §1). Player names and chat lines would be dropped before storage. |

### D4. Public replay datasets

All five were found on Hugging Face by searching for Showdown and VGC datasets. Each is built from D3, so D3's open question applies to all of them on top of their own licence.

| # | Dataset | Licence (as the card states) | Content | Verdict and reasoning |
|---|---|---|---|---|
| D4a | `cameronangliss/vgc-battle-logs` @ `74f2605` | `license: mit` [S21]; code repository `cameronangliss/vgc-bench` is MIT [S21] | 88,905 logs "scraped" from the Showdown replay database for Champions Reg M-A and M-B (Bo1 and Bo3), "all filtered for Open Team Sheets (OTS)"; latest log 2026-06-20; usernames not stated as removed [S21] | **Open question.** The card's MIT tag is the uploader's licence; the card does not say the uploader holds rights in the underlying replays (I6). The data is also off-target: regulations M-A and M-B, not M-C, and Open Team Sheets, not CTS. Not usable for M-C CTS training even if the rights question is settled; possibly useful only as an out-of-format reference. |
| D4b | `jakegrigsby/metamon-raw-replays` @ `792a86b` | No licence field [S22] | Showdown replays "publicly available from the Showdown replay API"; usernames replaced by consistent pseudonyms; singles formats (e.g. Gen 9 OU) [S22] | **Do not use.** No licence; singles formats only. |
| D4c | `jakegrigsby/metamon-parsed-replays` @ `f80c25b` | `license: cc-by-nc-4.0` [S23] | Replays "parsed (or 'reconstructed')" into RL trajectories, one archive per singles format [S23] | **Do not use.** Non-commercial licence (CC BY-NC 4.0): content under it cannot be distributed under Apache-2.0, which allows commercial use, so nothing derived from it could be distributed under the project licences (ADR-001). Whether the NC clause also reaches trained weights is a contested legal question that no source settles; it is treated as binding for engineering purposes (I5). Singles formats only. |
| D4d | `milkkarten/pokechamp` @ `b5820ff` | No licence field [S24] | "over 3 million competitive Pokémon battles", mostly singles [S24] | **Do not use.** No licence. |
| D4e | `HolidayOugi/pokemon-showdown-replays` @ `15a0a45` | No licence field [S25] | Public replays "pulled using the Pokémon Showdown API", including Champions formats; counts as of 2026-10-05 [S25] | **Do not use.** No licence. |

Derived artifact for any of them: as D3 (`private`, no valid `rights` value).

## 4. Statistics, libraries and data packages

### D5. Smogon usage statistics

| Field | Value |
|---|---|
| What | Monthly usage tables per format and rating cutoff at `www.smogon.com/stats/<YYYY-MM>/` (plus `chaos/`, `leads/`, `metagame/`, `moveset/`). The 2026-09 directory has `gen9championsvgc2026regmc-{0,1500,1630,1760}.txt`; the cutoff-0 file reports 1,631,943 battles [S7]. |
| Licence | **None found.** The directory listing carries no licence or terms [S7]. Smogon's forum terms cover content users post (users keep copyright and license it to Smogon); they say nothing on the statistics or on automated access [S8]. The `@pkmn/smogon` README (a third party, not Smogon) says "the aggregated stats information is freely available in the public domain" while set and analysis data is "copyrighted by Smogon and its contributors" [S9]. The Showdown set importer's licence calls stats-derived sets MIT [S1]. |
| Version, accessed | 2026-09 files dated 2026-10-01; accessed 2026-10-19 |
| May do | Undecided. |
| Attribution | Not stated; if used, credit Smogon and link the file. |
| Privacy | Aggregates only: no player names in the tables [S7]. |
| Collection rules | None published. If used: download each month's file once, cache it, cite the URL and date. |
| Verdict | **Open question.** |
| Reasoning | Smogon itself publishes no terms. The public-domain statement comes from a third party and is not Smogon's grant. Statistics would only shape synthetic team sampling or evaluation opponent mixes, never legality. |
| Where | Nowhere today. Possible future use: a synthetic team generator or the sealed test's opponent mix (planned `ml/`). |
| Derived artifact | A team sampled from the statistics: see Q4. The statistics file itself: not committed. |

### D6. Smogon sets and analyses

| Field | Value |
|---|---|
| What | Curated sets and analyses from Smogon's dex, shipped as `@smogon/sets`, mirrored at `data.pkmn.cc`, and embedded in calculator front ends (e.g. `src/js/data/sets/champions.js` in `smogon/damage-calc`, with Champions M-B sets) [S9][S10][S11]. |
| Licence | The pinned engine's `tools/set-import/sets/LICENSE`: dex sets "are copyrighted by Smogon.com and its contributors and you must first receive permission from Smogon.com before using sets from this data source"; only the code and the stats-derived sets are MIT [S1]. `@smogon/sets` 2.0.0 is `UNLICENSED` on npm [S10]. `@pkmn/smogon` repeats that set and analysis data is copyrighted by Smogon [S9]. |
| Verdict | **Do not use.** |
| Reasoning | Permission is required and none is held. They are also other people's teams, which ADR-001 never commits. The `@smogon/calc` npm package does not include the front-end set files (D7). |
| Derived artifact | None. |

### D7. `@smogon/calc`

| Field | Value |
|---|---|
| What | The Smogon damage calculator library, npm `@smogon/calc` 0.12.0 (modified 2026-09-18). The package contains `dist/mechanics/champions.js` and `src/mechanics/champions.ts` [S10][S11]. |
| Licence | MIT, "Copyright (c) 2013-2025 Honko and other contributors" (`LICENSE` at `smogon/damage-calc` `c1eb09b`) [S11]; npm `license: MIT` [S10]. |
| May do | Use as code, bundle, redistribute with the MIT notice. Its outputs computed on Areneon's own data are Areneon's data. |
| Attribution | MIT notice in `NOTICE` when it ships. |
| Privacy | None (a library). |
| Verdict | **Use with conditions.** |
| Reasoning | MIT is allowlisted (ADR-001). Conditions: damage estimates only as heuristic or policy features, never as a mechanics or legality authority (ADR-003: the pinned engine decides; Rust never reimplements mechanics); add it through `scripts/check_licenses.py`'s allowlist like any dependency; do not import the front-end set data (D6). |
| Where | Not a dependency today. Candidate for the legal heuristic baseline or feature extraction (planned `crates/` calculators or `ml/`). |
| Derived artifact | Inherits the input's class: on synthetic data, `synthetic` / `cc0-synthetic`. |

### D8. `@pkmn` packages

| Field | Value |
|---|---|
| What | `@pkmn/data`, `@pkmn/dex`, `@pkmn/sim`, `@pkmn/mods` 0.10.11 (repository `pkmn/ps`), and `@pkmn/smogon` 0.6.0 (repository `pkmn/smogon`) [S10]. `@pkmn/mods` 0.10.11 contains `build/champions` and `build/championsregma` [S10]; ADR-003 found no `champions` mod in `@pkmn/sim` 0.10.11 [S27]. |
| Licence | MIT: `pkmn/ps` `LICENSE` at `4fec887` [S12]; `pkmn/smogon` `LICENSE` at `d3cd03b` [S9]; npm `license: MIT` for each [S10]. `@pkmn/smogon`'s README separates the code (MIT) from the data it fetches (D5, D6) [S9]. |
| May do | Use as code and redistribute with the MIT notice. Data fetched through `@pkmn/smogon` follows D5 and D6, not MIT. |
| Verdict | **Use with conditions.** |
| Reasoning | MIT is allowlisted. Conditions: never as a mechanics or legality source (the pinned engine is, ADR-003); a `@pkmn` dex may lag the pin, so any table taken from it is checked against the pinned engine; `@pkmn/smogon` only to read D5 if the owner admits it, never for D6 sets. |
| Where | Not a dependency today. Candidate for typed dex lookups in feature code (planned `ml/`, `crates/`). |
| Derived artifact | As D7. |

### D9. PokéAPI

| Field | Value |
|---|---|
| What | A public REST API and data repository of Pokémon game data. Its version groups include `champions` (and `mega-dimension`, `legends-za`) [S16]. |
| Licence | `PokeAPI/pokeapi` `LICENSE.md` at `6fd69df`: BSD-3-Clause, copyright Paul Hallett and PokéAPI contributors, "Pokémon and Pokémon character names are trademarks of Nintendo" [S13]. `PokeAPI/api-data` `LICENSE.txt` at `8ed59f8`: BSD-3-Clause [S13]. The fair use policy: free and open, "Locally cache resources whenever you request them", IP bans for non-compliance, rate limiting removed after static hosting but requests should be limited [S14]. The about page says the data was gathered "from various resources": Veekun's Pokédex (an open source project) as the base, generation 8 data "scraped from different resources", and generation 9 data "scraped ... from the web" [S15]. |
| May do | Read with local caching; redistribute the repository data with the BSD notice. |
| Attribution | BSD-3 notice and disclaimer if any of its data is redistributed. |
| Privacy | None (game data only). |
| Verdict | **Use with conditions.** |
| Reasoning | Licensed and permissive. Conditions: reference cross-check only (for example in the roster comparison of #773), never an input that overrides the pinned engine; cached, low-volume reads; no data committed unless the BSD notice goes with it. The BSD-3-Clause notice on the repository does not settle the rights of the data inside it, which came from other sources with their own terms (Veekun, and web sources the about page does not name) [S15]. The origin of its `champions` data is not documented (Q6). |
| Where | Not used. Possible cross-check in `rules/gaps/` research. |
| Derived artifact | Not a training input; no `rights` value needed. |

### D10. Champions Calc

| Field | Value |
|---|---|
| What | Identified as `SebNotFound/champions-calc` @ `c900836`, titled "Champions Calc": a Pokémon Champions damage calculator built on `@smogon/calc`, `@pkmn/dex` and `@pkmn/data` [S17]. ADR-001 lists Champions Calc as AGPL-3.0 [S27]. |
| Licence | AGPL-3.0-or-later (`LICENSE`; README "License" section) [S17]. |
| Verdict | **Do not use.** |
| Reasoning | ADR-001: AGPL code is never copied, and the CI licence allowlist fails on AGPL dependencies; reuse would need a superseding ADR [S27]. Nothing it offers is missing from D7 (MIT), which it wraps. This note quotes none of its code or text. |
| Derived artifact | None. |

### D11. Pokémon Champions Battle Data API

| Field | Value |
|---|---|
| What | A fan-made site with usage statistics for Champions ranked play (Regulations M-A, M-B, M-C), "rebuilt daily" from in-game ranked snapshots; it names no operator [S20]. Endpoints: `GET /api`, `/api/pokemon/:name`, `/api/battle/:format/:name`, `/api/metadata/:name`, plus static CSVs and sprites [S20]. |
| Terms | The API rules allow "personal, community, nonprofit, and commercial projects"; require "clear attribution" ("Battle data provided by Pokémon Champions Battle Data", with a link); allow "Reasonable caching" but no "permanent mirror, archive, or replacement"; forbid reselling or redistributing the data "as a standalone product" and building "a competing public API, database dump, bulk download service"; no fixed rate limit, access may be limited or revoked; rules may change at any time [S18]. The licence page instead describes use "for personal, educational, and competitive analysis use", and says the project grants no rights to Pokémon names, images or game data [S19]. Neither page is dated [S18][S19]. |
| Privacy | Aggregates; no player names in the documented schema [S20]. |
| Verdict | **Open question.** Detailed assessment belongs to #679. |
| Reasoning | The two pages disagree on the allowed use, neither has a date or version, and the site disclaims rights in the underlying game data, whose origin (in-game ranked data) comes with no licence from its owner. If admitted, conditions from #679 apply: CI makes no network calls to it, no committed dumps, attribution in any published result. |
| Where | Nowhere today. |
| Derived artifact | Not committed; no valid `rights` value (as D3). |

### D12. Official Champions and Play! Pokémon publications

| Field | Value |
|---|---|
| What | The pages the format matrix cites: Champions news pages for Regulation M-C and its maintenance notice, the eligible Pokémon page, and the Play! Pokémon VGC handbook and update notice (`docs/research/regmc-format-matrix.md`, sources S2-S6) [S27]. `rules/regulations/M-C.json` records one of them as a source URL with a verification date [S27]. |
| Terms | Pokémon terms of use (last update 2023-06-15): content of the service is Pokémon's property; permission is for "personal, noncommercial home use"; users may not "(v) Download quantities of content to a database for any reason", nor "Decompose, disassemble, or reverse engineer any part of any Service" [S26]. The Champions news page shows no own terms link [S26]. |
| May do | Read the pages and record facts (dates, limits, names of rules) with a link and a verification date. |
| Attribution | Cite the URL; the trademark notice in `NOTICE` and `README.md` applies [S27]. |
| Verdict | **Use with conditions.** |
| Reasoning | Areneon takes facts, not content: no text, images or lists are copied, nothing is bulk-downloaded or collected automatically, and the data never becomes a training input. Rules records store the fact and the link (ADR-007). |
| Where | `rules/regulations/`, `rules/gaps/`, `docs/research/`. |
| Derived artifact | Rules records (Apache-2.0 repository content); not a training dataset. |

## 5. Explicitly excluded

| Excluded | Why |
|---|---|
| Any material without a licence or terms that grant reuse | Ticket rule. Applies to D3, D4b, D4d, D4e and, until decided, D4a, D5, D11. |
| Other people's teams in any form: Smogon sets (D6), team sheets in replays (`\|showteam\|`), shared team collections | ADR-001: other people's teams are never committed; D6 requires permission. |
| AGPL code and text: the Showdown client (AGPL-3.0, [S3]) and Champions Calc (D10) | ADR-001; CI licence allowlist. The client's `WEB-API.md` is read only for facts about the replay service [S2]. |
| Non-commercial datasets (D4c) | CC BY-NC is incompatible with distributing anything derived under Apache-2.0 (ADR-001), and its reach to trained weights is treated as binding for engineering purposes (I5). |
| Game files and data extracted from the game | No licence; the Pokémon terms forbid reverse engineering their services [S26]. Mechanics come from the pinned engine only. |
| Player names, chat lines and room messages from any battle log | Personal data with no consent for this use [S4]; they carry nothing a policy needs. |
| Opponent hidden information as policy input (an inputlog, an omniscient log, a split's secret half) | ADR-008 §5: observations are player-perspective only. |

## 6. Open questions for the owner

| # | Question | Sources | Default until answered |
|---|---|---|---|
| Q1 | May Areneon train on public Showdown replays (D3) that carry no licence, given the published JSON API? If yes, under which storage, retention and deletion rules, and which new `rights` value? | D3, §1 | Excluded |
| Q2 | Does an uploader's licence tag on a dataset scraped from D3 (D4a, MIT) change Q1's answer? | D4a | Excluded |
| Q3 | May Smogon usage statistics (D5) be used without terms from Smogon, relying on aggregates only? Should Smogon be asked? | D5 | Excluded |
| Q4 | If D5 or D11 is admitted: is a team sampled from third-party usage statistics still `synthetic` / `cc0-synthetic`? | D2, D5, D11 | Teams are hand-made or engine-generated only |
| Q5 | Which of the Battle Data API's two terms pages governs (D11), and does its disclaimer of game-data rights rule it out? Owned by #679. | D11 | Excluded |
| Q6 | (For the ticket owner, not ADR-009.) Where does PokéAPI's `champions` data come from? If it was extracted from the game, should the cross-check use (D9) stop? | D9 | Cross-check only |
| Q7 | (For the ticket owner, not ADR-009.) Is `SebNotFound/champions-calc` the "Champions Calc" ADR-001 means? The verdict (do not use) holds either way for any AGPL calculator. | D10 | Do not use |
| Q8 | (For the ticket owner, not ADR-009.) Format matrix task U4 proposes rendering the eligible Pokémon page with an automated renderer. Is one automated read of one page compatible with the Pokémon terms (D12), or must it be read by hand? | D12 | Read by hand |
| Q9 | What retention, deletion and takedown rule applies to every admitted source (a replay, a statistics file, an API response), not only to replays? Who handles a request from a player or rights holder to remove data? | All admitted sources | Nothing third-party is stored; if one is admitted, it is stored only with its source, access date and a deletion route, and removed on request |
| Q10 | Should the owner contact Smogon (D5, D6) or the Battle Data API operator (D11) for written permission? Who asks, and does a written grant change the verdict? | D5, D6, D11 | No contact; verdicts stay as above. A written grant would move the source to use with conditions, with the grant stored beside the data |
| Q11 | What happens when a dataset's licence tag changes or disappears upstream (D4, D9, D11)? | D4, D9, D11 | The recorded access date and revision are the evidence of what applied when the data was taken; data already used is kept only if the earlier terms allowed it, and new pulls stop until the owner decides |
| Q12 | Does third-party data permission need its own ADR, or does it live in ADR-009? | All | In ADR-009, as one section on admitted sources |

## 7. Inference

These follow from the facts above but are not stated by any source.

1. **I1.** Self-play on the pinned engine (D1 → D2) is the only data source with no open rights question. Any plan that needs third-party battles depends on Q1.
2. **I2.** Even with Q1 answered yes, D3 data does not fit ADR-008 without new work: a spectator log has no requests and no own team, so first-person reconstruction would infer parts of a `PlayerChannelLog`, and the result would differ from the engine-recorded channels the observation builder now trusts.
3. **I3.** Public Bo1 M-C replays mix CTS battles with battles where both players accepted Open Team Sheets; the `|showteam|` line separates them. The VGC-Bench logs (D4a) are by construction the OTS ones, the opposite of the target.
4. **I4.** Libraries (D7, D8) add no rights question for training data: their outputs on Areneon's own data keep that data's class. Their risk is mechanical drift from the pin, not licensing.
5. **I5.** Whether a non-commercial licence on training data reaches the trained weights is a contested legal question; no source consulted settles it. The note treats the restriction as binding for engineering purposes, which is the cautious reading, and separately CC BY-NC cannot be redistributed under Apache-2.0, so nothing derived from D4c could be distributed under the project licences.
6. **I6.** An uploader's licence tag on a dataset scraped from D3 (D4a) may not be effective for the underlying replays: the card does not say the uploader holds rights in them, and D3's terms grant none. This is the reason Q2 asks the question.
7. **I7.** The point that a new `rights` value needs a code change (§1) is read from `rightsFor` as it stands; ADR-009 could instead replace the derivation with a stored field. That choice is the ADR's, not this note's.

## Sources

All accessed 2026-10-19.

| Label | Title | URL | Publisher / version | Claim supported |
|---|---|---|---|---|
| S1 | Pokémon Showdown, pinned submodule | <https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b> | Smogon / Pokémon Showdown contributors; commit `7244fe8`, 2026-10-16 | MIT `LICENSE` and `README.md:72-77`; `tools/set-import/sets/LICENSE` (Smogon sets need permission; stats-derived sets MIT); replay upload fields and spectator log (`server/rooms.ts:2063-2072`, `:2099-2109`); private flags (`server/replays.ts:32-48`); chat in the room log (`server/chat.ts:727`, `server/rooms.ts:379-381`); `\|showteam\|` (`sim/battle.ts:3220`) |
| S2 | Pokémon Showdown website APIs (`WEB-API.md`) | <https://github.com/smogon/pokemon-showdown-client/blob/ab4b77b8f3a5712e8fbe03ce106ef2b70a874fdb/WEB-API.md> | Pokémon Showdown client repository, commit `ab4b77b`, 2026-10-18 | Replay and search endpoints, 51-result pages, inputlog only for generated-team formats, CORS, no scraping of the replay page |
| S3 | Pokémon Showdown client `README.md`, License section | <https://github.com/smogon/pokemon-showdown-client/blob/ab4b77b8f3a5712e8fbe03ce106ef2b70a874fdb/README.md> | same commit | Client is AGPLv3, not the server's licence |
| S4 | Pokémon Showdown privacy policy | <https://pokemonshowdown.com/privacy> | Pokémon Showdown; undated | Logging of chat in rooms and battles; no replay or reuse terms |
| S5 | Pokémon Showdown rules | <https://pokemonshowdown.com/rules> | Pokémon Showdown; undated | No rule on automated access, replays or reuse |
| S6 | Replay search and one replay JSON for `gen9championsvgc2026regmc` | <https://replay.pokemonshowdown.com/search.json?format=gen9championsvgc2026regmc> (the replay id is not recorded here, for the players' privacy) | Pokémon Showdown replay service; response of 2026-10-19 | Field lists, no `inputlog` in the replay JSON, line types in the log, Cloudflare front, no rate-limit headers |
| S7 | Smogon usage statistics index | <https://www.smogon.com/stats/>, <https://www.smogon.com/stats/2026-09/> | Smogon; directory dated 2026-10-01 | Files per format and cutoff, M-C files, battle count, no licence on the listing |
| S8 | Smogon forums terms and rules | <https://www.smogon.com/forums/help/terms/> | Smogon (XenForo terms page); undated | Users keep copyright of posts and license them to Smogon; no clause on statistics or automated access |
| S9 | `pkmn/smogon` README and LICENSE | <https://github.com/pkmn/smogon/tree/d3cd03badedbebe962cf4688ae51eec28d8dd2fa> | pkmn contributors; commit `d3cd03b` | MIT code; third-party statement that aggregated stats are public domain and sets/analyses are Smogon's copyright (secondary for D5) |
| S10 | npm registry metadata (`npm view`, `npm pack`) | <https://www.npmjs.com/package/@smogon/calc>, `@pkmn/data`, `@pkmn/dex`, `@pkmn/sim`, `@pkmn/mods`, `@pkmn/smogon`, `@smogon/sets`, `pokemon-showdown` | npm registry; queried 2026-10-19 | Versions and licences: `@smogon/calc` 0.12.0 MIT; `@pkmn/*` 0.10.11 MIT; `@pkmn/smogon` 0.6.0 MIT; `@smogon/sets` 2.0.0 `UNLICENSED`; `pokemon-showdown` 0.11.11 MIT; package contents (`champions` in `@smogon/calc` and `@pkmn/mods`) |
| S11 | `smogon/damage-calc` | <https://github.com/smogon/damage-calc/tree/c1eb09be483f0bbf65132bf3703ee677195132f3> | Smogon; commit `c1eb09b`, 2026-10-17 | MIT `LICENSE`; `calc/src/mechanics/champions.ts`; front-end set file `src/js/data/sets/champions.js` |
| S12 | `pkmn/ps` LICENSE | <https://github.com/pkmn/ps/blob/4fec8877c83d102528929100b9c45a3a1cc160d3/LICENSE> | pkmn contributors; commit `4fec887` | MIT for `@pkmn/data`, `@pkmn/dex`, `@pkmn/sim`, `@pkmn/mods` |
| S13 | PokéAPI licences | <https://github.com/PokeAPI/pokeapi/blob/6fd69dfe90465ba96534f6816fc817737d1f97f6/LICENSE.md>, <https://github.com/PokeAPI/api-data/blob/8ed59f87f79ed12dab7240257358eca1be54e707/LICENSE.txt> | PokéAPI contributors; commits `6fd69df`, `8ed59f8` | BSD-3-Clause for code and data repositories; trademark statement |
| S14 | PokéAPI documentation, Fair Use Policy | <https://pokeapi.co/docs/v2> | PokéAPI; undated | Cache locally, IP bans, no rate limit since static hosting |
| S15 | About PokéAPI | <https://pokeapi.co/about> | PokéAPI; undated | Data sources (Veekun base; gen 8 and gen 9 scraped from other web resources, not named); trademark statement |
| S16 | PokéAPI version groups | <https://pokeapi.co/api/v2/version-group/?limit=100> | PokéAPI; response of 2026-10-19 | 32 version groups including `champions` |
| S17 | `SebNotFound/champions-calc` | <https://github.com/SebNotFound/champions-calc/tree/c900836e6a382e8b170dd72b5cf7bb82ddf7b016> | repository owner; commit `c900836`, pushed 2026-10-11 | Name "Champions Calc"; AGPL-3.0-or-later; built on `@smogon/calc` and `@pkmn` |
| S18 | Pokémon Champions Battle Data, API rules | <https://championsbattledata.com/api-rules/> | site operator (unnamed); undated | Allowed use, attribution wording, caching, no mirror or dump, no fixed rate limit |
| S19 | Pokémon Champions Battle Data, License | <https://championsbattledata.com/license.html> | same; undated | Personal, educational and competitive analysis use; no rights to Pokémon game data; credit and link |
| S20 | Pokémon Champions Battle Data, home page and API guide | <https://championsbattledata.com/>, <https://championsbattledata.com/api_guide> | same; undated | Regulations covered, daily rebuild from ranked snapshots, endpoints and schema fields |
| S21 | VGC-Bench battle logs dataset card; `vgc-bench` repository | <https://huggingface.co/datasets/cameronangliss/vgc-battle-logs>, <https://github.com/cameronangliss/vgc-bench> | Hugging Face dataset revision `74f2605` (2026-06-21); repository MIT | `license: mit`; logs scraped from the replay database; Reg M-A/M-B; filtered for OTS |
| S22 | Metamon raw replay dataset card | <https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays> | revision `792a86b` (2026-05-21) | No licence field; pseudonymised usernames; singles formats |
| S23 | Metamon parsed replay dataset card | <https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays> | revision `f80c25b` (2026-05-21) | `license: cc-by-nc-4.0`; reconstructed trajectories |
| S24 | PokéChamp dataset card | <https://huggingface.co/datasets/milkkarten/pokechamp> | revision `b5820ff` (2025-03-28) | No licence field; size; mostly singles |
| S25 | Pokémon Showdown Replays dataset card | <https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays> | revision `15a0a45` (2026-10-05) | No licence field; pulled through the Showdown API; Champions formats included |
| S26 | Pokémon terms of use; Regulation Set M-C news page | <https://www.pokemon.com/us/terms-of-use>, <https://champions-news.pokemon-home.com/en/page/816.html> | The Pokémon Company International, last update 2023-06-15; news page updated 2026-09-09 | Personal, noncommercial use; no bulk download to a database; no reverse engineering; news page shows no own terms link |
| S27 | Areneon repository | `docs/adr/ADR-001-open-source-licence.md`, `docs/adr/ADR-003-pinned-showdown-reference-simulator.md`, `docs/adr/ADR-008-cross-language-data-contracts.md` §5, `NOTICE` (repo root), `README.md`, `fixtures/LICENSE`, `fixtures/teams/<format>/manifest.json`, `packages/schema/contracts/v1/episode.schema.json`, `packages/showdown-bridge/src/episode.ts`, `rules/regulations/M-C.json`, `docs/research/regmc-format-matrix.md` | Areneon, `main` at `047caa6` | Project licences, data classes and rights values, storage rule, AGPL exclusion, sources the rules records cite |

Notes on sources:

- [S6] is a two-request probe made for this note; no replay content was kept.
- Web search results (Hugging Face dataset search, GitHub repository search, general search for the Battle Data API and Champions Calc) were used only to find the primary pages above. No row relies on them.
