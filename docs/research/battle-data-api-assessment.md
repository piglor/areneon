# Champions Battle Data API assessment

Research for Redmine #679 (Notion Wave 1 acceptance gate). It answers the inventory's open question Q5 on source D11 ([data-sources-rights.md](data-sources-rights.md)) and feeds the Wave 2 go/no-go (#683) and row 4 of ADR-009 (Training data sources, Accepted).

**Scope.** The Pokémon Champions Battle Data API (`championsbattledata.com`), assessed as a possible data source for the offline training pipeline and as a cross-check of the rules records and the pinned engine data. The note records the API's endpoints, schema, availability and terms, classifies every field, compares the data with what the repository already has, and gives a verdict under ADR-009's admission template (§7 of that ADR).

**Not legal advice.** A verdict here is the project's engineering decision, made under the terms as quoted and on the date accessed. It is not a statement of what the law allows.

**How to read.**

- Every fact carries a source label `[S#]` (see [Sources](#sources)). Every request to the site was made on **2026-10-19** (UTC); [§2](#2-requests-made-for-this-note) lists each one.
- Verdicts use the inventory's words: **use**, **use with conditions**, **do not use**, **open question**.
- Field classes: **base-stat** (per-form numbers fixed by the game data), **usage** (aggregate statistics of what players chose), **legal-roster** (which species, forms, abilities and moves exist or are allowed), **other** (with a reason).
- Facts are what a source says, what a response contained, or what the pinned code does. Conclusions drawn from them are marked **Inference** and collected in [§12](#12-inference).
- **Attribution.** §6 reports counts computed from API responses. Battle data provided by Pokémon Champions Battle Data (<https://championsbattledata.com/>, API guide <https://championsbattledata.com/api_guide>). No response data is reproduced beyond the field names and the few example values needed to describe the schema.

## Summary

| Question | Answer |
|---|---|
| What is it? | An unofficial, fan-made, unauthenticated JSON API plus static CSV files of in-game Champions usage statistics (Doubles and Singles), species metadata and learnable moves, rebuilt daily; operator not named [S3][S4][S6] |
| Which terms page governs (Q5)? | Neither page says. Both apply; on each point the stricter one is followed (§4.4). For a non-commercial cross-check that publishes no data, both pages allow the use |
| Does the game-data disclaimer rule it out? | Not for a cross-check that copies no data into any artifact. It does count against training use: nobody with rights in the underlying data has granted anything, and the site does not say how it obtains in-game data (§4.5) |
| Training data, pool U, synthetic team generator | **Do not use.** ADR-009 row 4 stands: excluded until the operator confirms training use in writing and a revision of ADR-009 admits the source |
| Rules and engine cross-check | **Use with conditions**, owner-run and offline only, never in CI, no stored or committed responses beyond the run (§8, §9). Until the owner accepts this (OQ1), no further calls are made |
| What the one-time cross-check found | The `hp`…`speed` fields are level-50 stats at 0 Stat Points, not base stats: all 447 form rows equal the champions stat formula at the pin. The species set equals the engine's (231 base species). Types match on every row; abilities differ on 3 rows, learnable moves on 10 of the 257 records compared (§6) |
| CI | Makes no call to the API; nothing from the API is committed (§10) |

## 1. Decision questions and what stays open

| # | Question | Answered in | State |
|---|---|---|---|
| DQ1 | What does the API offer: endpoints, parameters, schema, authentication, versioning, availability, rate limits? | §3 | Answered from the guide and five responses |
| DQ2 | What do its terms say, which page governs, and do the disclaimers rule it out? | §4 | Answered (inventory Q5) |
| DQ3 | What is each field, and which Areneon consumer could use it? | §5 | Answered |
| DQ4 | What would it verify in the repository, and what would it only duplicate? | §6 | Answered for one dated index |
| DQ5 | How fresh is it, and how would a change be detected? | §7 | Answered; conditional requests not supported |
| DQ6 | Use verdict under ADR-009's template, and its conditions | §8, §9 | Proposed; the owner decides (§11) |
| DQ7 | How would it be consumed without live calls in CI? | §10 | Designed; nothing is built by this note |

**Left open:** whether the operator would grant training use (OQ2); whether ML training falls under the licence page's "educational, and competitive analysis use" (no page mentions it); how the operator obtains in-game data; which in-game season ids belong to Regulation M-C (the API has no regulation field, §3.5); and whether the daily history the guide promises for finished seasons is complete (§3.5).

**Out of scope:** legality and mechanics (the pinned engine decides, ADR-003); the official eligible list (format matrix task U4, which needs an official source); building any client.

## 2. Requests made for this note

Every request was made by hand, one at a time, with the `User-Agent` `Areneon-research/0.1 (+https://github.com/piglor/areneon; terms and schema assessment, manual)`, no cookies and no login. `robots.txt` was read first and allows every path [S5]. Thirteen requests (R1-R13) went to the site and six to the Internet Archive. The responses were kept in a session scratch directory outside the repository to compute the §6 counts and were not committed; nothing from them is in this repository except field names and the example values quoted in §3.

| # | Time (UTC) | URL | Purpose | Status, size | Kept in this note |
|---|---|---|---|---|---|
| R1 | 16:18:15 | `https://championsbattledata.com/robots.txt` | Crawl rules | 200 | `User-agent: *`, `Allow: /`, sitemap URL |
| R2 | 16:18:24 | `https://championsbattledata.com/api-rules/` | Terms page 1 | 200, 6,516 B | Text quoted in §4; SHA-256 |
| R3 | 16:18:25 | `https://championsbattledata.com/license.html` | Terms page 2 (inventory URL) | 308 → `/license` | The redirect |
| R4 | 16:18:25 | `https://championsbattledata.com/api_guide` | API documentation | 200, 21,715 B | Endpoints and documented samples, §3 |
| R5 | 16:18:26 | `https://championsbattledata.com/sitemap.xml` | Look for page dates | 200, 502,033 B | `lastmod` values (§4.3) |
| R6 | 16:18:34 | `https://championsbattledata.com/license` | Terms page 2 | 200, 4,094 B | Text quoted in §4; SHA-256 |
| R7 | 16:18:57 | `https://championsbattledata.com/` | Coverage and origin statement | 200 | §3.1 |
| R8 | 16:19:07 | `https://championsbattledata.com/api` | Index: schema and the §6 comparison | 200, 5,704,203 B | Field names, counts, headers |
| R9 | 16:20:25 | `…/api/pokemon/garchomp?format=Doubles` | Pokémon record shape | 200, 1,178,478 B | Field names |
| R10 | 16:20:25 | `…/api/battle/Doubles/garchomp` | Battle rows shape | 200, 14,681 B | Field names, row counts |
| R11 | 16:20:25 | `…/api/battle/Doubles/garchomp?days=1` | Daily snapshot shape | 200, 14,742 B | Field names |
| R12 | 16:20:26 | `…/api/metadata/garchomp` | Metadata shape | 200, 1,122 B | Field names |
| R13 | 16:20:46 | `…/api` with `If-None-Match` set to R8's ETag | Does the API support conditional requests? | 200, full body | The answer (no 304) |
| W1-W3 | 16:21:00-16:22:15 | `https://web.archive.org/cdx/search/cdx?url=championsbattledata.com/<page>*` for `api-rules`, `license`, `api_guide` | Earlier versions of the terms | 200 | Capture dates (§4.3) |
| W4-W5 | 16:23:07-16:23:08 | Archive captures `20260701234116` and `20260901023206` of `/license` | Has the licence text changed? | 200 | §4.3 |
| W6 | 16:23:23 | Archive capture `20260718153817` of `/api_guide` | Did the guide link the API rules then? | 200 | §4.3 |

**Request spacing.** Several requests fell under one second apart: R3 and R4 (both 16:18:25), R4 and R5 (16:18:25-26), R9, R10 and R11 (all three at 16:20:25) and R12 right after (16:20:26). The site states no rate limit [S1], so no stated site rule was broken, but this note's own default, at most one request per second (ADR-009 §7), was. R13 was avoidable: it downloaded the full 5.7 MB index a second time only to show that the API returns no `304`. A later run follows §9 condition 3 and skips such a probe.

**Comparison scope.** The learnable-move count in §6 (10 differing records) covers the 257 records that have a learnset entry of their own in the engine; the 6 records whose engine forms inherit their learnset were not compared.

The guide itself links a bulk archive, `pokemon_champions_assets/battle_data.zip` ("Download Battle Data") [S3]; it was not fetched.

## 3. The API

### 3.1 What it is

- The site calls itself a "fan-made data explorer for competitive research and personal use" [S2] and the API "unofficial" [S1]. No page names an operator; the only contact route shown is a Discord invite [S1][S2][S3].
- The home page describes "Pokemon Champions ranked usage stats for Regulation M-A, Regulation M-B and Regulation M-C, rebuilt daily" from in-game ranked snapshots [S4]. No page says how those snapshots are obtained.
- The site also offers a team builder, a damage calculator, a solver and paid "Pro" plans [S4]. None of these is assessed here.
- The guide: "Use Showdown internal names to fetch current battle data, dated daily snapshots, metadata, and direct CSV assets" [S3].

### 3.2 Endpoints

All are `GET`, return JSON and take a Pokémon name that is matched case-insensitively and may be a Showdown ID (`raichualola`), a human-readable name, a site slug or a `saved_name` [S3].

| Path | Parameters | Returns | Observed size |
|---|---|---|---|
| `/api` (aliases `/api/index`, `/data/pokemon-index.json` [S3]; only `/api` was fetched) | none | The index: every Pokémon with Showdown ids, CSV paths, learnable moves, form metadata and a usage summary | 5,704,203 B [S6] |
| `/api/pokemon/:name` | `format` (`Doubles` or `Singles`), `season` (e.g. `M4`), `days` | One indexed Pokémon; with `format`, its current CSV path and summary; with `days`, newest-first daily CSVs and summaries | 1,178,478 B for `garchomp?format=Doubles` [S6] |
| `/api/battle/:format/:name` | `days` ("Integer from 1 to 31"), `season` (optional; else "the newest daily snapshots across seasons") | Parsed rows of the current CSV, or with `days` the dated snapshots newest first | 14,681 B [S6] |
| `/api/metadata/:name` | base name (e.g. `tauros`) | One row per form from the metadata file | 1,122 B [S6] |
| Static files under `/pokemon_champions_assets/` | none | `battle_data/<Doubles\|Singles>/<Showdown name>.csv` (current), `battle_data/<season>/<DD_MM_YYYY>/<format>/<name>.csv` (daily), `metadata/<base name>.csv`, sprites `pokemon/<saved_name>.png`, type icons `types/<Type>.png` | not fetched [S3] |

The guide says dated CSVs are hosted only for "the season currently being collected"; for a finished season "every one of their daily rows stays available through the JSON API, which returns exactly the same values", flagged `"archived": true` [S3].

### 3.3 Response shapes

From the documented samples [S3], checked against R8-R12 [S6]. Only names and types are given; the full classification is §5.

- **Index (`/api`):** `generatedAt`, `dataVersion`, `assetRoot`, `battleDataFolders` (`M7` … `M1`), `dailyDataFolders` (`<season>/<DD_MM_YYYY>`), `seasons` (`Current`, `M7` … `M1`), `defaultSeason` (`Current`), `topicPages[]` and `pokemonPages[]` (site navigation), and `pokemon[]` with `name`, `battleName`, `slug`, `showdownId`, `showdownName`, `metadataCsv`, `battleDataCsvs[]` (`season`, `format`, `path`, `date`, `daily`), `learnableMoveNames[]` and `summary` (`dex`, `sprite`, `types`, `primary`, `forms[]`, `baseStats`, `baseStatTotal`, `battleSummary`). Each `forms[]` row has `pokemon_name`, `title`, `dex_number`, `base_dex_url`, `image_path`, `form_name`, `saved_name`, `showdown_name`, `slug`, `legacy_slug` (on 164 of 447 rows), `form_kind`, `types`, `types_raw`, `abilities` (pipe-separated), `hidden_ability`, `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`, `base_stat_total`. `battleSummary.<season>.<format>.top.<category>` holds the rank-1 row of each category [S6].
- **Battle rows (`/api/battle/:format/:name`):** `pokemon`, `showdownId`, `format`, `season`, `date` (`null` for `Current`), `source`, `columns`, `rows[]`. A row has `pokemon`, `column_position`, `category` (`move`, `held_item`, `teammate`, `stat_alignment`, `stat_points`, `ability`), `rank`, `name`, `percentage` (a string such as `"59.8%"`), `percentage_value` (a number, `null` for `teammate`), `stat_up`, `stat_down` (for `stat_alignment`), and `hp_points`, `attack_points`, `defense_points`, `sp_atk_points`, `sp_def_points`, `speed_points` (integers for `stat_points`, empty strings elsewhere). R10 held ranks 1-10 for five categories and 2 ability rows [S6].
- **Daily (`?days=`):** `pokemon`, `showdownId`, `format`, `requestedDays`, `season`, `daily[]` with `season`, `date`, `source`, `columns`, `rows[]` [S3][S6].
- **Pokémon record (`/api/pokemon/:name`):** the index entry plus `requestedFormat`, `requestedSeason`, `battleDataCsv`, `battleSummary` (`top`, `values`, `rows`) and `summary.dailyBattleSummary` [S6].
- **Metadata (`/api/metadata/:name`):** `pokemon`, `source`, `columns`, `rows[]` with `title`, `base_name`, `saved_name`, `types`, `abilities`, `weight` (a string in pounds), `image_path`, `form`, `hp`, `atk`, `def`, `spa`, `spd`, `spe`, `total` [S6].

**Documented versus observed.** The guide's metadata sample has no `weight` column; R12 has one [S3][S6]. `summary.dex` and every `dex_number` were `null` in R8 [S6]. No response carries a battle count, a sample size, a rating cut-off or a regulation id [S6].

### 3.4 Authentication, versioning, availability, rate limits

| Item | Value | Source |
|---|---|---|
| Authentication | None: no key, token, cookie or login | [S3][S6] |
| Cross-origin | `access-control-allow-origin: *`, methods `GET, OPTIONS` | [S6] |
| Front end | Cloudflare (`server: cloudflare`) | [S6] |
| Versioning | None in the path or headers. The index carries `generatedAt` (`2026-10-19T07:41:40.387Z` in R8) and `dataVersion` (`20261019074140387`). "Data, endpoints, fields, and access rules may change at any time." | [S1][S6] |
| HTTP caching | `cache-control: public, max-age=120, s-maxage=1800` on the index, `max-age=120, s-maxage=900` on the other endpoints; an `ETag` on the index; no `Last-Modified` | [S6] |
| Conditional requests | Not honoured: R13 sent R8's ETag in `If-None-Match` and got `200` with the full 5.7 MB body | [S6] |
| Rate limit | "There is currently no fixed public rate limit." Access may be limited, blocked or revoked "if usage causes service problems, security issues, excessive load, or violation of these rules." No rate-limit or `Retry-After` header was seen | [S1][S6] |
| Availability | "provided “as is.”" No guarantee that data is "complete, correct, current, or available"; no status page or SLA was found | [S1] |
| Crawl rules | `robots.txt`: `User-agent: *`, `Allow: /` | [S5] |

### 3.5 Seasons, formats and regulations

- The API's dimensions are **format** (`Doubles`, `Singles`) and **season** (`Current`, `M1` … `M7`); there is no regulation field [S3][S6].
- In R8, `dailyDataFolders` held 44 folders. The date ranges per season were: `M1` 2026-05-13, `M2` 2026-06-17, `M3` 2026-07-08, `M4` 2026-08-05, `M5` 2026-09-10 (one folder each); `M6` 2026-09-11 to 2026-10-06 (26 folders); `M7` 2026-10-07 to 2026-10-19 (13 folders) [S6].
- The guide's examples name folders (`M4/16_07_2026`, `M4/23_07_2026`, `M5/08_09_2026`) that are not in R8's `dailyDataFolders` [S3][S6]. Whether `?season=M4&days=31` returns them was not tested.
- Regulation M-C runs from 2026-09-09 02:00 UTC to 2026-12-02 01:59 UTC (format matrix AV1-AV2 [S11]). Which of the API's seasons fall inside it is not stated by the API.

## 4. Terms

### 4.1 The two pages

The inventory found two pages and noted that they disagree and are undated [S10, D11]. Both were read again in full (R2, R6).

| Point | API Usage Rules, `/api-rules/` [S1] | License, `/license` (canonical link `license.html`) [S2] |
|---|---|---|
| Heading | "API Terms" / "API Usage Rules" | "Site Terms" / "License" |
| What it covers | "the Pokémon Champions Battle Data API" | "The website, generated pages, API responses, and data presentation" |
| Allowed use | "You may use the Pokémon Champions Battle Data API for personal, community, nonprofit, and commercial projects." | "provided as-is for personal, educational, and competitive analysis use" |
| Commercial use | "Commercial Use Allowed … including paid apps, websites, tools, dashboards, videos, articles, and services" | Not mentioned; not in the list of allowed uses |
| Attribution | Required "If you display, publish, or use data from this API in a public project". "A simple attribution is enough: “Battle data provided by Pokémon Champions Battle Data”". "Please include a link to our website or API documentation when possible." | "When sharing or using data from this site, please credit Pokemon Champions Battle Data and link to https://championsbattledata.com/." |
| Caching | "Reasonable caching is allowed to improve performance and reduce repeated requests." "Cached data should not be used to create a permanent mirror, archive, or replacement for this API." | Not mentioned |
| Redistribution | Forbids reselling, sublicensing or redistributing the API or its data as a standalone product; forbids a competing public API, a database dump, a bulk download service or a paid data feed built on the data without written permission; forbids selling raw responses or presenting the data as one's own original source | Not mentioned |
| Third-party rights | Using the API gives no rights in the trademarks, logos, images, characters or other intellectual property of Pokémon, Nintendo, Creatures Inc., GAME FREAK or The Pokémon Company | Pokémon names, images, game data, trademarks and related intellectual property belong to their owners; the project grants no rights in them |
| Endorsement | A project must not state or imply that it is official, sponsored, endorsed or approved by those companies | Do not imply endorsement by the site, Pokémon, Nintendo, Game Freak, Creatures Inc. or Pokémon Champions |
| Warranty | "provided “as is.”"; no guarantee of complete, correct, current or available data | "provided as-is" |
| Changes | "Data, endpoints, fields, and access rules may change at any time." | Not mentioned |
| Access limits | No fixed rate limit; access may be limited, blocked or revoked | Not mentioned |
| Footer | Not affiliated with "Pokémon, Nintendo, GAME FREAK, Creatures Inc., or The Pokémon Company" | Not affiliated with "Pokemon, Nintendo, Game Freak, or Creatures Inc." |
| Linked from | The API guide ("API Rules") [S3] | The home page ("License") [S4]; the home page also links "API rules" [S4] |

### 4.2 Text-level differences

1. **Allowed use.** The rules list "personal, community, nonprofit, and commercial"; the licence lists "personal, educational, and competitive analysis". Only "personal" is in both.
2. **Commercial use** is allowed by the rules and absent from the licence.
3. **Scope overlaps.** The licence names "API responses" explicitly, so both pages claim to cover the API.
4. **Attribution wording.** The rules give an exact sentence and ask for a link to "our website or API documentation"; the licence asks for credit and a link to `https://championsbattledata.com/`. The rules make attribution a condition ("must") for public projects; the licence says "please".
5. **Only the rules** address caching, mirroring, dumps, bulk download, resale, rate limits and changes.
6. **Third-party rights.** The licence names "game data" among the materials it grants no rights to; the rules name "intellectual property" without naming game data.
7. **The endorsement list** of the licence adds "this site" and "Pokemon Champions"; the rules add The Pokémon Company.
8. **Spelling.** The rules write "Pokémon"; the licence writes "Pokemon".

### 4.3 Dates

- Neither page shows a date, version or "last updated" line [S1][S2].
- The sitemap gives `lastmod` 2026-10-19 for all 2,708 of its URLs, including both pages [S5]. **Inference:** that is the sitemap's generation date, not an edit date.
- The Internet Archive holds captures of `/license` from 2026-07-01 and 2026-09-01; their licence text is word for word the text read on 2026-10-19 [S7]. It holds no capture of `/api-rules/` [S7]. Its 2026-07-18 capture of the API guide already links `/api-rules/` [S7].
- So the licence text has been unchanged since at least 2026-07-01, and an API rules page existed by 2026-07-18; its text at that time is not known.

### 4.4 Which page governs, and why

Neither page says which one prevails, and neither refers to the other [S1][S2].

**Decision (this note).** Areneon treats both pages as binding at once and, on each point, follows the stricter one:

- allowed use: only what both lists allow. A non-commercial, open-source research project's own internal cross-check arguably fits the rules' "nonprofit" and the licence's "educational, and competitive analysis use". Commercial use is not relied on;
- attribution: the rules' exact sentence **and** the licence's link to `https://championsbattledata.com/`, in every public place that shows a result derived from the data;
- caching, mirroring, dumps, bulk download, resale: the rules' limits (the licence is silent);
- third-party rights: the licence's wider wording, which names game data.

**Why.** The rules page is the more specific text for the API and the guide links to it [S3]. The licence page names API responses explicitly. Neither page is dated, so "the newer one wins" cannot be applied. Following both costs Areneon nothing for the use proposed in §8, since that use publishes no data and makes no money. So the conflict needs no resolution unless Areneon ever wants a use that only one page allows, such as commercial use (none planned) or training (§4.5, OQ2).

### 4.5 The disclaimers

- **Trademarks and endorsement.** Areneon's `NOTICE` and `README.md` already carry the trademark and non-affiliation notice ADR-001 requires [S13]. Any public result that uses the data must also not imply endorsement by the site (licence) or by the rights holders (both pages).
- **Game data.** The licence says the project "does not grant rights" in Pokémon "game data" [S2]. Inventory D12 records that the Pokémon terms of use forbid "(v) Download quantities of content to a database for any reason" and reverse engineering [S10, D12]; they were not fetched again for this note. The site does not say how it collects in-game data [S3][S4].
  - **For a cross-check** (§8) the disclaimer does not rule the API out. Areneon would copy no game data from it into any artifact; the engine stays the only source of species, stats and learnsets (ADR-003), and the comparison only reports where the two agree.
  - **For training data** it counts against admission. The usage figures describe in-game choices of real players, collected in an undisclosed way. The site grants no rights in that material, and nobody who holds them has granted anything. ADR-009 row 4 already requires the operator's written confirmation; this note adds that the confirmation should also say where the data comes from (OQ2).

## 5. Field classification

Consumers: **R** rules records (`rules/`), **F** format matrix ([regmc-format-matrix.md](regmc-format-matrix.md)), **SF** stat-formula evidence (gap `stat-points`), **U** usage-statistics opponents (pool U, [evaluation-protocol.md](evaluation-protocol.md) §3), **G** synthetic team generator (ADR-009 1f). "—" means no Areneon consumer. **Inference** for the consumer column: the API is not an official source, so it can never be a source URL in `rules/regulations/` (which cites official publications) nor gap evidence (only `engine` and `fixture` kinds exist, and `verified` needs an in-game fixture [S13]).

| Field (where) | Class | What it is | Consumer |
|---|---|---|---|
| `forms[].hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`, `base_stat_total` (index); `hp`, `atk`, `def`, `spa`, `spd`, `spe`, `total` (metadata); `summary.baseStats`, `baseStatTotal` | **base-stat** | Labelled base stats, but each value equals the level-50 stat at 0 Stat Points and a neutral nature (§6) | SF (cross-check of the formula at 0 SP only); F (none: base stats are engine data) |
| `forms[].types`, `types_raw`; metadata `types`; `summary.types` | other: species data | Typing per form | Engine cross-check only |
| metadata `weight` | other: species data | Weight in pounds | — (engine has `weightkg`) |
| `pokemon[].showdownId`, `showdownName`, `forms[].showdown_name`, `form_kind`, `form_name` | **legal-roster** | Which species and forms the site indexes, in Showdown names | F (T8 / U4 lead only), engine cross-check |
| `forms[].abilities`, `hidden_ability`; metadata `abilities` | **legal-roster** | Abilities per form (pipe-separated; `hidden_ability` was empty on the rows read) | Engine cross-check |
| `pokemon[].learnableMoveNames` | **legal-roster** | Moves the site lists as learnable | Engine cross-check (gap `roster-and-learnsets`, as a lead) |
| `rows[]` with `category` `move`, `held_item`, `ability`, `teammate` and their `rank`, `name`, `percentage`, `percentage_value`, `column_position`, `position` | **usage** | Top-10 share of each choice per species | U, G: excluded (ADR-009 rows 3a, 4) |
| `rows[]` `stat_alignment` with `stat_up`, `stat_down` | **usage** | Top natures with their raised and lowered stat | U, G: excluded |
| `rows[]` `stat_points` with `hp_points` … `speed_points` | **usage** | Top Stat Point spreads | U, G: excluded; SF: the spreads' upper bounds are a weak consistency check (§6) |
| `battleSummary`, `dailyBattleSummary`, `top`, `values`, `statChanges` | **usage** | Pre-computed summaries of the rows | U, G: excluded |
| `generatedAt`, `dataVersion` | other: freshness | Build time and version string of the index | Change detection (§7) |
| `seasons`, `defaultSeason`, `battleDataFolders`, `dailyDataFolders`, `season`, `date`, `daily[]`, `requestedDays`, `requestedFormat`, `requestedSeason`, `format` | other: data versioning | In-game season and daily snapshot keys | — (no regulation mapping, §3.5) |
| `assetRoot`, `metadataCsv`, `battleDataCsvs`, `battleDataCsv`, `source`, `path`, `columns` | other: file locations | Paths of the static CSVs | — |
| `sprite`, `image_path`, `base_dex_url` | other: images and links | Sprite paths; third-party artwork | — (never used: no rights, ADR-001) |
| `name`, `battleName`, `slug`, `legacy_slug`, `title`, `saved_name`, `pokemon_name`, `base_name`, `form` | other: site naming | Display names and URL slugs | — (Showdown ids suffice) |
| `topicPages`, `pokemonPages` | other: site navigation | Search-page links | — |
| `summary.dex`, `dex_number` | other: empty | `null` on every row read | — |

## 6. Comparison with what the repository has

The repository already has the rules snapshot (`rules/`), the engine data of the pinned Showdown (`data/mods/champions/` at `7244fe8`), and the format matrix [S8][S11][S13]. The comparison below ran once, offline, on R8 (index, `dataVersion` `20261019074140387`) against the built engine (`Dex.mod('champions')`). It is evidence of agreement on one date, not verification: the engine stays the authority, and a gap record moves to `verified` only with an in-game fixture [S13].

| Check | Result | What it verifies, or only duplicates |
|---|---|---|
| Every form's `showdown_name` resolves in the champions dex | 447 of 447 form rows (347 distinct species ids) | Naming only |
| Species set | The API's 231 base species equal the engine's 231 base species that the champions formats data does not mark nonstandard (excluding the `Mythical` and `Restricted Legendary` tags). At form level the engine has 43 more: 19 Vivillon patterns, 7 Alcremie variants, Squawkabilly-Blue/-Yellow/-White, Polteageist-Antique, Sinistcha-Masterpiece, Maushold (the API lists Maushold-Four) and 11 battle-only formes. All six Megas M-C added (format matrix T9) are present | Corroborates the engine pool against a second, unofficial list. It is not the official eligible list (U4 still needs that) and the API does not say its list is a legality list |
| Stats | All 447 rows: `hp` = engine base HP + 75 and each other stat = engine base + 20 | Corroborates the champions `statModify` at 0 Stat Points and neutral nature (`data/mods/champions/scripts.ts:24-27`; gap `stat-points`). Base stats themselves only duplicate the engine |
| Types | 447 of 447 rows equal the engine's types | Duplicates the engine |
| Abilities | 444 of 447 rows equal the engine's ability set. Greninja: the engine also lists Battle Bond (slot `S`), the API does not. Squawkabilly (2 rows): the API lists Guts and Sheer Force on one form; the engine gives Guts to Squawkabilly and -Blue, Sheer Force to -Yellow and -White | Leads only; both look like differences of presentation, not of legality (unverified) |
| Learnable moves | 247 of 263 records equal the record's own learnset in `data/mods/champions/learnsets.ts`; 6 records have no learnset entry of their own in that file (forms that inherit) and were not compared. 10 differ: Beartic, Liepard, Pinsir, Samurott and Sharpedo have Slash in the engine but not in the API; the five Rotom appliance forms list Thunder Shock in the API, which neither their own engine entries nor the base Rotom entry list | Slash on five species and Thunder Shock on the Rotom forms are leads for gap `roster-and-learnsets`; which side is right is unverified |
| Usage-row names | Every top-ranked move, item, ability, teammate and nature (524 rows each) resolves in the champions dex and none is nonstandard | No consumer; usage stays excluded |
| Stat Point spreads | Of 524 top spreads, none has a stat above 32 or a total above 66; 520 total exactly 66, 511 have a stat of exactly 32 | Consistent with the engine caps (gap `stat-points`, format matrix T12, U5); it does not prove them, since it only shows the spreads players chose |
| Regulation, season, timer, team-sheet rules | Not in the API | Nothing to compare: the rules records' format and regulation facts come from official sources and the engine only |

**What would only be duplicated.** Base stats, types, abilities, learnsets and species names are already in the pinned engine under MIT (inventory D1 [S10]). Storing them from the API would add a second copy under worse terms and no new authority.

**What it could add.** An independent, daily-refreshed signal that the engine's champions data and the game agree on roster, stats at 0 SP and learnsets, as leads for the official checks of format matrix U4 and U5. It cannot close either task.

## 7. Freshness and change detection

- **Data.** The site says it is "rebuilt daily" [S4]; R8 was generated at 07:41:40 UTC on the day of access and its newest folder was the same day [S6]. The HTTP caches allow 120 s in clients and 900-1,800 s at the edge [S6].
- **Change signal.** `dataVersion` and `generatedAt` in the index change with each build [S6]. Conditional requests do not save the download (R13), so detecting a change costs a full 5.7 MB index read.
- **Shape changes.** No schema version exists and "fields … may change at any time" [S1]. A consumer must validate each response against a closed schema of the fields it reads and stop on any difference.
- **Terms changes.** Neither page is dated (§4.3). A consumer records the SHA-256 of each terms page as read (R2: `931c204b61188a90ae17b4a0107d439610bb917402add0c84df509e58965ca13`, R6: `0390b99cdb5cf85085dae794ed4acdd09f6427f8f8cf73b34b7500c919b94e4f`) and stops when either changes, until the owner has read the new text. The raw bytes include the site's navigation, so a navigation change also stops the run. That is deliberate (fail closed).

## 8. Verdict under ADR-009's admission template

ADR-009 (Accepted) admits only self-play and synthetic data for now, and its row 4 excludes this API ("Not called") until "#679 settles which terms page governs, and the operator confirms training use in writing" [S9]. §4.4 settles the first condition; the second is not met. Any admission, including the narrower cross-check use proposed here, is a delta revision of ADR-009 (its answer to Q12), not a new ADR [S9].

| Use | Verdict | Reasoning |
|---|---|---|
| Training data (any field as a policy input or label) | **Do not use** | ADR-009 decision 1 and row 4. No page mentions ML training; the licence page grants no rights in game data; the collection method is undisclosed (§4.5); the operator has not confirmed anything in writing |
| Pool U (usage-statistics opponents) | **Do not use** | As above; ADR-009 row 3a/4: pool U does not exist until a statistics source is admitted. Also no sample sizes or rating cut-offs (§3.3) and no regulation id (§3.5), so a dated M-C snapshot could not be defined from the API alone |
| Synthetic team generator | **Do not use** | ADR-009 Q4: a team sampled from third-party statistics is not automatically `synthetic` / `cc0-synthetic`; none is generated before a grant |
| Rules records (`rules/`) | **Do not use** | Not an official source, so never a regulation source URL; gap evidence admits only engine and fixture kinds [S13] |
| Cross-check of the engine data and format matrix (§6) | **Use with conditions** (§9), proposed; the owner decides (OQ1) | Appears allowed under both pages (§4.4); takes no data into any artifact; the API's own terms set no rate limit and ask only for attribution and no mirror. Its results are leads for U4, U5 and gap `roster-and-learnsets`, never evidence that moves a gap to `verified`. ADR-009 says "Not called", so until the owner accepts this row no further calls are made; the single comparison of §6 is this assessment's own evidence |

**Inference.** The cross-check row is consistent with ADR-009: that ADR decides training data sources, and its inventory already gives PokéAPI (D9) the same "cross-check only" status [S10]. It is still a change to row 4's "Not called", which is why it goes to the owner.

## 9. Conditions for any use

These apply to the cross-check. A later admission as a data source would add ADR-009 §7 in full (class `third-party` / `third-party-terms`, private storage, retention, deletion).

1. **Attribution.** Any public text, report or result that uses the data says: "Battle data provided by Pokémon Champions Battle Data" with a link to <https://championsbattledata.com/> (both pages' wishes, §4.4). It does not imply endorsement by the site or by the rights holders.
2. **Who and when.** Run by hand by the owner or a maintainer, outside CI, at most once per engine pin bump or regulation change, and only after reading both terms pages and comparing their hashes (§7).
3. **Requests.** One sequential client, at most one request per second (ADR-009 §7 default), the descriptive `User-Agent` of §2 and a `From` contact, `robots.txt` read first, no cookies, no login, stop on any challenge or error burst. The index alone answers the §6 checks: one request per run.
4. **Cache.** The raw response lives only for the run and at most 7 days after it *(inference: "reasonable caching", never "a permanent mirror, archive, or replacement")*, in the gitignored `private-data/` or the run's scratch space, then is deleted.
5. **No redistribution.** No response, CSV, extract or derived table of API values is committed, published or placed under `fixtures/` (rules page; ADR-009 §7; `fixtures/` is CC0 and Areneon cannot dedicate third-party data). The bulk archive linked from the guide is never downloaded.
6. **What may be kept.** The comparison's outcome as prose and counts in a research note (as §6 does), naming species and moves, never a copy of the API's tables.
7. **Provenance.** If a raw response is ever stored beyond the run, it gets a `SourceSnapshot` record (ADR-009 §7: `source_id` `D11`, `url`, `retrieved_at`, `terms_url`, `terms_sha256`, `terms_retrieved_at`, `grant_ref`, `licence: null`, `content_sha256`, `item_count`, `collector_commit`, `retention_until`), kept privately beside the bytes.
8. **Authority.** The engine decides. A difference opens a lead for the official sources; it never edits engine data, rules records or gap statuses.

## 10. CI-safe consumption design

Nothing is built by this note. If the owner accepts the cross-check (OQ1), a later ticket would build it as follows.

- **No live calls in CI.** The cross-check is a script run by hand, never invoked by `make check`, `make ts`, `make python` or any workflow. CI tests only its parser and comparison, on inputs that make no network call.
- **Test inputs.** A small hand-written response under the test's own directory, with the API's field names (§3.3) and invented values (species and numbers chosen so the comparison has one match and one difference of each kind). It is synthetic, carries `"data_class": "synthetic"` if team-shaped, and copies nothing from a real response. Field names are the schema, not the data.
- **Guard.** A test fails if the hostname `championsbattledata.com` appears in any source file outside `docs/` and the hand-run script, so a stray call cannot reach CI.
- **Recording a snapshot.** The terms forbid a permanent mirror and ADR-009 §7 forbids committing third-party data, so **no response is ever checked in**. The run prints a report (counts and named differences, as in §6), which a maintainer may paste into a research note with the attribution line, the access time and `dataVersion`. If the raw bytes must outlive the run, condition 7 of §9 applies and they stay private.
- **Schema drift.** The parser validates the fields it reads against a closed JSON Schema; a missing or renamed field stops the run with an error rather than producing a partial comparison.

## 11. Open questions for the owner

| # | Question | Default until answered |
|---|---|---|
| OQ1 | Accept the cross-check row of §8 (owner-run, offline, conditions of §9), as a delta revision of ADR-009 per its Q12 (admission is a delta revision, not a new ADR), changing row 4's "Not called" for that use only? | No further calls. §6 stands as this assessment's one-time evidence |
| OQ2 | Contact the operator (the only route shown is the site's Discord) to ask (a) whether training a policy on the usage data, and publishing the trained weights, is allowed, (b) how the in-game data is collected, (c) which page governs? | No contact (ADR-009 decision 7 limits contact to the one Smogon letter). The API stays excluded as training data |
| OQ3 | Should the §6 leads (Slash on five species, Greninja's Battle Bond, Squawkabilly plumage abilities, the Rotom forms' Thunder Shock) be checked against an official source? | Listed here as leads only. Gap records unchanged; they are checked with the U4 work in the format matrix |
| OQ4 | May the API's season ids (`M6` from 2026-09-11, `M7` from 2026-10-07) inform format matrix AV5 (Ranked seasons inside M-C)? | No: not a primary source. AV5 stays unverified (task U1) |
| OQ5 | If the API is ever admitted as a data source, which fields may enter: usage only, or also the legal-roster fields? | Usage only, under ADR-009 §7; legal-roster and base-stat fields never, since the engine already holds them (§6) |

Inventory Q5 is answered by §4.4 and §4.5; Q4, Q10 and Q11 keep the defaults ADR-009 adopted [S9].

## 12. Inference

These follow from the facts above but are not stated by any source.

1. **I1.** The fields labelled base stats are computed stats: every one of 447 rows equals the champions stat formula at 0 Stat Points and neutral nature. A consumer that took them as base stats would add 75 HP and 20 to every other stat twice.
2. **I2.** The sitemap's uniform `lastmod` is a build date, so it dates neither terms page.
3. **I3.** Because the API has no regulation id and its seasons are in-game season ids, any M-C cut of its data would rest on Areneon's own mapping of seasons to regulation dates, which no source provides.
4. **I4.** Without battle counts or rating cut-offs, usage percentages cannot be weighted or given confidence intervals, which pool U's evaluation rules would need ([evaluation-protocol.md](evaluation-protocol.md) §6).
5. **I5.** The licence's "educational, and competitive analysis use" may or may not cover ML training; no page mentions training, so only the operator can answer (OQ2).
6. **I6.** The cache limit of 7 days in §9 is the project's reading of "reasonable caching", not a value the site sets.

## Sources

All site requests on 2026-10-19 (UTC; times in §2). Repository files at `origin/main` `4b638ce`.

| Label | Title | URL | Publisher / version / date | Accessed | Claim supported |
|---|---|---|---|---|---|
| S1 | API Usage Rules | <https://championsbattledata.com/api-rules/> | Pokémon Champions Battle Data (operator unnamed); undated; SHA-256 of bytes in §7 | 2026-10-19 16:18:24Z | Allowed use, commercial use, attribution sentence, caching, no mirror, no dump or bulk service, no resale, no rights in Pokémon IP, no endorsement, as is, changes, no fixed rate limit |
| S2 | License (Site Terms) | <https://championsbattledata.com/license> (`/license.html` redirects here with 308) | same; undated; SHA-256 in §7 | 2026-10-19 16:18:34Z | Personal, educational and competitive analysis use; no rights in game data; credit and link; no endorsement |
| S3 | API Guide | <https://championsbattledata.com/api_guide> | same; undated | 2026-10-19 16:18:25Z | Endpoints, parameters, documented samples, name matching, folder layout, archived seasons, CORS, link to API rules and to the bulk archive |
| S4 | Home page | <https://championsbattledata.com/> | same; undated | 2026-10-19 16:18:57Z | Regulations covered, daily rebuild from in-game ranked snapshots, other tools and paid plans, links to both terms pages |
| S5 | `robots.txt` and `sitemap.xml` | <https://championsbattledata.com/robots.txt>, <https://championsbattledata.com/sitemap.xml> | same; responses of 2026-10-19 | 2026-10-19 16:18:15Z, 16:18:26Z | All paths allowed; 2,708 URLs, all `lastmod` 2026-10-19 |
| S6 | API responses | `/api`, `/api/pokemon/garchomp?format=Doubles`, `/api/battle/Doubles/garchomp`, `/api/battle/Doubles/garchomp?days=1`, `/api/metadata/garchomp` on `championsbattledata.com` | same; index `dataVersion` `20261019074140387`, `generatedAt` 2026-10-19T07:41:40.387Z | 2026-10-19 16:19:07Z-16:20:46Z | Field names and types, sizes, headers, no conditional requests, seasons and folders, §6 counts |
| S7 | Internet Archive CDX index and captures | <https://web.archive.org/cdx/search/cdx?url=championsbattledata.com/license*> (and `api-rules*`, `api_guide*`); captures `20260701234116`, `20260901023206` of `/license`, `20260718153817` of `/api_guide` | Internet Archive | 2026-10-19 16:21:00Z-16:23:23Z | Licence text unchanged since 2026-07-01; no capture of the rules page; guide linked the rules page on 2026-07-18 |
| S8 | Pokémon Showdown, pinned submodule | <https://github.com/smogon/pokemon-showdown/tree/7244fe8e28fa8bc4d0809ae30d183c608b62bb2b> | Smogon / Pokémon Showdown contributors (MIT); commit `7244fe8`, 2026-10-16 | 2026-10-19 | `data/mods/champions/scripts.ts:10-40` (`statModify`: HP `base + SP + 75`, others `base + SP + 20`, then nature; this branch does not depend on the level), `data/mods/champions/learnsets.ts`, `data/mods/champions/formats-data.ts`; the built `Dex.mod('champions')` used in §6 |
| S9 | ADR-009 Training data sources | Redmine wiki page `ADR-009_Training_Data_Sources` | Areneon; r5 (wiki version 6), status Accepted, acceptance dated 2026-10-20 on the page | 2026-10-19 about 16:17 UTC, read through the Redmine API before the site requests of §2. The page already showed r5 as Accepted with the date 2026-10-20, later than the read by the UTC clock; this note was drafted against that text, and acceptance was recorded on the page after work on #679 had started | Decision table rows 3a and 4, §7 admission template and `SourceSnapshot` fields, §8 defaults (Q4, Q5, Q10, Q11, Q12: admission is a delta revision), §9 decisions 1 and 7 |
| S10 | Data-source and reuse-rights inventory | [data-sources-rights.md](data-sources-rights.md) | Areneon, `4b638ce` | 2026-10-19 | D1, D9, D11, D12 (Pokémon terms of use as quoted there, not refetched), Q4, Q5, Q10, Q11 |
| S11 | Reg M-C target format matrix | [regmc-format-matrix.md](regmc-format-matrix.md) | Areneon, `4b638ce` | 2026-10-19 | AV1-AV2, AV5, T8, T9, T12, tasks U1, U4, U5 |
| S12 | Held-out evaluation protocol | [evaluation-protocol.md](evaluation-protocol.md) | Areneon, `4b638ce` | 2026-10-19 | Pool U is conditional on ADR-009; built from one dated snapshot with its hash |
| S13 | Areneon repository | `rules/regulations/M-C.json`, `rules/gaps/stat-points.json`, `rules/gaps/roster-and-learnsets.json`, `packages/schema/rules/v1/gap.schema.json`, `packages/schema/rules/v1/regulation.schema.json`, `NOTICE`, `docs/adr/ADR-001-open-source-licence.md`, `docs/adr/ADR-003-pinned-showdown-reference-simulator.md`, `docs/adr/ADR-007-rules-store-database-and-editing.md` | Areneon, `4b638ce` | 2026-10-19 | Regulation sources are official URLs with dates; gap evidence kinds `engine` and `fixture`, `verified` needs a fixture; trademark notice; engine as the only authority; replay with no live service |
