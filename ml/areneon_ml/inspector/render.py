"""The inspector's pages (ADR-017 section 4a): index, distributions, teams and games.

Each page is self-contained and static; pages link to each other by relative links built from fixed
names and hex ids. Nothing here depends on the clock, so the same run gives the same bytes.
"""

# ruff: noqa: E501  (HTML, CSS and SVG template lines are kept whole so the markup reads as written)

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from areneon_ml.inspector import charts
from areneon_ml.inspector.model import RESULTS, Game, Generation, InspectModel, TeamUse
from areneon_ml.inspector.page import document, esc, link_id
from areneon_ml.inspector.pool import PoolSet, PoolTeam

CHUNK = 500
#: A rate is shown only from this many games; below it the page shows counts alone.
MIN_RATE_GAMES = 10
MAX_GAME_LINKS = 12
STAT_LABELS = (
    ("hp", "HP"),
    ("atk", "Atk"),
    ("def", "Def"),
    ("spa", "SpA"),
    ("spd", "SpD"),
    ("spe", "Spe"),
)

INDEX = "index.html"
DISTRIBUTIONS = "distributions.html"


def games_page(generation: int) -> str:
    return f"games-g{generation:06d}.html"


def drawn_page(generation: int) -> str:
    return f"drawn-g{generation:06d}.html"


def teams_page(chunk: int) -> str:
    return f"teams-{chunk + 1:04d}.html"


def team_anchor(team_hash: str) -> str:
    return "t-" + link_id(team_hash)


def _nav(model: InspectModel) -> str:
    return (
        f'<a href="{INDEX}">Overview</a><a href="{DISTRIBUTIONS}">Distributions</a>'
        f'<a href="{teams_page(0)}">Teams</a>'
        f'<span class="muted">Run {esc(model.run_key)}</span>'
    )


def _stat(label: str, value: object) -> str:
    return f'<div class="stat">{esc(label)}<b>{esc(value)}</b></div>'


def _table(head: Iterable[tuple[str, bool]], rows: Iterable[str]) -> str:
    cols = "".join(
        f"<th{' class="n"' if numeric else ''}>{esc(name)}</th>" for name, numeric in head
    )
    return f'<div class="scroll"><table><thead><tr>{cols}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _short(value: str, size: int = 8) -> str:
    return value[:size]


def _page_of(model: InspectModel) -> dict[str, int]:
    return {team.hash: index // CHUNK for index, team in enumerate(model.pool.teams)}


def _team_link(pages: dict[str, int], team_hash: str) -> str:
    return f'<a class="m" href="{teams_page(pages[team_hash])}#{team_anchor(team_hash)}">{_short(team_hash)}</a>'


def _footer(model: InspectModel) -> str:
    return (
        f"Run config hash <code>{esc(model.config_hash)}</code>. Every generation's files were checked against "
        "its manifest before they were read. Counts are descriptive."
    )


def draw_histogram(counts: Counter[str]) -> str:
    """How often each drawn team was drawn: times drawn -> number of teams."""
    per = Counter(counts.values())
    rows = [
        f'<tr><td class="n">{times}</td><td class="n">{teams}</td></tr>'
        for times, teams in sorted(per.items())
    ]
    return _table([("Times drawn", True), ("Teams", True)], rows)


def index_page(model: InspectModel) -> str:
    pool = model.pool
    distinct = len(model.cumulative_drawn)
    appearances = sum(model.cumulative_drawn.values())
    stats = "".join(
        [
            _stat("Pool before the ability filter", pool.records_before),
            _stat("Pool after the filter", len(pool.teams)),
            _stat("Generations with games", len(model.generations)),
            _stat("Distinct teams drawn", distinct),
            _stat("Team draws (one per seat)", appearances),
        ]
    )
    excluded = ", ".join(pool.excluded_abilities) or "none"
    rows = []
    cumulative: set[str] = set()
    for gen in model.generations:
        cumulative |= set(gen.drawn)
        played = sum(1 for g in gen.games if g.row.outcome == "played")
        rows.append(
            f'<tr><td class="n">{gen.number}</td><td class="m">{esc(_short(gen.pair_policy_id.removeprefix("policy:"), 10))}</td>'
            f'<td class="n">{len(gen.games)}</td><td class="n">{played}</td><td class="n">{gen.forfeits}</td>'
            f'<td class="n">{gen.episodes}</td><td class="n">{gen.examples}</td><td class="n">{len(gen.drawn)}</td>'
            f'<td class="n">{len(cumulative)}</td>'
            f'<td><a href="{games_page(gen.number)}">games</a> · <a href="{drawn_page(gen.number)}">drawn teams</a></td></tr>'
        )
    head = [
        ("Generation", True),
        ("Pair", False),
        ("Games", True),
        ("Played", True),
        ("Forfeits", True),
        ("Episodes", True),
        ("Examples", True),
        ("Teams drawn", True),
        ("Distinct so far", True),
        ("Pages", False),
    ]
    manifests = "".join(
        f'<tr><td class="n">{n}</td><td class="m">{esc(h)}</td></tr>'
        for n, h in sorted(model.manifest_hashes.items())
    )
    body = (
        f"<h1>Training data of run {esc(model.run_key)}</h1>"
        f'<p class="note">Which teams and which games this run trained on. Lineage {esc(model.lineage_id)}, '
        f"format {esc(model.format_id)}. Synthetic data only; the pool was rebuilt with the loop's own function "
        "and every file below was checked against it.</p>"
        f'<div class="grid">{stats}</div>'
        f'<p class="note">Source of the pool: {esc(pool.source)}. Dataset hash: '
        f"<code>{esc(pool.dataset_hash or 'none (interim fixture pool)')}</code>. Abilities left out: {esc(excluded)}.</p>"
        "<h2>Generations</h2>"
        f"{_table(head, rows)}"
        '<p class="note">Generation 0 is the seeded pair and plays no games. "Teams drawn" counts distinct teams a '
        "generation's plan draws.</p>"
        f'<p><a href="{DISTRIBUTIONS}">Distributions over the pool and the drawn teams</a> · '
        f'<a href="{teams_page(0)}">All {len(pool.teams)} pool teams, {CHUNK} per page</a></p>'
        f"<details><summary>Manifest hashes of the generations read</summary>{_table([('Generation', True), ('SHA-256 of manifest.json', False)], [manifests])}</details>"
    )
    return document(f"Training data of {model.run_key}", _nav(model), body, footer=_footer(model))


def distributions_page(model: InspectModel) -> str:
    pool_dist = charts.distribution((t, 1) for t in model.pool.teams)
    drawn_dist = charts.distribution(
        (team, model.cumulative_drawn[team.hash])
        for team in model.pool.teams
        if team.hash in model.cumulative_drawn
    )
    body = (
        "<h1>Distributions</h1>"
        '<p class="note">Share of teams that have each value, over the whole pool (each team once) and over all '
        "teams drawn in the run (each draw once, so a team drawn twice counts twice). A skew between the two shows "
        "in the gap between the bars.</p>"
        f"<h2>How often drawn teams were drawn</h2>{draw_histogram(model.cumulative_drawn)}"
        "<h2>Per generation</h2><p>"
        + " · ".join(
            f'<a href="{drawn_page(g.number)}">generation {g.number}</a>' for g in model.generations
        )
        + "</p><h2>Pool and all drawn teams</h2>"
        + charts.section(pool_dist, drawn_dist, "Drawn teams")
    )
    return document("Distributions", _nav(model), body, footer=_footer(model))


def drawn_generation_page(model: InspectModel, gen: Generation) -> str:
    counts: Counter[str] = Counter()
    for game in gen.games:
        counts.update(game.hashes.values())
    by_hash = {t.hash: t for t in model.pool.teams}
    pool_dist = charts.distribution((t, 1) for t in model.pool.teams)
    drawn_dist = charts.distribution((by_hash[h], n) for h, n in counts.items())
    stats = "".join(
        [
            _stat("Games", len(gen.games)),
            _stat("Team draws", sum(counts.values())),
            _stat("Distinct teams", len(counts)),
            _stat("Teams recorded so far", gen.team_set_size),
        ]
    )
    body = (
        f"<h1>Teams drawn in generation {gen.number}</h1>"
        f'<p><a href="{games_page(gen.number)}">Games of generation {gen.number}</a></p>'
        f'<div class="grid">{stats}</div>'
        f"<h2>How often each team was drawn</h2>{draw_histogram(counts)}"
        f"<h2>Pool and this generation's drawn teams</h2>{charts.section(pool_dist, drawn_dist, f'Generation {gen.number}')}"
    )
    return document(
        f"Drawn teams of generation {gen.number}", _nav(model), body, footer=_footer(model)
    )


def _opponent(game: Game) -> str:
    if game.plan.opponent == "self":
        return "self-play"
    baseline_seat = "p2" if game.plan.policy_seat == "p1" else "p1"
    return f"{game.plan.opponent} ({baseline_seat})"


def _game_row(pages: dict[str, int], game: Game) -> str:
    row = game.row
    winner = "tie" if row.outcome == "played" and row.winner is None else (row.winner or "none")
    loser = cause = ""
    title = ""
    if row.outcome == "forfeit":
        loser = row.forfeit.loser if row.forfeit else "not recorded"
        cause = row.forfeit.cause if row.forfeit else "not recorded"
        if row.forfeit is not None and row.forfeit.detail:
            title = f' title="{esc(row.forfeit.detail)}"'
    return (
        f'<tr id="g{game.plan.game}"><td class="n">{game.plan.game}</td><td class="m">{esc(row.seed.removeprefix("sodium,")[:12])}</td>'
        f"<td>{_team_link(pages, game.hashes['p1'])}</td><td>{_team_link(pages, game.hashes['p2'])}</td>"
        f'<td>{esc(_opponent(game))}</td><td>{esc(winner)}</td><td class="n">{row.turns}</td>'
        f"<td>{esc(row.outcome)}</td><td>{esc(loser)}</td><td{title}>{esc(cause)}</td></tr>"
    )


def games_generation_page(model: InspectModel, gen: Generation) -> str:
    pages = _page_of(model)
    rows = [_game_row(pages, g) for g in gen.games]
    head = [
        ("Game", True),
        ("Seed", False),
        ("p1 team", False),
        ("p2 team", False),
        ("Opponent (baseline seat)", False),
        ("Winner", False),
        ("Turns", True),
        ("Outcome", False),
        ("Forfeit loser", False),
        ("Cause", False),
    ]
    causes = (
        ", ".join(f"{esc(k)} {v}" for k, v in sorted(gen.forfeit_causes.items()))
        if gen.forfeit_causes
        else ("none" if gen.forfeits == 0 else "not recorded")
    )
    ids = [f"g{g.plan.game}" for g in gen.games]
    body = (
        f"<h1>Games of generation {gen.number}</h1>"
        f'<p class="note">Every planned game, played by the frozen pair {esc(gen.pair_policy_id)}. '
        f"Forfeits: {gen.forfeits} ({causes}). Seeds are the battle seeds without the sodium prefix. "
        f'Replay pages are not part of this build. <a href="{drawn_page(gen.number)}">Teams drawn</a></p>'
        '<input class="filter" id="filter" type="search" hidden aria-label="Filter the games">'
        '<p class="muted" id="filter-count"></p>'
        f"{_table(head, rows)}"
    )
    data = {"ids": ids, "noun": "games", "run": model.run_key}
    return document(f"Games of generation {gen.number}", _nav(model), body, data, _footer(model))


def _set_html(item: PoolSet) -> str:
    name = item.species if item.name in ("", item.species) else f"{item.name} ({item.species})"
    points = " / ".join(
        f"{value} {label}"
        for key, label in STAT_LABELS
        for k, value in item.evs
        if k == key and value
    )
    mega = " mega" if item.mega_stone else ""
    return (
        f'<div class="set{mega}"><b>{esc(name)}</b> @ {esc(item.item)}<br>{esc(item.ability)} · {esc(item.nature)} · '
        f"Lv {item.level}<br>Stat Points: {esc(points or 'none')}<br>{esc(' · '.join(item.moves))}</div>"
    )


def _results_table(use: TeamUse) -> str:
    if not use.results:
        return ""
    rows = []
    for (kind, by), tally in sorted(use.results.items()):
        games = tally["games"]
        share = f"{100 * tally['win'] / games:.0f}%" if games >= MIN_RATE_GAMES else "&ndash;"
        cells = "".join(f'<td class="n">{tally[name]}</td>' for name in RESULTS)
        rows.append(
            f'<tr><td>{esc(kind)}</td><td>{esc(by)}</td><td class="n">{games}</td>{cells}<td class="n">{share}</td></tr>'
        )
    head = [
        ("Opponent kind", False),
        ("Team played by", False),
        ("Games", True),
        ("Wins", True),
        ("Ties", True),
        ("Losses", True),
        ("Forfeits", True),
        (f"Win share (from {MIN_RATE_GAMES} games)", True),
    ]
    return _table(head, rows)


def _team_card(model: InspectModel, team: PoolTeam, index: int) -> str:
    use = model.use[team.hash]
    chips = "".join(f'<span class="chip">{esc(a)} = {esc(b)}</span>' for a, b in team.labels)
    if team.split:
        chips += f'<span class="chip">split {esc(team.split)}</span>'
    if use.draws:
        links = ", ".join(
            f'<a href="{games_page(g)}#g{n}">g{g} #{n} {esc(seat)}</a>'
            for g, n, seat in use.games[:MAX_GAME_LINKS]
        )
        more = (
            f" and {len(use.games) - MAX_GAME_LINKS} more"
            if len(use.games) > MAX_GAME_LINKS
            else ""
        )
        drawn = (
            f"Drawn {use.draws} time{'s' if use.draws != 1 else ''} in generation"
            f"{'s' if len(use.generations) != 1 else ''} {', '.join(str(g) for g in sorted(use.generations))}. "
            f"Games: {links}{more}."
        )
    else:
        drawn = "Not drawn in any game of this run."
    canonical = (
        f" · canonical <code>{esc(team.canonical_team_hash)}</code>"
        if team.canonical_team_hash
        else ""
    )
    return (
        f'<article class="card" id="{team_anchor(team.hash)}"><h3>Team {index + 1} · <code>{esc(team.hash)}</code></h3>'
        f'<div class="chips">{chips}</div><div class="sets">{"".join(_set_html(s) for s in team.sets)}</div>'
        f'<p class="muted">{drawn}{canonical}</p>{_results_table(use)}</article>'
    )


def teams_chunk_page(model: InspectModel, chunk: int) -> str:
    teams = model.pool.teams
    start, stop = chunk * CHUNK, min(len(teams), (chunk + 1) * CHUNK)
    chunks = (len(teams) + CHUNK - 1) // CHUNK
    cards = [_team_card(model, teams[i], i) for i in range(start, stop)]
    ids = [team_anchor(teams[i].hash) for i in range(start, stop)]
    pager = " · ".join(
        f'<a href="{teams_page(c)}">{c + 1}</a>' if c != chunk else f"<b>{c + 1}</b>"
        for c in range(chunks)
    )
    body = (
        f"<h1>Pool teams {start + 1} to {stop} of {len(teams)}</h1>"
        '<p class="note">In team-hash order. Results are the counts of this run\'s training games, one appearance per '
        "seat a team played; a win share appears only from "
        f"{MIN_RATE_GAMES} games.</p>"
        f'<p class="muted">Pages: {pager}</p>'
        '<input class="filter" id="filter" type="search" hidden aria-label="Filter the teams on this page">'
        f'<p class="muted" id="filter-count"></p>{"".join(cards)}'
    )
    data = {"ids": ids, "noun": "teams", "run": model.run_key}
    return document(f"Pool teams {start + 1} to {stop}", _nav(model), body, data, _footer(model))


def build_pages(model: InspectModel) -> dict[str, str]:
    """Every page of the set by file name."""
    pages = {INDEX: index_page(model), DISTRIBUTIONS: distributions_page(model)}
    for gen in model.generations:
        pages[games_page(gen.number)] = games_generation_page(model, gen)
        pages[drawn_page(gen.number)] = drawn_generation_page(model, gen)
    for chunk in range((len(model.pool.teams) + CHUNK - 1) // CHUNK):
        pages[teams_page(chunk)] = teams_chunk_page(model, chunk)
    return pages
