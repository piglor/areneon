"""The inspector's pages for a population run (ADR-021 section 8, ADR-017 sections 4a and 4c).

The index lists the members of every generation (frozen pair, slot origin, ladder strength), the
copy decisions and the past pool; the games table has the member on each seat, the game kind and the
past-pool snapshot a past pair met. The distribution, drawn-team and pool-team pages are the single
lineage's own functions (``render``), fed by the population model. Every string from data goes
through ``page.esc``; links are built from fixed names and hex ids. Nothing depends on the clock.
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

from collections import Counter

from areneon_ml.inspector.page import document, esc
from areneon_ml.inspector.population_model import (
    PopGame,
    PopGeneration,
    PopulationModel,
)
from areneon_ml.inspector.render import (
    CHUNK,
    DISTRIBUTIONS,
    INDEX,
    _footer,
    _nav,
    _page_of,
    _short,
    _stat,
    _table,
    _team_link,
    distributions_page,
    drawn_generation_page,
    drawn_page,
    games_page,
    teams_chunk_page,
    teams_page,
)
from areneon_ml.inspector.replay_page import page_path as replay_path
from areneon_ml.loop.past_pool import Snapshot

KIND_ORDER = ("ladder", "self", "past", "anchor")
KIND_NOTE = (
    "ladder: a member against another member; self: a member against itself; past: a member "
    "against a frozen past-pool snapshot; anchor: a member against a scripted baseline"
)


def _holder(game: PopGame, seat: str) -> str:
    held = game.plan.seats[seat]
    label = esc(held.label)
    if held.kind == "member":
        return f"member {label}"
    if held.kind == "snapshot":
        return f"snapshot {label}"
    return f"baseline {label}"


def _winner(game: PopGame) -> str:
    row = game.row
    if row.outcome == "forfeit":
        return "none"
    if row.winner is None:
        return "tie"
    return f"{row.winner} ({esc(game.plan.seats[row.winner].label)})"


def _snapshot_cell(game: PopGame, generation: PopGeneration) -> str:
    for seat in ("p1", "p2"):
        held = game.plan.seats[seat]
        if held.kind == "snapshot":
            snap = generation.snapshot(held.label)
            return f"{esc(snap.id)} (member {esc(snap.member)}, generation {snap.generation})"
    return "&ndash;"


def _replay_cell(game: PopGame, replays: frozenset[tuple[int, int]]) -> str:
    if game.row.outcome == "forfeit":
        return "none (forfeit)"
    if (game.generation, game.plan.game) in replays:
        return f'<a href="{esc(replay_path(game.generation, game.plan.game))}">replay</a>'
    return "not built"


def _game_row(
    pages: dict[str, int],
    game: PopGame,
    generation: PopGeneration,
    replays: frozenset[tuple[int, int]],
) -> str:
    row = game.row
    loser = cause = ""
    title = ""
    if row.forfeit is not None:
        loser, cause = row.forfeit.loser, row.forfeit.cause
        if row.forfeit.detail:
            title = f' title="{esc(row.forfeit.detail)}"'
    return (
        f'<tr id="g{game.plan.game}"><td class="n">{game.plan.game}</td>'
        f'<td class="n">{game.plan.pair}.{game.plan.half}</td>'
        f'<td class="m">{esc(row.seed.removeprefix("sodium,")[:12])}</td>'
        f"<td>{esc(game.kind)}</td><td>{esc(game.plan.member)}</td>"
        f"<td>{_holder(game, 'p1')}</td><td>{_holder(game, 'p2')}</td>"
        f"<td>{_team_link(pages, game.hashes['p1'])}</td><td>{_team_link(pages, game.hashes['p2'])}</td>"
        f"<td>{_snapshot_cell(game, generation)}</td><td>{_winner(game)}</td>"
        f'<td class="n">{row.turns}</td><td>{esc(row.outcome)}</td><td>{esc(loser)}</td>'
        f"<td{title}>{esc(cause)}</td><td>{_replay_cell(game, replays)}</td></tr>"
    )


def population_games_page(
    model: PopulationModel, gen: PopGeneration, replays: frozenset[tuple[int, int]] = frozenset()
) -> str:
    pages = _page_of(model)
    rows = [_game_row(pages, g, gen, replays) for g in gen.games]
    head = [
        ("Game", True),
        ("Pair.half", True),
        ("Seed", False),
        ("Kind", False),
        ("Scheduled for", False),
        ("p1 played by", False),
        ("p2 played by", False),
        ("p1 team", False),
        ("p2 team", False),
        ("Past-pool snapshot", False),
        ("Winner", False),
        ("Turns", True),
        ("Outcome", False),
        ("Forfeit loser", False),
        ("Cause", False),
        ("Replay", False),
    ]
    causes = (
        ", ".join(f"{esc(k)} {v}" for k, v in sorted(gen.forfeit_causes.items()))
        if gen.forfeit_causes
        else "none"
    )
    kinds = Counter(g.kind for g in gen.games)
    mix = ", ".join(f"{k} {kinds[k]}" for k in KIND_ORDER if kinds[k])
    ids = [f"g{g.plan.game}" for g in gen.games]
    body = (
        f"<h1>Games of generation {gen.number}</h1>"
        f'<p class="note">Every planned game of the population, {len(gen.games)} in all ({esc(mix)}). '
        f"A pair is played twice on one battle seed (half 0 and 1). {esc(KIND_NOTE)}. "
        f"Forfeits: {gen.forfeits} ({causes}). Seeds are the battle seeds without the sodium prefix. "
        "Replay pages are built on demand (<code>python -m areneon_ml.loop replay</code>); a link shows "
        f'where one exists. <a href="{drawn_page(gen.number)}">Teams drawn</a> · <a href="{INDEX}#g{gen.number}">Members</a></p>'
        '<input class="filter" id="filter" type="search" hidden aria-label="Filter the games">'
        '<p class="muted" id="filter-count"></p>'
        f"{_table(head, rows)}"
    )
    data = {"ids": ids, "noun": "games", "run": model.run_key}
    return document(f"Games of generation {gen.number}", _nav(model), body, data, _footer(model))


def _strength(value: tuple[int, int, int] | None) -> str:
    if value is None:
        return "&ndash;"
    lower, middle, upper = value
    return f"{middle} ({lower} to {upper})"


def _members_table(gen: PopGeneration | None, model: PopulationModel) -> str:
    states = model.initial if gen is None else gen.members
    rows = []
    for m in states:
        origin = f"copy of {esc(m.copied_from)}" if m.copied_from else "own"
        pair = (
            esc(_short(m.pair_policy_id.removeprefix("policy:"), 10))
            if m.pair_policy_id
            else "&ndash;"
        )
        rows.append(
            f'<tr><td>{esc(m.member)}</td><td class="m">{pair}</td>'
            f'<td class="m">{esc(_short(m.preview, 10))}</td><td class="m">{esc(_short(m.battle, 10))}</td>'
            f'<td>{origin}</td><td>{esc(m.root or "none")}</td><td class="n">{_strength(m.strength)}</td>'
            f'<td class="n">{m.games}</td><td class="n">{m.played}</td><td class="n">{m.episodes}</td>'
            f'<td class="n">{m.examples}</td></tr>'
        )
    head = [
        ("Member", False),
        ("Pair played", False),
        ("Preview frozen", False),
        ("Battle frozen", False),
        ("Slot", False),
        ("Root", False),
        ("Strength (lower to upper)", True),
        ("Games", True),
        ("Played", True),
        ("Own Episodes", True),
        ("Examples", True),
    ]
    return _table(head, rows)


def _selection(gen: PopGeneration) -> str:
    s = gen.selection
    if s is None:
        return "no decision"
    if s["kind"] == "copy":
        return (
            f"copy: {esc(s['donor'])} into {esc(s['loser'])}; {s['pairs']} pairs, lower bound "
            f"{s['lower_bound_ppm']} ppm, margin {s['margin_ppm']} ppm; roots {s['roots_before']} to {s['roots_after']}; "
            f"applied at the start of generation {gen.number + 1}"
        )
    return f"none: {esc(s.get('failed', 'no reason'))}"


def _snapshot_names(snapshots: tuple[Snapshot, ...] | tuple[str, ...]) -> str:
    names = [s if isinstance(s, str) else s.id for s in snapshots]
    return esc(", ".join(names) or "none")


def population_index_page(model: PopulationModel) -> str:
    pool = model.pool
    distinct = len(model.cumulative_drawn)
    appearances = sum(model.cumulative_drawn.values())
    games = sum(len(g.games) for g in model.generations)
    episodes = sum(g.episodes for g in model.generations)
    stats = "".join(
        [
            _stat("Members", len(model.members)),
            _stat("Generations with games", len(model.generations)),
            _stat("Games (distinct)", games),
            _stat("Episodes", episodes),
            _stat("Pool after the filter", len(pool.teams)),
            _stat("Distinct teams drawn", distinct),
            _stat("Team draws (one per seat)", appearances),
        ]
    )
    p = model.population
    settings = _table(
        [("Setting", False), ("Value", False)],
        [
            f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
            for k, v in (
                ("members", p["members"]),
                ("seats per member", p["seats_per_member"]),
                ("shares (ppm)", ", ".join(f"{a} {b}" for a, b in sorted(p["shares"].items()))),
                ("past pool", ", ".join(f"{a} {b}" for a, b in sorted(p["past_pool"].items()))),
                ("selection", ", ".join(f"{a} {b}" for a, b in sorted(p["selection"].items()))),
                ("rating", ", ".join(f"{a} {b}" for a, b in sorted(p["rating"].items()))),
            )
        ],
    )
    cumulative: set[str] = set()
    gen_rows = []
    for gen in model.generations:
        cumulative |= set(gen.drawn)
        played = sum(1 for g in gen.games if g.row.outcome == "played")
        kinds = Counter(g.kind for g in gen.games)
        mix = " ".join(f"{kinds[k]}" for k in KIND_ORDER)
        gen_rows.append(
            f'<tr id="g{gen.number}"><td class="n">{gen.number}</td><td class="n">{len(gen.games)}</td>'
            f'<td class="n">{played}</td><td class="n">{gen.forfeits}</td><td class="n">{gen.episodes}</td>'
            f'<td>{mix}</td><td class="n">{len(gen.drawn)}</td><td class="n">{len(cumulative)}</td>'
            f'<td><a href="{games_page(gen.number)}">games</a> · <a href="{drawn_page(gen.number)}">drawn teams</a></td></tr>'
        )
    gen_head = [
        ("Generation", True),
        ("Games", True),
        ("Played", True),
        ("Forfeits", True),
        ("Episodes", True),
        ("ladder self past anchor", False),
        ("Teams drawn", True),
        ("Distinct so far", True),
        ("Pages", False),
    ]
    member_sections = [
        "<h3>Generation 0 (the seeded pairs, no games)</h3>" + _members_table(None, model)
    ]
    member_sections += [
        f"<h3>Generation {g.number}</h3>{_members_table(g, model)}" for g in model.generations
    ]
    copy_rows = [
        f'<tr><td class="n">{g.number}</td><td>{_selection(g)}</td></tr>' for g in model.generations
    ]
    past_rows = []
    for g in model.generations:
        met = Counter(
            gm.plan.seats[s].label
            for gm in g.games
            for s in ("p1", "p2")
            if gm.plan.seats[s].kind == "snapshot" and gm.row.half == 0
        )
        past_rows.append(
            f'<tr><td class="n">{g.number}</td><td>{_snapshot_names(g.snapshots)}</td>'
            f"<td>{esc(', '.join(f'{k} x{v}' for k, v in sorted(met.items())) or 'none')}</td>"
            f"<td>{_snapshot_names(g.committed_snapshots)}</td></tr>"
        )
    manifests = "".join(
        f'<tr><td class="n">{n}</td><td class="m">{esc(h)}</td></tr>'
        for n, h in sorted(model.manifest_hashes.items())
    )
    body = (
        f"<h1>Training data of population run {esc(model.run_key)}</h1>"
        f'<p class="note">Which teams and which games this population run trained on. Lineage {esc(model.lineage_id)}, '
        f"format {esc(model.format_id)}. Synthetic data only; the pool was rebuilt with the loop's own function "
        "and every file below was checked against it and against the other files of its generation.</p>"
        f'<div class="grid">{stats}</div>'
        f'<p class="note">Source of the pool: {esc(pool.source)}. Dataset hash: '
        f"<code>{esc(pool.dataset_hash or 'none (interim fixture pool)')}</code>.</p>"
        f"<h2>Population settings</h2>{settings}"
        f"<h2>Generations</h2>{_table(gen_head, gen_rows)}"
        '<p class="note">Every pair of teams is played twice on one battle seed, so games come in halves. '
        f"{esc(KIND_NOTE)}.</p>"
        "<h2>Members by generation</h2>"
        + "".join(member_sections)
        + '<p class="note">Strength is the ladder\'s fit in integer units (10^9 is the reference anchor, actor-rule@v1). '
        '"Slot: copy of m" means the copy step made the slot train from member m\'s frozen pair at the start of that '
        "generation (its new records name m's checkpoints as parent); the pair it played is its own, frozen the "
        "generation before. Preview and Battle frozen are the checkpoints the generation produced. Own Episodes are "
        "the Episodes of the member's own seats.</p>"
        f"<h2>Copy decisions</h2>{_table([('Decided at generation', True), ('Decision', False)], copy_rows)}"
        f"<h2>Past pool</h2>{_table([('Generation', True), ('Pool it played', False), ('Snapshots met (pairs)', False), ('Pool it committed', False)], past_rows)}"
        f'<p><a href="{DISTRIBUTIONS}">Distributions over the pool and the drawn teams</a> · '
        f'<a href="{teams_page(0)}">All {len(pool.teams)} pool teams, {CHUNK} per page</a></p>'
        f"<details><summary>Manifest hashes of the generations read</summary>{_table([('Generation', True), ('SHA-256 of manifest.json', False)], [manifests])}</details>"
    )
    return document(f"Training data of {model.run_key}", _nav(model), body, footer=_footer(model))


def build_population_pages(
    model: PopulationModel,
    replays: frozenset[tuple[int, int]] | set[tuple[int, int]] = frozenset(),
) -> dict[str, str]:
    """Every page of the set by file name."""
    pages = {INDEX: population_index_page(model), DISTRIBUTIONS: distributions_page(model)}
    for gen in model.generations:
        pages[games_page(gen.number)] = population_games_page(model, gen, frozenset(replays))
        pages[drawn_page(gen.number)] = drawn_generation_page(model, gen)
    for chunk in range((len(model.pool.teams) + CHUNK - 1) // CHUNK):
        pages[teams_page(chunk)] = teams_chunk_page(model, chunk)
    return pages


__all__ = ["build_population_pages", "population_games_page", "population_index_page"]
