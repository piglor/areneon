"""Distributions over the pool and over the drawn teams, as small inline SVG bar charts.

Two series, the pool (series 1) and the drawn teams (series 2), drawn side by side per value, with
the share written at the end of each bar so identity never rests on colour alone. Every string comes
through ``esc``; the SVG holds no script, no style attribute and no URL.
"""

# ruff: noqa: E501  (HTML, CSS and SVG template lines are kept whole so the markup reads as written)

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from areneon_ml.inspector.page import esc
from areneon_ml.inspector.pool import PoolTeam

CATEGORIES = (
    ("labels", "Generator labels"),
    ("species", "Species"),
    ("items", "Items"),
    ("moves", "Moves"),
    ("abilities", "Abilities"),
    ("mega", "Mega Stone holders"),
)
TOP = 15
LABEL_WIDTH = 190
BAR_WIDTH = 330
ROW = 28
_MAX_LABEL = 30


def team_values(team: PoolTeam) -> dict[str, set[str]]:
    """The values a team has in each category (a team counts once per value)."""
    out: dict[str, set[str]] = {key: set() for key, _ in CATEGORIES}
    for label_id, value in team.labels:
        out["labels"].add(f"{label_id} = {value}")
    for item in team.sets:
        out["species"].add(item.species)
        out["items"].add(item.item)
        out["abilities"].add(item.ability)
        out["moves"].update(item.moves)
        if item.mega_stone:
            out["mega"].add(f"{item.species} @ {item.item}")
    return out


@dataclass(frozen=True)
class Distribution:
    """Counts per category value over a weighted set of teams, and the weight."""

    total: int
    counts: dict[str, Counter[str]]


def distribution(teams: Iterable[tuple[PoolTeam, int]]) -> Distribution:
    counts: dict[str, Counter[str]] = {key: Counter() for key, _ in CATEGORIES}
    total = 0
    for team, weight in teams:
        total += weight
        for key, values in team_values(team).items():
            for value in values:
                counts[key][value] += weight
    return Distribution(total, counts)


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percent(share: float) -> str:
    return f"{100 * share:.1f}%"


def _clip(value: str) -> str:
    return value if len(value) <= _MAX_LABEL else value[: _MAX_LABEL - 1] + "…"


def ordered(pool: Distribution, drawn: Distribution, key: str) -> list[str]:
    """Values by pool count (then drawn count, then name), so the order never depends on the filter."""
    values = set(pool.counts[key]) | set(drawn.counts[key])
    return sorted(values, key=lambda v: (-pool.counts[key][v], -drawn.counts[key][v], v))


def legend(drawn_label: str) -> str:
    return (
        '<div class="legend"><span><i class="sw s1"></i>Pool</span>'
        f'<span><i class="sw s2"></i>{esc(drawn_label)}</span></div>'
    )


def bar_chart(
    title: str, pool: Distribution, drawn: Distribution, key: str, drawn_label: str
) -> str:
    """The top values of a category as grouped horizontal bars (pool above, drawn below)."""
    values = ordered(pool, drawn, key)[:TOP]
    if not values:
        return '<p class="muted">No values.</p>'
    shares = [
        (_share(pool.counts[key][v], pool.total), _share(drawn.counts[key][v], drawn.total))
        for v in values
    ]
    scale = max(max(a, b) for a, b in shares) or 1.0
    width = LABEL_WIDTH + BAR_WIDTH + 70
    height = ROW * len(values) + 8
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}: share of teams per value, pool and {esc(drawn_label)}">',
        f"<title>{esc(title)}</title>",
        f'<line x1="{LABEL_WIDTH}" y1="0" x2="{LABEL_WIDTH}" y2="{height - 4}"/>',
    ]
    for index, (value, (a, b)) in enumerate(zip(values, shares, strict=True)):
        y = 4 + index * ROW
        parts.append(
            f'<text x="{LABEL_WIDTH - 6}" y="{y + 15}" text-anchor="end"><title>{esc(value)}</title>{esc(_clip(value))}</text>'
        )
        for offset, share, css, name, count in (
            (0, a, "b1", "pool", pool.counts[key][value]),
            (11, b, "b2", drawn_label, drawn.counts[key][value]),
        ):
            length = round(BAR_WIDTH * share / scale, 1)
            parts.append(
                f'<rect class="{css}" x="{LABEL_WIDTH}" y="{y + offset}" width="{length}" height="9" rx="2">'
                f"<title>{esc(value)}: {esc(name)} {count} ({_percent(share)})</title></rect>"
                f'<text class="v" x="{LABEL_WIDTH + length + 4}" y="{y + offset + 8}">{_percent(share)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def value_table(pool: Distribution, drawn: Distribution, key: str, drawn_label: str) -> str:
    """Every value with its counts: the chart's data, for a reader who wants all of it."""
    rows = "".join(
        f'<tr><td>{esc(v)}</td><td class="n">{pool.counts[key][v]}</td>'
        f'<td class="n">{_percent(_share(pool.counts[key][v], pool.total))}</td>'
        f'<td class="n">{drawn.counts[key][v]}</td>'
        f'<td class="n">{_percent(_share(drawn.counts[key][v], drawn.total))}</td></tr>'
        for v in ordered(pool, drawn, key)
    )
    return (
        f"<details><summary>All {len(set(pool.counts[key]) | set(drawn.counts[key]))} values</summary>"
        '<div class="scroll"><table><thead><tr><th>Value</th><th class="n">Pool teams</th>'
        f'<th class="n">Pool share</th><th class="n">{esc(drawn_label)}</th>'
        '<th class="n">Share</th></tr></thead><tbody>'
        f"{rows}</tbody></table></div></details>"
    )


def section(pool: Distribution, drawn: Distribution, drawn_label: str) -> str:
    """All categories: heading, chart (top values) and the full table."""
    out = [legend(drawn_label)]
    for key, title in CATEGORIES:
        out.append(f"<h3>{esc(title)}</h3>")
        out.append(bar_chart(title, pool, drawn, key, drawn_label))
        out.append(value_table(pool, drawn, key, drawn_label))
    return "".join(out)
