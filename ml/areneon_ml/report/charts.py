"""Hand-written inline SVG charts: lines with optional intervals, and stacked bars.

Every chart is a ``<figure>`` with a title, an SVG (a group with an aria-label; each point or segment
has a ``<title>`` and a ``data-t`` index into the page's tip list, which the one script writes into
the chart's readout with ``textContent``), a legend (identity is never colour alone: markers differ
and the legend names each series), a caption and a collapsed data table with every number. All text
goes through ``esc``; the SVG holds no script, no style attribute and no URL.
"""

# ruff: noqa: E501  (SVG template lines are kept whole so the markup reads as written)

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from areneon_ml.inspector.page import esc

W, H = 400, 254
LEFT, RIGHT, TOP, BOTTOM = 54, 14, 12, 36
MAX_TICKS = 12
PW, PH = W - LEFT - RIGHT, H - TOP - BOTTOM


class Tips:
    """The readout texts of a page, by index (they travel in the page's JSON data block)."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, text: str) -> int:
        self.items.append(text)
        return len(self.items) - 1


@dataclass(frozen=True)
class Pt:
    """One value of a series at the generation ``x``; ``low``/``high`` draw a whisker."""

    x: int
    y: float
    tip: str
    cell: str
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Series:
    name: str
    slot: int  # 1..9: the categorical colour slot, fixed by the entity
    shape: str  # "circle" or "square"
    points: tuple[Pt, ...]


def nice_ticks(low: float, high: float, want: int = 4) -> list[float]:
    """Round tick values covering ``[low, high]``."""
    if high <= low:
        high = low + (abs(low) or 1.0)
    raw = (high - low) / want
    base = 10 ** math.floor(math.log10(raw))
    step = next(m * base for m in (1, 2, 2.5, 5, 10) if m * base >= raw)
    first = math.floor(low / step) * step
    out, value = [], first
    while value < high + step * 0.999:
        out.append(round(value, 12))
        value += step
        if len(out) > MAX_TICKS:
            break
    return out


def _label(value: float) -> str:
    text = f"{value:.3g}"
    return "0" if text in ("-0", "0.00") else text


def _fx(index: int, count: int) -> float:
    return LEFT + (PW / 2 if count == 1 else index * PW / (count - 1))


def _xticks(xs: Sequence[int], x_of: dict[int, float]) -> str:
    step = max(1, math.ceil(len(xs) / 8))
    out = []
    for i, g in enumerate(xs):
        if i % step == 0:
            out.append(f'<text x="{x_of[g]:.1f}" y="{H - 16}" text-anchor="middle">{g}</text>')
    out.append(
        f'<text x="{LEFT + PW / 2:.1f}" y="{H - 2}" text-anchor="middle" class="v">generation</text>'
    )
    return "".join(out)


def _legend(series: Sequence[tuple[str, int, str]]) -> str:
    if len(series) <= 1:
        return ""
    items = "".join(
        f'<span><i class="sw sw{slot} {"ci" if shape == "circle" else "sq"}"></i>{esc(name)}</span>'
        for name, slot, shape in series
    )
    return f'<div class="legend">{items}</div>'


def _table(xs: Sequence[int], columns: Sequence[tuple[str, dict[int, str]]], missing: str) -> str:
    head = "".join(f'<th scope="col" class="n">{esc(name)}</th>' for name, _ in columns)
    rows = "".join(
        f'<tr><th scope="row">{g}</th>'
        + "".join(f'<td class="n">{esc(cells.get(g, missing))}</td>' for _, cells in columns)
        + "</tr>"
        for g in xs
    )
    return (
        '<details><summary>Data table</summary><div class="scroll"><table>'
        f'<thead><tr><th scope="col">Generation</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></details>'
    )


def _frame(
    title: str,
    xs: Sequence[int],
    ticks: list[float],
    y_of: dict[float, float],
    y_fmt: dict[float, str],
) -> list[str]:
    parts = [f"<title>{esc(title)}</title>"]
    for t in ticks:
        y = y_of[t]
        parts.append(f'<line class="axis" x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="v" x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt[t])}</text>'
        )
    return parts


def line_chart(  # noqa: PLR0913
    title: str,
    caption: str,
    xs: Sequence[int],
    series: Sequence[Series],
    tips: Tips,
    *,
    y_range: tuple[float, float] | None = None,
    y_label: Callable[[float], str] = _label,
    ref: float | None = None,
    empty: str = "Not recorded for any generation.",
) -> str:
    """Lines per series over the generations; a point with ``low``/``high`` carries a whisker."""
    values = [v for s in series for p in s.points for v in (p.y, p.low, p.high) if v is not None]
    if not values:
        return f'<figure><h3>{esc(title)}</h3><p class="muted">{esc(empty)}</p></figure>'
    if y_range is None:
        ticks = nice_ticks(min(values), max(values))
    else:
        ticks = (
            nice_ticks(*y_range) if y_range[1] - y_range[0] != 1 else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    low_t, high_t = ticks[0], ticks[-1]
    span = (high_t - low_t) or 1.0

    def y_pos(v: float) -> float:
        return TOP + PH - (v - low_t) / span * PH

    x_of = {g: _fx(i, len(xs)) for i, g in enumerate(xs)}
    parts = _frame(title, xs, ticks, {t: y_pos(t) for t in ticks}, {t: y_label(t) for t in ticks})
    if ref is not None and low_t < ref < high_t:
        parts.append(
            f'<line class="ref" x1="{LEFT}" y1="{y_pos(ref):.1f}" x2="{W - RIGHT}" y2="{y_pos(ref):.1f}"/>'
        )
    parts.append(_xticks(xs, x_of))
    offsets = {s.name: (i - (len(series) - 1) / 2) * 6 for i, s in enumerate(series)}
    for s in series:
        dx = offsets[s.name] if len(xs) > 1 else 0.0
        run: list[str] = []
        for g in xs:
            point = next((p for p in s.points if p.x == g), None)
            if point is None:
                if len(run) > 1:
                    parts.append(f'<polyline class="ln l{s.slot}" points="{" ".join(run)}"/>')
                run = []
                continue
            run.append(f"{x_of[g] + dx:.1f},{y_pos(point.y):.1f}")
        if len(run) > 1:
            parts.append(f'<polyline class="ln l{s.slot}" points="{" ".join(run)}"/>')
        for p in s.points:
            cx, cy, t = x_of[p.x] + dx, y_pos(p.y), tips.add(p.tip)
            if p.low is not None and p.high is not None:
                parts.append(
                    f'<line class="wh l{s.slot}" x1="{cx:.1f}" y1="{y_pos(p.low):.1f}" x2="{cx:.1f}" y2="{y_pos(p.high):.1f}"/>'
                )
            mark = (
                f'<circle class="ring f{s.slot}" cx="{cx:.1f}" cy="{cy:.1f}" r="4"'
                if s.shape == "circle"
                else f'<rect class="ring f{s.slot}" x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8"'
            )
            parts.append(
                f'{mark} tabindex="0" data-t="{t}"><title>{esc(p.tip)}</title></{"circle" if s.shape == "circle" else "rect"}>'
            )
    legend = _legend([(s.name, s.slot, s.shape) for s in series])
    table = _table(xs, [(s.name, {p.x: p.cell for p in s.points}) for s in series], "not recorded")
    return (
        f"<figure><h3>{esc(title)}</h3>{legend}"
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="group" aria-label="{esc(title)}: {esc(caption)}">{"".join(parts)}</svg>'
        f'<div class="readout" aria-live="polite"></div><figcaption>{esc(caption)}</figcaption>{table}</figure>'
    )


@dataclass(frozen=True)
class Stack:
    name: str
    slot: int
    counts: dict[int, int]


def stacked_chart(  # noqa: PLR0913, PLR0917
    title: str, caption: str, xs: Sequence[int], stacks: Sequence[Stack], tips: Tips, empty: str
) -> str:
    """Stacked bars of counts per generation; segments are 2px apart and each says its count."""
    totals = {g: sum(s.counts.get(g, 0) for s in stacks) for g in xs}
    top = max(totals.values(), default=0)
    if top == 0:
        return f'<figure><h3>{esc(title)}</h3><p class="muted">{esc(empty)}</p></figure>'
    ticks = nice_ticks(0, float(top))
    ticks = [t for t in ticks if t == int(t)] or [0.0, float(top)]
    high_t = ticks[-1] if ticks[-1] >= top else float(top)

    def y_pos(v: float) -> float:
        return TOP + PH - v / high_t * PH

    x_of = {g: _fx(i, len(xs)) for i, g in enumerate(xs)}
    parts = _frame(title, xs, ticks, {t: y_pos(t) for t in ticks}, {t: str(int(t)) for t in ticks})
    parts.append(_xticks(xs, x_of))
    bar = min(28.0, PW / len(xs) * 0.7)
    for g in xs:
        base = 0
        for s in stacks:
            n = s.counts.get(g, 0)
            if n == 0:
                continue
            y1, y0 = y_pos(base + n), y_pos(base)
            t = tips.add(f"Generation {g}, {s.name}: {n} of {totals[g]} games")
            parts.append(
                f'<rect class="ring f{s.slot}" x="{x_of[g] - bar / 2:.1f}" y="{y1:.1f}" width="{bar:.1f}" height="{max(y0 - y1, 1):.1f}" '
                f'rx="2" tabindex="0" data-t="{t}"><title>Generation {g}, {esc(s.name)}: {n} of {totals[g]} games</title></rect>'
            )
            base += n
    legend = _legend([(s.name, s.slot, "square") for s in stacks])
    table = _table(xs, [(s.name, {g: str(s.counts.get(g, 0)) for g in xs}) for s in stacks], "0")
    return (
        f"<figure><h3>{esc(title)}</h3>{legend}"
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="group" aria-label="{esc(title)}: {esc(caption)}">{"".join(parts)}</svg>'
        f'<div class="readout" aria-live="polite"></div><figcaption>{esc(caption)}</figcaption>{table}</figure>'
    )
