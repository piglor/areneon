"""The dashboard's ladder view of a population run (ADR-021 section 11, ADR-017 sections 4 and 4c).

Every number comes from ``population.py`` (the run's verified files). The ladder is a training
signal: it is fitted on training-pool teams in sampling mode, it is not validation and it is not
held out, and the page says so on top and at every table. The display rating is computed here from
the stored integer strength (``400 * log10(strength / 10^9)``, decimal arithmetic so the bytes do
not depend on the platform's libm) and is never stored. Every string that comes from data goes
through ``esc`` or the JSON data block; the SVG holds no script, no style attribute and no URL;
colour is never the only carrier (a heatmap cell states its value and its pair count).
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, localcontext
from typing import Any

from areneon_ml.inspector.page import esc
from areneon_ml.report.charts import Pt, Series, Tips, line_chart
from areneon_ml.report.model import NOT_RECORDED, ReportModel
from areneon_ml.report.page import STYLE, document
from areneon_ml.report.population import MemberGeneration, PopulationFacts, PopulationGeneration

POP_BANNER = (
    "Population ladder view: a training signal, never a measure of strength. "
    "Not the Training gate. Loss is not skill."
)
LADDER_LABEL = "Training-pool teams, sampling mode; not validation, not held out."
RATING_NOTE = (
    "Display rating = 400 x log10(strength / 10^9), computed on this page from the stored integer "
    "strength (the reference anchor actor-rule@v1 is 0.0) and never stored. The interval is the "
    "record's percentile bootstrap, descriptive. The ratings are fitted on the games of the "
    "rating window; anchors and frozen-rated snapshots are fixed rows, not re-fitted."
)
NOT_FINITE = "not finite"
FAILED = {
    "no_ready_member": "no member was ready",
    "too_few_pairs": "too few pairs of the donor against the loser",
    "lower_bound": "the lower bound did not clear the margin",
    "anchor_minimum": "the donor's weakest anchor score is below the loser's",
    "root_floor": "a copy would leave fewer roots than the minimum",
}
HEALTH_CAUSES = {
    "non_finite_loss": "a non-finite number in its update",
    "clip_fraction": "clip fraction above the limit for three generations",
    "approx_kl": "approximate KL above the limit for three generations",
    "forfeits": "forfeits above the limit",
}

POPULATION_CSS = """
:root{--h0:#e4a8a7;--h1:#f1d0cf;--h2:#e8e8e3;--h3:#c8e0d6;--h4:#8ec8b0}
@media (prefers-color-scheme:dark){:root{--h0:#6b2f2e;--h1:#4d3736;--h2:#3a3a37;--h3:#2f4a41;--h4:#245a46}}
table.heat td{text-align:center}
table.heat td.hm0{background:var(--h0)}table.heat td.hm1{background:var(--h1)}table.heat td.hm2{background:var(--h2)}table.heat td.hm3{background:var(--h3)}table.heat td.hm4{background:var(--h4)}
table.heat td.hmx{color:var(--muted)}
svg.tree .dash{stroke-dasharray:4 3}svg.tree .lane{stroke:var(--line)}svg.tree text{font-size:11px;fill:var(--muted)}
""".strip()
POPULATION_STYLE = STYLE + "\n" + POPULATION_CSS

TW, TLEFT, TRIGHT, TTOP, LANE = 640, 40, 16, 18, 30


def rating(strength: int) -> Decimal:
    """``400 * log10(strength / 10^9)`` to one decimal, in decimal arithmetic (never stored)."""
    with localcontext() as ctx:
        ctx.prec = 40
        value = (Decimal(strength) / Decimal(10**9)).log10() * 400
        out = value.quantize(Decimal("0.1"))
    return Decimal("0.0") if out == 0 else out


def shown(strength: int) -> str:
    return format(rating(strength), "+.1f")


def _pct(ppm: int) -> str:
    tenths = (ppm + 500) // 1000
    return f"{tenths // 10}.{tenths % 10}%"


def _table(head: list[tuple[str, bool]], rows: Iterable[list[str]], caption: str) -> str:
    """A table whose cells the caller has already escaped; the first column is the row header."""
    heads = "".join(
        f'<th scope="col"{' class="n"' if numeric and index else ""}>{esc(name)}</th>'
        for index, (name, numeric) in enumerate(head)
    )
    body = ""
    for row in rows:
        cells = [f'<th scope="row">{row[0]}</th>']
        for (_, numeric), cell in zip(head[1:], row[1:], strict=True):
            cells.append(f'<td class="{"n" if numeric else "txt"}">{cell}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"
    return (
        f'<div class="scroll"><table><caption class="muted">{esc(caption)}</caption>'
        f"<thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _slot(index: int) -> int:
    return index % 8 + 1


def _number(label: str) -> int:
    return int(label[1:])


# -- the events: copies and quarantines -----------------------------------------------------------


def _strongest_live(ladder: dict[str, Any], quarantined: set[str]) -> str | None:
    """The replacement donor: the greatest strength among the members not quarantined (ties: lower)."""
    live = [m for m in ladder["members"] if m["member"] not in quarantined]
    if not live:
        return None
    best = max(live, key=lambda m: (m["strength"]["strength"], -_number(m["member"])))
    return str(best["member"])


def events(facts: PopulationFacts) -> list[dict[str, Any]]:
    """Every copy and health replacement, in generation order, with the record's evidence.

    A replacement's donor is not stored (the record names the cause only): it is derived the way
    the loop derives it, the strongest member that is not quarantined.
    """
    out: list[dict[str, Any]] = []
    for g in facts.generations:
        if g.ladder is None:
            continue
        first, *rest = g.ladder["selections"]
        if first["kind"] == "copy":
            out.append({"kind": "copy", "at": g.number, **first})
        stopped = {e["member"] for e in rest}
        for entry in rest:
            out.append(
                {
                    "kind": "health",
                    "at": g.number,
                    "member": entry["member"],
                    "cause": entry["cause"],
                    "donor": _strongest_live(g.ladder, stopped),
                }
            )
    return out


# -- the ladder table and the heatmap -------------------------------------------------------------


def _notes(ladder: dict[str, Any]) -> dict[str, str]:
    first, *rest = ladder["selections"]
    notes: dict[str, str] = {}
    if first["kind"] == "copy":
        notes[first["donor"]] = "donor of this generation's copy"
        notes[first["loser"]] = f"loser: takes {first['donor']}'s weights next generation"
    for entry in rest:
        notes[entry["member"]] = f"quarantined: {HEALTH_CAUSES[entry['cause']]}"
    return notes


def ladder_table(ladder: dict[str, Any]) -> str:
    notes = _notes(ladder)
    rows: list[list[str]] = []
    members = sorted(
        ladder["members"], key=lambda m: (-m["strength"]["strength"], _number(m["member"]))
    )
    for rank, m in enumerate(members, start=1):
        s = m["strength"]
        rows.append(
            [
                esc(m["member"]),
                "member",
                shown(s["strength"]),
                f"{shown(s['lower'])} to {shown(s['upper'])}",
                str(s["strength"]),
                esc(m["root"]),
                esc(f"rank {rank}" + (f"; {notes[m['member']]}" if m["member"] in notes else "")),
            ]
        )
    for a in ladder["anchors"]:
        rows.append(
            [
                esc(a["anchor_id"]),
                "anchor",
                shown(a["strength"]),
                "fixed",
                str(a["strength"]),
                "-",
                "fixed: frozen by the generation-0 calibration",
            ]
        )
    for snap in ladder["snapshots"]:
        s = snap["strength"]
        rows.append(
            [
                esc(snap["snapshot_id"]),
                "snapshot",
                shown(s["strength"]),
                f"{shown(s['lower'])} to {shown(s['upper'])}",
                str(s["strength"]),
                "-",
                "fixed: frozen when it had enough pairs",
            ]
        )
    return _table(
        [
            ("Row", False),
            ("Kind", False),
            ("Display rating", True),
            ("Bootstrap interval", False),
            ("Strength (10^9 scale)", True),
            ("Root", False),
            ("Note", False),
        ],
        rows,
        f"Ladder at generation {ladder['generation']}: window generations "
        f"{ladder['window']['first_generation']} to {ladder['window']['last_generation']}. {LADDER_LABEL}",
    )


#: Upper bounds (inclusive for the middle band) of the heat classes, in parts per million.
HEAT_EDGES = (250_000, 450_000, 550_000, 750_000)


def _heat_class(ppm: int) -> str:
    low, mid_low, mid_high, high = HEAT_EDGES
    if ppm < low:
        return "hm0"
    if ppm < mid_low:
        return "hm1"
    if ppm <= mid_high:
        return "hm2"
    return "hm3" if ppm <= high else "hm4"


def heatmap(ladder: dict[str, Any]) -> str:
    """Pair scores of each member (row) against each opponent (column): members, snapshots, anchors."""
    cells = {(c["member"], c["opponent"]["kind"], c["opponent"]["id"]): c for c in ladder["cells"]}
    columns = (
        [("member", m["member"]) for m in ladder["members"]]
        + [("snapshot", s["snapshot_id"]) for s in ladder["snapshots"]]
        + [("anchor", a["anchor_id"]) for a in ladder["anchors"]]
    )
    head = "".join(
        f'<th scope="col">{esc(i)} <span class="muted">{k}</span></th>' for k, i in columns
    )
    body = ""
    for m in ladder["members"]:
        row = f'<th scope="row">{esc(m["member"])}</th>'
        for kind, ident in columns:
            if kind == "member" and ident == m["member"]:
                row += '<td class="hmx">self</td>'
                continue
            cell = cells.get((m["member"], kind, ident))
            if cell is None:
                row += '<td class="hmx">no pairs</td>'
                continue
            mean = cell["score_sum_ppm"] // cell["pairs"]
            row += f'<td class="{_heat_class(mean)}">{_pct(mean)} <span class="muted">{cell["pairs"]} pairs</span></td>'
        body += f"<tr>{row}</tr>"
    caption = (
        f"Pair scores at generation {ladder['generation']}: the row member's mean pair score "
        "against the column opponent over the rating window (a pair is two games, one from each "
        f"seat). {LADDER_LABEL}"
    )
    return (
        f'<div class="scroll"><table class="heat"><caption class="muted">{esc(caption)}</caption>'
        f'<thead><tr><th scope="col">Member</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _ladder_section(facts: PopulationFacts, tips: Tips) -> str:
    played = [g for g in facts.generations if g.ladder is not None]
    if not played:
        return (
            '<h2>Ladder</h2><p class="muted">No generation has played yet: there is no ladder.</p>'
        )
    xs = [g.number for g in played]
    series = []
    for k, label in enumerate(facts.members):
        points = []
        for g in played:
            assert g.ladder is not None
            m = next(m for m in g.ladder["members"] if m["member"] == label)
            s = m["strength"]
            tip = (
                f"Generation {g.number}, {label}: display rating {shown(s['strength'])}, "
                f"bootstrap interval {shown(s['lower'])} to {shown(s['upper'])}"
            )
            points.append(
                Pt(
                    g.number,
                    float(rating(s["strength"])),
                    tip,
                    f"{shown(s['strength'])} ({shown(s['lower'])} to {shown(s['upper'])})",
                    float(rating(s["lower"])),
                    float(rating(s["upper"])),
                )
            )
        series.append(Series(label, _slot(k), "circle" if k % 2 == 0 else "square", tuple(points)))
    chart = line_chart(
        "Display rating by generation",
        "Each member's display rating with its bootstrap interval; 0 is the reference anchor actor-rule@v1. "
        + LADDER_LABEL,
        xs,
        series,
        tips,
        ref=0.0,
        empty="No ladder yet.",
    )
    blocks = []
    for g in played:
        assert g.ladder is not None
        latest = g is played[-1]
        blocks.append(
            f"<details{' open' if latest else ''}><summary>Generation {g.number}: ladder table and pair-score heatmap</summary>"
            + ladder_table(g.ladder)
            + heatmap(g.ladder)
            + "</details>"
        )
    return (
        "<h2>Ladder</h2>"
        f'<p class="note">{esc(LADDER_LABEL)} {esc(RATING_NOTE)}</p>'
        f'<div class="charts">{chart}</div>' + "".join(reversed(blocks))
    )


# -- cyclic triples -------------------------------------------------------------------------------


def _cycles_section(facts: PopulationFacts, tips: Tips) -> str:
    played = [g for g in facts.generations if g.ladder is not None]
    if not played:
        return ""
    points = tuple(
        Pt(
            g.number,
            float(g.ladder["cyclic_triples"]) if g.ladder else 0.0,
            f"Generation {g.number}: {g.ladder['cyclic_triples'] if g.ladder else 0} cyclic triples",
            str(g.ladder["cyclic_triples"] if g.ladder else 0),
        )
        for g in played
    )
    chart = line_chart(
        "Cyclic triples by generation",
        "Triples of members where A beats B, B beats C and C beats A, each with a one-sided lower bound above even odds.",
        [g.number for g in played],
        [Series("cyclic triples", 1, "circle", points)],
        tips,
    )
    cyclic = [g for g in played if g.ladder and g.ladder["cyclic_triples"] > 0]
    words = (
        "".join(
            f"<p>Generation {g.number}: {g.ladder['cyclic_triples']} cyclic triple"
            f"{'s' if g.ladder['cyclic_triples'] != 1 else ''}: non-transitive at this generation, "
            "so one ordering of these members hides a cycle.</p>"
            for g in cyclic
            if g.ladder
        )
        if cyclic
        else ""
    )
    return (
        "<h2>Cyclic triples</h2>"
        f'<p class="note">{esc(LADDER_LABEL)} A non-zero count means the pairwise results are not '
        "transitive: the ranking above is not a total order of strength.</p>"
        + (words if words else '<p class="muted">No cyclic triple at any generation.</p>')
        + f'<div class="charts">{chart}</div>'
    )


# -- copies, quarantines and the tree -------------------------------------------------------------


def _member_row(g: PopulationGeneration, member: str) -> MemberGeneration | None:
    return next((m for m in g.members if m.member == member), None)


def _health_stats(facts: PopulationFacts, at: int, member: str) -> str:
    g = next(x for x in facts.generations if x.number == at)
    row = _member_row(g, member)
    if row is None or row.train is None:
        return NOT_RECORDED
    battle = row.train.roles["battle"]
    nulls = dict(row.nulls).get("battle", ())
    clip = NOT_FINITE if "clip_fraction" in nulls else f"{battle['clip_fraction']:.3g}"
    kl = NOT_FINITE if "approx_kl" in nulls else f"{battle['approx_kl']:.3g}"
    return f"Battle clip fraction {clip}, approximate KL {kl}, {row.forfeits} of {row.games} games forfeited"


def _evidence(facts: PopulationFacts, e: dict[str, Any]) -> list[str]:
    if e["kind"] == "copy":
        return [
            "copy",
            esc(e["donor"]),
            esc(e["loser"]),
            esc(
                f"{e['pairs']} pairs of the donor against the loser; lower bound {_pct(e['lower_bound_ppm'])} "
                f"above 50.0% plus margin {_pct(e['margin_ppm'])}; weakest anchor score donor {_pct(e['donor_anchor_min_ppm'])}, "
                f"loser {_pct(e['loser_anchor_min_ppm'])}; roots {e['roots_before']} to {e['roots_after']}"
            ),
        ]
    donor = e["donor"] if e["donor"] is not None else "none"
    return [
        "quarantine and replacement",
        esc(f"{donor} (derived: the strongest member not quarantined)"),
        esc(e["member"]),
        esc(f"{HEALTH_CAUSES[e['cause']]}; {_health_stats(facts, e['at'], e['member'])}"),
    ]


def _events_section(facts: PopulationFacts) -> str:
    found = events(facts)
    outcomes = []
    for g in facts.generations:
        if g.ladder is None:
            continue
        first = g.ladder["selections"][0]
        if first["kind"] == "none":
            outcomes.append([str(g.number), "no copy", "-", "-", esc(FAILED[first["failed"]])])
    rows = [[str(e["at"]), *_evidence(facts, e)] for e in found]
    table = _table(
        [
            ("Generation", True),
            ("Event", False),
            ("Donor", False),
            ("Slot replaced", False),
            ("Evidence", False),
        ],
        sorted(rows + outcomes, key=lambda r: int(r[0])),
        "Copies and quarantines with the record's evidence; a generation without a copy states the condition that failed. "
        "A copy is decided in the generation shown and applied at the start of the next one.",
    )
    return (
        "<h2>Copies and quarantines</h2>"
        f'<p class="note">{esc(LADDER_LABEL)} A copy needs the donor to beat the loser by a margin on '
        "the training games of the rating window; it is a training step, not a claim about strength.</p>"
        + table
    )


def _lane_y(index: int) -> float:
    return TTOP + index * LANE + LANE / 2


def copy_tree(facts: PopulationFacts, tips: Tips) -> str:
    """Members as lanes over the generations; copies and replacements as edges; roots as colours."""
    played = [g for g in facts.generations if g.ladder is not None]
    if not played:
        return ""
    numbers = [g.number for g in played]
    height = TTOP + len(facts.members) * LANE + 20
    span = TW - TLEFT - TRIGHT
    step = span / max(len(numbers) - 1, 1)

    def x_of(g: int) -> float:
        return TLEFT + (numbers.index(g) * step if len(numbers) > 1 else span / 2)

    index = {label: i for i, label in enumerate(facts.members)}
    parts = ["<title>Copy tree</title>"]
    for label, i in index.items():
        y = _lane_y(i)
        parts.append(
            f'<line class="lane" x1="{TLEFT}" y1="{y:.1f}" x2="{TW - TRIGHT}" y2="{y:.1f}"/>'
        )
        parts.append(f'<text x="4" y="{y + 4:.1f}">{esc(label)}</text>')
    stride = max(1, -(-len(numbers) // 12))
    for n, number in enumerate(numbers):
        if n % stride == 0:
            parts.append(
                f'<text x="{x_of(number):.1f}" y="{height - 4}" text-anchor="middle">{number}</text>'
            )
    roots_by_gen: dict[int, int] = {}
    for g in played:
        assert g.ladder is not None
        roots = sorted({m["root"] for m in g.ladder["members"]})
        roots_by_gen[g.number] = len(roots)
        for m in g.ladder["members"]:
            slot = _slot(_number(m["root"]))
            parts.append(
                f'<circle class="ring f{slot}" cx="{x_of(g.number):.1f}" cy="{_lane_y(index[m["member"]]):.1f}" r="4">'
                f"<title>{esc(m['member'])} at generation {g.number}: root {esc(m['root'])}</title></circle>"
            )
    for e in events(facts):
        if e["kind"] == "copy":
            donor, loser, dash, what = e["donor"], e["loser"], "", "copy"
        else:
            donor, loser, dash, what = (
                e["donor"],
                e["member"],
                " dash",
                "quarantine and replacement",
            )
        if donor is None:
            continue
        end = e["at"] + 1 if e["at"] + 1 in numbers else e["at"]
        text = f"Generation {e['at']}: {what}, {donor} to {loser}" + (
            "" if end != e["at"] else " (decided in the last generation; applied in the next)"
        )
        tip = tips.add(text)
        parts.append(
            f'<line class="wh l{_slot(_number(donor))}{dash}" x1="{x_of(e["at"]):.1f}" y1="{_lane_y(index[donor]):.1f}" '
            f'x2="{x_of(end):.1f}" y2="{_lane_y(index[loser]):.1f}" tabindex="0" data-t="{tip}"><title>{esc(text)}</title></line>'
        )
    roots = sorted(
        {m["root"] for g in played if g.ladder for m in g.ladder["members"]}, key=_number
    )
    legend = "".join(
        f'<span><i class="sw sw{_slot(_number(r))} ci"></i>root {esc(r)}</span>' for r in roots
    )
    legend += '<span><i class="sw sw9 sq"></i>solid edge: copy</span><span><i class="sw sw9 sq"></i>dashed edge: health replacement</span>'
    caption = (
        "Members are lanes, generations run left to right; a dot is coloured by the generation-0 root "
        "its weights descend from. An edge goes from the donor at the generation a copy was decided to the "
        "slot it replaces at the next generation."
    )
    table = _table(
        [("Generation", True), ("Distinct roots", True), ("Roots by member", False)],
        [
            [
                str(g.number),
                str(roots_by_gen[g.number]),
                esc(", ".join(f"{m['member']}: {m['root']}" for m in g.ladder["members"]))
                if g.ladder
                else "",
            ]
            for g in played
        ],
        "Roots counted per generation, from the members of each ladder record.",
    )
    return (
        "<h2>Copy tree</h2>"
        f'<p class="note">{esc(LADDER_LABEL)}</p>'
        f'<figure><h3>Copy tree</h3><div class="legend">{legend}</div>'
        f'<svg class="chart tree" viewBox="0 0 {TW} {height}" role="group" aria-label="Copy tree: {esc(caption)}">{"".join(parts)}</svg>'
        f'<div class="readout" aria-live="polite"></div><figcaption>{esc(caption)}</figcaption></figure>'
        + table
    )


# -- member settings, forfeits and health ---------------------------------------------------------


def _settings_section(facts: PopulationFacts, tips: Tips) -> str:
    played = [g for g in facts.generations if g.ladder is not None]
    if not played:
        return ""
    rows = []
    for g in played:
        assert g.ladder is not None
        for m in g.ladder["members"]:
            s = m["settings"]
            sh = s["shares"]
            rows.append(
                [
                    f"{g.number} {esc(m['member'])}",
                    str(s["learning_rate_micro"]),
                    str(s["entropy_coef_micro"]),
                    str(s["epochs"]),
                    str(s["clip_micro"]),
                    str(s["target_kl_micro"]) if "target_kl_micro" in s else "not set",
                    esc(
                        f"{_pct(sh['ladder_ppm'])} / {_pct(sh['self_play_ppm'])} / {_pct(sh['past_pool_ppm'])}"
                    ),
                    esc(m["root"]),
                ]
            )
    table = _table(
        [
            ("Generation and member", False),
            ("Learning rate (micro)", True),
            ("Entropy weight (micro)", True),
            ("Epochs", True),
            ("Clip (micro)", True),
            ("Target KL (micro)", True),
            ("Shares ladder / self-play / past pool", False),
            ("Root", False),
        ],
        rows,
        "Each member's effective PPO settings and opponent shares in each generation's ladder record "
        "(the state at the decision, before that generation's copy takes effect).",
    )
    charts = []
    for key, name in (
        ("learning_rate_micro", "Learning rate (micro)"),
        ("entropy_coef_micro", "Entropy weight (micro)"),
    ):
        series = []
        for k, label in enumerate(facts.members):
            pts = []
            for g in played:
                assert g.ladder is not None
                m = next(m for m in g.ladder["members"] if m["member"] == label)
                value = m["settings"][key]
                pts.append(
                    Pt(
                        g.number,
                        float(value),
                        f"Generation {g.number}, {label}: {name} {value}",
                        str(value),
                    )
                )
            series.append(Series(label, _slot(k), "circle" if k % 2 == 0 else "square", tuple(pts)))
        charts.append(
            line_chart(
                name + " by generation",
                "Explore steps change a copied slot's settings within the configured bounds.",
                [g.number for g in played],
                series,
                tips,
            )
        )
    return (
        "<h2>Member PPO settings</h2>"
        f'<p class="note">{esc(LADDER_LABEL)}</p><div class="charts">{"".join(charts)}</div>'
        f"<details><summary>Settings table, every generation</summary>{table}</details>"
    )


def _health_section(facts: PopulationFacts) -> str:
    played = [g for g in facts.generations if g.ladder is not None]
    if not played:
        return ""
    totals: dict[str, dict[str, int]] = {label: {} for label in facts.members}
    games: dict[str, list[int]] = {label: [0, 0, 0] for label in facts.members}
    rows = []
    for g in played:
        for m in g.members:
            games[m.member][0] += m.games
            games[m.member][1] += m.played
            games[m.member][2] += m.forfeits
            for cause, count in m.forfeit_causes:
                totals[m.member][cause] = totals[m.member].get(cause, 0) + count
            clip = kl = ent = NOT_RECORDED
            if m.train is not None:
                nulls = dict(m.nulls).get("battle", ())
                b = m.train.roles["battle"]
                clip = NOT_FINITE if "clip_fraction" in nulls else f"{b['clip_fraction']:.3g}"
                kl = NOT_FINITE if "approx_kl" in nulls else f"{b['approx_kl']:.3g}"
                ent = NOT_FINITE if "entropy" in nulls else f"{b['entropy']:.3g}"
            stopped = {e["member"] for e in (g.ladder or {}).get("selections", [])[1:]}
            rows.append(
                [
                    f"{g.number} {esc(m.member)}",
                    str(m.games),
                    str(m.played),
                    str(m.forfeits),
                    esc(", ".join(f"{c}: {n}" for c, n in m.forfeit_causes) or "none"),
                    clip,
                    kl,
                    ent,
                    "update discarded (no training set)"
                    if m.training_set_hash is None
                    else "update kept",
                    "quarantined" if m.member in stopped else "healthy",
                ]
            )
    summary = _table(
        [
            ("Member", False),
            ("Games", True),
            ("Played", True),
            ("Forfeits", True),
            ("Forfeits by cause", False),
        ],
        [
            [
                esc(label),
                str(games[label][0]),
                str(games[label][1]),
                str(games[label][2]),
                esc(", ".join(f"{c}: {n}" for c, n in sorted(totals[label].items())) or "none"),
            ]
            for label in facts.members
        ],
        "Games and forfeits of each member over every committed generation (the member's own seats; self-play counts its games).",
    )
    detail = _table(
        [
            ("Generation and member", False),
            ("Games", True),
            ("Played", True),
            ("Forfeits", True),
            ("Causes", False),
            ("Battle clip fraction", True),
            ("Battle approximate KL", True),
            ("Battle entropy", True),
            ("Update", False),
            ("Health", False),
        ],
        rows,
        "Health statistics from each member's train-report.json: a quarantined member stores null for a non-finite number, "
        "shown as not finite, never as zero. Diagnostics, not skill.",
    )
    return (
        "<h2>Member forfeits and health</h2>"
        f'<p class="note">{esc(LADDER_LABEL)}</p>{summary}'
        f"<details><summary>Per generation</summary>{detail}</details>"
    )


def _counts_section(facts: PopulationFacts) -> str:
    if not facts.member_counts:
        return ""
    rows = [
        [f"{g} {esc(label)}", *(str(v) for v in values)]
        for g, per in sorted(facts.member_counts.items())
        for label, values in per.items()
    ]
    return (
        "<h2>Per-member counts</h2>"
        '<p class="note">From <code>metrics/members/gNNNNNN.json</code>, outside every manifest and hash: '
        "its SHA-256 is listed under Inputs.</p>"
        + _table(
            [
                ("Generation and member", False),
                ("Games", True),
                ("Decisions", True),
                ("Optimiser steps", True),
            ],
            rows,
            "Games, decisions and optimiser steps of each member in each generation.",
        )
    )


def _pool_section(facts: PopulationFacts) -> str:
    pooled = [g for g in facts.generations if g.pool_snapshots is not None]
    if not pooled:
        return ""
    rows = [
        [
            str(g.number),
            str(len(g.pool_snapshots or ())),
            esc(", ".join(g.pool_snapshots or ()) or "none"),
        ]
        for g in pooled
    ]
    return "<h2>Past pool</h2>" + _table(
        [("Generation", True), ("Snapshots", True), ("Pool the next generation plays", False)],
        rows,
        "Frozen snapshots of the members that the next generation's past-pool games draw from.",
    )


# -- the page -------------------------------------------------------------------------------------


def build_population_report(model: ReportModel) -> str:
    from areneon_ml.report import render  # noqa: PLC0415  (render builds this page for a population)

    facts = model.population
    assert facts is not None
    tips = Tips()
    found = events(facts)
    copies = sum(1 for e in found if e["kind"] == "copy")
    health = len(found) - copies
    latest = next((g for g in reversed(facts.generations) if g.ladder is not None), None)
    head = (
        f"<h1>Population report: {esc(model.run_key)}</h1>"
        f'<p class="banner">{esc(POP_BANNER)}</p>'
        '<div class="grid">'
        f'<div class="stat"><b>{len(model.generations)}</b>generations committed</div>'
        f'<div class="stat"><b>{len(facts.members)}</b>learning members</div>'
        f'<div class="stat"><b>{copies}</b>copies</div>'
        f'<div class="stat"><b>{health}</b>quarantines</div>'
        f'<div class="stat"><b>{len(latest.ladder["snapshots"]) if latest and latest.ladder else NOT_RECORDED}</b>frozen-rated snapshots</div>'
        f'<div class="stat"><b>{esc(model.format_id)}</b>format</div>'
        f'<div class="stat"><b>{esc(model.lineage_id)}</b>lineage</div></div>'
        f'<p class="note">Run config hash <code>{esc(model.config_hash)}</code>. Synthetic, public-lineage runs only. {esc(LADDER_LABEL)}</p>'
    )
    body = "".join(
        [
            head,
            _ladder_section(facts, tips),
            _cycles_section(facts, tips),
            _events_section(facts),
            copy_tree(facts, tips),
            _settings_section(facts, tips),
            _health_section(facts),
            _pool_section(facts),
            _counts_section(facts),
            render._metrics_section(model),
            render._inputs(model),
        ]
    )
    return document(
        f"Population report {model.run_key}",
        body,
        tips.items,
        f"{esc(POP_BANNER)} Built by <code>python -m areneon_ml.loop report</code> from the files listed above; it reads no Episode content.",
        POPULATION_STYLE,
    )
