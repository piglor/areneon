"""The report page (ADR-017 section 4, slice 6): every number comes from ``model.py``.

Every string that comes from data goes through ``esc`` (HTML) or the JSON data block (tips); a link
is built only from a name this module's caller checked against ``[a-z0-9_-]+`` and a file that exists.
The page says on top, and in every progress section, what it is not: a progress view on the fixed
validation plan, never the Training gate, and loss is not skill.
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial

from areneon_ml.inspector.page import esc
from areneon_ml.progress.gap import GAP_LABEL, SAME_MODE_NOTE
from areneon_ml.progress.records import (
    SEARCH_EVIDENCE_NOTE,
    SEARCH_LABEL,
    SEARCH_PRIOR_LABEL,
)
from areneon_ml.report.charts import Pt, Series, Stack, Tips, line_chart, stacked_chart
from areneon_ml.report.dominated import INTERVAL_LABEL as DOMINATED_INTERVAL_LABEL
from areneon_ml.report.dominated import LABEL as DOMINATED_LABEL
from areneon_ml.report.model import (
    METRICS,
    NOT_RECORDED,
    GenerationFacts,
    MatchFacts,
    ReportModel,
    Tally,
)
from areneon_ml.report.move_use import CLASS_ORDER, share_ppm
from areneon_ml.report.move_use import LABEL as MOVE_USE_LABEL
from areneon_ml.report.page import document

BANNER = "Progress view on validation teams. Not the Training gate. Loss is not skill."
CLAIM = (
    "Allowed claim: the frozen pair of generation G scored s against opponent O on these "
    "validation teams under this fixed plan (interval conditional on the plan). Nothing about "
    "the generator's distribution of teams; never stronger, never a rating, never a gate claim."
)
CONDITIONAL = (
    "Wilson 95% intervals on the score, conditional on the fixed validation plan: the same pairs "
    "and seeds in every match, mode and generation. Several generations viewed together are a "
    "sequence of looks, so their intervals are not simultaneous."
)
CONDITIONAL_PAIRS = (
    "Pair-level 95% percentile bootstrap intervals on the score (every pair is played from both "
    "seats, so the pair is the unit), descriptive and conditional on the fixed validation plan: "
    "the same pairs and seeds in every match, mode and generation, no multiplicity control "
    "across matches. Several generations viewed together are a sequence of looks, so their "
    "intervals are not simultaneous."
)
WILSON = "Wilson 95%"
PAIRS = "pair bootstrap 95%, descriptive"


def _interval_name(match: MatchFacts) -> str:
    return PAIRS if match.interval_kind == "pairs" else WILSON


SEEN = "seen, same-lineage comparison: it can cycle"
SEAT_CAPTION = (
    "The candidate alternates seats by game (p1 on even games). Descriptive, no interval."
)
SEAT_CAPTION_PAIRS = (
    "Every pair is played from both seats: the candidate sits in p1 in even games and in p2 in odd "
    "games (a Wilson-only record alternates by game instead). Descriptive, no interval."
)
SAME_MODE_NOTE_SENTENCE = "Anchor and previous-generation rows: " + SAME_MODE_NOTE + "."
ANCHOR_NOTE = (
    "An anchor is a fixed earlier generation's frozen pair, played "
    "from both seats on the same validation pairs and seeds. The paired difference is the pair "
    "score minus one half; above zero means the candidate scored more than the anchor on these "
    "pairs. Descriptive pair-level bootstrap intervals, conditional on the fixed validation plan, "
    "no multiplicity control across anchors and modes: none is a test or a claim. "
    + SAME_MODE_NOTE_SENTENCE
)
VERSUS_NOTE = (
    "A cross-run comparison: another run's frozen pair, played from both seats on this run's "
    "validation pairs and seeds. The paired difference is the pair score minus one half; above "
    "zero means this run's pair scored more than the other run's on these pairs. Descriptive "
    "pair-level bootstrap intervals, conditional on the fixed validation plan: not held-out "
    "evidence, not a gate result, not a test or a claim. The two runs may share their seeded "
    "start (equal initial seeds): the progress command prints the lineage relation, "
    '"independent lineage" or "related lineage: shared start".'
)
GAP_NOTE = (
    "Each row is a scripted baseline or the cross-run head-to-head: argmax and the mode in the "
    "row were played on the same validation pairs, seats and seeds. The gap is the mean over "
    "pairs of the argmax pair score minus the mode's pair score; above zero means argmax scored "
    "more. Derived on display, never stored: descriptive pair-level bootstrap intervals, "
    "conditional on the fixed validation plan, no multiplicity control; not a test, a claim or a "
    "gate. " + SAME_MODE_NOTE_SENTENCE
)
ABLATION_NOTE = (
    "Team Preview ablation: this generation's Battle network plays both scripted baselines "
    "(argmax) with three Previews: its own learned one, a uniformly random one and an earlier "
    "generation's. Every variant plays the same validation pairs on the same seeds from both "
    "seats. A pair scores the mean of its two games. The difference to the learned Preview is "
    "per pair, with its pair-level bootstrap interval: descriptive, conditional on the fixed "
    "validation plan, no multiplicity control; not a test or a claim."
)
NOT_EVALUATED = "not evaluated yet"
MODES = (("sample", 1, "circle"), ("argmax", 2, "square"))
LABELS = {
    "policy_loss": "Policy loss",
    "value_loss": "Value loss",
    "entropy": "Entropy",
    "approx_kl": "Approximate KL",
    "clip_fraction": "Clip fraction",
}


@dataclass(frozen=True)
class Links:
    """Relative links into an inspector output that exists next to the report."""

    games: dict[int, str] = field(default_factory=dict)
    replays: dict[int, tuple[tuple[str, str], ...]] = field(default_factory=dict)


def permille(ppm: int) -> int:
    return (ppm + 500) // 1000


def pct(ppm: int, *, up: bool = False, down: bool = False) -> str:
    """A ppm value as a percentage with one decimal, in integers (rounded outward when asked)."""
    tenths = ppm // 1000 if down else (ppm + 999) // 1000 if up else permille(ppm)
    return f"{tenths // 10}.{tenths % 10}%"


def tenths(total: int, games: int) -> str:
    """``total / games`` to one decimal, in integers."""
    t = (10 * total + games // 2) // games
    return f"{t // 10}.{t % 10}"


def _table(head: list[tuple[str, bool]], rows: Iterable[list[str]], caption: str) -> str:
    """A table; the first column is the row header, numeric columns right-aligned."""
    heads = "".join(
        f'<th scope="col"{' class="n"' if numeric and index else ""}>{esc(name)}</th>'
        for index, (name, numeric) in enumerate(head)
    )
    body = ""
    for row in rows:
        cells = [f'<th scope="row">{esc(row[0])}</th>']
        for (_, numeric), cell in zip(head[1:], row[1:], strict=True):
            cells.append(f'<td class="{"n" if numeric else "txt"}">{cell}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"
    return (
        f'<div class="scroll"><table><caption class="muted">{esc(caption)}</caption>'
        f"<thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _stem(match: MatchFacts) -> str:
    return match.stem or match.match_id.removesuffix("-" + match.mode)


def _mode(match: MatchFacts) -> str:
    """What a score cell says of the play mode (``sample, T 0.5``, ``argmax``)."""
    return match.mode_label or match.mode


OTHER_DATASET_LABEL = (
    "Not the run's own validation split: scores are comparable only with runs measured on the "
    "same dataset."
)


def _evaluation_dataset_note(model: ReportModel) -> str:
    """Which generations were measured on another dataset's validation teams (the eval-dataset option)."""
    other = {
        g: p.evaluation_dataset
        for g, p in model.progress.items()
        if p.evaluation_dataset is not None
    }
    if not other:
        return ""
    groups: dict[str, list[int]] = {}
    for g, found in sorted(other.items()):
        groups.setdefault(found, []).append(g)
    listed = "; ".join(
        f"generation {', '.join(map(str, gs))}: dataset {found}" for found, gs in groups.items()
    )
    return (
        '<p class="note">Evaluation teams: the validation split of another verified dataset than '
        f"the run's own ({esc(listed)}). {esc(OTHER_DATASET_LABEL)}</p>"
    )


def _evaluation_key_note(model: ReportModel) -> str:
    """Which generations keyed their pairs and seeds by an evaluation key (#1177)."""
    groups: dict[str, list[int]] = {}
    for g, p in sorted(model.progress.items()):
        if p.evaluation_key is not None:
            groups.setdefault(p.evaluation_key, []).append(g)
    if not groups:
        return ""
    listed = "; ".join(f"generation {', '.join(map(str, gs))}: {k}" for k, gs in groups.items())
    return (
        '<p class="note">Evaluation key: the validation pairs, seats and battle seeds are keyed '
        f"by it instead of the run key ({esc(listed)}). Runs evaluated with the same key play "
        "the same pairs and seeds.</p>"
    )


def _looks_note(model: ReportModel) -> str:
    if model.looks is None:
        return ""
    return (
        f'<p class="note">Looks at validation results logged for this run: {model.looks}. '
        "The validation teams are used again at every look, so a validation result is a "
        "progress view, never held-out evidence or a strength claim.</p>"
    )


def _progress_section(model: ReportModel, tips: Tips) -> str:  # noqa: PLR0915
    out = ["<h2>Progress evaluation</h2>", f'<p class="banner">{esc(BANNER)}</p>']
    if not model.progress_root:
        out.append(
            f"<p>Progress evaluation: {NOT_EVALUATED}. No progress directory was given, so no "
            "generation has a score here; the training curves below do not replace it.</p>"
        )
        return "".join(out)
    xs = [g.number for g in model.generations]
    done = sorted(model.progress)
    if not done:
        out.append(
            f"<p>Progress evaluation: {NOT_EVALUATED}. The progress directory holds no committed generation.</p>"
        )
        return "".join(out)
    missing = [str(g) for g in xs if g not in model.progress]
    out.append(_looks_note(model) + _evaluation_dataset_note(model) + _evaluation_key_note(model))
    kinds = {m.interval_kind for p in model.progress.values() for m in p.matches}
    notes = "".join(
        f'<p class="note">{esc(text)}</p>'
        for kind, text in (("wilson", CONDITIONAL), ("pairs", CONDITIONAL_PAIRS))
        if kind in kinds
    )
    out.append(f'<p class="claim">{esc(CLAIM)}</p>{notes}')
    anchored = sorted({str(p.anchor_games) for p in model.progress.values() if p.anchor_games})
    anchor_games = f"; games per anchor match: {', '.join(anchored)}" if anchored else ""
    out.append(
        f'<p class="note">Evaluated generations: {esc(", ".join(map(str, done)))}. '
        f"Not evaluated yet: {esc(', '.join(missing) if missing else 'none')}. Games per match: "
        f"{esc(', '.join(sorted({str(p.games_per_match) for p in model.progress.values()})) + anchor_games)}.</p>"
    )
    xs = [x for x in xs if x <= done[-1]]
    stems = sorted(
        {
            _stem(m)
            for p in model.progress.values()
            for m in p.matches
            if m.variant is None and not m.extra_mode
        }
    )
    charts = []
    for stem in stems:
        series = []
        seen_label = None
        for mode, slot, shape in MODES:
            pts = []
            for g in done:
                for m in model.progress[g].matches:
                    if (
                        _stem(m) == stem
                        and m.mode == mode
                        and m.variant is None
                        and not m.extra_mode
                    ):
                        seen_label = m.seen_label or seen_label
                        low, high = pct(m.low_ppm, down=True), pct(m.high_ppm, up=True)
                        text = (
                            f"Generation {g}, {mode} vs {stem}: score {pct(m.score_ppm)} "
                            f"({_interval_name(m)} {low} to {high}, conditional on the fixed validation plan); "
                            f"{m.wins} wins, {m.ties} ties, {m.losses} losses of {m.games} games"
                        )
                        pts.append(
                            Pt(
                                g,
                                m.score_ppm / 1e6,
                                text,
                                f"{pct(m.score_ppm)} ({low} to {high})",
                                m.low_ppm // 1000 / 1000,
                                ((m.high_ppm + 999) // 1000) / 1000,
                            )
                        )
            if pts:
                series.append(Series(mode, slot, shape, tuple(pts)))
        title = f"Score vs {stem}" + (f" ({seen_label})" if seen_label else "")
        charts.append(
            line_chart(
                title,
                "Progress score by generation with its interval, conditional on the fixed validation plan. Dashed line: 50%. Lines join looks, not a trend.",
                xs,
                series,
                tips,
                y_range=(0.0, 1.0),
                y_label=lambda v: f"{round(v * 100)}%",
                ref=0.5,
                missing="not applicable"
                if stem in ("previous", "run") or stem.startswith("anchor-")
                else NOT_EVALUATED,
            )
        )
    out.append(f'<div class="charts">{"".join(charts)}</div>')
    rows = []
    for g in done:
        for m in model.progress[g].matches:
            low, high = pct(m.low_ppm, down=True), pct(m.high_ppm, up=True)
            forfeits = sum(n for _, _, n in m.forfeits)
            rows.append(
                [
                    str(g),
                    esc(m.match_id)
                    + (f' <span class="muted">({esc(m.seen_label)})</span>' if m.seen_label else "")
                    + (f' <span class="muted">({esc(SEARCH_LABEL)})</span>' if m.search else ""),
                    esc(_mode(m)),
                    str(m.games),
                    f"{m.wins} / {m.ties} / {m.losses}",
                    pct(m.score_ppm),
                    f"{low} to {high}",
                    str(forfeits),
                    tenths(m.turns_total, m.games),
                    str(m.turn_cap_ties),
                ]
            )
    out.append("<h3>Matches</h3>")
    out.append(
        _table(
            [
                ("Generation", True),
                ("Match", False),
                ("Mode", False),
                ("Games", True),
                ("Wins / ties / losses", True),
                ("Score", True),
                (
                    "Wilson 95%, conditional on the plan"
                    if kinds == {"wilson"}
                    else "Interval (Wilson per game, or pair bootstrap), conditional on the plan",
                    True,
                ),
                ("Forfeits", True),
                ("Mean turns", True),
                ("Turn-cap ties", True),
            ],
            rows,
            "Each row: the frozen pair of that generation against the opponent, in the mode named, on the validation plan. "
            + SAME_MODE_NOTE_SENTENCE,
        ).replace('<td class="txt">', '<td class="txt">')
    )
    out.append(_gap_section(model, done))
    out.append(_anchor_section(model, done))
    out.append(_damage_aware_section(model, done))
    out.append(_search_section(model, done))
    out.append(_versus_section(model, done))
    out.append("<h3>Score by seat (diagnostic)</h3>")
    out.append(
        _table(
            [
                ("Generation", True),
                ("Match", False),
                ("Candidate on p1: W / T / L", True),
                ("Candidate on p2: W / T / L", True),
            ],
            [
                [str(g), esc(m.match_id), *(f"{w} / {t} / {lo}" for _, w, t, lo in m.seats)]
                for g in done
                for m in model.progress[g].matches
            ],
            SEAT_CAPTION if kinds == {"wilson"} else SEAT_CAPTION_PAIRS,
        )
    )
    causes = [
        [str(g), m.match_id, side, cause, str(n)]
        for g in done
        for m in model.progress[g].matches
        for side, cause, n in m.forfeits
    ]
    out.append("<h3>Progress forfeits by cause</h3>")
    if causes:
        out.append(
            _table(
                [
                    ("Generation", True),
                    ("Match", False),
                    ("Side", False),
                    ("Cause", False),
                    ("Count", True),
                ],
                [[r[0], *(esc(c) for c in r[1:4]), r[4]] for r in causes],
                "A forfeit is a scored loss for the failing side; side is candidate or opponent.",
            )
        )
    else:
        out.append('<p class="muted">No forfeit in any evaluated match.</p>')
    out.append(_ablation_section(model, done))
    out.append(_move_use_section(model, done))
    out.append(_dominated_section(model, done))
    return "".join(out)


def _baseline(kind: str, g: GenerationFacts) -> Tally | None:
    return g.baselines.get(kind)


def _avg(t: Tally) -> float | None:
    return None if t.played == 0 else t.turns / t.played


def _training_section(model: ReportModel, tips: Tips) -> str:
    played = [g for g in model.generations if g.number > 0]
    xs = [g.number for g in played]
    out = ["<h2>Training games</h2>"]
    out.append(
        '<p class="note">Games the frozen pair played to train on: self-play and games against the scripted baselines, '
        "sampling at temperature 1. Descriptive counts of the run, not a progress measure.</p>"
    )
    if not played:
        out.append(
            '<p class="muted">The run has no generation with games yet (generation 0 is the seeded pair).</p>'
        )
        return "".join(out)
    sp = [g for g in played if g.self_play is not None and g.self_play.games]
    stacks = [
        Stack("p1 won", 1, {g.number: g.self_play.a for g in sp if g.self_play}),
        Stack("p2 won", 2, {g.number: g.self_play.b for g in sp if g.self_play}),
        Stack("tied", 3, {g.number: g.self_play.ties for g in sp if g.self_play}),
        Stack("forfeited", 8, {g.number: g.self_play.forfeits for g in sp if g.self_play}),
    ]
    charts = [
        stacked_chart(
            "Self-play outcomes",
            "Games per generation by result, the frozen pair against itself (the seats are p1 and p2).",
            [g.number for g in sp],
            stacks,
            tips,
            "No self-play games.",
        )
    ]
    kinds = sorted({k for g in played for k in g.baselines})
    lengths = []
    kinds_list: list[tuple[str, Callable[[GenerationFacts], Tally | None]]] = [
        ("self-play", lambda g: g.self_play)
    ]
    kinds_list += [(k, partial(_baseline, k)) for k in kinds]
    for slot, (name, pick) in enumerate(kinds_list, start=1):
        pts = []
        for g in played:
            tally = pick(g)
            avg = None if tally is None else _avg(tally)
            if tally is not None and avg is not None:
                text = f"Generation {g.number}, {name}: {tenths(tally.turns, tally.played)} turns on average over {tally.played} played games"
                pts.append(Pt(g.number, avg, text, tenths(tally.turns, tally.played)))
        if pts:
            lengths.append(Series(name, slot, "circle" if slot % 2 else "square", tuple(pts)))
    charts.append(
        line_chart(
            "Average game length",
            "Mean turns of the played games per generation, by opponent kind (forfeited games excluded).",
            xs,
            lengths,
            tips,
        )
    )
    out.append(f'<div class="charts">{"".join(charts)}</div>')
    rows = []
    for g in played:
        for name, tally, a, b in [
            ("self-play", g.self_play, "p1 won", "p2 won"),
            *[(k, g.baselines[k], "pair won", "baseline won") for k in sorted(g.baselines)],
        ]:
            if tally is None or not tally.games:
                continue
            rows.append(
                [
                    str(g.number),
                    esc(name),
                    str(tally.games),
                    f"{tally.a} ({a})",
                    f"{tally.b} ({b})",
                    str(tally.ties),
                    str(tally.forfeits),
                    tenths(tally.turns, tally.played) if tally.played else "none played",
                ]
            )
    out.append("<h3>Games by opponent kind</h3>")
    out.append(
        _table(
            [
                ("Generation", True),
                ("Opponent", False),
                ("Games", True),
                ("First side", True),
                ("Second side", True),
                ("Ties", True),
                ("Forfeits", True),
                ("Mean turns", True),
            ],
            rows,
            "Self-play: p1 and p2. Against a scripted baseline: the frozen pair and the baseline.",
        )
    )
    return "".join(out)


def _cause_stacks(played: list[GenerationFacts]) -> list[Stack]:
    names = sorted({c for g in played for c in (g.forfeit_causes or {}) if c != NOT_RECORDED})
    shown, folded = names[:6], names[6:]
    stacks = [
        Stack(name, slot, {g.number: (g.forfeit_causes or {}).get(name, 0) for g in played})
        for slot, name in enumerate(shown, start=1)
    ]
    if folded:
        stacks.append(
            Stack(
                f"other causes ({len(folded)})",
                7,
                {g.number: sum((g.forfeit_causes or {}).get(n, 0) for n in folded) for g in played},
            )
        )
    if any((g.forfeit_causes or {}).get(NOT_RECORDED) for g in played):
        stacks.append(
            Stack(
                "cause not recorded",
                9,
                {g.number: (g.forfeit_causes or {}).get(NOT_RECORDED, 0) for g in played},
            )
        )
    return stacks


def _forfeit_section(model: ReportModel, tips: Tips) -> str:
    played = [g for g in model.generations if g.number > 0]
    out = ["<h2>Training forfeits by cause</h2>"]
    if not played:
        return "".join(out) + '<p class="muted">No generation with games yet.</p>'
    stacks = _cause_stacks(played)
    total = sum(g.forfeits or 0 for g in played)
    out.append(
        '<p class="note">A forfeit is a game one side lost by failing (for example a late or invalid answer). '
        "The cause is the one the loop recorded for that game; where none was recorded it says so.</p>"
    )
    out.append(
        stacked_chart(
            "Forfeits per generation by cause",
            "Training games lost by forfeit, per generation, by recorded cause.",
            [g.number for g in played],
            stacks,
            tips,
            "No training game was forfeited in any generation."
            if total == 0
            else "No forfeit causes.",
        )
    )
    return "".join(out)


def _counts_section(model: ReportModel, tips: Tips) -> str:
    played = [g for g in model.generations if g.number > 0]
    xs = [g.number for g in played]
    if not played:
        return ""

    def one(
        title: str, caption: str, pick: Callable[[GenerationFacts], int | None], name: str
    ) -> str:
        pts = [
            Pt(g.number, float(v), f"Generation {g.number}: {v} {name}", str(v))
            for g in played
            if (v := pick(g)) is not None
        ]
        return line_chart(
            title,
            caption,
            xs,
            [Series(name, 1, "circle", tuple(pts))],
            tips,
            y_range=(0.0, float(max((p.y for p in pts), default=1.0))),
        )

    steps = []
    for role, slot, shape in (("battle", 1, "circle"), ("preview", 2, "square")):
        pts = [
            Pt(
                g.number,
                float(g.train.roles[role]["steps"]),
                f"Generation {g.number}, {role}: {int(g.train.roles[role]['steps'])} steps",
                str(int(g.train.roles[role]["steps"])),
            )
            for g in played
            if g.train is not None
        ]
        steps.append(Series(role, slot, shape, tuple(pts)))
    out = ["<h2>Games and examples per generation</h2>"]
    out.append(
        '<p class="note">Steps are the decisions the learner trained on in that generation. '
        "Wall-clock time and throughput are in the next section.</p>"
    )
    out.append(
        '<div class="charts">'
        + one("Games per generation", "Games played or forfeited.", lambda g: g.games, "games")
        + one(
            "Examples per generation",
            "Training examples written.",
            lambda g: g.examples,
            "examples",
        )
        + line_chart(
            "Learner steps per generation",
            "Decisions trained on, by network.",
            xs,
            steps,
            tips,
            empty="No train-report.json was recorded.",
        )
        + "</div>"
    )
    return "".join(out)


def _seconds(ms: int) -> str:
    return f"{ms // 1000}.{ms % 1000:03d} s"


def _memory(value: int | None) -> str:
    if value is None:
        return "none (CPU learner)"
    return f"{value // 2**20} MiB"


def _reserved(m: dict[str, object]) -> str:
    value = m.get("peak_accelerator_reserved_bytes")
    return NOT_RECORDED if not isinstance(value, int) else _memory(value)


def _resident(m: dict[str, object]) -> str:
    """The driver, game process and policy worker peaks that were measured, in MiB."""
    peaks = m.get("peak_rss_bytes")
    if not isinstance(peaks, dict) or not peaks:
        return NOT_RECORDED
    names = (("driver", "driver"), ("game_process", "game process"), ("policy_worker", "worker"))
    return ", ".join(f"{label} {peaks[key] // 2**20} MiB" for key, label in names if key in peaks)


def _metrics_section(model: ReportModel) -> str:
    rows = []
    for g in model.generations:
        m = g.metrics
        if m is None:
            rows.append([str(g.number), *[NOT_RECORDED] * 13])
            continue
        p, c, r = m["phase_ms"], m["counts"], m["per_hour"]
        rows.append(
            [
                str(g.number),
                *(_seconds(p[name]) for name in ("play", "load", "train", "write", "seal")),
                str(c["games"]),
                str(r["games"]),
                str(r["decisions"]),
                str(r["optimiser_steps"]),
                _memory(m["peak_accelerator_memory_bytes"]),
                _reserved(m),
                _resident(m),
                f"{m['device']}, {m['threads']} thread{'s' if m['threads'] != 1 else ''}",
            ]
        )
    return (
        "<h2>Wall-clock time and throughput</h2>"
        '<p class="note">From <code>metrics/gNNNNNN.json</code>, written after each generation '
        "committed and outside every manifest and hash: it has no tamper evidence of its own, and "
        "its SHA-256 is listed under Inputs. A generation whose file is missing (a crash between "
        "its commit and the write) says not recorded; no resume writes it again. Games per "
        "hour are over the play phase, decisions and optimiser steps per hour over the train phase. "
        "Memory figures are the peaks of the generation (resident sets are Linux high-water marks; the "
        "driver's covers the whole run so far); they are optional and say not recorded when absent. "
        "Timings differ on every run and say nothing about skill.</p>"
        + _table(
            [
                ("Generation", True),
                ("Play", True),
                ("Load", True),
                ("Train", True),
                ("Write", True),
                ("Seal", True),
                ("Games", True),
                ("Games per hour", True),
                ("Decisions per hour", True),
                ("Optimiser steps per hour", True),
                ("Peak accelerator memory", True),
                ("Peak reserved accelerator memory", True),
                ("Peak resident set", False),
                ("Learner", False),
            ],
            rows,
            "Wall-clock time per phase of each committed generation, and the throughput that follows.",
        )
    )


def _diagnostics_section(model: ReportModel, tips: Tips) -> str:
    played = [g for g in model.generations if g.number > 0]
    xs = [g.number for g in played]
    out = [
        "<h2>Training diagnostics</h2>",
        '<p class="note">What the learner reports about its own optimisation. These show whether training is '
        "stable, not whether the policy is any good: loss is not skill.</p>",
    ]
    charts = []
    for metric in METRICS:
        series = []
        for role, slot, shape in (("battle", 1, "circle"), ("preview", 2, "square")):
            pts = [
                Pt(
                    g.number,
                    g.train.roles[role][metric],
                    f"Generation {g.number}, {role} {LABELS[metric].lower()}: {g.train.roles[role][metric]:.4g}",
                    f"{g.train.roles[role][metric]:.4g}",
                )
                for g in played
                if g.train is not None
            ]
            if pts:
                series.append(Series(role, slot, shape, tuple(pts)))
        charts.append(
            line_chart(
                LABELS[metric],
                "Per generation, from train-report.json.",
                xs,
                series,
                tips,
                empty="No train-report.json was recorded.",
            )
        )
    anchored = [g for g in played if g.train is not None and g.train.anchors]
    if anchored:
        series = []
        for role, slot, shape in (("battle", 1, "circle"), ("preview", 2, "square")):
            pts = [
                Pt(
                    g.number,
                    g.train.anchors[role][0],
                    f"Generation {g.number}, {role} anchor KL: {g.train.anchors[role][0]:.4g} "
                    f"(beta {g.train.anchors[role][1] / 1_000_000:.4g})",
                    f"{g.train.anchors[role][0]:.4g}",
                )
                for g in anchored
                if g.train is not None and role in g.train.anchors
            ]
            if pts:
                series.append(Series(role, slot, shape, tuple(pts)))
        charts.append(
            line_chart(
                "Anchor KL",
                "KL from the warm-start pair, per generation, from train-report.json; the "
                "weight beta used is in each point's tip.",
                xs,
                series,
                tips,
                empty="No anchor was recorded.",
            )
        )
    valued = [g for g in played if g.train is not None and g.train.value_heads]
    if valued:
        series = []
        for role, slot, shape in (("battle", 1, "circle"), ("preview", 2, "square")):
            pts = [
                Pt(
                    g.number,
                    g.train.value_heads[role][0],
                    f"Generation {g.number}, {role} explained variance: "
                    f"{g.train.value_heads[role][0]:.4g}",
                    f"{g.train.value_heads[role][0]:.4g}",
                )
                for g in valued
                if g.train is not None and role in g.train.value_heads
            ]
            if pts:
                series.append(Series(role, slot, shape, tuple(pts)))
        charts.append(
            line_chart(
                "Explained variance of the value head",
                "1 - Var(return - V) / Var(return) over the returns the learner trains on, from "
                "train-report.json; 0 is a constant predictor and a return with no variance. "
                "A diagnostic, not skill.",
                xs,
                series,
                tips,
                empty="No value diagnostics were recorded.",
            )
        )
    out.append(f'<div class="charts">{"".join(charts)}</div>')
    if valued:
        out.append(_calibration_tables(valued[-1]))
    return "".join(out)


def _calibration_tables(g: GenerationFacts) -> str:
    assert g.train is not None
    out = []
    for role in ("battle", "preview"):
        if role not in g.train.value_heads:
            continue
        rows = [
            [
                f"{i / 10:.1f} to {(i + 1) / 10:.1f}",
                str(count),
                f"{mean_value:.4f}" if count else "-",
                f"{mean_score:.4f}" if count else "-",
            ]
            for i, (count, mean_value, mean_score) in enumerate(g.train.value_heads[role][1])
        ]
        out.append(
            f"<h3>Value calibration, {esc(role)}, generation {g.number}</h3>"
            + _table(
                [
                    ("Predicted value", False),
                    ("Decisions", True),
                    ("Mean predicted value", True),
                    ("Mean final score", True),
                ],
                rows,
                "Diagnostic, not skill: the frozen-weight value of each decision against the "
                "final game score (win 1, tie 0.5, loss 0). The first and last bins also hold "
                "values below 0 and above 1.",
            )
        )
    return "".join(out)


def _score_cell(model: ReportModel, g: int) -> str:
    if g not in model.progress:
        return NOT_EVALUATED
    parts = [f"{_stem(m)} {_mode(m)} {pct(m.score_ppm)}" for m in model.progress[g].matches]
    looks = "" if model.looks is None else f" (looks logged: {model.looks})"
    return esc("; ".join(parts) + looks)


def _links_cell(g: int, links: Links) -> str:
    parts = []
    if g in links.games:
        parts.append(f'<a href="{esc(links.games[g])}">games</a>')
    for label, href in links.replays.get(g, ()):
        parts.append(f'<a href="{esc(href)}">{esc(label)}</a>')
    return " ".join(parts) if parts else '<span class="muted">none</span>'


def _comparison(model: ReportModel, links: Links) -> str:
    rows = []
    for g in model.generations:

        def cell(v: int | None) -> str:
            return NOT_RECORDED if v is None else str(v)

        rows.append(
            [
                str(g.number),
                cell(g.games),
                cell(g.played),
                cell(g.forfeits),
                cell(g.episodes),
                cell(g.examples),
                _score_cell(model, g.number) if model.progress_root else NOT_EVALUATED,
                _links_cell(g.number, links),
            ]
        )
    return "<h2>Generations</h2>" + _table(
        [
            ("Generation", True),
            ("Games", True),
            ("Played", True),
            ("Forfeits", True),
            ("Episodes", True),
            ("Examples", True),
            ("Progress scores (conditional on the plan)", False),
            ("Inspector", False),
        ],
        rows,
        "One row per committed generation. Generation 0 is the seeded pair and plays no training games.",
    )


def _inputs(model: ReportModel) -> str:
    rows = "".join(
        f'<tr><th scope="row" class="m">{esc(path)}</th><td class="m">{esc(digest)}</td></tr>'
        for path, digest in model.inputs.items()
    )
    return (
        f'<h2>Inputs</h2><p class="note">The SHA-256 of every file this page was built from ({len(model.inputs)} files). '
        "The same files give the same page, byte for byte.</p>"
        '<details><summary>Show the input hashes</summary><div class="scroll"><table>'
        f'<thead><tr><th scope="col">File</th><th scope="col">SHA-256</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    )


def build_report(model: ReportModel, links: Links | None = None) -> str:
    links = links or Links()
    tips = Tips()
    evaluated = len(model.progress)
    head = (
        f"<h1>Run report: {esc(model.run_key)}</h1>"
        f'<p class="banner">{esc(BANNER)}</p>'
        '<div class="grid">'
        f'<div class="stat"><b>{len(model.generations)}</b>generations committed</div>'
        f'<div class="stat"><b>{evaluated if model.progress_root else NOT_EVALUATED}</b>generations with a progress evaluation</div>'
        f'<div class="stat"><b>{esc(model.format_id)}</b>format</div>'
        f'<div class="stat"><b>{esc(model.lineage_id)}</b>lineage</div></div>'
        f'<p class="note">Run config hash <code>{esc(model.config_hash)}</code>. Synthetic, public-lineage runs only.</p>'
        + (
            ""
            if model.family is None
            else f'<p class="note">Policy family: {esc(model.family)}. Its decision latency includes the probe; '
            "a comparison with damage-aware@v1 is between two users of the same probe.</p>"
        )
    )
    body = "".join(
        [
            head,
            _progress_section(model, tips),
            _training_section(model, tips),
            _forfeit_section(model, tips),
            _counts_section(model, tips),
            _metrics_section(model),
            _diagnostics_section(model, tips),
            _comparison(model, links),
            _inputs(model),
        ]
    )
    return document(
        f"Run report {model.run_key}",
        body,
        tips.items,
        f"{esc(BANNER)} Built by <code>python -m areneon_ml.loop report</code> from the files listed above; it reads no Episode content beyond counts.",
    )


def signed_pct(ppm: int, *, up: bool = False, down: bool = False) -> str:
    """A signed ppm value as a percentage with one decimal, in integers (outward when asked)."""
    tenths = ppm // 1000 if down else -((-ppm) // 1000) if up else (ppm + 500) // 1000
    sign = "-" if tenths < 0 else "+" if tenths > 0 else ""
    return f"{sign}{abs(tenths) // 10}.{abs(tenths) % 10}%"


def _damage_aware_section(model: ReportModel, done: list[int]) -> str:
    """The damage-aware baseline's counts per match (ADR-018 sections 4 and 5), summed over its
    games and shown as the record states them: a baseline drifting towards actor-rule shows as
    unsupported queries and fallbacks."""
    rows = [
        [
            str(g),
            esc(m.match_id),
            str(m.games),
            str(m.baseline["counts"]["decisions"]),
            str(m.baseline["counts"]["scored_decisions"]),
            str(m.baseline["counts"]["queries"]),
            str(m.baseline["counts"]["probe_runs"]),
            str(m.baseline["counts"]["unsupported_queries"]),
            str(m.baseline["counts"]["reader_fallbacks"]),
            str(m.baseline["counts"]["assumed_set_fallbacks"]),
            f'<span class="m">{esc(m.baseline["identity_hash"][:16])}</span>',
        ]
        for g in done
        for m in model.progress[g].matches
        if m.baseline is not None
    ]
    if not rows:
        return ""
    return "<h3>Damage-aware baseline counts</h3>" + _table(
        [
            ("Generation", True),
            ("Match", False),
            ("Games", True),
            ("Decisions", True),
            ("Scored", True),
            ("Queries", True),
            ("Probe runs", True),
            ("Unsupported", True),
            ("Reader fallbacks", True),
            ("Assumed-set fallbacks", True),
            ("Identity (first 16 hex)", False),
        ],
        rows,
        "Summed over each match's games, as the record states them. A seen scripted baseline: "
        "its rows are never held-out evidence. Unsupported queries scored 0 and fallbacks "
        "played the actor-rule candidate; games with another identity hash are another baseline.",
    )


def _search_section(model: ReportModel, done: list[int]) -> str:
    """The decision-time search's counts per match (ADR-026 section 5), summed over its games, and
    the no-search control's beside them (searched decisions 0). Labelled, never held-out evidence;
    wall time is not recorded."""
    rows = []
    for g in done:
        for m in model.progress[g].matches:
            for who, block in (("search", m.search), ("control", m.opponent_search)):
                if block is None:
                    continue
                c = block["counts"]
                rows.append(
                    [
                        str(g),
                        esc(m.match_id),
                        who,
                        str(c["decisions"]),
                        str(c["searched_decisions"]),
                        str(sum(c["fallbacks"].values())),
                        str(c["worlds_built"]),
                        str(c["worlds_refused"]),
                        str(c["worlds_scored"]),
                        str(c["turns"]),
                        str(c["leaf_queries"]),
                        f'<span class="m">{esc(block["identity_hash"][:16])}</span>',
                    ]
                )
    if not rows:
        return ""
    return (
        f"<h3>Decision-time search counts</h3>"
        f'<p class="note">{esc(SEARCH_LABEL)}. Prior: {esc(SEARCH_PRIOR_LABEL)}. '
        f"{esc(SEARCH_EVIDENCE_NOTE.capitalize())}. The control row is the same frozen pair played "
        "without search; its decisions and structural fallbacks are counted, nothing else.</p>"
        + _table(
            [
                ("Generation", True),
                ("Match", False),
                ("Side", False),
                ("Decisions", True),
                ("Searched", True),
                ("Fallbacks", True),
                ("Worlds built", True),
                ("Worlds refused", True),
                ("Worlds scored", True),
                ("Turns simulated", True),
                ("Leaf queries", True),
                ("Search identity (first 16 hex)", False),
            ],
            rows,
            "Summed over each match's games, as the record states them. Fallbacks are decisions "
            "the pair played itself (Team Preview, forced switches, fewer than two candidates, "
            "too few valid worlds, a belief refusal); wall time is printed by the command, "
            "never stored.",
        )
    )


def _gap_section(model: ReportModel, done: list[int]) -> str:
    """The play-mode gap (ADR-023 section 2, #1042): argmax minus each other mode on the same
    pairs, per scripted baseline and for the cross-run head-to-head. Derived on display."""
    rows = []
    unavailable = []
    for g in done:
        gap = model.progress[g].play_mode_gap
        if "status" in gap:
            unavailable.append(f"generation {g}: {gap['status']}")
        for r in gap.get("rows", []):
            rows.append(
                [
                    str(g),
                    esc(r["opponent"]),
                    esc(r["mode"]),
                    str(r["pairs"]),
                    pct(r["argmax_ppm"]),
                    pct(r["mode_ppm"]),
                    _signed_cell(r["mean_ppm"], r["low_ppm"], r["high_ppm"]),
                    f"{r['better']} / {r['equal']} / {r['worse']}",
                ]
            )
    if not rows and not unavailable:
        return ""
    out = [f'<h3>Play-mode gap (argmax minus mode)</h3><p class="note">{esc(GAP_LABEL)}.</p>']
    if rows:
        out.append(
            _table(
                [
                    ("Generation", True),
                    ("Opponent", False),
                    ("Mode", False),
                    ("Pairs", True),
                    ("Argmax score", True),
                    ("Mode score", True),
                    (f"Gap, argmax minus mode ({PAIRS})", True),
                    ("Pairs argmax better / equal / worse", True),
                ],
                rows,
                GAP_NOTE,
            )
        )
    if unavailable:
        out.append(f'<p class="muted">{esc("; ".join(unavailable))}.</p>')
    return "".join(out)


def _anchor_section(model: ReportModel, done: list[int]) -> str:
    rows = [
        [
            str(g),
            esc(m.match_id) + f' <span class="muted">({esc(SEEN)})</span>',
            str(m.games),
            str(m.pairs),
            f"{m.wins} / {m.ties} / {m.losses}",
            pct(m.score_ppm),
            signed_pct(m.diff[0]),
            f"{signed_pct(m.diff[1], down=True)} to {signed_pct(m.diff[2], up=True)}",
        ]
        for g in done
        for m in model.progress[g].matches
        if m.anchor_generation is not None and m.diff is not None
    ]
    if not rows:
        return ""
    return "<h3>Anchor comparisons (paired differences)</h3>" + _table(
        [
            ("Generation", True),
            ("Match", False),
            ("Games", True),
            ("Pairs", True),
            ("Wins / ties / losses", True),
            ("Pair score", True),
            ("Difference to one half", True),
            (f"{PAIRS}, conditional on the plan", True),
        ],
        rows,
        ANCHOR_NOTE,
    )


def _versus_section(model: ReportModel, done: list[int]) -> str:
    rows = [
        [
            str(g),
            esc(m.match_id) + f' <span class="muted">({esc(m.seen_label or "")})</span>',
            esc(f"{m.versus[0]} g{m.versus[1]}") if m.versus else "",
            str(m.games),
            str(m.pairs),
            f"{m.wins} / {m.ties} / {m.losses}",
            pct(m.score_ppm),
            signed_pct(m.diff[0]),
            f"{signed_pct(m.diff[1], down=True)} to {signed_pct(m.diff[2], up=True)}",
        ]
        for g in done
        for m in model.progress[g].matches
        if m.versus is not None and m.diff is not None
    ]
    if not rows:
        return ""
    return "<h3>Head to head against another run (cross-run comparison)</h3>" + _table(
        [
            ("Generation", True),
            ("Match", False),
            ("Other run", False),
            ("Games", True),
            ("Pairs", True),
            ("Wins / ties / losses", True),
            ("Pair score", True),
            ("Difference to one half", True),
            (f"{PAIRS}, conditional on the plan", True),
        ],
        rows,
        VERSUS_NOTE,
    )


def _signed_cell(mean: int, low: int, high: int) -> str:
    return f"{signed_pct(mean)} ({signed_pct(low, down=True)} to {signed_pct(high, up=True)})"


def _ablation_section(model: ReportModel, done: list[int]) -> str:
    out = []
    for g in done:
        ablation = model.progress[g].ablation
        if not ablation:
            continue
        rows = []
        for baseline, variants in sorted(ablation["baselines"].items()):
            for variant, row in sorted(variants.items()):
                diff = row.get("minus_learned")
                rows.append(
                    [
                        str(g),
                        esc(baseline),
                        esc(variant),
                        str(row["pairs"]),
                        f"{pct(row['score_ppm'])} ({pct(row['low_ppm'], down=True)} to {pct(row['high_ppm'], up=True)})",
                        "-"
                        if diff is None
                        else _signed_cell(diff["mean_ppm"], diff["low_ppm"], diff["high_ppm"]),
                        "-"
                        if diff is None
                        else f"{diff['better']} / {diff['equal']} / {diff['worse']}",
                    ]
                )
        out.append(
            _table(
                [
                    ("Generation", True),
                    ("Baseline", False),
                    ("Preview", False),
                    ("Pairs", True),
                    (f"Mean pair score ({PAIRS})", True),
                    ("Difference to the learned Preview", True),
                    ("Pairs better / equal / worse", True),
                ],
                rows,
                ABLATION_NOTE,
            )
        )
    return ("<h3>Team Preview ablation</h3>" + "".join(out)) if out else ""


CLASS_NAMES = {
    "status": "Status move",
    "lowers_target_stat": "Lowers a foe's stat",
    "raises_own_stat": "Raises own or ally stat",
    "side_condition": "Side condition",
    "field_condition": "Weather, terrain, field",
    "protection": "Protect family",
    "switch": "Switch",
}
MOVE_USE_NOTE = (
    "Counted over the stored games only (the first games of each match), one count per slot "
    "of a move request: a slot action is a move or a voluntary switch; a forced replacement is not "
    "counted. A move may be in several classes. Classes come from the move-class table, a sibling "
    "of the engine-data export bound to it by hash. Unresolved: slot actions whose move could not "
    "be named from the log and the team (a locked, charging or recharging move, or a copied "
    "move set); they are left out of every other count."
)


def _fraction(part: int, whole: int) -> str:
    share = share_ppm(part, whole)
    return f"{part} ({'-' if share is None else pct(share)})"


def _move_use_section(model: ReportModel, done: list[int]) -> str:
    """The move-use diagnostic per match and seat (#1164): descriptive counts, never a score."""
    shown = [(g, used) for g in done if (used := model.progress[g].move_use) is not None]
    if not shown:
        return ""
    heads = [(CLASS_NAMES[name], True) for name in CLASS_ORDER]
    ident = [
        ("Generation", True),
        ("Match", False),
        ("Seat", False),
        ("Policy", False),
        ("Slot actions", True),
        ("Unresolved", True),
    ]
    chosen, passed = [], []
    for g, used in shown:
        for match in used["matches"]:
            for role in ("candidate", "opponent"):
                seat = match[role]
                reference = role == "opponent" and match["reference"]
                label = role + (" (damage-aware reference)" if reference else "")
                prefix = [
                    str(g),
                    esc(match["match_id"]),
                    esc(label),
                    f'<span class="m">{esc(seat["policy_id"][:24])}</span>',
                    str(seat["slot_actions"]),
                    str(seat["unresolved"]),
                ]
                total = seat["slot_actions"]
                chosen.append(
                    [
                        *prefix,
                        *(_fraction(seat["classes"][c]["actions"], total) for c in CLASS_ORDER),
                    ]
                )
                passed.append(
                    [
                        *prefix,
                        *(
                            f"{seat['classes'][c]['passed']} of {seat['classes'][c]['offered']}"
                            + (
                                ""
                                if seat["classes"][c]["offered"] == 0
                                else f" ({pct(seat['classes'][c]['passed'] * 1_000_000 // seat['classes'][c]['offered'])})"
                            )
                            for c in CLASS_ORDER
                        ),
                    ]
                )
    first = shown[0][1]
    return (
        "<h3>Move-use diagnostic</h3>"
        f'<p class="banner">{esc(MOVE_USE_LABEL)}</p>'
        f'<p class="note">Move-class table <code>{esc(first["class_table_sha256"])}</code> on engine-data '
        f"<code>{esc(first['engine_data_hash'])}</code>. A seat that plays only for immediate damage "
        "and one that also sets up, controls speed or protects differ here; neither is ranked.</p>"
        + _table(
            [*ident, *heads],
            chosen,
            "Slot actions in each class: count and share of the seat's resolved slot actions. "
            + MOVE_USE_NOTE,
        )
        + _table(
            [*ident, *heads],
            passed,
            "Slot decisions where a move of the class (or a switch) was among the candidates and was "
            "not chosen: not chosen of offered. " + MOVE_USE_NOTE,
        )
    )


DOMINATED_NOTE = (
    "Over the stored games only. Each damaging-move slot action gets a static score: base power "
    "times the type-chart cell values for the target's shown types (and a same-type factor), a "
    "spread move summed over the alive foes with a spread factor; it is dominated when below the "
    "threshold share of the best score among the Pokemon's enabled damaging moves on any target, "
    "and immune when its score is 0 while that best is positive (immune actions are also "
    "dominated). The score ignores abilities, items, stats, boosts, weather, accuracy and "
    "variable power. Scored: actions with a score; the rates divide by it. Left out of the rates: "
    "status moves, moves aimed at the ally, moves with no base power, actions the table cannot "
    "score, and unresolved slots."
)


def _rate(part: int, interval: dict[str, int] | None) -> str:
    if interval is None:
        return f"{part} (-)"
    return (
        f"{part} ({pct(interval['ppm'])}; {pct(interval['low_ppm'], down=True)} to "
        f"{pct(interval['high_ppm'], up=True)})"
    )


def _dominated_section(model: ReportModel, done: list[int]) -> str:
    """The statically dominated damaging-move rate (#1025): an approximate secondary diagnostic."""
    shown = [(g, used) for g in done if (used := model.progress[g].dominated) is not None]
    if not shown:
        return ""
    head = [
        ("Generation", True),
        ("Match", False),
        ("Seat", False),
        ("Policy", False),
        ("Scored", True),
        ("Dominated", True),
        ("Immune target", True),
        ("Status", True),
        ("Ally target", True),
        ("No base power", True),
        ("Unscored", True),
        ("Unresolved", True),
    ]
    rows = []
    for g, used in shown:
        for match in used["matches"]:
            for role in ("candidate", "opponent"):
                seat = match[role]
                reference = role == "opponent" and match["reference"]
                label = role + (" (damage-aware reference)" if reference else "")
                rows.append(
                    [
                        str(g),
                        esc(match["match_id"]),
                        esc(label),
                        f'<span class="m">{esc(seat["policy_id"][:24])}</span>',
                        str(seat["scored"]),
                        _rate(seat["dominated"], seat["dominated_interval"]),
                        _rate(seat["immune"], seat["immune_interval"]),
                        str(seat["status"]),
                        str(seat["ally_target"]),
                        str(seat["no_power"]),
                        str(seat["unscored"]),
                        str(seat["unresolved"]),
                    ]
                )
    first = shown[0][1]
    return (
        "<h3>Statically dominated damaging moves</h3>"
        f'<p class="banner">{esc(DOMINATED_LABEL)}</p>'
        f'<p class="note">Config record <code>{esc(first["config_hash"])}</code> on engine-data '
        f"<code>{esc(first['engine_data_hash'])}</code>. A secondary diagnostic that decides "
        "nothing; compare a seat with another seat on the same games, not with zero.</p>"
        + _table(
            head,
            rows,
            f"Share of scored actions, with a {DOMINATED_INTERVAL_LABEL}. " + DOMINATED_NOTE,
        )
    )
