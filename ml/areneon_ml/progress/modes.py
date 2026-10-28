"""The play modes of the progress evaluation (ADR-023 section 1, ticket #1044).

``argmax`` and ``sample`` (temperature 1, no truncation) are always played. An EXTRA mode is a
sample at a declared temperature and/or probability threshold, named by one closed grammar::

    mode  := "argmax" | "sample" | "sample" [":t" ppm] [":th" ppm]        (ppm: 1 to 999999)
    match := <opponent>-sample [-t<ppm>] [-th<ppm>]

so no mode string or match id is free text. The explicit default temperature (1,000,000) is not a
mode: the policy id never writes it. The TS side (``progress-modes.ts``) holds the same grammar.
All integers; a label shows a ppm as a decimal by digit manipulation, never a float.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

MAX_PPM = 999_999
_MODE = re.compile(r"sample(?::t([1-9][0-9]{0,5}))?(?::th([1-9][0-9]{0,5}))?")
#: A sample match id: ``<opponent>-sample`` and the play suffix, if any.
SAMPLE_ID = re.compile(r"(?P<stem>.+)-sample(?P<suffix>(?:-t[0-9]+)?(?:-th[0-9]+)?)")
_DIGITS = 6


class ModeError(ValueError):
    """A play mode outside the grammar, or listed twice."""


@dataclass(frozen=True, order=True)
class ExtraMode:
    """A sample with a declared temperature and/or threshold (at least one)."""

    temperature_ppm: int | None = None
    min_ppm: int | None = None

    def __post_init__(self) -> None:
        if self.temperature_ppm is None and self.min_ppm is None:
            raise ModeError("an extra play mode sets a temperature or a threshold")
        for value in (self.temperature_ppm, self.min_ppm):
            if value is not None and not 1 <= value <= MAX_PPM:
                raise ModeError(f"a play mode's ppm is 1 to {MAX_PPM}")

    @property
    def truncation(self) -> dict[str, Any] | None:
        return None if self.min_ppm is None else {"kind": "threshold", "min_ppm": self.min_ppm}

    @property
    def name(self) -> str:
        """The command-line form, ``sample:t500000:th30000``."""
        t = "" if self.temperature_ppm is None else f":t{self.temperature_ppm}"
        th = "" if self.min_ppm is None else f":th{self.min_ppm}"
        return f"sample{t}{th}"

    @property
    def suffix(self) -> str:
        """The match-id suffix, ``-t500000-th30000``."""
        t = "" if self.temperature_ppm is None else f"-t{self.temperature_ppm}"
        th = "" if self.min_ppm is None else f"-th{self.min_ppm}"
        return t + th


def parse_mode(text: str) -> ExtraMode | None:
    """The extra mode a string names; None for ``argmax`` and plain ``sample`` (always played)."""
    if text in ("argmax", "sample"):
        return None
    found = _MODE.fullmatch(text)
    if found is None:
        raise ModeError(
            "unknown play mode: argmax, sample, sample:t<ppm>, sample:th<ppm> or "
            f"sample:t<ppm>:th<ppm> (ppm 1 to {MAX_PPM})"
        )
    t, th = found.groups()
    return ExtraMode(None if t is None else int(t), None if th is None else int(th))


def parse_modes(texts: Iterable[str]) -> tuple[ExtraMode, ...]:
    """A ``--modes`` list: the extra modes it names, each once, in the order given."""
    seen: set[str] = set()
    out: list[ExtraMode] = []
    for text in texts:
        if text in seen:
            raise ModeError(f"play mode {text!r} is listed twice")
        seen.add(text)
        mode = parse_mode(text)
        if mode is not None:
            out.append(mode)
    return tuple(out)


def match_extra(match: dict[str, Any]) -> ExtraMode | None:
    """The extra mode a match record was played in, or None for argmax and plain sample."""
    truncation = match.get("truncation")
    t = match.get("temperature_ppm")
    if t is None and truncation is None:
        return None
    return ExtraMode(t, None if truncation is None else truncation["min_ppm"])


def ppm_decimal(ppm: int) -> str:
    """A ppm below one as a decimal, by digits: 500000 is ``0.5``, 30000 is ``0.03``."""
    return "0." + f"{ppm:0{_DIGITS}d}".rstrip("0")


def mode_label(match: dict[str, Any]) -> str:
    """What a score cell says of the play mode: ``argmax``, ``sample`` (temperature 1, no
    threshold), ``sample, T 0.5``, ``sample, T 1, threshold 0.03``."""
    extra = match_extra(match)
    if extra is None:
        return str(match["mode"])
    temperature = "1" if extra.temperature_ppm is None else ppm_decimal(extra.temperature_ppm)
    threshold = "" if extra.min_ppm is None else f", threshold {ppm_decimal(extra.min_ppm)}"
    return f"sample, T {temperature}{threshold}"


def match_stem(match: dict[str, Any]) -> str:
    """A match id without its mode part: ``random-legal`` for ``random-legal-sample-t500000``."""
    name: str = match["match_id"]
    if match["mode"] == "argmax":
        return name.removesuffix("-argmax")
    found = SAMPLE_ID.fullmatch(name)
    return name if found is None else found["stem"]
