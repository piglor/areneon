"""Optional model options (ADR-010 erratum A4 item 2).

An option is an optional key of ``model.options`` (network) or ``model.encoder.options``
(encoder) of the ``TrainingRunConfig``. Absent keeps today's behaviour and today's bytes:
``EncoderConfig.as_dict`` and ``ModelConfig.as_dict`` omit the ``options`` key when no option
is set, so ``encoder_version``, the checkpoint metadata and every existing hash stay as they
were. Present, an option is part of the config JSON, so it moves the run config hash, the
``encoder_version`` and the checkpoint metadata.

This module is the seam and the registry: each experiment arm's ticket adds its own
``OptionSpec`` to ``REGISTRY`` (and a property to the contract's closed ``options`` object).
A name that is not registered is refused, never ignored.

Defined so far:

- ``candidate_matchup`` (encoder, layout changing; ADR-022 tier B): per move candidate, the
  engine-data type-chart cells of the move's type against the shown types of each position, and
  a same-type flag. Only ``true`` is a value: leave the key out to turn it off.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

SCOPES = ("encoder", "network")
OptionValue = bool | int
Options = tuple[tuple[str, OptionValue], ...]  # sorted by name; hashable, so configs stay frozen


class OptionError(ValueError):
    """An option set is not usable (unknown name, wrong scope, invalid value)."""


@dataclass(frozen=True)
class OptionSpec:
    """One option: its value check and whether it changes the network's tensor layout."""

    name: str
    check: Callable[[OptionValue], bool]
    layout_changing: bool = False


CANDIDATE_MATCHUP = "candidate_matchup"

REGISTRY: dict[str, dict[str, OptionSpec]] = {
    "encoder": {
        CANDIDATE_MATCHUP: OptionSpec(CANDIDATE_MATCHUP, lambda value: value is True, True),
    },
    "network": {},
}


def read_options(scope: str, source: object) -> Options:
    """The options of ``source`` (a JSON object) as a sorted tuple, refusing any unknown name."""
    if scope not in REGISTRY:
        raise OptionError(f"unknown option scope {scope!r}")
    if not isinstance(source, Mapping):
        raise OptionError(f"{scope} options must be an object")
    found: list[tuple[str, OptionValue]] = []
    for name in sorted(source):
        spec = REGISTRY[scope].get(name)
        if spec is None:
            raise OptionError(f"unknown {scope} option {name!r}")
        value = source[name]
        if not isinstance(value, bool | int) or not spec.check(value):
            raise OptionError(f"{scope} option {name!r} has an invalid value")
        found.append((name, value))
    return tuple(found)


def options_dict(options: Options) -> dict[str, OptionValue]:
    return dict(options)


def with_options(body: dict[str, Any], options: Options) -> dict[str, Any]:
    """``body`` plus an ``options`` key, only when an option is set (absent keeps the bytes)."""
    if options:
        body["options"] = options_dict(options)
    return body


def layout_changing(scope: str, options: Options) -> bool:
    """Whether any set option changes the tensor layout of a written checkpoint."""
    return any(REGISTRY[scope][name].layout_changing for name, _ in options)


def is_set(options: Options, name: str) -> bool:
    """Whether the option ``name`` is present (a boolean option is present only as ``true``)."""
    return any(key == name and value for key, value in options)
