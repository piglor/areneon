"""The evaluation-key option (#1177): one key for the pairs and seeds of several runs.

``progress-pairing/1`` and ``progress-seed/1`` are keyed by the evaluated run's key. With
``--evaluation-key KEY`` the key replaces it for every match of the evaluation, so the plan is a
function of (KEY, validation teams) alone and two runs evaluated with the same KEY play the same
pairs, seats, battle seeds and baseline streams (a paired comparison between runs). The sampling
labels are unchanged (seed, seat, policy id), and the bootstrap stream keeps the run key. A key
equal to the evaluated run's own key is refused: omit the flag instead (one spelling per plan).
"""

from __future__ import annotations

import argparse
import re

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class EvaluationKeyError(ValueError):
    """A refused evaluation key."""


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evaluation-key",
        help="key the validation pairs and battle seeds by this id instead of the run key, so "
        "another run evaluated with the same key plays the same pairs, seats and seeds",
    )


def check(key: str | None, run_key: str) -> str | None:
    """The key as given: an id token that is not the run's own key; None when not given."""
    if key is None:
        return None
    if _TOKEN.fullmatch(key) is None:
        raise EvaluationKeyError(
            "--evaluation-key must be an id token ([A-Za-z0-9][A-Za-z0-9._:-]{0,127})"
        )
    if key == run_key:
        raise EvaluationKeyError(
            "--evaluation-key is the run's own key: omit the option to key the plan by it"
        )
    return key


def node_args(key: str | None) -> tuple[str, ...]:
    return () if key is None else ("--evaluation-key", key)
