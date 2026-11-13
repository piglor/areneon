"""The train report's ``ppo.anchor.threatened`` keys (ADR-010 erratum A11.2, #1366) as the
dashboard reader takes them: optional, together, on the anchored Battle role only."""

from __future__ import annotations

import json
from typing import Any

import pytest
from areneon_ml.inspector.readers import InspectError
from areneon_ml.report.model import read_train_report

from tests.ml.report.support import TRAIN_REPORT

KEYS = {
    "anchor_threatened_steps": 7,
    "anchor_kl_threatened": 0.25,
    "anchor_threatened_beta_micro": 1_000_000,
}


def report() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(json.dumps(TRAIN_REPORT))
    raw["battle"].update(anchor_kl=0.1, anchor_beta_micro=0)
    return raw


def test_the_keys_are_read_on_the_battle_role_with_the_anchor() -> None:
    raw = report()
    raw["battle"].update(KEYS)
    facts = read_train_report(raw)
    assert facts.anchors == {"battle": (0.1, 0)}
    assert "anchor_kl_threatened" not in facts.roles["battle"]
    assert read_train_report(report()).anchors == {"battle": (0.1, 0)}


def without(role: dict[str, Any], key: str) -> None:
    """``role`` holding all the Threatened keys except ``key``, which is also dropped."""
    role.update(KEYS)
    del role[key]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["battle"].update(anchor_threatened_steps=1), "come together"),
        (lambda r: without(r["battle"], "anchor_threatened_steps"), "come together"),
        (lambda r: r["preview"].update(KEYS), "come together"),
        (lambda r: without(r["battle"], "anchor_kl"), "come together"),
        (lambda r: r["battle"].update(KEYS, anchor_threatened_steps=-1), "integer"),
        (lambda r: r["battle"].update(KEYS, anchor_threatened_beta_micro=-1), "integer"),
        (lambda r: r["battle"].update(KEYS, anchor_kl_threatened=-0.5), "cannot be negative"),
        (lambda r: r["battle"].update(KEYS, anchor_kl_threatened=True), "finite"),
    ],
)
def test_the_keys_are_refused_when_malformed(change: Any, message: str) -> None:
    raw = report()
    change(raw)
    with pytest.raises(InspectError, match=message):
        read_train_report(raw)
