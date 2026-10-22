"""One reduction per channel gives the tensors of re-reducing every example (#958).

``trajectories`` encodes an Episode's examples through an ``EncoderSession`` and an
``IncrementalView``. The reference here is the per-example path the loader used before:
``Encoder.encode(encoder_input(query))``, which reduces the log from its first line. The
tensors must be bit-identical on every committed battle channel, rqid cleaning included.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from typing import Any

import pytest
import torch
from areneon_ml.learner.data import trajectories
from areneon_ml.model.encoder import EncodedQuery, Encoder, EncodingError
from areneon_ml.model.network import ROLES
from areneon_ml.reducer import OpponentSecretError, ReducerError
from areneon_ml.worker.view import IncrementalView, encoder_input

from tests.ml.learner.support import POLICY, episode_lines, stream_of, tiny_config
from tests.ml.model.support import (
    FORMAT_DIR,
    RANDOM,
    TINY_ENCODER,
    candidates_for,
    game,
    request_points,
)

# Eight moves a Pokemon can show: the transform battle needs them (TINY_ENCODER has four).
WIDE = Encoder(game(), dataclasses.replace(TINY_ENCODER, max_moves=8))


def encoder() -> Encoder:
    return WIDE


BATTLES = [
    *RANDOM,
    *sorted((FORMAT_DIR / "unavailable").glob("b*.json")),
    FORMAT_DIR / "illusion" / "battle.json",
    FORMAT_DIR / "transform" / "battle.json",
]


def same(a: EncodedQuery, b: EncodedQuery) -> None:
    assert a.kind == b.kind
    assert a.n_own == b.n_own
    left, right = a.tensors(), b.tensors()
    for name, tensor in left.items():
        assert tensor.dtype == right[name].dtype, name
        assert torch.equal(tensor, right[name]), name


def with_rqids(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The examples with a distinct rqid in each |request| line (the cleaner must drop it)."""
    out = copy.deepcopy(examples)
    for example in out:
        log = example["observation"]["log"]
        for position, line in enumerate(log):
            if line.startswith("|request|"):
                body = json.loads(line[len("|request|") :])
                body["rqid"] = position + 1000 * example["observation"]["identity"]["request_seq"]
                log[position] = "|request|" + json.dumps(body)
        example["observation"]["identity"]["rqid"] = 7
    return out


def battle_examples(path: Any, player: str) -> list[dict[str, Any]]:
    battle = json.loads(path.read_text())
    lines = battle["channels"][player]["lines"]
    team = {"team": battle["teams"][player]}
    examples = []
    for position, body in request_points(lines):
        examples.append(
            {
                "observation": {
                    "kind": "preview" if body.get("teamPreview") else "battle",
                    "player": player,
                    "log": lines[: position + 1],
                    "team": team,
                    "identity": {"request_seq": 1},
                },
                "candidates": candidates_for(body),
            }
        )
    return examples


@pytest.mark.parametrize("path", BATTLES, ids=lambda p: f"{p.parent.name}-{p.stem}")
@pytest.mark.parametrize("player", ["p1", "p2"])
def test_session_equals_per_example_reduction_at_every_request(path: Any, player: str) -> None:
    examples = battle_examples(path, player)
    assert examples
    for variant in (examples, with_rqids(examples)):
        view, session = IncrementalView(), encoder().session()
        for example in variant:
            expected = encoder().encode(encoder_input(example))
            got = session.encode(view.encoder_input(example))
            same(got, expected)


def test_a_session_serves_two_players_and_restarts_on_a_log_that_is_not_an_extension() -> None:
    mine = battle_examples(RANDOM[0], "p1")
    theirs = battle_examples(RANDOM[0], "p2")
    other = battle_examples(RANDOM[1], "p1")
    session = encoder().session()
    # forward, backward, another player, another battle, a repeat: always the full reduction
    for example in [*mine[:5], mine[1], *theirs[:3], *other[:3], mine[-1], mine[-1]]:
        same(session.encode(example), encoder().encode(example))


def test_the_incremental_view_equals_encoder_input_and_restarts_on_a_different_log() -> None:
    examples = with_rqids(battle_examples(RANDOM[0], "p1"))
    view = IncrementalView()
    for example in [*examples, examples[2], *with_rqids(battle_examples(RANDOM[1], "p2"))[:3]]:
        assert view.encoder_input(example) == encoder_input(example)


def test_a_refusal_is_the_same_error_and_leaves_the_session_usable() -> None:
    examples = battle_examples(RANDOM[0], "p1")
    session = encoder().session()
    session.encode(examples[0])
    bad = copy.deepcopy(examples[1])
    bad["observation"]["log"].insert(len(bad["observation"]["log"]) - 1, "|split|p2")
    with pytest.raises(OpponentSecretError) as incremental:
        session.encode(bad)
    with pytest.raises(OpponentSecretError) as reference:
        encoder().encode(bad)
    assert str(incremental.value) == str(reference.value)
    # the canary still holds at every request point of a session fed in order
    for example in examples:
        same(session.encode(example), encoder().encode(example))
    wrong = copy.deepcopy(examples[0])
    wrong["observation"]["kind"] = "battle"
    with pytest.raises(EncodingError):
        session.encode(wrong)
    with pytest.raises(EncodingError):
        encoder().encode(wrong)
    same(session.encode(examples[1]), encoder().encode(examples[1]))
    assert issubclass(OpponentSecretError, ReducerError)


@pytest.mark.parametrize("index", range(min(len(RANDOM), 4)))
def test_trajectories_hold_the_tensors_of_the_per_example_path(index: int) -> None:
    config = tiny_config()
    encoders = {role: Encoder(game(), config.model.encoder) for role in ROLES}
    picker = random.Random(index)
    examples = with_rqids(
        [*episode_lines(index, "p1", picker), *episode_lines(index, "p2", picker)]
    )
    text, manifest = stream_of(examples)
    found = trajectories(text, manifest, encoders, POLICY)
    ordered = sorted(
        examples, key=lambda e: (e["episode_hash"], e["observation"]["identity"]["request_seq"])
    )
    expected: list[tuple[str, list[EncodedQuery]]] = []
    for episode in sorted({e["episode_hash"] for e in ordered}):
        for role in ROLES:
            group = [
                e
                for e in ordered
                if e["episode_hash"] == episode and e["observation"]["kind"] == role
            ]
            if group:
                expected.append((role, [encoders[role].encode(encoder_input(e)) for e in group]))
    assert len(found) == len(expected)
    for trajectory, (role, queries) in zip(found, expected, strict=True):
        assert trajectory.role == role
        assert len(trajectory.steps) == len(queries)
        for step, query in zip(trajectory.steps, queries, strict=True):
            same(step.query, query)
