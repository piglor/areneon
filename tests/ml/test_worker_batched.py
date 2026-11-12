"""The batched worker session (ADR-028 section 2 F2, #1195): the handshake states the batching and
refuses any other, one padded forward pass answers several queued queries, each query is refused or
answered on its own, and a batch of one is exactly the unbatched scorer."""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.worker.__main__ import main
from areneon_ml.worker.protocol import check
from areneon_ml.worker.sampler import ScoreError
from areneon_ml.worker.serve import EXIT_FRAME, EXIT_OK, EXIT_REFUSED, Worker
from areneon_ml.worker.view import encoder_input

from tests.ml.helpers import (
    ENGINE_DATA,
    ENGINE_DATA_HASH,
    MAX_LINE,
    battle_query,
    hello,
    line,
    make_run_dir,
    make_worker,
    preview_query,
    session,
)

INFERENCE = {"kind": "batched", "max_batch": 3}
BATCH2 = {"kind": "batched", "max_batch": 2}


def at(root: Path, name: str) -> Path:
    found = root / name
    found.mkdir()
    return found


def hello_with(inference: dict[str, Any] | None) -> dict[str, Any]:
    frame = hello()
    if inference is not None:
        frame["message"]["inference"] = inference
    return frame


def evaluate_query(query_id: int, base: dict[str, Any] | None = None) -> dict[str, Any]:
    query = copy.deepcopy(base or battle_query())
    query["query_id"] = query_id
    query["sampling"] = {"kind": "evaluate"}
    return query


def without_elapsed(reply: dict[str, Any]) -> dict[str, Any]:
    found = copy.deepcopy(reply)
    recommendation = found["reply"].get("recommendation")
    if recommendation is not None:
        recommendation["elapsed_ms"] = 0
    return found


def test_a_batched_worker_states_its_batching_and_a_plain_one_states_none(tmp_path: Path) -> None:
    batched = make_worker(at(tmp_path, "a"), inference_max_batch=3)
    code, [ready] = session(batched, line(hello_with(INFERENCE)))
    assert code == EXIT_OK
    check("policy-handshake", ready)
    assert ready["message"]["inference"] == INFERENCE
    _, [plain] = session(make_worker(at(tmp_path, "b")), line(hello()))
    assert "inference" not in plain["message"]


def test_a_hello_that_expects_another_batching_ends_the_session(tmp_path: Path) -> None:
    for index, (worker_batch, asked) in enumerate(
        [(3, None), (None, INFERENCE), (3, {"kind": "batched", "max_batch": 4})]
    ):
        worker = make_worker(at(tmp_path, str(index)), inference_max_batch=worker_batch)
        code, frames = session(worker, line(hello_with(asked)), line(battle_query(query_id=1)))
        assert code == EXIT_REFUSED
        assert len(frames) == 1  # the ready, so the client can name the difference; no reply


def test_a_worker_cannot_be_asked_for_a_batch_of_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        make_worker(tmp_path, inference_max_batch=1)


def test_a_batched_session_answers_every_query_in_order(tmp_path: Path) -> None:
    queries = [
        battle_query(query_id=1),
        preview_query(query_id=2),
        battle_query(query_id=3),
        battle_query(query_id=4),
        preview_query(query_id=5),
    ]
    code, frames = session(
        make_worker(tmp_path, inference_max_batch=3),
        line(hello_with(INFERENCE)),
        *(line(q) for q in queries),
    )
    assert code == EXIT_OK
    replies = frames[1:]
    assert [r["query_id"] for r in replies] == [1, 2, 3, 4, 5]
    for query, reply in zip(queries, replies, strict=True):
        check("policy-reply", reply)
        assert reply["reply"]["kind"] == "recommendation"
        assert reply["reply"]["recommendation"]["choice"] in {
            c["choice"] for c in query["candidates"]
        }


def test_a_frame_error_ends_a_batched_session_after_the_frames_before_it(tmp_path: Path) -> None:
    code, frames = session(
        make_worker(tmp_path, inference_max_batch=3),
        line(hello_with(INFERENCE)),
        line(battle_query(query_id=1)),
        b"not json\n",
    )
    assert code == EXIT_FRAME
    assert [f.get("query_id") for f in frames[1:]] == [1]


def test_a_batch_of_one_is_the_serial_answer(tmp_path: Path) -> None:
    serial = make_worker(at(tmp_path, "s"))
    batched = make_worker(at(tmp_path, "b"), inference_max_batch=2)
    for query in (battle_query(query_id=1), preview_query(query_id=2), evaluate_query(3)):
        [alone] = batched.answer_batch([query])
        assert without_elapsed(alone) == without_elapsed(serial.answer(query))


def test_the_queries_of_a_batch_are_scored_in_one_pass(tmp_path: Path) -> None:
    worker = make_worker(tmp_path, inference_max_batch=4)
    passes: list[int] = []
    scorer: Any = worker._scorer
    original = scorer.score_batch

    def spy(items: Any) -> Any:
        passes.append(len(items))
        return original(items)

    scorer.score_batch = spy
    queries = [evaluate_query(i) for i in range(1, 4)] + [preview_query(query_id=4)]
    replies = worker.answer_batch(queries)
    assert passes == [4]
    assert [r["query_id"] for r in replies] == [1, 2, 3, 4]


def test_a_batched_prior_is_the_batch_one_prior_up_to_rounding(tmp_path: Path) -> None:
    serial = make_worker(at(tmp_path, "s"))
    batched = make_worker(at(tmp_path, "b"), inference_max_batch=3)
    queries = [evaluate_query(1), evaluate_query(2, preview_query()), evaluate_query(3)]
    together = batched.answer_batch(queries)
    for query, reply in zip(queries, together, strict=True):
        alone = serial.answer(query)
        a, b = reply["reply"]["evaluation"], alone["reply"]["evaluation"]
        assert len(a["prior_ppm"]) == len(b["prior_ppm"])
        assert max(abs(x - y) for x, y in zip(a["prior_ppm"], b["prior_ppm"], strict=True)) <= 2
        assert abs(a["value_ppm"] - b["value_ppm"]) <= 2


def test_a_refused_query_is_refused_alone_and_the_others_are_answered(tmp_path: Path) -> None:
    bad_observation = battle_query(query_id=2)
    bad_observation["observation"]["log"].insert(-1, "|nosuchline|SECRET-FIELD")
    bad_candidate = battle_query(query_id=3)
    bad_candidate["candidates"] = [{"choice": "move 9, move 1", "pending": False}]
    unknown_shape = battle_query(query_id=4)
    unknown_shape["candidates"] = [
        *unknown_shape["candidates"],
        {"choice": "shift, pass", "pending": 0},
    ]
    unknown_shape["candidates"][-1]["pending"] = False
    worker = make_worker(tmp_path, inference_max_batch=5)
    replies = worker.answer_batch(
        [
            battle_query(query_id=1),
            bad_observation,
            bad_candidate,
            unknown_shape,
            battle_query(query_id=5),
        ]
    )
    kinds = [(r["query_id"], r["reply"]["kind"], r["reply"].get("error")) for r in replies]
    assert kinds == [
        (1, "recommendation", None),
        (2, "error", "invalid_query"),
        (3, "error", "unsupported_candidate"),
        (4, "error", "unsupported_candidate"),
        (5, "recommendation", None),
    ]
    assert "SECRET-FIELD" not in json.dumps(replies[1])


def test_the_scorer_returns_the_refusal_of_one_query_beside_the_answers(tmp_path: Path) -> None:
    worker = make_worker(tmp_path, inference_max_batch=3)
    bad = battle_query()
    bad["observation"]["log"].insert(-1, "|nosuchline|x")
    scorer: Any = worker._scorer
    found = scorer.score_batch(
        [
            (encoder_input(battle_query()), None),
            (encoder_input(bad), None),
            (encoder_input(battle_query()), 500_000),
        ]
    )
    assert isinstance(found[1], Exception)
    first, third = found[0], found[2]
    assert not isinstance(first, Exception)
    assert not isinstance(third, Exception)
    assert first[0] == pytest.approx(scorer.probabilities(encoder_input(battle_query())), abs=1e-5)
    assert third[0] == pytest.approx(
        scorer.probabilities(encoder_input(battle_query()), temperature_ppm=500_000), abs=1e-5
    )
    # A batch of one is the unbatched forward pass, bit for bit.
    [alone] = scorer.score_batch([(encoder_input(battle_query()), None)])
    assert not isinstance(alone, Exception)
    assert alone[0] == scorer.probabilities(encoder_input(battle_query()))
    assert abs(sum(first[0]) - 1) < 1e-5


class BrokenScorer:
    encoder_version = "broken-1"

    def probabilities(self, view: dict[str, Any], temperature_ppm: int = 1_000_000) -> list[float]:
        return [1.0]

    def evaluate(self, view: dict[str, Any]) -> tuple[list[float], float]:
        return [1.0], 0.0


class OneAtATime:
    """A scorer without ``score_batch`` (a stand-in): the worker asks it one query at a time."""

    encoder_version = "one-at-a-time-1"

    def probabilities(self, view: dict[str, Any], temperature_ppm: int = 1_000_000) -> list[float]:
        count = len(view["candidates"])
        return [1 / count] * count

    def evaluate(self, view: dict[str, Any]) -> tuple[list[float], float]:
        return self.probabilities(view), 0.5


def test_a_scorer_without_a_batch_method_is_asked_one_query_at_a_time() -> None:
    worker = Worker(OneAtATime(), ENGINE_DATA_HASH, [], MAX_LINE, inference_max_batch=2)
    plain = battle_query(query_id=1)
    tempered = battle_query(
        query_id=2, sampling={"kind": "sample", "u": 5, "temperature_ppm": 500_000}
    )
    replies = worker.answer_batch([plain, tempered, evaluate_query(3)])
    assert [r["query_id"] for r in replies] == [1, 2, 3]
    assert replies[2]["reply"]["evaluation"]["value_ppm"] == 500_000


def test_a_defect_in_the_scorer_ends_a_batched_session_with_the_error(tmp_path: Path) -> None:
    worker = Worker(BrokenScorer(), ENGINE_DATA_HASH, [], MAX_LINE, inference_max_batch=2)
    with pytest.raises(ScoreError):
        session(worker, line(hello_with(BATCH2)), line(battle_query(query_id=1)))


def cli(run_dir: Path, preview: str, battle: str, *extra: str) -> list[str]:
    return [
        "--run-dir", str(run_dir), "--preview", preview, "--battle", battle,
        "--engine-data", str(ENGINE_DATA), "--max-line-bytes", str(MAX_LINE), *extra,
    ]  # fmt: skip


class Stdio:
    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def test_main_serves_a_batched_session_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    out = Stdio(b"")
    frames = (
        line(hello_with(BATCH2)) + line(battle_query(query_id=1)) + line(preview_query(query_id=2))
    )
    monkeypatch.setattr(sys, "stdin", Stdio(frames))
    monkeypatch.setattr(sys, "stdout", out)
    assert main(cli(run_dir, preview, battle, "--inference-max-batch", "2")) == EXIT_OK
    ready, *replies = (json.loads(text) for text in out.buffer.getvalue().splitlines())
    assert ready["message"]["inference"] == BATCH2
    assert [r["query_id"] for r in replies] == [1, 2]


def test_the_command_line_refuses_a_batch_of_one(tmp_path: Path) -> None:
    run_dir, preview, battle = make_run_dir(tmp_path)
    with pytest.raises(SystemExit):
        main(cli(run_dir, preview, battle, "--inference-max-batch", "1"))


@pytest.mark.parametrize("make", [battle_query, preview_query])
def test_a_batch_of_one_is_bit_identical_to_the_unbatched_pass_for_each_role(
    tmp_path: Path, make: Any
) -> None:
    scorer: Any = make_worker(tmp_path, inference_max_batch=2)._scorer
    view = encoder_input(make())
    [alone] = scorer.score_batch([(view, None)])
    assert not isinstance(alone, Exception)
    probabilities, value = scorer.evaluate(view)
    assert alone == (probabilities, value)  # float for float, value included
    [tempered] = scorer.score_batch([(view, 500_000)])
    assert not isinstance(tempered, Exception)
    assert tempered[0] == scorer.probabilities(view, temperature_ppm=500_000)
