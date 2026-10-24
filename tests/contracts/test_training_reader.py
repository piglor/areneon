"""The shortcut for a repeated log prefix (#964) changes no verdict and no message.

``ExampleReader`` must accept and refuse exactly what ``parse("training-example", ...)`` does, with
the same error text, for every sequence of examples a stream can hold. The corpus samples, growing
logs and seeded random damage are fed to both; a rule the shortcut relies on is pinned too.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts import training
from areneon_contracts.contracts import ContractError, parse, schema, validate
from areneon_contracts.training import ExampleReader

CORPUS = Path(__file__).resolve().parents[2] / "fixtures/contracts/v1/training-example"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def outcome(call: Any, raw: Any) -> tuple[str, str]:
    """Accept or refuse, with the message: what a caller can tell apart."""
    try:
        call(raw)
    except ContractError as error:
        return ("refused", str(error))
    except Exception as error:  # a model error is a refusal too, with its own text
        return ("error", f"{type(error).__name__}: {error}")
    return ("accepted", "")


def full(raw: Any) -> object:
    return parse("training-example", copy.deepcopy(raw))


def growing(base: dict[str, Any], steps: int) -> list[dict[str, Any]]:
    """Examples whose logs each begin with the previous log, as an Episode's do."""
    out = []
    log = list(base["observation"]["log"])
    for step in range(steps):
        example = copy.deepcopy(base)
        example["observation"]["log"] = list(log)
        out.append(example)
        log = [*log, f"|turn|{step + 2}", f"|request|{step}"]
    return out


def agree(sequence: list[Any]) -> None:
    """One reader over the sequence says what a fresh full check says of each item."""
    reader = ExampleReader()
    for position, raw in enumerate(sequence):
        assert outcome(reader.parse, copy.deepcopy(raw)) == outcome(full, raw), position


def test_the_corpus_samples_agree() -> None:
    for kind in ("valid", "invalid", "reader-invalid"):
        for path in sorted((CORPUS / kind).glob("*.json")):
            agree([load(path)])
            # A valid example before it changes nothing for this one.
            agree([load(CORPUS / "valid/battle.json"), load(path)])


def test_growing_logs_are_accepted_by_both() -> None:
    sequence = growing(load(CORPUS / "valid/battle.json"), 6)
    agree(sequence)
    assert all(outcome(full, raw)[0] == "accepted" for raw in sequence)


def test_the_shortcut_checks_only_new_strings_and_builds_the_whole_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = growing(load(CORPUS / "valid/battle.json"), 4)
    seen: list[int] = []
    real = validate

    def spy(name: str, instance: Any, version: int = 1) -> None:
        seen.append(len(instance["observation"]["log"]))
        real(name, instance, version)

    monkeypatch.setattr("areneon_contracts.training.validate", spy)
    reader = ExampleReader()
    for raw in sequence:
        reader.parse(copy.deepcopy(raw))
    whole = [len(raw["observation"]["log"]) for raw in sequence]
    assert seen[0] == whole[0]  # nothing known before the first example
    assert seen[1:] == [2, 2, 2]  # the two new strings each time
    # ... and the model is built from the whole of each example.
    last = ExampleReader()
    for raw in sequence:
        model = last.parse(copy.deepcopy(raw))
    dumped = model.model_dump(mode="json", by_alias=True)
    assert dumped["observation"]["log"] == sequence[-1]["observation"]["log"]


@pytest.mark.parametrize("seed", range(40))
def test_random_damage_to_a_growing_stream_gets_the_same_verdicts(seed: int) -> None:
    rng = random.Random(seed)
    base = load(CORPUS / "valid/battle.json")
    sequence = growing(base, 5)
    damage: list[Any] = [
        lambda e: e["observation"]["log"].__setitem__(
            rng.randrange(len(e["observation"]["log"])),
            rng.choice([1, None, ["x"], {"a": 1}, 2.5, True]),
        ),
        lambda e: e["observation"].__setitem__("log", []),
        lambda e: e["observation"].__setitem__("log", "text"),
        lambda e: e["observation"].__setitem__("log", None),
        lambda e: e["observation"].pop("log"),
        lambda e: e["observation"]["log"].insert(0, "|changed|"),
        lambda e: e["observation"]["log"].pop(0),
        lambda e: e["observation"]["log"].append(7),
        lambda e: e.__setitem__("observation", rng.choice([None, [], "x", 3])),
        lambda e: e.pop("choice"),
        lambda e: e.__setitem__("extra", 1),
        lambda e: e["observation"].__setitem__("kind", rng.choice(["preview", "other", 1])),
        lambda e: e["candidates"].append({"choice": "x"}),
        lambda e: e["observation"]["identity"].__setitem__("request_seq", -1),
    ]
    for _ in range(rng.randrange(1, 4)):
        rng.choice(damage)(sequence[rng.randrange(len(sequence))])
    agree(sequence)
    # The same damage with the stream in another order (a log no longer extends the last).
    shuffled = sequence[:]
    rng.shuffle(shuffled)
    agree(shuffled)


def test_a_log_that_repeats_the_previous_one_exactly_is_checked() -> None:
    first = load(CORPUS / "valid/battle.json")
    again = copy.deepcopy(first)
    bad = copy.deepcopy(first)
    bad["choice"] = 5  # not a string
    agree([first, again, bad])


def test_the_shortcut_needs_the_schema_to_say_nothing_more_about_a_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert training._log_shortcut_sound() is True
    real = schema

    def stricter(name: str, version: int = 1) -> dict[str, Any]:
        found = copy.deepcopy(real(name, version))
        found["$defs"]["BattleObservation"]["properties"]["log"]["maxItems"] = 3
        return found

    monkeypatch.setattr("areneon_contracts.training.schema", stricter)
    assert training._log_shortcut_sound() is False
    # Without the shortcut a long log is judged by the schema, as before.
    long = growing(load(CORPUS / "valid/battle.json"), 4)[-1]
    reader = ExampleReader()
    assert outcome(reader.parse, long)[0] == "accepted"  # validate() reads the real schema file

    def odd(name: str, version: int = 1) -> dict[str, Any]:
        found = copy.deepcopy(real(name, version))
        found["$defs"]["PreviewObservation"]["properties"].pop("log")
        return found

    monkeypatch.setattr("areneon_contracts.training.schema", odd)
    assert training._log_shortcut_sound() is False
