"""The shortcut for a repeated log prefix (#964) changes no verdict and no message.

``ExampleReader`` must accept and refuse exactly what ``parse("training-example", ...)`` does, with
the same error text, for every sequence of examples a stream can hold. The corpus samples, growing
logs and seeded random damage are fed to both; a rule the shortcut relies on is pinned too.
"""

from __future__ import annotations

import copy
import dataclasses
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


class Watch:
    """A validator that records how long the log of each instance it is asked about is."""

    def __init__(self, inner: Any, seen: list[int]) -> None:
        self.inner = inner
        self.seen = seen

    def is_valid(self, instance: Any) -> bool:
        self.seen.append(len(instance["observation"]["log"]))
        return bool(self.inner.is_valid(instance))


def test_the_shortcut_checks_only_new_strings_and_builds_the_whole_example() -> None:
    sequence = growing(load(CORPUS / "valid/battle.json"), 4)
    seen: list[int] = []
    reader = ExampleReader()
    assert reader._split is not None
    reader._split = dataclasses.replace(reader._split, reduced=Watch(reader._split.reduced, seen))
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


def test_the_log_shortcut_still_runs_when_the_split_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = growing(load(CORPUS / "valid/battle.json"), 4)
    seen: list[int] = []
    real = validate

    def spy(name: str, instance: Any, version: int = 1) -> None:
        seen.append(len(instance["observation"]["log"]))
        real(name, instance, version)

    monkeypatch.setattr("areneon_contracts.training.validate", spy)
    monkeypatch.setattr("areneon_contracts.training._static_shortcut", lambda: None)
    reader = ExampleReader()
    for raw in sequence:
        reader.parse(copy.deepcopy(raw))
    assert seen[1:] == [2, 2, 2]
    agree(sequence)


def mutated(base: dict[str, Any], change: Any) -> dict[str, Any]:
    example = copy.deepcopy(base)
    change(example)
    return example


def test_the_split_gives_the_verdicts_and_messages_of_the_whole_check() -> None:
    base = load(CORPUS / "valid/battle.json")
    changes: list[Any] = [
        lambda e: e["observation"]["team"]["team"].pop(),
        lambda e: e["observation"]["team"]["team"][0].__setitem__("level", 0),
        lambda e: e["observation"]["team"].__setitem__("extra", 1),
        lambda e: e["observation"].__setitem__("team", None),
        lambda e: e["observation"].__setitem__("team", {"set": 1}),
        lambda e: e["observation"].pop("team"),
        lambda e: e["observation"].pop("manifest"),
        lambda e: e["observation"]["manifest"].__setitem__("showdown_commit", "x"),
        lambda e: e["observation"]["manifest"].__setitem__("regulation", 5),
        lambda e: e["observation"].__setitem__("manifest", [1]),
        lambda e: e["candidates"].append({"choice": "x", "pending": 1}),
        lambda e: e["candidates"].append({"choice": "x", "pending": True, "more": 1}),
        lambda e: e["candidates"].append({"choice": "", "pending": False}),
        lambda e: e["candidates"].append({"choice": 3, "pending": False}),
        lambda e: e["candidates"].append(None),
        lambda e: e["candidates"].append([]),
        lambda e: e["candidates"].clear(),
        lambda e: e.__setitem__("candidates", None),
        lambda e: e.__setitem__("candidates", {"choice": "x", "pending": False}),
        lambda e: e.__setitem__("candidates", [{"choice": "move 1", "pending": False}] * 1025),
        lambda e: e.pop("candidates"),
        lambda e: e["observation"].__setitem__("kind", "preview"),
        lambda e: e["observation"].__setitem__("kind", "other"),
    ]
    sequence = [mutated(base, change) for change in changes]
    agree(sequence)
    # A valid example first, so the memories hold good values, then each damaged one.
    reader = ExampleReader()
    reader.parse(copy.deepcopy(base))
    for position, raw in enumerate(sequence):
        assert outcome(reader.parse, copy.deepcopy(raw)) == outcome(full, raw), position
        reader.parse(copy.deepcopy(base))
    # A damaged value is never remembered as good.
    assert outcome(reader.parse, copy.deepcopy(sequence[0]))[0] == outcome(full, sequence[0])[0]


def test_a_value_the_memory_holds_is_not_checked_again_but_a_new_one_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load(CORPUS / "valid/battle.json")
    reader = ExampleReader()
    reader.parse(copy.deepcopy(base))
    assert reader._split is not None
    assert len(reader._good["team"]) == 1
    assert reader._pairs
    reader.parse(copy.deepcopy(base))
    assert len(reader._good["team"]) == 1
    other = copy.deepcopy(base)
    other["observation"]["team"]["team"][0]["ivs"]["hp"] = 30
    reader.parse(other)
    assert len(reader._good["team"]) == 2
    # Both memories are bounded.
    monkeypatch.setattr(training, "_STATIC_MEMORY", 3)
    for hp in range(10):
        again = copy.deepcopy(base)
        again["observation"]["team"]["team"][0]["ivs"]["hp"] = hp
        reader.parse(again)
    assert 1 <= len(reader._good["team"]) <= 3
    many = copy.deepcopy(base)
    many["candidates"] = [{"choice": f"move {i}", "pending": False} for i in range(1, 1000)]
    monkeypatch.setattr(training, "_PAIR_MEMORY", 100)
    reader.parse(many)
    assert len(reader._pairs) <= 100
    # A value that is no JSON value is judged by the schema, not remembered.
    odd = copy.deepcopy(base)
    odd["observation"]["team"] = {1: 2, "a": 3}
    assert outcome(reader.parse, odd) == outcome(full, odd)


def test_the_split_needs_the_schema_to_say_exactly_what_it_assumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert training._static_shortcut() is not None
    real = schema

    def altered(change: Any) -> Any:
        def fake(name: str, version: int = 1) -> dict[str, Any]:
            found = copy.deepcopy(real(name, version))
            if name == "training-example":
                change(found)
            return found

        return fake

    breaks = [
        lambda s: s.__setitem__("allOf", []),
        lambda s: s["properties"]["observation"].__setitem__("if", {}),
        lambda s: s["properties"].pop("observation"),
        lambda s: s["$defs"]["BattleObservation"].__setitem__("dependentRequired", {}),
        lambda s: s["$defs"]["BattleObservation"].__setitem__("additionalProperties", True),
        lambda s: s["$defs"]["BattleObservation"]["properties"]["team"].__setitem__("maxItems", 1),
        lambda s: s["$defs"]["PreviewObservation"]["properties"]["manifest"].pop("$ref"),
        lambda s: s["$defs"]["PreviewObservation"]["required"].remove("team"),
        lambda s: s["$defs"].pop("PreviewObservation"),
        lambda s: s["required"].append("candidates"),
        lambda s: s["properties"]["candidates"].__setitem__("uniqueItems", True),
        lambda s: s["properties"]["candidates"].__setitem__("items", {"type": "object"}),
        lambda s: s["properties"].pop("candidates"),
    ]
    for position, change in enumerate(breaks):
        monkeypatch.setattr("areneon_contracts.training.schema", altered(change))
        assert training._static_shortcut() is None, position
        # With the shortcut off, the reader still decides as the whole check does.
        reader = ExampleReader()
        sample = load(CORPUS / "valid/battle.json")
        assert outcome(reader.parse, copy.deepcopy(sample))[0] in ("accepted", "refused")


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
