"""Check a TrainingExample stream against its TrainingSet (ADR-009 §4).

The TS training loader writes the stream and its manifest. Python training
reads only that stream: ``read_stream`` validates the manifest, refuses a
stream whose digest or line count differs from it, then validates and parses
every line as a ``training-example``. What the contract profile cannot say
(it has no if/then) is checked here: a ``learner_ready`` set names its
``behaviour_policy_id`` and every one of its examples carries ``candidates``
and ``outcome`` (ADR-010 section 5), and an example's ``choice`` is among its
candidates. When examples carry ``candidate_features`` (ADR-022 §3) their rows are sorted and
unique and the whole stream carries one feature set, or none. A stream with no matching manifest is
not readable at all: there is no function that reads one alone.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from areneon_contracts.contracts import ContractError, _model, parse, schema, sha256_hex, validate


class TrainingStreamError(ValueError):
    """A stream does not match its TrainingSet manifest."""


def _strictly_ascending(hashes: Sequence[Any]) -> bool:
    return all(a < b for a, b in itertools.pairwise(hashes))


def _check_candidates(number: int, raw: dict[str, Any], ready: bool) -> None:
    """The cross-field rules of one example that the profile cannot say."""
    if ready:
        for field in ("candidates", "outcome"):
            if field not in raw:
                raise TrainingStreamError(
                    f"line {number}: the set is learner_ready, its example has no {field}"
                )
    candidates = raw.get("candidates")
    if candidates is not None and raw["choice"] not in [c["choice"] for c in candidates]:
        raise TrainingStreamError(f"line {number}: its choice is not among its candidates")


#: The arrays of the `probe-damage/2` feature set and the sort key of each array's rows.
_FEATURE_SET_WITH_ARRAYS = "probe-damage/2"
_ARRAY_KEYS: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
    "threats": lambda r: (r["own"], int(r["mega"]), r["foe"], r["move"]),
    "speeds": lambda r: (r["own"], int(r["mega"]), r["foe"]),
    "own_priorities": lambda r: (r["slot"], r["move"], int(r["mega"])),
    "foe_priorities": lambda r: (r["foe"], r["move"]),
}


def _check_features(number: int, raw: dict[str, Any]) -> None:
    """The reader rules of ``candidate_features`` the profile cannot say.

    Rows sorted and unique; under ``probe-damage/2`` the four arrays present, under any other set
    absent, and each sorted and unique by its key (ADR-022 erratum E-22.3.2). What needs the request
    (positions, Mega flags, coverage of the candidates) is the worker's, as for the rows.
    """
    features = raw.get("candidate_features")
    if features is None:
        return
    keys = [(r["slot"], r["move"], r["target"], int(r["mega"])) for r in features["rows"]]
    if not _strictly_ascending(keys):
        raise TrainingStreamError(
            f"line {number}: the candidate_features rows must be sorted and unique"
            " by (slot, move, target, mega)"
        )
    present = [name for name in _ARRAY_KEYS if name in features]
    wanted = len(_ARRAY_KEYS) if features["feature_set"] == _FEATURE_SET_WITH_ARRAYS else 0
    if len(present) != wanted:
        raise TrainingStreamError(
            f"line {number}: the candidate_features arrays are present exactly under"
            f" {_FEATURE_SET_WITH_ARRAYS}"
        )
    for name in present:
        if not _strictly_ascending([_ARRAY_KEYS[name](r) for r in features[name]]):
            raise TrainingStreamError(
                f"line {number}: the candidate_features {name} must be sorted and unique"
                " by their key"
            )


# The feature set of an example (or none) is recorded in `StreamChecker.classes` under this
# prefix, so the pieces of a stream checked apart (learner/parallel.py) bring it together with
# the data classes.
_FEATURES = "features:"


def _check_manifest(manifest: object) -> bool:
    """The manifest's own rules; True when the set is learner_ready."""
    parse("training-set", manifest)
    if not isinstance(manifest, dict):  # parse() has validated an object; this narrows it
        raise TrainingStreamError("training-set: not an object")
    for field in ("episode_hashes", "evaluation_manifest_hashes"):
        if not _strictly_ascending(manifest[field]):
            raise TrainingStreamError(f"training-set: {field} must be sorted and unique")
    ready = manifest.get("learner_ready") is True
    if ready and "behaviour_policy_id" not in manifest:
        raise TrainingStreamError("a learner_ready TrainingSet must name its behaviour_policy_id")
    return ready


#: What the only array in an example that grows from one example to the next must be for the
#: prefix shortcut below to be sound: strings, at least one, and nothing else said about the array.
_LOG_SHAPE = {"type": "array", "items": {"type": "string"}, "minItems": 1}


def _log_shortcut_sound() -> bool:
    """True when the schema says nothing about an observation's ``log`` but ``_LOG_SHAPE``."""
    definitions = schema("training-example").get("$defs", {})
    for name in ("PreviewObservation", "BattleObservation"):
        log = definitions.get(name, {}).get("properties", {}).get("log")
        if not isinstance(log, dict):
            return False
        if {k: v for k, v in log.items() if k != "description"} != _LOG_SHAPE:
            return False
    return True


class ExampleReader:
    """``parse("training-example", raw)`` for one stream's lines, with a cheaper check of the log.

    An Episode's examples list each log as a prefix of the next, so a generation's stream
    repeats the same strings about a hundred times over, and python-jsonschema checks every one
    of them every time (about 80% of the load phase, #964). The schema says of the log that it is
    an array of at least one string and nothing more (``_log_shortcut_sound`` checks that it
    still says exactly that), so if an example's log begins with the strings of the previous
    valid example's log, the example is valid exactly when the same example with only the new
    strings (or the last one, if there are none) as its log is valid. That verdict, a pass, is
    taken; anything else, a failure included, is decided by the full check of the whole example,
    so every refusal keeps its message. The typed model is always built from the whole example.
    """

    def __init__(self) -> None:
        self._shortcut = _log_shortcut_sound()
        self._previous: list[str] | None = None

    def parse(self, raw: object) -> BaseModel:
        log = _log_of(raw)
        previous = self._previous
        if (
            self._shortcut
            and isinstance(raw, dict)
            and previous is not None
            and log is not None
            and log[: len(previous)] == previous
        ):
            tail = log[len(previous) :] or log[-1:]
            reduced = {**raw, "observation": {**raw["observation"], "log": tail}}
            try:
                validate("training-example", reduced)
            except ContractError:
                validate("training-example", raw)  # raises with the whole example's message
            else:
                return self._build(raw, log)
        validate("training-example", raw)
        return self._build(raw, log)

    def _build(self, raw: object, log: list[str] | None) -> BaseModel:
        model = _model("training-example").model_validate(raw, strict=True)
        # Only a validated example is remembered, and its log is then an array of strings.
        self._previous = log
        return model


def _log_of(raw: object) -> list[str] | None:
    """The example's log if it is where the schema puts it and is a list, else None."""
    observation = raw.get("observation") if isinstance(raw, dict) else None
    log = observation.get("log") if isinstance(observation, dict) else None
    return log if isinstance(log, list) else None


class StreamChecker:
    """The per-line rules of ``read_stream``, one line at a time.

    A stream can be checked in pieces (each piece with its own checker, from the line number its
    first line has in the stream) and the pieces' ``seen`` and ``classes`` brought together before
    ``finish`` runs the rules that need the whole stream. The loader's workers do that.
    """

    def __init__(self, manifest: object) -> None:
        self.ready = _check_manifest(manifest)
        assert isinstance(manifest, dict)
        self.manifest: dict[str, Any] = manifest
        self.known = set(manifest["episode_hashes"])
        self.seen: set[str] = set()
        self.classes: set[str] = set()
        self._reader = ExampleReader()

    def line(self, number: int, text: str) -> tuple[dict[str, Any], BaseModel]:
        """Check one line (numbered from 1 in the whole stream); its raw value and typed model."""
        try:
            raw = json.loads(text)
            example = self._reader.parse(raw)
        except (json.JSONDecodeError, ContractError) as error:
            raise TrainingStreamError(f"line {number}: {error}") from error
        _check_candidates(number, raw, self.ready)
        _check_features(number, raw)
        if raw["episode_hash"] not in self.known:
            raise TrainingStreamError(f"line {number}: its Episode is not in the TrainingSet")
        self.seen.add(raw["episode_hash"])
        self.classes.add(raw["observation"]["data_class"])
        features = raw.get("candidate_features")
        self.classes.add(_FEATURES + ("" if features is None else features["feature_set_hash"]))
        return raw, example

    def finish(self) -> None:
        """The rules that need every line: each Episode is there, and the data class is right."""
        if self.seen != self.known:
            missing = sorted(self.known - self.seen)
            raise TrainingStreamError(f"the stream holds no example of Episode {missing[0]}")
        if sum(1 for c in self.classes if c.startswith(_FEATURES)) > 1:
            raise TrainingStreamError(
                "the stream's examples must all carry the same candidate feature set, or none"
            )
        highest = "private" if "private" in self.classes else "synthetic"
        if self.manifest["data_class"] != highest:
            raise TrainingStreamError(
                f"the TrainingSet says {self.manifest['data_class']}, its examples are {highest}"
            )


def _check_lines(lines: Iterable[str], manifest: dict[str, Any], keep: bool) -> list[BaseModel]:
    """Validate every line against the manifest; the examples are kept only when asked."""
    checker = StreamChecker(manifest)
    examples: list[BaseModel] = []
    for number, line in enumerate(lines, start=1):
        _, example = checker.line(number, line)
        if keep:
            examples.append(example)
    checker.finish()
    return examples


def read_stream(stream: str, manifest: object) -> list[BaseModel]:
    """The stream's examples, once the manifest and every line check out."""
    _check_manifest(manifest)
    assert isinstance(manifest, dict)
    if sha256_hex(stream.encode("utf-8")) != manifest["stream_sha256"]:
        raise TrainingStreamError("the stream does not hash to its TrainingSet's stream_sha256")
    if not stream.endswith("\n"):
        raise TrainingStreamError("the stream's last line must end with a newline")
    lines = stream.split("\n")[:-1]
    if len(lines) != manifest["examples"]:
        raise TrainingStreamError(
            f"the stream has {len(lines)} lines, its TrainingSet says {manifest['examples']}"
        )
    return _check_lines(lines, manifest, keep=True)


CHUNK_BYTES = 1 << 20


def stream_lines(path: Path) -> Iterator[str]:
    """The lines of a file, each with its newline, split on newline only."""
    with path.open("rb") as handle:
        for raw in handle:
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise TrainingStreamError("the stream is not UTF-8") from error


def check_stream_digest(path: Path, manifest: object, marks: Sequence[int] = ()) -> list[int]:
    """The file-level checks of ``read_stream``: the digest, the last newline and the line count.

    The file is read once in chunks. For each byte offset in ``marks`` (ascending) the result
    holds the number of newlines before it, which is the number of lines that end before the
    offset: a line that starts at a mark is numbered one more than that.
    """
    _check_manifest(manifest)
    assert isinstance(manifest, dict)
    digest = hashlib.sha256()
    newlines = 0
    last = b""
    before: list[int] = []
    position = 0
    pending = list(marks)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            while pending and pending[0] < position + len(chunk):
                before.append(newlines + chunk.count(b"\n", 0, pending.pop(0) - position))
            newlines += chunk.count(b"\n")
            position += len(chunk)
            last = chunk[-1:]
    before.extend(newlines for _ in pending)  # a mark at or past the end
    if digest.hexdigest() != manifest["stream_sha256"]:
        raise TrainingStreamError("the stream does not hash to its TrainingSet's stream_sha256")
    if last != b"\n":
        raise TrainingStreamError("the stream's last line must end with a newline")
    if newlines != manifest["examples"]:
        raise TrainingStreamError(
            f"the stream has {newlines} lines, its TrainingSet says {manifest['examples']}"
        )
    return before


def check_stream_file(path: Path, manifest: object) -> None:
    """The checks of ``read_stream`` on a stream file, read in chunks and line by line.

    A generation's stream can be larger than a gigabyte, so no string holds it and no
    example is kept: the file is hashed and its newlines counted in one pass, then each
    line is validated in a second. The refusals and their order are ``read_stream``'s.
    """
    check_stream_digest(path, manifest)
    assert isinstance(manifest, dict)
    _check_lines(stream_lines(path), manifest, keep=False)


_EPISODE_KEY = re.compile(rb'"episode_hash":"([0-9a-f]{64})"')


def _episode_of(line: bytes) -> bytes | None:
    """The Episode hash a stream line names, read without parsing the line (None if it names none).

    Quotes inside a JSON string are escaped, so the text ``"episode_hash":"`` can only be the
    example's own key.
    """
    found = _EPISODE_KEY.search(line)
    return None if found is None else found.group(1)


def stream_split_points(path: Path, parts: int) -> list[int]:
    """Byte offsets that cut a stream file into about ``parts`` pieces between two Episodes.

    Every offset is the start of a line whose Episode differs from the line before it, so no
    Episode's examples are split (as a stream lists them, together). Fewer pieces come back when
    the file is small, and none when it cannot be cut: the result is always strictly ascending,
    starts with 0 and ends with the file size. A reader of the pieces still checks the claim
    (an Episode in two pieces is a stream that does not list its Episodes together).
    """
    size = path.stat().st_size
    points = [0]
    with path.open("rb") as handle:
        for index in range(1, parts):
            target = size * index // parts
            if target <= points[-1]:
                continue
            handle.seek(target)
            handle.readline()  # to the next line start
            before = None
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    start = size
                    break
                episode = _episode_of(line)
                if before is not None and episode != before:
                    break
                before = episode
            if points[-1] < start < size:
                points.append(start)
    points.append(size)
    return points
