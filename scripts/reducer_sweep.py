"""The reducer sweep's check side (#919).

Reads the JSON Lines files of ``packages/team-dataset/scripts/reducer-sweep.ts`` (seeded
battles: both players' own channels, their teams, and per decision the channel length at
the request and the candidates) and, at every request of every player, builds the query
the policy worker would get and runs the encoder on it (which runs ``reduce_channel``).
Every refusal is grouped by its kind (the error class and its message, with the channel
line kept out of the key); a group keeps a count and the example with the fewest lines.
A channel is checked up to its first refusal. The one named exclusion is the Illusion
``|replace|`` line. Exit status 0 only when nothing else was refused.

    uv run python scripts/reducer_sweep.py [--engine-data FILE] [--encoder-config JSON] OUT.jsonl...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from areneon_ml.model.encoder import Encoder, EncoderConfig
from areneon_ml.model.gamedata import GameData
from areneon_ml.reducer import ReducerError

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DATA = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc" / "engine-data.json"
# The training config's encoder section (max_moves 8: a Pokemon may show more than four moves).
DEFAULT_ENCODER = {
    "volatile_buckets": 64, "condition_buckets": 64, "max_moves": 8, "max_team": 6,
    "max_picks": 4, "stat_scale": 500, "turn_scale": 100,
}  # fmt: skip
# Line types the reducer names as unsupported (Illusion ending); a refusal at one is excluded.
EXCLUDED_LINES = frozenset({"replace"})


@dataclass
class Group:
    kind: str
    count: int = 0
    excluded: bool = False
    example: dict[str, Any] | None = None


@dataclass
class Report:
    battles: int = 0
    failed_battles: Counter[str] = field(default_factory=Counter)
    failed_example: dict[str, dict[str, Any]] = field(default_factory=dict)
    channels: int = 0
    queries: int = 0
    groups: dict[str, Group] = field(default_factory=dict)

    @property
    def unexplained(self) -> int:
        return sum(g.count for g in self.groups.values() if not g.excluded) + sum(
            self.failed_battles.values()
        )


def refusal_kind(error: Exception) -> str:
    """The error class and its message, without the offending line or any Pokemon name."""
    message = error.message if isinstance(error, ReducerError) else str(error)
    return f"{type(error).__name__}: {re.sub(r'[0-9]+', 'N', message)[:160]}"


def _excluded(error: Exception) -> bool:
    line = error.line if isinstance(error, ReducerError) else None
    return line is not None and line.startswith("|") and line[1:].split("|", 1)[0] in EXCLUDED_LINES


def queries(battle: dict[str, Any], player: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Each decision of `player` as the policy query it would be asked (position, query)."""
    channel = battle["channels"][player]
    team = {"schema_version": 1, "data_class": "synthetic", "team": battle["teams"][player]}
    for position, decision in enumerate(battle["decisions"][player]):
        yield (
            position,
            {
                "observation": {
                    "player": player,
                    "kind": decision["kind"],
                    "log": channel[: decision["n"]],
                    "team": team,
                },
                "candidates": decision["candidates"],
            },
        )


def check_battle(battle: dict[str, Any], encoder: Encoder, report: Report) -> None:
    report.battles += 1
    failure = battle.get("failure")
    if failure is not None:
        key = f"{failure['kind']}: {re.sub(r'[0-9]+', 'N', failure['message'])[:160]}"
        report.failed_battles[key] += 1
        if key not in report.failed_example:
            report.failed_example[key] = {
                "mix": battle["mix"],
                "seed": battle["seed"],
                "message": failure["message"],
            }
        return
    for player in ("p1", "p2"):
        report.channels += 1
        for position, query in queries(battle, player):
            report.queries += 1
            try:
                encoder.encode(query)
            except Exception as error:  # every refusal is a finding, not a crash
                record(report, error, (battle, player, position), query)
                break


def record(
    report: Report, error: Exception, where: tuple[dict[str, Any], str, int], query: dict[str, Any]
) -> None:
    battle, player, position = where
    kind = refusal_kind(error)
    group = report.groups.setdefault(kind, Group(kind=kind, excluded=_excluded(error)))
    group.count += 1
    lines = len(query["observation"]["log"])
    if group.example is None or lines < group.example["log_lines"]:
        group.example = {
            "mix": battle["mix"], "seed": battle["seed"],
            "player": player, "decision": position + 1,
            "log_lines": lines, "line": error.line if isinstance(error, ReducerError) else None,
            "message": str(error)[:300],
        }  # fmt: skip


def run(paths: Iterable[Path], encoder: Encoder) -> Report:
    report = Report()
    for path in paths:
        with path.open() as handle:
            for text in handle:
                if text.strip():
                    check_battle(json.loads(text), encoder, report)
    return report


def render(report: Report) -> str:
    out = [f"battles {report.battles}, channels {report.channels}, queries {report.queries}"]
    for key, count in sorted(report.failed_battles.items()):
        out.append(
            f"BATTLE FAILURE x{count}: {key}  example {json.dumps(report.failed_example[key])}"
        )
    for group in sorted(report.groups.values(), key=lambda g: -g.count):
        tag = "excluded" if group.excluded else "REFUSAL"
        out.append(f"{tag} x{group.count}: {group.kind}  example {json.dumps(group.example)}")
    out.append(f"unexplained: {report.unexplained}")
    return "\n".join(out)


def build_encoder(engine_data: Path, config: dict[str, Any]) -> Encoder:
    game = GameData.from_table(json.loads(engine_data.read_text()))
    return Encoder(game, EncoderConfig(**config))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--engine-data", type=Path, default=ENGINE_DATA)
    parser.add_argument("--encoder-config", default=json.dumps(DEFAULT_ENCODER))
    args = parser.parse_args(argv)
    report = run(args.files, build_encoder(args.engine_data, json.loads(args.encoder_config)))
    print(render(report))
    return 0 if report.unexplained == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
