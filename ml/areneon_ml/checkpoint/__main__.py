"""``python -m areneon_ml.checkpoint``: initial checkpoints and the C1/C6 checker.

* ``init``: a freshly (seeded) initialised network of one role, stored as
  ``<run dir>/checkpoints/<id>.safetensors``; prints its id. This is
  generation 0 of a lineage and a plumbing input for tests, not a trained model.
* ``classify``: C1, whether an opponent is ``seen`` or ``held_out`` relative to a
  candidate, from the CheckpointRecords of a directory and a sources registry.
  The registry is a hand-supplied, not yet run-bound input (see ``SOURCES_HELP``);
  both checks print its hash.
* ``independent``: C6, whether two checkpoints (or pairs) are independent.

Output is one JCS JSON line. Exit 0 on success, 2 for a usage error, 3 when the
records, the registry or a checkpoint are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, sha256_hex

from areneon_ml.checkpoint.files import (
    CheckpointFileError,
    checkpoint_bytes,
    write_checkpoint,
)
from areneon_ml.checkpoint.provenance import ProvenanceError, ProvenanceGraph
from areneon_ml.checkpoint.record import RecordError, read_record
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ConfigError, ModelConfig, init_network

EXIT_REFUSED = 3


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


SOURCES_HELP = (
    "the sources registry: a canonical (RFC 8785) JSON object, training-set hash to the sorted "
    "checkpoint ids that generated it ([] for an outside source). It is supplied by hand and not "
    "yet bound to any run record (follow-up #853); its hash is printed so a plan can bind it"
)


def _graph(records: Path, sources: Path) -> tuple[ProvenanceGraph, str]:
    files = sorted(records.glob("*.json"))
    if not files:
        raise ProvenanceError(f"{records} holds no CheckpointRecord (*.json)")
    raw = sources.read_bytes()
    registry = json.loads(raw)
    if not isinstance(registry, dict):
        raise ProvenanceError("the sources registry must be a JSON object")
    canonical = jcs(registry)
    if raw not in (canonical, canonical + b"\n"):
        raise ProvenanceError("the sources registry is not canonical JSON (RFC 8785)")
    graph = ProvenanceGraph([read_record(_load_json(f)) for f in files], registry)
    return graph, sha256_hex(canonical)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="areneon_ml.checkpoint")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write a seeded, untrained checkpoint")
    init.add_argument("--role", choices=("battle", "preview"), required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--engine-data", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="a ModelConfig JSON file: the model section of a TrainingRunConfig",
    )
    classify = commands.add_parser("classify", help="C1: seen or held out")
    independent = commands.add_parser("independent", help="C6: independence of two checkpoints")
    for sub in (classify, independent):
        sub.add_argument(
            "--records", type=Path, required=True, help="a directory of CheckpointRecord *.json"
        )
        sub.add_argument(
            "--sources",
            type=Path,
            required=True,
            help=SOURCES_HELP,
        )
    classify.add_argument("--candidate", nargs="+", required=True)
    classify.add_argument("--opponent", nargs="+", required=True)
    independent.add_argument("--a", nargs="+", required=True)
    independent.add_argument("--b", nargs="+", required=True)
    return parser


def _init(args: argparse.Namespace) -> dict[str, Any]:
    game = GameData.from_table(_load_json(args.engine_data))
    config = ModelConfig.from_dict(_load_json(args.model_config))
    data = checkpoint_bytes(init_network(args.role, game, config, args.seed), game)
    found = write_checkpoint(args.run_dir, data)
    return {
        "checkpoint_id": found,
        "role": args.role,
        "encoder_version": encoder_version(game, config.encoder),
        "engine_data_hash": game.engine_data_hash,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "init":
            result = _init(args)
        elif args.command == "classify":
            graph, sources_hash = _graph(args.records, args.sources)
            opponent = args.opponent[0] if len(args.opponent) == 1 else args.opponent
            found = graph.classify(args.candidate, opponent)
            result = {"label": found.label, "reason": found.reason, "sources_hash": sources_hash}
        else:
            graph, sources_hash = _graph(args.records, args.sources)
            found_independence = graph.independent(args.a, args.b)
            result = {
                "sources_hash": sources_hash,
                "independent": found_independence.independent,
                "shared_checkpoints": list(found_independence.shared_checkpoints),
                "shared_sources": list(found_independence.shared_sources),
            }
    except (
        ProvenanceError,
        RecordError,
        ContractError,
        CheckpointFileError,
        ConfigError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"areneon_ml.checkpoint: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
