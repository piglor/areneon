"""``python -m areneon_ml.checkpoint``: initial checkpoints and the C1/C6 checker.

* ``init``: a freshly (seeded) initialised network of one role, stored as
  ``<run dir>/checkpoints/<id>.safetensors``; prints its id. This is
  generation 0 of a lineage and a plumbing input for tests, not a trained model.
* ``classify``: C1, whether an opponent is ``seen`` or ``held_out`` relative to a
  candidate, from the CheckpointRecords of a directory and a sources registry.
  The registry is hash-bound to the run that exported it (``provenance.json``, see
  ``binding.py``) or derived again from ``--run-dir``; both checks print its hash.
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

from areneon_ml.checkpoint.binding import BINDING_NAME, BindingError, verify_binding
from areneon_ml.checkpoint.files import (
    CheckpointFileError,
    checkpoint_bytes,
    write_checkpoint,
)
from areneon_ml.checkpoint.provenance import ProvenanceError, ProvenanceGraph
from areneon_ml.checkpoint.record import RecordError, read_record
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ConfigError, ModelConfig, init_network, shared_trunk_of

EXIT_REFUSED = 3


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


SOURCES_HELP = (
    "the sources registry written by `loop provenance` or `learner clone`: a canonical (RFC 8785) "
    "JSON object, training-set hash to the sorted checkpoint ids that generated it ([] for an "
    "outside source). It is accepted only with the provenance.json beside it, which binds its "
    "hash and the records' hashes; its hash is printed so a plan can bind it"
)


def _graph(args: argparse.Namespace) -> tuple[ProvenanceGraph, str]:
    if args.run_dir is not None:
        if args.records is not None or args.sources is not None:
            raise ProvenanceError("--run-dir replaces --records and --sources")
        from areneon_ml.loop.driver import (  # noqa: PLC0415 (the loop imports this package)
            LoopError,
            provenance_graph,
            sources_registry,
        )

        try:
            graph = provenance_graph(args.run_dir)
            return graph, sha256_hex(jcs(sources_registry(args.run_dir)))
        except LoopError as exc:
            raise ProvenanceError(str(exc)) from exc
    if args.records is None or args.sources is None:
        raise ProvenanceError("give --run-dir, or both --records and --sources")
    files = sorted(args.records.glob("*.json"))
    if not files:
        raise ProvenanceError(f"{args.records} holds no CheckpointRecord (*.json)")
    raw = args.sources.read_bytes()
    registry = json.loads(raw)
    if not isinstance(registry, dict):
        raise ProvenanceError("the sources registry must be a JSON object")
    canonical = jcs(registry)
    if raw not in (canonical, canonical + b"\n"):
        raise ProvenanceError("the sources registry is not canonical JSON (RFC 8785)")
    records = [read_record(_load_json(f)) for f in files]
    verify_binding(
        args.sources.parent / BINDING_NAME,
        canonical,
        {r.checkpoint_id: f.read_bytes() for r, f in zip(records, files, strict=True)},
    )
    graph = ProvenanceGraph(records, registry)
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
        sub.add_argument("--records", type=Path, help="a directory of CheckpointRecord *.json")
        sub.add_argument("--sources", type=Path, help=SOURCES_HELP)
        sub.add_argument(
            "--run-dir",
            type=Path,
            help="derive the records and the registry from this run directory instead",
        )
    classify.add_argument("--candidate", nargs="+", required=True)
    classify.add_argument("--opponent", nargs="+", required=True)
    independent.add_argument("--a", nargs="+", required=True)
    independent.add_argument("--b", nargs="+", required=True)
    return parser


def _init(args: argparse.Namespace) -> dict[str, Any]:
    game = GameData.from_table(_load_json(args.engine_data))
    config = ModelConfig.from_dict(_load_json(args.model_config))
    if shared_trunk_of(config):  # one role alone has no partner trunk: the loop makes the pair
        raise ConfigError("shared_trunk: a Preview and Battle pair is made by the generation loop")
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
            graph, sources_hash = _graph(args)
            opponent = args.opponent[0] if len(args.opponent) == 1 else args.opponent
            found = graph.classify(args.candidate, opponent)
            result = {"label": found.label, "reason": found.reason, "sources_hash": sources_hash}
        else:
            graph, sources_hash = _graph(args)
            found_independence = graph.independent(args.a, args.b)
            result = {
                "sources_hash": sources_hash,
                "independent": found_independence.independent,
                "shared_checkpoints": list(found_independence.shared_checkpoints),
                "shared_sources": list(found_independence.shared_sources),
            }
    except (
        ProvenanceError,
        BindingError,
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
