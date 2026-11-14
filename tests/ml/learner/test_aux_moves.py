"""``ppo.aux_moves`` and the move-reveal labels (ADR-010 erratum A12.2, #1384): labels worked out by
hand on built channels, the masks, the loss, the loader and its worker transport, the trainer, the
value warm-up, the report, the resume and the privacy rules."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml import learner as learner_package
from areneon_ml.learner import aux_moves, parallel
from areneon_ml.learner.aux_moves import (
    AuxTarget,
    Seen,
    aux_terms,
    labels,
    snapshot,
    targets_tensors,
)
from areneon_ml.learner.config import MICRO, RunConfig, RunConfigError, read_run_config
from areneon_ml.learner.data import (
    Collected,
    LearnerDataError,
    Trajectory,
    _Collector,
    trajectories,
)
from areneon_ml.learner.state import pack_state, restore_state
from areneon_ml.learner.trainer import Learner
from areneon_ml.model.encoder import Encoder
from areneon_ml.model.network import ROLES
from areneon_ml.reducer.reducer import reduce_channel
from areneon_ml.reducer.state import BattleState, PokemonState, SideState

from tests.ml.learner.support import POLICY, edited_config, learner, make_stream, tiny_config
from tests.ml.learner.test_config import raw
from tests.ml.model.support import game

COEF = 50_000


def aux_config(coef: int = COEF, **ppo: Any) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"aux_move_head": True}
        config["ppo"]["aux_moves"] = {"coef_micro": coef}
        config["ppo"].update(ppo)

    return edited_config(edit)


def warm_config(generations: int = 2) -> RunConfig:
    def edit(config: dict[str, Any]) -> None:
        config["model"]["options"] = {"aux_move_head": True}
        config["ppo"]["aux_moves"] = {"coef_micro": COEF}
        config["warm_start"] = {"battle": "b" * 64}
        config["ppo"]["value_warmup"] = {"generations": generations}

    return edited_config(edit)


def encoders(config: RunConfig) -> dict[str, Encoder]:
    return {role: Encoder(game(), config.model.encoder) for role in ROLES}


def aux_trajectories(config: RunConfig, seed: int = 0) -> list[Trajectory]:
    text, manifest = make_stream(seed=seed)
    return trajectories(text, manifest, encoders(config), POLICY, aux_moves=True)


# -- labels, by hand ------------------------------------------------------------------------------


def seen(key: int, row: int, *moves: str, transformed: bool = False) -> Seen:
    return Seen(key, row, frozenset(moves), transformed)


def index(move: str) -> int:
    return {"a": 2, "b": 3, "c": 4, "d": 5, "e": 6}[move]


def test_the_label_is_what_the_entry_reveals_at_the_first_later_decision_that_reveals_any() -> None:
    decisions = [
        (seen(0, 5), seen(1, 6)),
        (seen(0, 5), seen(1, 6, "a")),
        (seen(0, 5, "b"), seen(1, 6, "a")),
        (seen(0, 5, "b", "c"), seen(1, 6, "a")),
    ]
    got = labels(decisions, index)
    # decision 0: entry 0 first reveals b (at decision 2), entry 1 first reveals a (at decision 1)
    assert got[0] == (AuxTarget(5, (), (3,)), AuxTarget(6, (), (2,)))
    # decision 1: entry 1 knows a and never reveals another one: masked; entry 0 -> b
    assert got[1] == (AuxTarget(5, (), (3,)),)
    # decision 2: entry 0 knows b and reveals c next
    assert got[2] == (AuxTarget(5, (3,), (4,)),)
    # the last decision has no later one: nothing is a label (moves revealed after it are unknown)
    assert got[3] == ()


def test_several_moves_first_seen_in_one_interval_share_the_target() -> None:
    got = labels([(seen(0, 2),), (seen(0, 2),), (seen(0, 2, "d", "a"),)], index)
    assert got[0] == (AuxTarget(2, (), (index("a"), index("d"))),)
    assert got[1] == got[0]


def test_an_entry_that_never_reveals_a_move_is_masked_never_a_negative() -> None:
    assert labels([(seen(0, 2),), (seen(0, 2),), (seen(0, 2),)], index) == [(), (), ()]
    assert labels([], index) == []


def test_a_transformed_entry_is_masked_at_the_decision_or_before_the_reveal() -> None:
    # transformed at decision t itself
    assert labels([(seen(0, 2, transformed=True),), (seen(0, 2, "a"),)], index)[0] == ()
    # transformed at the decision that reveals: the copied list is no reveal of the entry's own
    got = labels([(seen(0, 2),), (seen(0, 2, "a", transformed=True),)], index)
    assert got[0] == ()
    # transformed between (a decision in the middle), then it reveals later: still masked
    got = labels(
        [(seen(0, 2),), (seen(0, 2, transformed=True),), (seen(0, 2, "b"),)],
        index,
    )
    assert got[0] == ()
    # the middle decision's own label is masked too (transformed there)
    assert got[1] == ()


def test_a_list_that_shrinks_between_decisions_is_masked() -> None:
    # a Transform or an Illusion that ended between two decisions replaced the list
    got = labels([(seen(0, 2, "a", "b"),), (seen(0, 2, "c"),), (seen(0, 2, "c", "a", "b"),)], index)
    assert got[0] == ()


def test_an_entry_missing_later_is_masked_and_the_key_not_the_row_names_it() -> None:
    assert labels([(seen(0, 2), seen(1, 3)), (seen(1, 3, "a"),)], index)[0] == (
        AuxTarget(3, (), (2,)),
    )
    # the row can move (an own phantom entry changes n_own); the key is the entry
    got = labels([(seen(0, 2),), (seen(0, 3, "a"),)], index)
    assert got[0] == (AuxTarget(2, (), (2,)),)


def side(*mons: PokemonState) -> BattleState:
    state = BattleState(player="p1")
    state.opponent = SideState(player="p2", roster=list(mons))
    return state


def test_a_snapshot_skips_phantoms_and_struggle_and_marks_a_transform() -> None:
    state = side(
        PokemonState(own=False, moves_revealed=["a", aux_moves.STRUGGLE]),
        PokemonState(own=False, moves_revealed=["b"], phantom=True),
        PokemonState(own=False, moves_revealed=["c"], moves_base=["d"]),
    )
    # n_own = 4 own rows come first; the phantom keeps its row but has no key
    assert snapshot(state, 4) == (seen(0, 4, "a"), seen(1, 6, "c", transformed=True))


def test_struggle_is_in_the_engine_data_vocabulary() -> None:
    assert aux_moves.STRUGGLE in game().moves.index


# -- the loss ------------------------------------------------------------------------------------


def test_the_cross_entropy_masks_the_known_moves_and_the_reserved_indices_by_hand() -> None:
    logits = torch.tensor([[9.0, 9.0, 1.0, 2.0, 3.0, 9.0]])
    # known: move 5 (huge logit, masked); label: moves 3 and 4 (uniform); index 2 is an option
    batch, rows, known, target = targets_tensors([(AuxTarget(0, (5,), (3, 4)),)], 6)
    assert (batch.tolist(), rows.tolist()) == ([0], [0])
    ce, hit = aux_terms(logits, known, target)
    probs = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), 0)  # moves 2, 3, 4
    assert ce.item() == pytest.approx(-0.5 * (probs[1].log() + probs[2].log()).item(), abs=1e-6)
    assert bool(hit[0])  # the top unmasked logit (move 4) is in the label
    # a label the top does not hit
    _, rows, known, target = targets_tensors([(AuxTarget(0, (5,), (2,)),)], 6)
    assert not bool(aux_terms(logits, known, target)[1][0])


def test_the_targets_of_a_minibatch_are_indexed_by_decision_and_row() -> None:
    steps = [(AuxTarget(7, (2,), (3,)),), (), (AuxTarget(1, (), (4,)), AuxTarget(9, (3,), (2, 5)))]
    batch, rows, known, target = targets_tensors(steps, 8)
    assert (batch.tolist(), rows.tolist()) == ([0, 2, 2], [7, 1, 9])
    assert known.shape == target.shape == (3, 8)
    assert target.sum(-1).tolist() == [1.0, 1.0, 1.0]
    none = targets_tensors([(), ()], 8)
    assert [t.shape[0] for t in none] == [0, 0, 0, 0]


# -- the loader ----------------------------------------------------------------------------------


def test_the_loader_gives_every_battle_step_its_targets_and_a_run_without_the_flag_none() -> None:
    config = aux_config()
    found = aux_trajectories(config)
    battle = [t for t in found if t.role == "battle"]
    assert battle
    assert all(s.aux is not None for t in battle for s in t.steps)
    assert all(s.aux is None for t in found if t.role == "preview" for s in t.steps)
    assert any(s.aux for t in battle for s in t.steps)  # the random games do reveal moves
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(config), POLICY)
    assert all(s.aux is None for t in plain for s in t.steps)
    # the flag adds the labels and nothing else
    for a, b in zip(found, plain, strict=True):
        assert [s.chosen for s in a.steps] == [s.chosen for s in b.steps]
        for x, y in zip(a.steps, b.steps, strict=True):
            assert all(torch.equal(v, y.query.tensors()[k]) for k, v in x.query.tensors().items())


def test_the_incremental_labels_equal_labels_from_a_full_reduction_of_each_log() -> None:
    config = aux_config()
    text, manifest = make_stream(seed=3)
    found = [t for t in trajectories(text, manifest, encoders(config), POLICY, aux_moves=True)]
    vocab = game().vocabulary
    lookup = game().moves.lookup
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        example = json.loads(line)
        if example["observation"]["kind"] == "battle":
            by_episode.setdefault(example["episode_hash"], []).append(example)
    for t in (t for t in found if t.role == "battle"):
        examples = by_episode[t.episode]
        snaps = []
        for example in examples:
            obs = example["observation"]
            state = reduce_channel(obs["player"], obs["log"], obs["team"]["team"], vocab)
            n_own = len(state.own.roster)
            snaps.append(snapshot(state, n_own))
        assert tuple(labels(snaps, lookup)) == tuple(s.aux for s in t.steps)


def test_every_label_is_a_move_line_the_player_saw_later_in_its_own_channel() -> None:
    config = aux_config()
    text, manifest = make_stream(seed=5)
    found = trajectories(text, manifest, encoders(config), POLICY, aux_moves=True)
    ids = {v: k for k, v in game().moves.index.items()}
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        example = json.loads(line)
        if example["observation"]["kind"] == "battle":
            by_episode.setdefault(example["episode_hash"], []).append(example)
    checked = 0
    for t in (t for t in found if t.role == "battle"):
        examples = by_episode[t.episode]
        last_log = examples[-1]["observation"]["log"]
        for example, step in zip(examples, t.steps, strict=True):
            player = example["observation"]["player"]
            later = last_log[len(example["observation"]["log"]) :]
            foe = "p2" if player == "p1" else "p1"
            used = {
                re.sub(r"[^a-z0-9]", "", line.split("|")[3].lower())
                for line in later
                if line.startswith(f"|move|{foe}")
            }
            for target in step.aux or ():
                names = {ids[i] for i in target.next}
                assert names <= used  # public, from the channel the player holds
                assert not names & {ids[i] for i in target.known}
                checked += 1
    assert checked


def test_the_label_code_reads_only_the_reduced_state_of_the_players_channel() -> None:
    root = Path(learner_package.__file__).parent
    tree = ast.parse((root / "aux_moves.py").read_text())
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    assert not {m for m in imported if re.search(r"omniscient|runner|team|engine|verdict", m)}
    source = (root / "aux_moves.py").read_text()
    for forbidden in ("omniscient", "hidden", "moves_own", "team"):
        assert not re.search(rf"\b{forbidden}\b", source.split('"""', 2)[2]), forbidden


# -- the worker transport ------------------------------------------------------------------------


def test_the_targets_travel_through_the_loader_workers_bit_for_bit() -> None:
    config = aux_config()
    text, _ = make_stream(seed=2)
    collector = _Collector(encoders(config), aux=True)
    for line in text.splitlines():
        collector.add(json.loads(line))
    collector.finish()
    piece = parallel._Piece(collector.collected(), set(), set())
    got = parallel.unpack_piece(parallel.pack_piece(piece))
    sent = [s.aux for steps in piece.collected.steps.values() for s in steps]
    back = [s.aux for steps in got.collected.steps.values() for s in steps]
    assert sent == back
    assert any(sent)
    assert any(a is None for a in back) or all(a is not None for a in back if a is not None)
    plain = _Collector(encoders(config))
    for line in text.splitlines():
        plain.add(json.loads(line))
    plain.finish()
    again = parallel.unpack_piece(
        parallel.pack_piece(parallel._Piece(plain.collected(), set(), set()))
    )
    assert all(s.aux is None for steps in again.collected.steps.values() for s in steps)


def test_finishing_twice_changes_nothing() -> None:
    config = aux_config()
    text, _ = make_stream()
    collector = _Collector(encoders(config), aux=True)
    for line in text.splitlines():
        collector.add(json.loads(line))
    first = collector.trajectories()
    again = collector.trajectories()
    assert [s.aux for t in first for s in t.steps] == [s.aux for t in again for s in t.steps]
    assert isinstance(collector.collected(), Collected)


# -- the readers ---------------------------------------------------------------------------------


def test_the_reader_needs_the_head_for_the_term_and_the_term_for_the_head() -> None:
    config = aux_config()
    assert config.ppo.aux_moves is not None
    assert config.ppo.aux_moves.coef_micro == COEF
    assert tiny_config().ppo.aux_moves is None
    with pytest.raises(RunConfigError, match=r"needs model\.options\.aux_move_head"):
        edited_config(lambda c: c["ppo"].update(aux_moves={"coef_micro": COEF}))
    with pytest.raises(RunConfigError, match="head nothing trains"):
        edited_config(lambda c: c["model"].update(options={"aux_move_head": True}))
    for bad in ({"coef_micro": 0}, {"coef_micro": MICRO + 1}, {"coef_micro": 1.5}, {}):
        data = raw()
        data["model"]["options"] = {"aux_move_head": True}
        data["ppo"]["aux_moves"] = bad
        with pytest.raises(RunConfigError):
            read_run_config(data)
    data = raw()
    data["model"]["options"] = {"aux_move_head": False}
    with pytest.raises(RunConfigError):
        read_run_config(data)


def test_a_config_with_the_block_hashes_differently_and_one_without_keeps_its_hash() -> None:
    assert aux_config().hash != tiny_config().hash
    assert "aux_moves" not in tiny_config().raw["ppo"]


# -- the trainer ---------------------------------------------------------------------------------


def test_a_steps_without_labels_are_refused_when_the_term_is_on() -> None:
    config = aux_config()
    student = learner(config)
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(config), POLICY)
    with pytest.raises(LearnerDataError, match="move-reveal labels"):
        student.train(plain)


def test_the_head_and_the_trunk_learn_from_the_term_and_the_report_has_its_keys() -> None:
    config = aux_config()
    student = learner(config)
    before = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    report = student.train(aux_trajectories(config))
    stored = report.as_dict()
    battle = stored["battle"]
    assert battle["aux_moves_steps"] > 0
    assert battle["aux_moves_loss"] > 0
    assert 0.0 <= battle["aux_moves_top1"] <= 1.0
    assert not {"aux_moves_steps", "aux_moves_loss", "aux_moves_top1"} & set(stored["preview"])
    after = student.networks["battle"].state_dict()
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("aux_moves."))
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("blocks."))


def test_the_steps_counted_are_the_tables_unmasked_targets() -> None:
    config = aux_config()
    found = aux_trajectories(config)
    total = sum(len(s.aux or ()) for t in found if t.role == "battle" for s in t.steps)
    report = learner(config).train(found)
    assert report.roles["battle"].aux_moves_steps == total


def test_a_run_without_the_block_has_no_aux_keys_and_the_same_bytes() -> None:
    text, manifest = make_stream()
    plain_config = tiny_config()
    found = trajectories(text, manifest, encoders(plain_config), POLICY)
    stored = learner(plain_config).train(found).as_dict()
    assert not any(k.startswith("aux_moves") for role in stored.values() for k in role)


def test_the_term_gradient_reaches_the_trunk_and_the_head_but_not_the_value() -> None:
    config = aux_config()
    student = learner(config)
    network = student.networks["battle"]
    from areneon_ml.model.batch import collate  # noqa: PLC0415

    steps = [s for t in aux_trajectories(config) if t.role == "battle" for s in t.steps][:8]
    batch = collate([s.query for s in steps])
    _, logits = network.forward_aux(batch)
    b, rows, known, target = targets_tensors([s.aux or () for s in steps], logits.shape[-1])
    ce, _ = aux_terms(logits[b, rows], known, target)
    ce.mean().backward()
    live = {
        n for n, p in network.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert any(n.startswith("aux_moves.") for n in live)
    assert any(n.startswith("blocks.") for n in live)
    assert not any(n.startswith("value.") for n in live)
    assert not any(n.startswith(("score.", "cross.")) for n in live)


def test_the_value_warm_up_trains_the_value_and_the_move_head_with_the_trunk_frozen() -> None:
    config = warm_config(2)
    student = learner(config)
    before = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    for _ in range(2):
        report = student.train(aux_trajectories(config))
        assert report.roles["battle"].value_warmup is True
        assert report.roles["battle"].aux_moves_steps
    after = {n: t.clone() for n, t in student.networks["battle"].state_dict().items()}
    moved = {n for n in before if not torch.equal(before[n], after[n])}
    assert moved
    assert all(n.startswith(("value.", "aux_moves.")) for n in moved)
    assert any(n.startswith("aux_moves.") for n in moved)
    # after the warm-up the term joins PPO and the trunk moves
    student.train(aux_trajectories(config))
    later = student.networks["battle"].state_dict()
    assert any(not torch.equal(after[n], later[n]) for n in after if n.startswith("blocks."))


def test_the_term_weight_scales_the_loss_the_policy_sees() -> None:
    small, large = aux_config(1), aux_config(MICRO)
    found = aux_trajectories(small)
    a, b = learner(small), learner(large)
    a.train(found)
    b.train(found)
    wa, wb = a.networks["battle"].state_dict(), b.networks["battle"].state_dict()  # live views
    assert any(not torch.equal(wa[n], wb[n]) for n in wa if n.startswith("blocks."))


def test_a_resume_gives_the_continuous_run_bit_for_bit() -> None:
    config = aux_config()
    found = aux_trajectories(config)
    straight = learner(config)
    straight.train(found)
    straight.train(found)
    first = learner(config)
    first.train(found)
    packed = pack_state(first)
    resumed = learner(config)
    restore_state(resumed, packed.tensors, packed.sidecar, packed.hash)
    resumed.train(found)
    assert pack_state(resumed).hash == pack_state(straight).hash
    sidecar = json.loads(packed.sidecar)
    assert sidecar["ppo"]["aux_moves"] == {"coef_micro": COEF}
    assert sidecar["model_options"]["network"] == {"aux_move_head": True}


def test_the_labels_are_deterministic_and_the_rerun_is_byte_identical() -> None:
    config = aux_config()
    one = [s.aux for t in aux_trajectories(config) for s in t.steps]
    two = [s.aux for t in aux_trajectories(config) for s in t.steps]
    assert one == two
    first, second = learner(config), learner(config)
    first.train(aux_trajectories(config))
    second.train(aux_trajectories(config))
    assert pack_state(first).hash == pack_state(second).hash
    assert jcs({"x": 1})  # the canonical bytes used for the hash are available


def test_the_learner_time_overhead_is_measured_and_small() -> None:
    # ADR-010 A12.2 expects under 5% at a real size; the tiny test network is dominated by the
    # loader and the optimiser, so only a loose bound is asserted (the figure is printed).
    plain_config, config = tiny_config(), aux_config()
    text, manifest = make_stream()
    plain = trajectories(text, manifest, encoders(plain_config), POLICY)
    withaux = trajectories(text, manifest, encoders(config), POLICY, aux_moves=True)

    def timed(student: Learner, found: list[Trajectory]) -> float:
        student.train(found)  # warm up
        start = time.perf_counter()
        for _ in range(2):
            student.train(found)
        return time.perf_counter() - start

    base = timed(learner(plain_config), plain)
    extra = timed(learner(config), withaux)
    print(f"aux_moves learner overhead: {extra / base - 1:+.1%} (tiny network)")
    assert extra < base * 3


def test_a_dataclass_step_keeps_its_old_positional_form() -> None:
    step = aux_trajectories(aux_config())[0].steps[0]
    assert dataclasses.replace(step, aux=None).aux is None
