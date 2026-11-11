"""The ``probe-outcome/1`` feature set as encoder input (ADR-022 erratum E-22.5.4, slice H5).

The five arrays are read exactly as ``probe-damage/2``'s (E-22.3.3), with one more reason category
(``history_unknown``) in every kind; the ``outcomes`` rows go into the move part: per foe location
a pooled summary over the part's branches and, for the part's own foe, the per-branch values. The
sample's numbers are invented
(``fixtures/contracts/v1/policy-query/valid/battle-features-outcome.json``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest
import torch
from areneon_ml.checkpoint.files import (
    FORMAT_OPTIONS,
    CheckpointOptionsError,
    checkpoint_bytes,
    checkpoint_id,
    load_checkpoint,
    read_header,
)
from areneon_ml.learner.config import VALUE_FREE_FEATURE_SETS
from areneon_ml.loop.records import FEATURES_OUTCOME_FAMILY
from areneon_ml.model import features as rows_of
from areneon_ml.model.batch import collate
from areneon_ml.model.candidates import parse_choice
from areneon_ml.model.encoder import (
    BASE_CAND_COLUMNS,
    Encoder,
    EncodingError,
    Layout,
    encoder_version,
)
from areneon_ml.model.gamedata import NONE, RESERVED
from areneon_ml.model.network import init_network
from areneon_ml.model.options import (
    CANDIDATE_FEATURES,
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    HISTORY_FACTS,
    RECENT_HISTORY,
    OptionError,
    read_options,
)
from areneon_ml.worker.features import coverage_problem, damaging_moves, rows_problem

from .support import ROOT, TINY_ENCODER, TINY_MODEL, game, sample
from .test_features_2 import SAMPLE as SAMPLE2
from .test_features_2 import candidate
from .test_options import TODAY_ENCODER_VERSION

SAMPLE = sample("policy-query/valid/battle-features-outcome.json")
SET = {key: SAMPLE["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
SET2 = {key: SAMPLE2["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
TWO = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET2}))
ENCODE2 = Encoder(game(), TWO)
LAYOUT = Layout.of(ON)
SLOTS = 2
BRANCHES = rows_of.MAX_BRANCHES
OUTCOMES = SAMPLE["candidate_features"]["outcomes"]

# The first outcome column of a move part: after the base, the feature kinds and the priority ones.
CAT_AT = BASE_CAND_COLUMNS + SLOTS + 1 + SLOTS
NUM_AT = len(Layout.of(TWO).cand_num)  # the `/2` numbers: rows, then the own priority
KIND = {name: RESERVED + i for i, name in enumerate(rows_of.OUTCOME_KINDS)}
PRE = {name: RESERVED + i for i, name in enumerate(rows_of.PRECONDITIONS)}
BRANCH = {name: RESERVED + i for i, name in enumerate(rows_of.BRANCH_KINDS)}
MOVES = game().moves


def edited(edit: Any) -> dict[str, Any]:
    q = copy.deepcopy(SAMPLE)
    edit(q["candidate_features"])
    return q


def net(seed: int = 1) -> Any:
    return init_network("battle", game(), ON_MODEL, seed)


def covering(q: dict[str, Any]) -> list[dict[str, Any]]:
    """Fewest candidates (greedy) that still use every outcome part of the query."""
    need = {(r["slot"], r["move"], r["target"], r["mega"]) for r in OUTCOMES}
    chosen: list[dict[str, Any]] = []
    for c in q["candidates"]:
        parts = {
            (slot, a.index, a.target or 0, a.event == "mega")
            for slot, a in enumerate(parse_choice(c["choice"]).slots)
            if a.kind == "move"
        }
        if parts & need:
            chosen.append(c)
            need -= parts
    assert not need
    return chosen


def part_rows(slot: int, move: int, target: int) -> list[dict[str, Any]]:
    return [
        r
        for r in OUTCOMES
        if (r["slot"], r["move"], r["target"], r["mega"]) == (slot, move, target, False)
    ]


# -- the set, the option and the identity ----------------------------------------------------


def test_the_option_reads_the_outcome_set_and_it_has_its_own_layout_and_version() -> None:
    assert read_options("encoder", {CANDIDATE_FEATURES: SET})
    assert rows_of.reads_arrays("probe-outcome/1")
    assert rows_of.reads_outcomes("probe-outcome/1")
    assert not rows_of.reads_outcomes("probe-damage/2")
    assert not rows_of.reads_belief_threats("probe-outcome/1")
    assert rows_of.array_names("probe-outcome/1") == (*rows_of.ARRAY_NAMES, "outcomes")
    assert LAYOUT.outcome
    assert not Layout.of(TWO).outcome
    assert encoder_version(game(), ON) != encoder_version(game(), TWO)
    assert len(LAYOUT.cand_cat) == len(Layout.of(TWO).cand_cat) + 2 * SLOTS + 2 * BRANCHES
    assert len(LAYOUT.cand_num) == NUM_AT + 3 * SLOTS + 2 * BRANCHES
    with pytest.raises(OptionError):
        read_options("encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-outcome/2"}})


def test_the_second_set_keeps_its_layout_version_and_kinds() -> None:
    """The `/2` columns, categories and version do not move: history_unknown is appended only
    under the outcome set (its kinds are longer by exactly that category)."""
    assert encoder_version(game(), replace(TINY_ENCODER)) == TODAY_ENCODER_VERSION
    assert "outcome" not in str(Layout.of(TWO).as_dict())
    assert rows_of.KIND_CATEGORIES_OUTCOME == rows_of.KIND_CATEGORIES + 1
    assert rows_of.PROBE_KIND_CATEGORIES_OUTCOME == rows_of.PROBE_KIND_CATEGORIES + 1
    assert rows_of.SPEED_CATEGORIES_OUTCOME == rows_of.SPEED_CATEGORIES + 1
    assert rows_of.KINDS_OUTCOME[:-1] == rows_of.KINDS
    assert rows_of.PROBE_KINDS_OUTCOME[:-1] == rows_of.PROBE_KINDS
    assert rows_of.SPEED_KINDS_OUTCOME[:-1] == rows_of.SPEED_KINDS
    for reason in (*rows_of.REASONS, *rows_of.ALL_REASONS):
        row = {"kind": "unsupported", "reason": reason}
        if reason in rows_of.REASONS:
            assert rows_of.category(row, True) == rows_of.category(row)
        assert rows_of.probe_category(row, True) == rows_of.probe_category(row)
        assert rows_of.speed_category(row, True) == rows_of.speed_category(row)
    assert VALUE_FREE_FEATURE_SETS == ("probe-damage/1", "probe-damage-belief/1")


def test_the_outcome_words_are_part_of_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    base = encoder_version(game(), ON)
    for name, value in (
        ("MAX_BRANCHES", 4),
        ("OUTCOME_POOLED", ("mean", "min")),
        ("PRECONDITIONS", ("passes", "fails")),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(rows_of, name, value)
            # the constants are hashed, whether or not a column count moves with them
            assert encoder_version(game(), ON) != base, name


def test_the_family_label_names_the_history_and_the_outcomes() -> None:
    assert "move outcomes" in FEATURES_OUTCOME_FAMILY
    assert "battle history" in FEATURES_OUTCOME_FAMILY


# -- the outcome columns ----------------------------------------------------------------------


def test_a_part_s_block_is_pooled_in_its_foe_s_group_and_listed_per_branch() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 1 1, move 1 1")
    cat, num = encoded.cand_cat[i, 0], encoded.cand_num[i, 0]
    # Part (0, 1, 1) against foe 1: switch passes (95%), Protect fails (0), Thunderbolt (80%).
    assert cat[CAT_AT].item() == KIND["computed"]
    assert cat[CAT_AT + 1].item() == PRE["chance"]  # mixed: some pass, one fails
    assert num[NUM_AT].item() == pytest.approx((0.95 + 0.0 + 0.8) / 3)
    assert num[NUM_AT + 1].item() == 0.0  # the minimum
    assert num[NUM_AT + 2].item() == 1.0  # known
    # Foe 2's group is empty: no category, zero numbers.
    assert cat[CAT_AT + 2 : CAT_AT + 4].eq(NONE).all()
    assert not num[NUM_AT + 3 : NUM_AT + 6].any()
    # Per branch: kind and the foe's move, then success and the assumed flag.
    branch = CAT_AT + 2 * SLOTS
    expected = [("passes", None), ("fails", "protect"), ("passes", "thunderbolt")]
    for b, (kind, move) in enumerate(expected):
        assert cat[branch + 2 * b].item() == BRANCH[kind]
        assert cat[branch + 2 * b + 1].item() == (NONE if move is None else MOVES.lookup(move))
    assert cat[branch + 6 :].eq(NONE).all()  # the two absent branches
    successes = NUM_AT + 3 * SLOTS
    assert [num[successes + 2 * b].item() for b in range(3)] == pytest.approx([0.95, 0.0, 0.8])
    assert not num[successes + 6 :].any()
    assert not num[successes + 1 : successes + 6 : 2].any()  # no branch is assumed


def test_an_assumed_branch_is_flagged_and_its_foe_move_is_listed() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 1 2, move 1 1")  # part (0, 1, 2): switch, then the assumed Protect
    cat, num = encoded.cand_cat[i, 0], encoded.cand_num[i, 0]
    group = CAT_AT + 2  # foe 2
    assert cat[group].item() == KIND["computed"]
    assert cat[group + 1].item() == PRE["passes"]  # every computed branch passes
    assert num[NUM_AT + 3].item() == pytest.approx((0.95 + 1.0) / 2)
    assert num[NUM_AT + 4].item() == pytest.approx(0.95)
    branch = CAT_AT + 2 * SLOTS
    assert cat[branch + 3].item() == MOVES.lookup("protect")
    assert num[NUM_AT + 3 * SLOTS + 3].item() == 1.0  # assumed


def test_an_ally_targeted_part_is_about_the_lowest_living_foe() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 1 -2, move 1 1")  # part (0, 1, -2): about foe location 1
    cat, num = encoded.cand_cat[i, 0], encoded.cand_num[i, 0]
    assert cat[CAT_AT + 1].item() == PRE["chance"]
    assert num[NUM_AT].item() == pytest.approx((0.95 + 0.5 + 0.8) / 3)
    assert cat[CAT_AT + 2 : CAT_AT + 4].eq(NONE).all()
    # An untargeted part (target 0) is the same.
    j = candidate(SAMPLE, "move 1 1, move 3")
    assert encoded.cand_cat[j, 1, CAT_AT + 1].item() == PRE["chance"]


def test_an_unsupported_branch_names_its_reason_and_adds_no_number() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 2 -2, move 1 1")  # part (0, 2, -2): Thunderbolt is history_unknown
    cat, num = encoded.cand_cat[i, 0], encoded.cand_num[i, 0]
    assert cat[CAT_AT].item() == KIND["unsupported:history_unknown"]
    assert cat[CAT_AT + 1].item() == PRE["chance"]
    assert num[NUM_AT].item() == pytest.approx((0.95 + 0.5) / 2)  # the computed branches only
    branch = CAT_AT + 2 * SLOTS
    assert cat[branch + 4].item() == BRANCH["unsupported:history_unknown"]
    assert num[NUM_AT + 3 * SLOTS + 4].item() == 0.0


def test_a_part_with_no_computed_branch_has_no_precondition_and_known_zero() -> None:
    def every_branch_unknown(f: dict[str, Any]) -> None:
        for r in f["outcomes"]:
            if (r["slot"], r["move"], r["target"]) == (0, 1, 1):
                for key in ("precondition", "precondition_ppm", "hit_ppm", "success_ppm"):
                    r.pop(key)
                r.update(kind="unsupported", reason="random_effect")

    encoded = ENCODE.encode(edited(every_branch_unknown))
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 1 1, move 1 1")
    assert encoded.cand_cat[i, 0, CAT_AT].item() == KIND["unsupported:random_effect"]
    assert encoded.cand_cat[i, 0, CAT_AT + 1].item() == NONE
    assert not encoded.cand_num[i, 0, NUM_AT : NUM_AT + 3].any()


def test_a_switch_a_pass_and_a_part_off_the_foes_carry_no_outcome() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    i = candidate(SAMPLE, "move 1 1, switch 3")
    assert encoded.cand_cat[i, 1, CAT_AT:].eq(0).all()
    assert not encoded.cand_num[i, 1, NUM_AT:].any()


def test_history_unknown_reaches_its_own_category_in_every_array() -> None:
    def unknown(f: dict[str, Any]) -> None:
        for key, row in (
            ("rows", f["rows"][0]),
            ("threats", f["threats"][0]),
            ("speeds", f["speeds"][0]),
            ("own_priorities", f["own_priorities"][0]),
            ("foe_priorities", f["foe_priorities"][0]),
        ):
            keep = {k: row[k] for k in row if k in ("slot", "move", "target", "mega", "own", "foe")}
            row.clear()
            row.update(keep, kind="unsupported", reason="history_unknown")
            assert key

    q = edited(unknown)
    assert rows_problem(q["candidate_features"]) is None
    encoded = ENCODE.encode(q)
    assert encoded.thr_cat is not None
    assert encoded.thr_cat[0, 1].item() == RESERVED + 1 + rows_of.ALL_REASONS_OUTCOME.index(
        "history_unknown"
    )
    assert rows_of.category({"kind": "unsupported", "reason": "history_unknown"}, True) == (
        RESERVED + 1 + 3
    )


def test_history_unknown_is_not_a_reason_of_the_other_sets() -> None:
    q = copy.deepcopy(SAMPLE2)
    row = q["candidate_features"]["threats"][0]
    keep = {k: row[k] for k in ("own", "mega", "foe", "move")}
    row.clear()
    row.update(keep, kind="unsupported", reason="history_unknown")
    with pytest.raises(EncodingError, match="history_unknown"):
        ENCODE2.encode(q)


# -- fail closed -----------------------------------------------------------------------------


def test_a_missing_array_block_or_branch_is_refused() -> None:
    with pytest.raises(EncodingError, match="lacks the array outcomes"):
        ENCODE.encode(edited(lambda f: f.pop("outcomes")))
    # A part with a foe to ask about and no block.
    gone = edited(
        lambda f: f.update(
            outcomes=[r for r in f["outcomes"] if (r["slot"], r["move"], r["target"]) != (0, 1, 1)]
        )
    )
    with pytest.raises(EncodingError, match="no candidate_features outcomes"):
        ENCODE.encode(gone)
    # A block with no switch branch first.
    no_switch = edited(
        lambda f: f.update(outcomes=[r for r in f["outcomes"] if r["action"] != "switch"])
    )
    with pytest.raises(EncodingError, match="switch branch"):
        ENCODE.encode(no_switch)
    # A block with too many branches.
    crowded = edited(lambda f: f["outcomes"].extend(copy.deepcopy(f["outcomes"][1:2]) * 5))
    with pytest.raises(EncodingError):
        ENCODE.encode(crowded)
    # A block that names a move branch before its switch branch.
    swapped = edited(lambda f: f["outcomes"].insert(0, f["outcomes"].pop(1)))
    with pytest.raises(EncodingError, match="switch branch"):
        ENCODE.encode(swapped)


def test_a_block_of_a_part_no_candidate_has_is_refused() -> None:
    foreign = edited(
        lambda f: f["outcomes"].extend(
            [{**copy.deepcopy(r), "move": 9} for r in part_rows(0, 1, 1)]
        )
    )
    with pytest.raises(EncodingError, match="no candidate has"):
        ENCODE.encode(foreign)


def test_a_block_about_a_foe_the_state_lacks_is_refused() -> None:
    far = edited(
        lambda f: f["outcomes"].extend(
            [{**copy.deepcopy(r), "move": 9, "target": 9} for r in part_rows(0, 1, 1)]
        )
    )
    # The part is about foe location 9, which the format lacks; no candidate has it either.
    with pytest.raises(EncodingError):
        ENCODE.encode(far)
    with pytest.raises(rows_of.RowError, match="foe location"):
        rows_of.outcome_columns(part_rows(0, 1, 1), 9, MOVES.lookup_or_unknown)


def test_an_outcomes_array_twice_is_refused() -> None:
    twice = edited(lambda f: f["outcomes"].insert(1, copy.deepcopy(f["outcomes"][0])))
    with pytest.raises(EncodingError, match="twice"):
        ENCODE.encode(twice)


def test_outcomes_under_another_set_are_refused_and_the_outcome_set_wants_them() -> None:
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE2.encode(SAMPLE)
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE.encode(SAMPLE2)
    # An outcomes array under a config that reads the second set (hash equal, so only the array
    # tells): refused, since the checkpoint was not trained with it.
    sneaky = copy.deepcopy(SAMPLE2)
    sneaky["candidate_features"]["outcomes"] = copy.deepcopy(OUTCOMES)
    with pytest.raises(EncodingError, match="outcomes this feature set does not read"):
        ENCODE2.encode(sneaky)
    plain = copy.deepcopy(SAMPLE)
    with pytest.raises(EncodingError, match="does not read"):
        Encoder(game(), TINY_ENCODER).encode(plain)


def test_the_readers_and_the_encoder_agree_on_the_sample() -> None:
    features = SAMPLE["candidate_features"]
    assert rows_problem(features) is None
    assert coverage_problem(SAMPLE, features, damaging_moves(game())) is None


# -- the network and the checkpoint -----------------------------------------------------------


def test_the_outcomes_are_an_input_of_the_score() -> None:
    network = net()
    base = network(collate([ENCODE.encode(SAMPLE)]))
    changed = copy.deepcopy(SAMPLE)
    for r in changed["candidate_features"]["outcomes"]:
        if r["kind"] == "computed" and r["success_ppm"] == 800000:
            r.update(hit_ppm=400000, success_ppm=400000)
    other = network(collate([ENCODE.encode(changed)]))
    assert not torch.equal(base.logits, other.logits)


def test_a_training_step_reaches_the_outcome_columns() -> None:
    network: Any = net(seed=3)
    batch = collate([ENCODE.encode(SAMPLE)] * 2)
    out = network(batch)
    loss: Any = -torch.log_softmax(out.logits.masked_fill(~batch.cand_mask, -1e9), -1)[:, 0].mean()
    loss.backward()
    first = NUM_AT - 0
    assert network.cand_features.weight.grad[:, first:].abs().sum() > 0
    for kind in ("outcome_kind", "outcome_pre", "outcome_branch_kind"):
        assert network.embeddings[kind].weight.grad is not None


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    network: Any = net(seed=3).eval()
    one = ENCODE.encode(SAMPLE)
    other = ENCODE.encode(edited(lambda f: f["outcomes"][0].update(success_ppm=1)))
    with torch.no_grad():
        alone = network(collate([one]))
        network(collate([other]))
        after = network(collate([one]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)


def test_a_padded_batch_matches_its_queries_scored_alone() -> None:
    network: Any = net(seed=3).eval()
    one = ENCODE.encode(SAMPLE)
    smaller = copy.deepcopy(SAMPLE)
    smaller["candidates"] = covering(SAMPLE)
    two = ENCODE.encode(smaller)
    with torch.no_grad():
        both = network(collate([one, two]))
        alone_one, alone_two = network(collate([one])), network(collate([two]))
    n = two.pending.shape[0]
    assert torch.allclose(both.logits[0], alone_one.logits[0], atol=1e-5)
    assert torch.allclose(both.logits[1, :n], alone_two.logits[0, :n], atol=1e-5)


def test_the_a1_and_a2_forms_with_the_same_arrays_get_identical_logits() -> None:
    a2 = sample("policy-query/valid/battle-a2.json")
    a2["candidate_features"] = copy.deepcopy(SAMPLE["candidate_features"])
    model = net()
    a1 = ENCODE.encode(SAMPLE)
    b = ENCODE.encode(a2)
    assert all(torch.equal(v, b.tensors()[k]) for k, v in a1.tensors().items())
    assert torch.equal(model(collate([a1])).logits, model(collate([b])).logits)


def test_an_incremental_session_gives_the_tensors_of_a_fresh_encode() -> None:
    fresh = ENCODE.encode(SAMPLE)
    session = ENCODE.session().encode(SAMPLE)
    for name, tensor in fresh.tensors().items():
        assert torch.equal(session.tensors()[name], tensor), name


def test_the_checkpoint_is_format_two_and_loads_only_under_the_outcome_set() -> None:
    network = net(seed=3)
    data = checkpoint_bytes(network, game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.layout.outcome
    other = replace(TINY_MODEL, encoder=TWO)
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=other)
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=TINY_MODEL)


def test_the_preview_network_reads_the_speed_columns_and_ignores_the_outcomes() -> None:
    preview = init_network("preview", game(), ON_MODEL, seed=3)
    assert "outcome_kind" not in preview.embeddings
    assert preview.embeddings["speed_relation"].num_embeddings == rows_of.SPEED_CATEGORIES_OUTCOME


# -- composition ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "others",
    [
        {HISTORY_FACTS: True},
        {RECENT_HISTORY: True},
        {CANDIDATE_MATCHUP: True},
        {CANDIDATE_TARGET: True},
        {
            HISTORY_FACTS: True,
            RECENT_HISTORY: True,
            CANDIDATE_MATCHUP: True,
            CANDIDATE_TARGET: True,
        },
    ],
    ids=["history", "recent", "matchup", "target", "all"],
)
def test_it_composes_with_the_other_encoder_options(others: dict[str, Any]) -> None:
    config = replace(
        TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET, **others})
    )
    assert encoder_version(game(), config) != ENCODE.version
    encoder = Encoder(game(), config)
    encoded = encoder.encode(SAMPLE)
    plain = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    assert plain.cand_num is not None
    assert encoded.cand_num.shape[2] == len(Layout.of(config, game()).cand_num)
    network: Any = init_network("battle", game(), replace(TINY_MODEL, encoder=config), seed=3)
    out = network(collate([encoded]))
    assert torch.isfinite(out.logits[:, 0]).all()
    # The outcome columns are the same numbers whatever else is set.
    if CANDIDATE_MATCHUP not in others:
        i = candidate(SAMPLE, "move 1 1, move 1 1")
        assert torch.equal(
            encoded.cand_num[i, 0, -(3 * SLOTS + 2 * BRANCHES) :],
            plain.cand_num[i, 0, -(3 * SLOTS + 2 * BRANCHES) :],
        )


@pytest.mark.parametrize(
    "network_options",
    [{"shared_trunk": True}, {"separate_value": "network"}, {"separate_value": "detached"}],
)
def test_it_composes_with_the_network_options(network_options: dict[str, Any]) -> None:
    model = replace(ON_MODEL, options=read_options("network", network_options))
    for role in ("battle", "preview"):
        network: Any = init_network(role, game(), model, seed=3)
        assert network is not None
    battle: Any = init_network("battle", game(), model, seed=3)
    out = battle(collate([ENCODE.encode(SAMPLE)]))
    assert torch.isfinite(out.logits[:, 0]).all()
    assert torch.isfinite(out.value).all()


def test_the_pooling_helpers_cover_every_branch_count() -> None:
    cats, nums = rows_of.outcome_no_columns()
    assert len(cats) == 2 * SLOTS + 2 * BRANCHES
    assert not any(cats)
    assert len(nums) == 3 * SLOTS + 2 * BRANCHES
    assert not any(nums)
    assert len(rows_of.outcome_blocks(OUTCOMES)) == 18
    cols = rows_of.outcome_columns(part_rows(0, 1, 1), 2, MOVES.lookup_or_unknown)
    assert cols[0][2] == KIND["computed"]  # the group of foe 2
    assert cols[0][0] == NONE


def test_a_block_of_one_row_a_dead_foe_and_the_pooled_extremes_are_handled() -> None:
    one = edited(
        lambda f: f.update(
            outcomes=[
                r
                for r in f["outcomes"]
                if (r["slot"], r["move"], r["target"]) != (0, 1, 1) or r["action"] == "switch"
            ]
        )
    )
    with pytest.raises(EncodingError, match="one to four move branches"):
        ENCODE.encode(one)
    # Every computed branch fails, or every one passes: the pooled precondition says so.
    fails = [
        {"kind": "computed", "precondition": "fails", "success_ppm": 0},
        {"kind": "computed", "precondition": "fails", "success_ppm": 0},
    ]
    assert rows_of.outcome_pooled(fails)[0][1] == PRE["fails"]
    assert rows_of.outcome_pooled([])[0][1] == NONE


def test_a_part_about_no_living_foe_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Encoder, "_foe_alive", lambda self, state, loc: False)
    with pytest.raises(EncodingError, match="foe the state lacks"):
        ENCODE.encode(SAMPLE)


def test_the_closed_outcome_enums_are_the_contract_s_in_their_order() -> None:
    defs = json.loads((ROOT / "packages/schema/contracts/v1/policy-query.schema.json").read_text())[
        "$defs"
    ]
    assert tuple(defs["UnsupportedOutcomeRow"]["properties"]["reason"]["enum"]) == (
        rows_of.OUTCOME_REASONS
    )
    assert tuple(defs["ComputedOutcomeRow"]["properties"]["precondition"]["enum"]) == (
        rows_of.PRECONDITIONS
    )
    assert rows_of.OUTCOME_REASONS == rows_of.REASONS_OUTCOME
