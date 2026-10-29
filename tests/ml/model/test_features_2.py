"""The ``probe-damage/2`` arrays as encoder inputs (ADR-022 erratum E-22.3.3, #1175).

Threat rows go to the own Pokemon they are about, speed relations to their (own, foe) category,
own priorities to their slot part, a Mega variant's summaries to the Mega candidate's slot part,
and foe priorities to the foe's token. A row the checkpoint needs and the query lacks is refused;
an unsupported row reaches its learned category. The sample's numbers are invented
(``fixtures/contracts/v1/policy-query/valid/battle-features-2.json``).
"""

from __future__ import annotations

import ast
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
from areneon_ml.model import features as rows_of
from areneon_ml.model.batch import collate
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
    OptionError,
    read_options,
)
from areneon_ml.worker.features import coverage_problem, damaging_moves
from areneon_ml.worker.view import encoder_input, view_digest

from .support import (
    RANDOM,
    TINY_ENCODER,
    TINY_MODEL,
    candidates_for,
    encoder,
    game,
    observation,
    request_points,
    sample,
)
from .test_features import SAMPLE as ONE_SAMPLE
from .test_features import SOURCES
from .test_options import TODAY_ENCODER_VERSION

SAMPLE = sample("policy-query/valid/battle-features-2.json")
SET = {key: SAMPLE["candidate_features"][key] for key in ("feature_set", "feature_set_hash")}
ON = replace(TINY_ENCODER, options=read_options("encoder", {CANDIDATE_FEATURES: SET}))
ON_MODEL = replace(TINY_MODEL, encoder=ON)
ENCODE = Encoder(game(), ON)
SLOTS = 2
FEATURE_COLUMNS = BASE_CAND_COLUMNS + SLOTS  # the first column after the `/1` categories
COMPUTED = RESERVED
REASON = {reason: RESERVED + 1 + i for i, reason in enumerate(rows_of.ALL_REASONS)}
RELATION = {name: RESERVED + i for i, name in enumerate(rows_of.RELATIONS)}
SPEED_REASON = {r: RESERVED + len(RELATION) + i for i, r in enumerate(rows_of.ALL_REASONS)}
SPEED_AT = {"mon": len(Layout.of(TINY_ENCODER).mon_cat)}  # first speed column of a token


def query(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(SAMPLE), **changes}


def edited(edit: Any) -> dict[str, Any]:
    q = query()
    edit(q["candidate_features"])
    return q


def net(seed: int = 1) -> Any:
    return init_network("battle", game(), ON_MODEL, seed)


def request_of(q: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(q["observation"]["log"][-1][len("|request|") :])
    return loaded


def token_of(q: dict[str, Any], own: int, encoded: Any) -> int:
    """The token row of the own Pokemon at ``own`` in the request, found by its species."""
    details = request_of(q)["side"]["pokemon"][own]["details"].split(",")[0]
    species = game().species.lookup(details.lower().replace("-", "").replace(" ", ""))
    rows = [
        i
        for i in range(encoded.n_own)
        if encoded.mon_cat[i, 0].item() == species or encoded.mon_cat[i, 1].item() == species
    ]
    assert len(rows) == 1
    return rows[0]


def candidate(q: dict[str, Any], choice: str) -> int:
    return [c["choice"] for c in q["candidates"]].index(choice)


# -- the option and the identity --------------------------------------------------------------


def test_the_option_reads_the_second_set() -> None:
    assert read_options("encoder", {CANDIDATE_FEATURES: SET})
    with pytest.raises(OptionError):
        read_options("encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/3"}})


def test_the_second_set_has_its_own_layout_and_version() -> None:
    one = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder", {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/1"}}
        ),
    )
    assert encoder_version(game(), ON) != encoder_version(game(), one)
    layout = Layout.of(ON)
    assert layout.thr_num == rows_of.THREAT_NUMBERS
    assert [kind for _, kind in layout.mon_cat[-SLOTS:]] == ["speed_relation"] * SLOTS
    assert layout.cand_num[-1] == "own_priority"
    assert Layout.of(one).thr_num == ()
    assert "thr_num" not in Layout.of(one).as_dict()
    assert "thr_num" in layout.as_dict()


def test_the_absent_option_and_the_first_set_keep_their_versions_and_tensors() -> None:
    assert encoder().version == TODAY_ENCODER_VERSION
    plain = encoder().encode(sample("policy-query/valid/battle.json"))
    assert plain.thr_idx is None
    assert "thr_idx" not in plain.tensors()
    one_query = ONE_SAMPLE
    one = Encoder(
        game(),
        replace(
            TINY_ENCODER,
            options=read_options(
                "encoder",
                {
                    CANDIDATE_FEATURES: {
                        key: one_query["candidate_features"][key]
                        for key in ("feature_set", "feature_set_hash")
                    }
                },
            ),
        ),
    )
    assert one.encode(one_query).thr_idx is None
    assert len(one.layout.cand_num) == len(rows_of.NUMBERS) * SLOTS


def test_the_second_set_scales_are_part_of_the_version() -> None:
    changed = rows_of.PRIORITY_SCALE
    try:
        rows_of.PRIORITY_SCALE = changed * 2
        assert encoder_version(game(), ON) != ENCODE.version
    finally:
        rows_of.PRIORITY_SCALE = changed
    assert encoder_version(game(), ON) == ENCODE.version


def test_the_checkpoint_is_format_two_and_loads_only_under_the_second_set() -> None:
    data = checkpoint_bytes(net(), game())
    assert read_header(data)["__metadata__"]["areneon.format"] == FORMAT_OPTIONS
    loaded = load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=ON_MODEL)
    assert loaded.config == ON_MODEL
    assert loaded.probing
    first = replace(
        ON_MODEL,
        encoder=replace(
            ON,
            options=read_options(
                "encoder",
                {CANDIDATE_FEATURES: {**SET, "feature_set": "probe-damage/1"}},
            ),
        ),
    )
    with pytest.raises(CheckpointOptionsError):
        load_checkpoint(data, game(), checkpoint_id(data), "battle", configured=first)


# -- where each array goes --------------------------------------------------------------------


def test_each_threat_row_names_its_own_pokemon_and_its_foe() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.thr_idx is not None
    assert encoded.thr_cat is not None
    rows = SAMPLE["candidate_features"]["threats"]
    assert encoded.thr_idx.shape == (len(rows), 3)
    for i, row in enumerate(rows):
        own, foe, mega = encoded.thr_idx[i].tolist()
        assert own == token_of(SAMPLE, row["own"], encoded)
        assert foe == encoded.n_own + (row["foe"] - 1)  # the foe in active slot foe - 1
        assert mega == 0
        assert encoded.thr_cat[i, 0].item() == game().moves.lookup(row["move"])


def test_a_threat_row_carries_its_numbers_and_its_foe_move_s_priority() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.thr_num is not None
    first = SAMPLE["candidate_features"]["threats"][0]  # protect from foe 1 into own 0
    expected = [first["mean_ppm"] / 1e6, first["max_ppm"] / 1e6, first["min_ppm"] / 1e6, 1.0, 0.0]
    assert encoded.thr_num[0, :5].tolist() == pytest.approx(expected)
    assert encoded.thr_num[0, 5].item() == pytest.approx(4 / rows_of.PRIORITY_SCALE)
    assert encoded.thr_num[1, 5].item() == pytest.approx(0.0)  # thunderbolt: priority 0


def test_an_unsupported_threat_row_reaches_its_category_with_no_number() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.thr_cat is not None
    assert encoded.thr_num is not None
    rows = SAMPLE["candidate_features"]["threats"]
    index = [i for i, r in enumerate(rows) if r["kind"] == "unsupported"]
    assert index
    for i in index:
        assert encoded.thr_cat[i, 1].item() == REASON["random_effect"]
        assert not encoded.thr_num[i, :5].any()
    assert encoded.thr_cat[0, 1].item() == COMPUTED
    for reason in ("query_kind", "scenario_rule_table"):
        changed = edited(
            lambda f, reason=reason: [
                r.update(reason=reason) for r in f["threats"] if "reason" in r
            ]
        )
        again = ENCODE.encode(changed).thr_cat
        assert again is not None
        assert again[index[0], 1].item() == REASON[reason]


def test_the_foe_priority_rows_name_their_foe_and_move() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.foe_idx is not None
    assert encoded.foe_cat is not None
    assert encoded.foe_num is not None
    rows = SAMPLE["candidate_features"]["foe_priorities"]
    for i, row in enumerate(rows):
        assert encoded.foe_idx[i, 0].item() == encoded.n_own + row["foe"] - 1
        assert encoded.foe_cat[i, 0].item() == game().moves.lookup(row["move"])
        assert encoded.foe_cat[i, 1].item() == COMPUTED
        assert encoded.foe_num[i, 0].item() == pytest.approx(row["priority"] / 8)
    unsupported = edited(
        lambda f: (
            f["foe_priorities"][0].update(
                kind="unsupported", reason="fractional_priority", priority=None
            )
            or f["foe_priorities"][0].pop("priority")
        )
    )
    again = ENCODE.encode(unsupported)
    assert again.foe_cat is not None
    assert again.foe_num is not None
    assert again.foe_cat[0, 1].item() == REASON["fractional_priority"]
    assert again.foe_num[0, 0].item() == 0.0


def test_each_speed_relation_lands_on_its_own_token_and_foe_location() -> None:
    encoded = ENCODE.encode(SAMPLE)
    first = SPEED_AT["mon"]
    expected = {
        (0, 1): RELATION["tie"],
        (0, 2): RELATION["foe_first"],
        (1, 1): RELATION["foe_first"],
        (1, 2): SPEED_REASON["unqueueable"],
        (2, 1): RELATION["own_first"],
        (2, 2): RELATION["tie"],
        (3, 1): RELATION["tie"],
        (3, 2): RELATION["foe_first"],
    }
    for (own, foe), category in expected.items():
        token = token_of(SAMPLE, own, encoded)
        assert encoded.mon_cat[token, first + foe - 1].item() == category
    # A foe's token carries none.
    assert encoded.mon_cat[encoded.n_own :, first:].eq(NONE).all()


def test_a_pokemon_with_no_speed_row_has_none() -> None:
    few = edited(lambda f: f.update(speeds=[r for r in f["speeds"] if r["own"] != 3]))
    # the candidates name no Pokemon 3 (a switch to slot 4 does): the encoder refuses that
    with pytest.raises(EncodingError, match="speed row"):
        ENCODE.encode(few)
    q = query()
    q["candidates"] = [c for c in q["candidates"] if "switch 4" not in c["choice"]]
    q["candidate_features"]["speeds"] = [
        r for r in q["candidate_features"]["speeds"] if r["own"] != 3
    ]
    encoded = ENCODE.encode(q)
    assert encoded.mon_cat[token_of(q, 3, encoded), SPEED_AT["mon"] :].eq(NONE).all()


def test_each_own_priority_lands_on_its_slot_part() -> None:
    encoded = ENCODE.encode(SAMPLE)
    assert encoded.cand_num is not None
    at = FEATURE_COLUMNS  # the own priority category (after the `/1` feature kinds)
    for choice, slot, kind, value in (
        ("move 1 1, move 1 1", 0, COMPUTED, 0.0),
        ("move 2 1, move 1 1", 0, REASON["fractional_priority"], 0.0),
        ("move 3, move 1 1", 0, COMPUTED, 1 / 8),
        ("move 1 1, move 3", 1, COMPUTED, 1 / 8),
    ):
        i = candidate(SAMPLE, choice)
        assert encoded.cand_cat[i, slot, at].item() == kind, choice
        assert encoded.cand_num[i, slot, -1].item() == pytest.approx(value), choice
    # A switch part and a pass carry none.
    i = candidate(SAMPLE, "move 1 1, switch 3")
    assert encoded.cand_cat[i, 1, at].item() == NONE
    assert encoded.cand_num[i, 1, -1].item() == 0.0


def mega_query() -> dict[str, Any]:
    """The sample plus a Mega candidate, with the Mega variant's rows of slot 0."""
    q = query()
    q["candidates"].append({"choice": "move 1 1 mega, move 1 1", "pending": False})
    f = q["candidate_features"]
    f["rows"].append(
        {**copy.deepcopy(f["rows"][0]), "slot": 0, "move": 1, "target": 1, "mega": True}
    )
    f["rows"].sort(key=lambda r: (r["slot"], r["move"], r["target"], r["mega"]))
    f["own_priorities"].insert(
        1, {"kind": "computed", "slot": 0, "move": 1, "mega": True, "priority": 2}
    )
    f["speeds"] += [
        {"kind": "computed", "own": 0, "mega": True, "foe": 1, "relation": "own_first"},
        {"kind": "unsupported", "own": 0, "mega": True, "foe": 2, "reason": "random_effect"},
    ]
    f["speeds"].sort(key=lambda r: (r["own"], r["mega"], r["foe"]))
    f["threats"] += [
        {
            "kind": "computed",
            "own": 0,
            "mega": True,
            "foe": 1,
            "move": "thunderbolt",
            "mean_ppm": 200000,
            "max_ppm": 230000,
            "min_ppm": 180000,
            "ohko_rolls": 3,
        }
    ]
    f["threats"].sort(key=lambda r: (r["own"], r["mega"], r["foe"], r["move"]))
    return q


def test_a_mega_part_gets_its_variant_s_speeds_and_priority() -> None:
    q = mega_query()
    encoded = ENCODE.encode(q)
    assert encoded.cand_num is not None
    i = candidate(q, "move 1 1 mega, move 1 1")
    at = FEATURE_COLUMNS
    assert encoded.cand_cat[i, 0, at].item() == COMPUTED
    assert encoded.cand_num[i, 0, -1].item() == pytest.approx(2 / 8)
    assert encoded.cand_cat[i, 0, at + 1].item() == RELATION["own_first"]
    assert encoded.cand_cat[i, 0, at + 2].item() == SPEED_REASON["random_effect"]
    # the same move without the Evolution has no Mega speed categories
    plain = candidate(q, "move 1 1, move 1 1")
    assert encoded.cand_cat[plain, 0, at + 1 : at + 3].eq(NONE).all()
    assert encoded.thr_idx is not None
    assert encoded.thr_idx[:, 2].sum().item() == 1  # one Mega threat row


def test_a_mega_part_without_the_variant_s_rows_is_refused() -> None:
    q = mega_query()
    q["candidate_features"]["speeds"] = [
        r for r in q["candidate_features"]["speeds"] if not r["mega"]
    ]
    with pytest.raises(EncodingError, match="speed row"):
        ENCODE.encode(q)
    q = mega_query()
    q["candidate_features"]["own_priorities"] = [
        r for r in q["candidate_features"]["own_priorities"] if not r["mega"]
    ]
    with pytest.raises(EncodingError, match="priority row"):
        ENCODE.encode(q)


# -- the network ------------------------------------------------------------------------------


def summaries(model: Any, q: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = collate([ENCODE.encode(q)])
    d = model.config.d_model
    base = torch.zeros(1, batch.mon_cat.shape[1], d)
    added, mega = model._probe_summaries(batch, base)
    return added[0], mega[0]


def test_a_threat_reaches_only_the_token_of_its_own_pokemon() -> None:
    model = net()
    before, _ = summaries(model, SAMPLE)
    encoded = ENCODE.encode(SAMPLE)
    bench = token_of(SAMPLE, 2, encoded)
    changed = edited(
        lambda f: [
            r.update(mean_ppm=500000) for r in f["threats"] if r["own"] == 2 and "mean_ppm" in r
        ]
    )
    after, _ = summaries(model, changed)
    differs = [i for i in range(before.shape[0]) if not torch.equal(before[i], after[i])]
    assert differs == [bench]


def test_a_foe_priority_reaches_only_its_foe_token() -> None:
    # Foe priority rows add to the foe's token; the threat rows also carry a priority, so the
    # own tokens move too: the check is that the other foe's token stays.
    model = net()
    encoded = ENCODE.encode(SAMPLE)
    before, _ = summaries(model, SAMPLE)
    changed = edited(
        lambda f: [r.update(priority=-3) for r in f["foe_priorities"] if r["foe"] == 1]
    )
    after, _ = summaries(model, changed)
    first, second = encoded.n_own, encoded.n_own + 1
    assert not torch.equal(before[first], after[first])
    assert torch.equal(before[second], after[second])


def test_a_mega_variant_s_threats_reach_only_the_mega_candidate() -> None:
    q = mega_query()
    model = net()
    base = model(collate([ENCODE.encode(q)])).logits[0]
    changed = copy.deepcopy(q)
    for row in changed["candidate_features"]["threats"]:
        if row["mega"]:
            row["mean_ppm"] = 700000
    out = model(collate([ENCODE.encode(changed)])).logits[0]
    mega = candidate(q, "move 1 1 mega, move 1 1")
    assert out[mega] != base[mega]
    others = [i for i in range(len(base)) if i != mega]
    assert torch.equal(out[others], base[others])


def test_the_arrays_are_an_input_of_the_score() -> None:
    model = net()
    base = model(collate([ENCODE.encode(SAMPLE)])).logits
    for array, edit in (
        ("threats", lambda f: f["threats"][0].update(mean_ppm=999999)),
        ("speeds", lambda f: f["speeds"][0].update(relation="own_first")),
        ("own_priorities", lambda f: f["own_priorities"][0].update(priority=3)),
        ("foe_priorities", lambda f: f["foe_priorities"][0].update(priority=-2)),
    ):
        changed = ENCODE.encode(edited(edit))
        assert not torch.equal(base, model(collate([changed])).logits), array


def test_a_query_scores_the_same_alone_and_after_other_queries() -> None:
    model = net()
    target = ENCODE.encode(SAMPLE)
    other = edited(lambda f: f["threats"][0].update(mean_ppm=5))
    alone = model(collate([target]))
    model(collate([ENCODE.encode(other)]))
    after = model(collate([target]))
    assert torch.equal(alone.logits, after.logits)
    assert torch.equal(alone.value, after.value)
    assert torch.equal(alone.logits, net()(collate([target])).logits)


def test_the_a1_and_a2_forms_with_the_same_arrays_get_identical_logits() -> None:
    a2 = sample("policy-query/valid/battle-a2.json")
    a2["candidate_features"] = copy.deepcopy(SAMPLE["candidate_features"])
    model = net()
    a1 = ENCODE.encode(SAMPLE)
    b = ENCODE.encode(a2)
    assert all(torch.equal(v, b.tensors()[k]) for k, v in a1.tensors().items())
    assert torch.equal(model(collate([a1])).logits, model(collate([b])).logits)


def test_a_padded_batch_matches_its_queries_scored_alone() -> None:
    model = net()
    short = edited(lambda f: f.update(threats=f["threats"][:6]))
    short["candidates"] = short["candidates"][:10]
    a, b = ENCODE.encode(SAMPLE), ENCODE.encode(short)
    assert a.thr_idx is not None
    assert b.thr_idx is not None
    assert a.thr_idx.shape[0] != b.thr_idx.shape[0]
    both = model(collate([a, b]))
    for row, single in enumerate((a, b)):
        alone = model(collate([single]))
        count = single.pending.shape[0]
        assert torch.allclose(both.logits[row, :count], alone.logits[0], atol=1e-5)


def test_a_query_with_no_threat_and_no_foe_rows_still_scores() -> None:
    empty = edited(
        lambda f: f.update(
            threats=[],
            foe_priorities=[],
        )
    )
    encoded = ENCODE.encode(empty)
    assert encoded.thr_idx is not None
    assert encoded.thr_idx.shape == (0, 3)
    model = net()
    batch = collate([encoded, ENCODE.encode(SAMPLE)])
    assert batch.thr_idx is not None
    assert model(batch).logits.isfinite().any()
    assert model(collate([encoded])).logits.shape[0] == 1


def test_collate_refuses_queries_of_two_layouts() -> None:
    one = Encoder(
        game(),
        replace(
            TINY_ENCODER,
            options=read_options(
                "encoder",
                {
                    CANDIDATE_FEATURES: {
                        key: ONE_SAMPLE["candidate_features"][key]
                        for key in ("feature_set", "feature_set_hash")
                    }
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="feature layout"):
        collate([ENCODE.encode(SAMPLE), one.encode(ONE_SAMPLE)])


def test_a_network_that_reads_the_arrays_refuses_a_batch_without_them() -> None:
    batch = collate([ENCODE.encode(SAMPLE)])
    with pytest.raises(ValueError, match="probe-damage/2 rows"):
        net()(replace(batch, thr_idx=None))
    with pytest.raises(ValueError, match="probe-damage/2 rows"):
        net()(replace(batch, foe_num=None))
    assert batch.thr_cat is not None
    with pytest.raises(ValueError, match="layout"):
        net()(replace(batch, thr_cat=batch.thr_cat[..., :2]))


def test_the_preview_network_ignores_the_threat_rows_but_reads_the_speed_columns() -> None:
    preview = init_network("preview", game(), ON_MODEL, seed=1)
    assert not preview.probing
    q = sample("policy-query/valid/preview-features-2.json")
    logits = preview(collate([ENCODE.encode(q)])).logits
    assert logits.shape[0] == 1
    assert ENCODE.encode(q).mon_cat[:, -SLOTS:].eq(NONE).all()


def test_the_mega_and_matchup_options_compose_with_the_arrays() -> None:
    both = replace(
        TINY_ENCODER,
        options=read_options("encoder", {CANDIDATE_FEATURES: SET, CANDIDATE_MATCHUP: True}),
    )
    encoder_both = Encoder(game(), both)
    encoded = encoder_both.encode(SAMPLE)
    layout = encoder_both.layout
    names = [name for name, _ in layout.cand_cat]
    assert (
        names.index("same_type") < names.index("feature_kind_1") < names.index("own_priority_kind")
    )
    assert encoded.cand_cat.shape[-1] == len(layout.cand_cat)
    model = init_network("battle", game(), replace(TINY_MODEL, encoder=both), seed=1)
    assert model(collate([encoded])).logits.shape == (1, len(SAMPLE["candidates"]))


# -- fail closed ------------------------------------------------------------------------------


def test_a_missing_array_or_row_is_refused() -> None:
    for name in rows_of.ARRAY_NAMES:
        q = query()
        del q["candidate_features"][name]
        with pytest.raises(EncodingError, match=f"lacks the array {name}"):
            ENCODE.encode(q)
    with pytest.raises(EncodingError, match="priority row"):
        ENCODE.encode(edited(lambda f: f["own_priorities"].pop(0)))
    with pytest.raises(EncodingError, match="speed row"):
        ENCODE.encode(edited(lambda f: f["speeds"].pop(0)))
    with pytest.raises(EncodingError, match="no foe priority row"):
        ENCODE.encode(edited(lambda f: f.update(foe_priorities=f["foe_priorities"][1:])))


def test_a_row_for_a_pokemon_or_foe_the_state_lacks_is_refused() -> None:
    with pytest.raises(EncodingError, match="own Pokémon the request lacks"):
        ENCODE.encode(edited(lambda f: f["threats"][-1].update(own=9)))

    def far(f: dict[str, Any]) -> None:
        f["threats"][-1].update(foe=3)
        f["foe_priorities"].append(dict(f["foe_priorities"][0], foe=3, move="thunderbolt"))

    with pytest.raises(EncodingError, match="foe location the state lacks"):
        ENCODE.encode(edited(far))
    with pytest.raises(EncodingError, match="foe location the format lacks"):
        ENCODE.encode(edited(lambda f: f["speeds"][-1].update(foe=3)))
    gone = query()
    gone["observation"]["log"].insert(-1, "|faint|p2a: Venusaur")
    # the builder gives a fainted foe no rows, so a row for its location has no token to find
    with pytest.raises(EncodingError, match="no Pokémon"):
        ENCODE.encode(gone)


def test_a_row_twice_is_refused() -> None:
    with pytest.raises(EncodingError, match="twice"):
        ENCODE.encode(edited(lambda f: f["speeds"].append(dict(f["speeds"][0]))))


def test_the_first_set_encoder_refuses_the_arrays_and_the_second_wants_them() -> None:
    one = Encoder(
        game(),
        replace(
            TINY_ENCODER,
            options=read_options(
                "encoder",
                {
                    CANDIDATE_FEATURES: {
                        key: ONE_SAMPLE["candidate_features"][key]
                        for key in ("feature_set", "feature_set_hash")
                    }
                },
            ),
        ),
    )
    smuggled = copy.deepcopy(ONE_SAMPLE)
    smuggled["candidate_features"]["speeds"] = []
    with pytest.raises(EncodingError, match="arrays this feature set does not read"):
        one.encode(smuggled)
    with pytest.raises(EncodingError, match="another feature set"):
        ENCODE.encode(ONE_SAMPLE)


def test_the_arrays_are_part_of_the_view_only_through_candidate_features() -> None:
    view = encoder_input(SAMPLE)
    assert view["candidate_features"] == SAMPLE["candidate_features"]
    assert view_digest(view) != view_digest(encoder_input(query(candidate_features=None)))


def test_an_incremental_session_gives_the_tensors_of_a_fresh_encode() -> None:
    session = ENCODE.session()
    a, b = session.encode(SAMPLE), ENCODE.encode(SAMPLE)
    assert all(torch.equal(x, b.tensors()[k]) for k, x in a.tensors().items())


def test_no_game_content_is_written_in_the_array_columns() -> None:
    source = ast.parse((SOURCES / "features.py").read_text())
    strings = {
        n.value
        for n in ast.walk(source)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert not {"protect", "thunderbolt", "tailwind", "trickroom", "quickclaw"} & strings


def test_the_closed_enums_are_the_probe_contract_s_in_their_order() -> None:
    schema = json.loads(
        (SOURCES.parents[2] / "packages/schema/contracts/v1/policy-query.schema.json").read_text()
    )
    defs = schema["$defs"]
    assert tuple(defs["UnsupportedSpeedRow"]["properties"]["reason"]["enum"]) == rows_of.ALL_REASONS
    assert (
        tuple(defs["UnsupportedFoePriorityRow"]["properties"]["reason"]["enum"])
        == rows_of.ALL_REASONS
    )
    assert tuple(defs["UnsupportedThreatRow"]["properties"]["reason"]["enum"]) == rows_of.REASONS
    assert tuple(defs["ComputedSpeedRow"]["properties"]["relation"]["enum"]) == rows_of.RELATIONS


def test_the_first_set_keeps_its_version_and_checkpoint_ids() -> None:
    # Computed on the commit before the encoder learned the second set (#1175).
    one = replace(
        TINY_ENCODER,
        options=read_options(
            "encoder",
            {
                CANDIDATE_FEATURES: {
                    key: ONE_SAMPLE["candidate_features"][key]
                    for key in ("feature_set", "feature_set_hash")
                }
            },
        ),
    )
    assert (
        encoder_version(game(), one)
        == "enc1:d23d915a1fb4d288da367db13ef78a9a8504af51ff3750c2793e19850199e696"
    )
    model = replace(TINY_MODEL, encoder=one)
    assert (
        checkpoint_id(checkpoint_bytes(init_network("battle", game(), model, 1), game()))
        == "218bb4daba96f7d0f0fec0c09de97b8511749680db10646dcd6eb0b35a42a1fe"
    )
    assert (
        checkpoint_id(checkpoint_bytes(init_network("preview", game(), model, 1), game()))
        == "4cba4e313bbf62673467b5b90c56a502a80163f79a13639f958244289653806c"
    )


def test_the_target_option_composes_with_the_arrays() -> None:
    # candidate_target (ADR-010 A4 item 7) adds a token column to a candidate part; the arrays add
    # categories, a number and the Mega summary: both reach the slot part and nothing collides.
    both = replace(
        TINY_ENCODER,
        options=read_options("encoder", {CANDIDATE_FEATURES: SET, CANDIDATE_TARGET: True}),
    )
    encoder_both = Encoder(game(), both)
    q = mega_query()
    encoded = encoder_both.encode(q)
    plain = ENCODE.encode(q)
    assert encoded.cand_tok.shape[-1] == plain.cand_tok.shape[-1] + 1
    assert torch.equal(encoded.cand_cat, plain.cand_cat)
    assert torch.equal(encoded.cand_num, plain.cand_num)
    assert torch.equal(encoded.thr_idx, plain.thr_idx)
    model = init_network("battle", game(), replace(TINY_MODEL, encoder=both), seed=1)
    alone = model(collate([encoded])).logits
    assert alone.shape == (1, len(q["candidates"]))
    model(collate([encoder_both.encode(SAMPLE)]))
    assert torch.equal(alone, model(collate([encoded])).logits)
    assert encoder_both.version != ENCODE.version
    mega = candidate(q, "move 1 1 mega, move 1 1")
    changed = copy.deepcopy(q)
    for row in changed["candidate_features"]["threats"]:
        if row["mega"]:
            row["mean_ppm"] = 700000
    out = model(collate([encoder_both.encode(changed)])).logits[0]
    others = [i for i in range(alone.shape[1]) if i != mega]
    assert out[mega] != alone[0, mega]
    assert torch.equal(out[others], alone[0, others])


def forced_switch_query() -> dict[str, Any]:
    """A forced-switch query in the builder's shape: switch-in threats and no foe priorities."""
    for path in RANDOM:
        raw = json.loads(path.read_text())
        channel = raw["p1"] if "p1" in raw else raw["channels"]["p1"]
        lines = channel["lines"] if isinstance(channel, dict) else channel
        for index, body in request_points(lines):
            if "forceSwitch" not in body:
                continue
            log = lines[: index + 1]
            if not any(line.startswith("|switch|p2a") for line in log):
                continue
            cands = [c for c in candidates_for(body) if "switch" in c["choice"]]
            if not cands:
                continue
            slots = sorted(
                {
                    int(part.split()[1]) - 1
                    for c in cands
                    for part in c["choice"].split(", ")
                    if part.startswith("switch")
                }
            )
            q = {
                "observation": observation("p1", log, raw["teams"]["p1"], "battle"),
                "candidates": cands,
                "candidate_features": {
                    **SET,
                    "rows": [],
                    "threats": [
                        {
                            "kind": "computed",
                            "own": own,
                            "mega": False,
                            "foe": 1,
                            "move": "thunderbolt",
                            "mean_ppm": 1,
                            "max_ppm": 2,
                            "min_ppm": 1,
                            "ohko_rolls": 0,
                        }
                        for own in slots
                    ],
                    "speeds": [
                        {"kind": "computed", "own": o, "mega": False, "foe": 1, "relation": "tie"}
                        for o in slots
                    ],
                    "own_priorities": [],
                    "foe_priorities": [],
                },
            }
            try:
                ENCODE.encode(q)
            except EncodingError as exc:
                if "foe priority" in str(exc):
                    raise
                continue
            return q
    raise AssertionError("no forced switch with a shown foe in the corpus")


def test_a_forced_switch_threat_needs_no_foe_priority_row() -> None:
    q = forced_switch_query()
    encoded = ENCODE.encode(q)
    assert encoded.thr_cat is not None
    assert encoded.thr_num is not None
    assert encoded.thr_cat.shape[0] == len(q["candidate_features"]["threats"])
    assert encoded.thr_cat[:, 2].eq(NONE).all()  # the NONE priority kind
    assert not encoded.thr_num[:, 5].any()  # and a zero priority
    assert encoded.foe_idx is not None
    assert encoded.foe_idx.shape[0] == 0
    out = net()(collate([encoded]))
    assert out.logits.shape == (1, len(q["candidates"]))


def test_a_move_request_threat_without_its_foe_priority_row_is_refused() -> None:
    with pytest.raises(EncodingError, match="no foe priority row"):
        ENCODE.encode(edited(lambda f: f.update(foe_priorities=f["foe_priorities"][1:])))
    gone = edited(lambda f: f.update(foe_priorities=f["foe_priorities"][1:]))
    problem = coverage_problem(gone, gone["candidate_features"], damaging_moves(game()))
    assert problem is not None
    assert "foe_priorities must hold a row" in problem
