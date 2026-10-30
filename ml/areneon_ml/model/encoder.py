"""The entity-token encoder (ADR-010 §3): a player's observation to tensors.

Input is exactly what the policy boundary allows (ADR-010 §2): the
observation's own channel (reduced by the clean-room reducer), the player's
own team, and the candidate list with each candidate's ``pending`` flag. It
reads nothing else of a query: not ``decision_context``, not ``identity``
(so not ``rqid``), not the ``rqid`` inside the log's ``|request|`` lines, and
never a verdict of the CloneJudge or an opponent's hidden information, none of
which the observation holds. The A1 and A2 forms of one decision therefore
give identical tensors.

Tokens: one per own Pokémon, one per opponent Pokémon (species from Team
Preview, then only what the reducer extracted, with explicit unknown values),
and the network adds a field token (``field_*``) and an aggregation token. A
candidate is described by its parsed parts and by which tokens it names
(the acting Pokémon, the switch-in, the picked team positions), taken from the
player's own last request. Free-form ids (volatiles, field and side
conditions) are bucketed by SHA-256 into a configured number of buckets: the
engine owns those names, not this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.model import features as rows_of
from areneon_ml.model.candidates import (
    EVENTS,
    SLOTS,
    ParsedChoice,
    UnsupportedCandidate,
    parse_choice,
)
from areneon_ml.model.gamedata import NONE, RESERVED, UNKNOWN, GameData
from areneon_ml.model.options import (
    CANDIDATE_MATCHUP,
    CANDIDATE_TARGET,
    EXACT_VOLATILES,
    RECENT_HISTORY,
    Options,
    feature_set,
    is_set,
    with_options,
)
from areneon_ml.reducer import (
    BattleState,
    ChannelReducer,
    PokemonState,
    SideState,
    VocabularyError,
    reduce_channel,
)
from areneon_ml.reducer.reducer import TYPELESS
from areneon_ml.reducer.state import STAT_KEYS
from areneon_ml.reducer.vocab import PROTOCOL_VOLATILES

# Bump when the meaning of a feature changes without its name or width changing.
ENCODER_FORMAT = 1

REQUEST_KINDS = ("preview", "move", "forceSwitch")
GENDERS = (None, "M", "F", "N")
MAX_SLOT_LETTERS = 3  # triples is the widest gametype the reducer reads
TARGET_OFFSET = 10
MATCHUP_TYPES = 2  # types a position shows: a Pokémon has at most two
BASE_CAND_TOKENS = (
    2  # the candidate token columns every layout has (actor, switch-in); the target follows
)
BASE_CAND_COLUMNS = 5  # the candidate columns every layout has; option columns follow
SAME_TYPE_NOT = RESERVED  # same_type categories: none, unknown, then these two
SAME_TYPE_YES = RESERVED + 1
TARGET_CODES = 20  # 0 untargeted, else TARGET_OFFSET + the engine's target location (-9..9)
_REQUEST = "|request|"
_SIDE = re.compile(r"^p[1-4]: ")
STAT_NAMES = ("atk", "def", "spa", "spd", "spe")
# recent_history: the divisors that are not config fields (turn counts use ``turn_scale``, exact
# hit points ``stat_scale``). Hashed into ``encoder_version``.
PERCENT_SCALE = 100.0  # an opponent's hit points are points of 100
ACTION_ORDER_SCALE = 8.0  # the place among a turn's actions (moves and `|cant|` lines)
STREAK_SCALE = 4.0  # consecutive stalling uses
SIDES = ("own", "opponent")
# exact_volatiles: a volatile the engine prints with a number (`stockpile2`, `perish3`, `fallen4`)
# has the value 1 + number / DERIVED_SCALE in its id's column; one printed with a stat
# (`protosynthesisatk`) has 1 and sets that stat's flag. Hashed into ``encoder_version``.
DERIVED_SCALE = 8.0


class EncodingError(ValueError):
    """The observation or a candidate cannot be encoded; nothing is guessed."""


@dataclass(frozen=True)
class EncoderConfig:
    """Feature widths and scales: configuration, part of ``encoder_version``."""

    volatile_buckets: int
    condition_buckets: int
    max_moves: int  # moves per Pokémon
    max_team: int  # Pokémon per side
    max_picks: int  # Team Preview positions an order names
    stat_scale: float  # divisor of a Pokémon's exact stats
    turn_scale: float  # divisor of the turn counter
    options: Options = ()  # optional encoder options (A4 item 2); empty keeps today's bytes

    def __post_init__(self) -> None:
        limits = {
            "volatile_buckets": 4096,
            "condition_buckets": 4096,
            "max_moves": 16,
            "max_team": 16,
            "max_picks": 16,
        }
        for name, limit in limits.items():
            if not 1 <= getattr(self, name) <= limit:
                raise ValueError(f"{name} must be at least 1 and at most {limit}")
        if self.stat_scale <= 0 or self.turn_scale <= 0:
            raise ValueError("scales must be positive")
        if self.max_picks > self.max_team:
            raise ValueError("max_picks cannot exceed max_team")

    def as_dict(self) -> dict[str, Any]:
        return with_options(
            {
                "volatile_buckets": self.volatile_buckets,
                "condition_buckets": self.condition_buckets,
                "max_moves": self.max_moves,
                "max_team": self.max_team,
                "max_picks": self.max_picks,
                "stat_scale": self.stat_scale,
                "turn_scale": self.turn_scale,
            },
            self.options,
        )


def _matchup_columns(config: EncoderConfig) -> tuple[tuple[str, str], ...]:
    """The ``candidate_matchup`` columns: per position the move's chart cell for each shown type.

    Positions are the opposing slots (one per candidate slot) and the ally; the last column
    is the same-type flag. Empty unless the option is set.
    """
    if not is_set(config.options, CANDIDATE_MATCHUP):
        return ()
    places = (*(f"foe_{i + 1}" for i in range(SLOTS)), "ally")
    cells = tuple(
        (f"matchup_{place}_{k + 1}", "matchup") for place in places for k in range(MATCHUP_TYPES)
    )
    return (*cells, ("same_type", "same_type"))


def _feature_columns(config: EncoderConfig) -> tuple[tuple[str, str], ...]:
    """The ``candidate_features`` categories: one row category per foe location; none unless set."""
    if feature_set(config.options) is None:
        return ()
    return tuple((f"feature_kind_{loc + 1}", "feature_kind") for loc in range(SLOTS))


def _feature_numbers(config: EncoderConfig) -> tuple[str, ...]:
    """The ``candidate_features`` numbers per move part: five per foe location. Empty unless set."""
    if feature_set(config.options) is None:
        return ()
    return tuple(f"feature_{name}_{loc + 1}" for loc in range(SLOTS) for name in rows_of.NUMBERS)


def _reads_arrays(config: EncoderConfig) -> bool:
    """Whether the config reads ``probe-damage/2`` (the set with threat, speed, priority arrays)."""
    named = feature_set(config.options)
    return named is not None and rows_of.reads_arrays(named.feature_set)


def _speed_columns(config: EncoderConfig) -> tuple[tuple[str, str], ...]:
    """Per own token, its speed relation to each foe location (``/2``; none for a foe token)."""
    if not _reads_arrays(config):
        return ()
    return tuple((f"speed_{loc + 1}", "speed_relation") for loc in range(SLOTS))


def _priority_columns(config: EncoderConfig) -> tuple[tuple[str, str], ...]:
    """A move part's own priority category and a Mega part's Mega-variant speed relations."""
    if not _reads_arrays(config):
        return ()
    return (
        ("own_priority_kind", "probe_kind"),
        *((f"speed_mega_{loc + 1}", "speed_relation") for loc in range(SLOTS)),
    )


def _recent(config: EncoderConfig) -> bool:
    return is_set(config.options, RECENT_HISTORY)


def _recent_cat(config: EncoderConfig) -> tuple[tuple[str, str], ...]:
    """The ``recent_history`` category of a Pokémon token: its last move. Empty unless set."""
    return (("last_move", "move"),) if _recent(config) else ()


def _recent_mon_num(config: EncoderConfig) -> tuple[str, ...]:
    """The ``recent_history`` numbers of a Pokémon token. Empty unless set."""
    if not _recent(config):
        return ()
    return (
        "active_turns",
        "last_move_age",
        "last_action_known",
        "last_action_age",
        "last_action_order",
        "hp_taken_this_turn",
        "hp_taken_last_turn",
        "hp_dealt_this_turn",
        "hp_dealt_last_turn",
        "protect_streak",
        *(f"last_target_{side}_{chr(97 + i)}" for side in SIDES for i in range(MAX_SLOT_LETTERS)),
    )


def _exact(config: EncoderConfig) -> bool:
    return is_set(config.options, EXACT_VOLATILES)


def exact_ids(game: GameData) -> tuple[str, ...]:
    """The ids ``exact_volatiles`` gives a column each, sorted: the engine's condition ids, the
    ``PROTOCOL_VOLATILES`` and the moves, abilities and items (the engine prints a volatile, a
    field condition or a side condition under the name of what started it). Needs a version 2
    engine-data table."""
    if game.conditions is None:
        raise EncodingError(
            "exact_volatiles needs a version 2 engine-data table (it has no condition list)"
        )
    return tuple(
        sorted(
            {
                *game.conditions.ids,
                *PROTOCOL_VOLATILES,
                *game.moves.ids,
                *game.abilities.ids,
                *game.items.ids,
            }
        )
    )


def _widths(config: EncoderConfig, game: GameData | None) -> tuple[int, int]:
    """(volatile columns, condition columns) of a token: the buckets, or one per exact id."""
    if not _exact(config):
        return config.volatile_buckets, config.condition_buckets
    if game is None:
        raise EncodingError("exact_volatiles needs the game data to lay out its columns")
    width = len(exact_ids(game))
    return width, width


def _typechange_num(config: EncoderConfig, game: GameData | None) -> tuple[str, ...]:
    """The ``exact_volatiles`` columns of a Pokémon token: whether a ``typechange`` set the types,
    the typeless marker and one flag per type of the engine data, then one flag per stat for the
    stat a Protosynthesis or Quark Drive boosts. Empty unless set."""
    if not _exact(config):
        return ()
    if game is None:
        raise EncodingError("exact_volatiles needs the game data to lay out its columns")
    return (
        "types_known",
        "types_typeless",
        *(f"type_{name}" for name in game.types.ids),
        *(f"boosted_stat_{name}" for name in STAT_NAMES),
    )


def _recent_field_num(config: EncoderConfig, game: GameData | None) -> tuple[str, ...]:
    """The ``recent_history`` numbers of the field token: turns since each condition began."""
    if not _recent(config):
        return ()
    buckets = _widths(config, game)[1]
    return (
        "weather_elapsed",
        "terrain_elapsed",
        *(f"field_turns_{i}" for i in range(buckets)),
        *(f"own_side_turns_{i}" for i in range(buckets)),
        *(f"opponent_side_turns_{i}" for i in range(buckets)),
    )


@dataclass(frozen=True)
class Layout:
    """Names and kinds of every feature column, derived from the configuration."""

    mon_cat: tuple[tuple[str, str], ...]  # (column, vocabulary kind)
    mon_num: tuple[str, ...]
    field_cat: tuple[tuple[str, str], ...]
    field_num: tuple[str, ...]
    cand_cat: tuple[tuple[str, str], ...]
    cand_num: tuple[str, ...] = ()  # numeric columns of a candidate part; only with features
    # The rows of ``probe-damage/2`` (variable count per query, so not part of a token or a part):
    # a threat row is (own token, foe token, Mega flag) with its move and kinds and its numbers;
    # a foe priority row is (foe token) with its move and kind and its number.
    thr_cat: tuple[tuple[str, str], ...] = ()
    thr_num: tuple[str, ...] = ()
    foe_cat: tuple[tuple[str, str], ...] = ()
    foe_num: tuple[str, ...] = ()
    cand_tok: tuple[str, ...] = ("actor", "switch_in")  # token columns; candidate_target adds one

    @classmethod
    def of(cls, config: EncoderConfig, game: GameData | None = None) -> Layout:
        """The layout of ``config``. ``game`` is needed only by ``exact_volatiles``, whose
        column count is the engine data's id count."""
        moves = tuple((f"move_{i + 1}", "move") for i in range(config.max_moves))
        volatiles, buckets = _widths(config, game)
        arrays = _reads_arrays(config)
        return cls(
            mon_cat=(
                ("base_species", "species"),
                ("species", "species"),
                ("temp_species", "species"),
                ("item", "item"),
                ("mega_stone", "item"),
                ("ability", "ability"),
                ("status", "status"),
                ("gender", "gender"),
                ("active_slot", "slot"),
                *moves,
                *_recent_cat(config),
                *_speed_columns(config),
            ),
            mon_num=(
                "own",
                "hp_known",
                "hp_fraction",
                "fainted",
                "brought_known",
                "brought",
                "shiny",
                "mega",
                "item_known",
                "item_lost",
                "ability_suppressed",
                "can_mega_evo",
                "level",
                "stats_known",
                "hp_stat",
                *(f"stat_{name}" for name in STAT_NAMES),
                *(f"boost_{name}" for name in STAT_KEYS),
                *(f"pp_{i + 1}" for i in range(config.max_moves)),
                *(f"volatile_{i}" for i in range(volatiles)),
                *_typechange_num(config, game),
                *_recent_mon_num(config),
            ),
            field_cat=(("weather", "weather"), ("terrain", "terrain"), ("request", "request")),
            field_num=(
                "turn",
                "started",
                *(f"field_{i}" for i in range(buckets)),
                *(f"own_side_{i}" for i in range(buckets)),
                *(f"opponent_side_{i}" for i in range(buckets)),
                *_recent_field_num(config, game),
            ),
            cand_cat=(
                ("action", "action"),
                ("move", "move"),
                ("move_slot", "move_slot"),
                ("target", "target"),
                ("event", "event"),
                *_matchup_columns(config),
                *_feature_columns(config),
                *_priority_columns(config),
            ),
            cand_num=(*_feature_numbers(config), *(rows_of.OWN_NUMBERS if arrays else ())),
            thr_cat=(
                (("move", "move"), ("kind", "probe_kind"), ("foe_priority_kind", "probe_kind"))
                if arrays
                else ()
            ),
            thr_num=rows_of.THREAT_NUMBERS if arrays else (),
            foe_cat=(("move", "move"), ("kind", "probe_kind")) if arrays else (),
            foe_num=rows_of.FOE_NUMBERS if arrays else (),
            cand_tok=(
                ("actor", "switch_in", "target")
                if is_set(config.options, CANDIDATE_TARGET)
                else ("actor", "switch_in")
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mon_cat": [list(c) for c in self.mon_cat],
            "mon_num": list(self.mon_num),
            "field_cat": [list(c) for c in self.field_cat],
            "field_num": list(self.field_num),
            "cand_cat": [list(c) for c in self.cand_cat],
        }
        if self.cand_num:  # absent without the option, so today's layout bytes stay
            out["cand_num"] = list(self.cand_num)
        if self.thr_num:  # likewise: only probe-damage/2 has these rows
            out["thr_cat"] = [list(c) for c in self.thr_cat]
            out["thr_num"] = list(self.thr_num)
            out["foe_cat"] = [list(c) for c in self.foe_cat]
            out["foe_num"] = list(self.foe_num)
        if (
            len(self.cand_tok) != BASE_CAND_TOKENS
        ):  # likewise: only candidate_target adds the target column
            out["cand_tok"] = list(self.cand_tok)
        return out


def encoder_version(game: GameData, config: EncoderConfig) -> str:
    """``enc<format>:`` + SHA-256 of everything the feature values depend on.

    That is the engine-data hash, the config (including the scales), the feature
    layout, the attribute tables as built, and the constants that map a value to
    an index (genders, request kinds, events, target codes, slot letters, the
    bucket hash). A pin bump changes the engine-data hash, so it yields a new
    encoder version, never a silent remap (ADR-010 §3). A change to how a value
    is computed from the state, with the same names and constants, is not
    visible to the hash: bump ``ENCODER_FORMAT``; the golden test in
    ``tests/ml/model/test_encoder.py`` (a pinned version and a tensor digest of
    one fixed query) fails on such a change if nobody did.
    """
    body: dict[str, Any] = {
        "encoder_format": ENCODER_FORMAT,
        "engine_data_hash": game.engine_data_hash,
        "attribute_digest": game.attribute_digest,
        "config": config.as_dict(),
        "layout": Layout.of(config, game).as_dict(),
        "constants": {
            "genders": list(GENDERS),
            "request_kinds": list(REQUEST_KINDS),
            "events": list(EVENTS),
            "target_codes": TARGET_CODES,
            "target_offset": TARGET_OFFSET,
            "max_slot_letters": MAX_SLOT_LETTERS,
            "bucket": "sha256 first 4 bytes big-endian modulo buckets",
        },
    }
    if is_set(config.options, CANDIDATE_MATCHUP):
        # The cell categories are the engine's codes shifted by RESERVED: the count is data,
        # the offsets and the same-type indices are index constants, so they are hashed.
        body["constants"]["matchup"] = {
            "types_shown": MATCHUP_TYPES,
            "reserved": RESERVED,
            "same_type": [SAME_TYPE_NOT, SAME_TYPE_YES],
            "chart_codes": game.chart_codes,
        }
    if is_set(config.options, CANDIDATE_TARGET):
        # The token column is in the layout; the location convention is an index constant.
        body["constants"]["target"] = {
            "location": "foe in active slot L-1 for L>0; own active slot -L-1 for L<0",
        }
    if _exact(config):
        # One column per id, in sorted order: the ids are derived from the engine data (hashed
        # above) and the fixed protocol words, so their hash is the constant; the value rules
        # are index constants.
        body["constants"]["exact_volatiles"] = {
            "ids_sha256": sha256_hex(jcs(list(exact_ids(game)))),
            "protocol_volatiles": sorted(PROTOCOL_VOLATILES),
            "typeless": TYPELESS,
            "volatile": "1 per id; the latest derived number n gives 1 + n / derived_scale in the "
            "id's column, a derived stat suffix sets that stat's boosted_stat flag",
            "derived_scale": DERIVED_SCALE,
            "stats": list(STAT_NAMES),
            "condition": "layers per id on a side, 1 on the field",
        }
    if _recent(config):
        # The columns are in the layout; the divisors that are not config fields and the unit
        # rules are constants (the scales themselves are in the config).
        body["constants"]["recent_history"] = {
            "percent_scale": PERCENT_SCALE,
            "action_order_scale": ACTION_ORDER_SCALE,
            "streak_scale": STREAK_SCALE,
            "slot_letters": MAX_SLOT_LETTERS,
            "units": "taken: own exact hp / stat_scale, opponent points / percent_scale; "
            "dealt: own actor points / percent_scale, opponent actor exact hp / stat_scale",
        }
    if feature_set(config.options) is not None:
        # The set and its hash are in the config; these are the index constants and scales.
        body["constants"]["features"] = {
            "reserved": RESERVED,
            "kinds": list(rows_of.KINDS),
            "numbers": list(rows_of.NUMBERS),
            "ppm": rows_of.PPM,
            "ohko_rolls": rows_of.OHKO_ROLLS,
            "foe_locations": SLOTS,
        }
        if _reads_arrays(config):
            body["constants"]["features"]["arrays"] = {
                "probe_kinds": list(rows_of.PROBE_KINDS),
                "speed_kinds": list(rows_of.SPEED_KINDS),
                "priority_scale": rows_of.PRIORITY_SCALE,
            }
    return f"enc{ENCODER_FORMAT}:{sha256_hex(jcs(body))}"


def bucket(text: str, buckets: int) -> int:
    """A stable bucket of an engine-owned name (SHA-256, not Python's salted hash)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % buckets


@dataclass(frozen=True)
class EncodedQuery:
    """One query as tensors, without a batch dimension.

    ``mon_*`` rows are the own Pokémon first (``n_own``), then the opponent's.
    Candidate token columns hold a row of ``mon_*`` or -1 for none.
    """

    kind: str
    n_own: int
    mon_cat: torch.Tensor  # int64 [M, len(Layout.mon_cat)]
    mon_num: torch.Tensor  # float32 [M, len(Layout.mon_num)]
    field_cat: torch.Tensor  # int64 [len(Layout.field_cat)]
    field_num: torch.Tensor  # float32 [len(Layout.field_num)]
    cand_cat: torch.Tensor  # int64 [N, SLOTS, len(Layout.cand_cat)]
    cand_tok: torch.Tensor  # int64 [N, SLOTS, len(Layout.cand_tok)]: actor, switch-in, target
    team_tok: torch.Tensor  # int64 [N, max_picks]: picked Pokémon in order (Team Preview)
    pending: torch.Tensor  # float32 [N]
    cand_num: torch.Tensor | None = None  # float32 [N, SLOTS, len(Layout.cand_num)]; with features
    # probe-damage/2 only: R threat rows (own token, foe token, Mega flag; move, kinds; numbers)
    # and P foe priority rows (foe token; move, kind; number). R and P vary per query.
    thr_idx: torch.Tensor | None = None  # int64 [R, 3]
    thr_cat: torch.Tensor | None = None  # int64 [R, len(Layout.thr_cat)]
    thr_num: torch.Tensor | None = None  # float32 [R, len(Layout.thr_num)]
    foe_idx: torch.Tensor | None = None  # int64 [P, 1]
    foe_cat: torch.Tensor | None = None  # int64 [P, len(Layout.foe_cat)]
    foe_num: torch.Tensor | None = None  # float32 [P, len(Layout.foe_num)]

    def tensors(self) -> dict[str, torch.Tensor]:
        out = {
            "mon_cat": self.mon_cat,
            "mon_num": self.mon_num,
            "field_cat": self.field_cat,
            "field_num": self.field_num,
            "cand_cat": self.cand_cat,
            "cand_tok": self.cand_tok,
            "team_tok": self.team_tok,
            "pending": self.pending,
        }
        if self.cand_num is not None:  # absent without the option: today's tensors stay the same
            out["cand_num"] = self.cand_num
        for name in ("thr_idx", "thr_cat", "thr_num", "foe_idx", "foe_cat", "foe_num"):
            if (value := getattr(self, name)) is not None:
                out[name] = value
        return out


def _last_request(log: Sequence[str]) -> str | None:
    """The last ``|request|`` line of a channel, or None."""
    found = [line for line in log if line.startswith(_REQUEST)]
    return found[-1] if found else None


@dataclass
class _Request:
    """The parts of the player's own last request a candidate refers to."""

    order: list[int] = field(default_factory=list)  # own token of side.pokemon[0], [1], ...
    moves: list[list[int]] = field(default_factory=list)  # per active slot: engine move indices
    targets: list[list[str | None]] = field(default_factory=list)  # per slot: request target types


@dataclass
class _Probe:
    """The ``probe-damage/2`` arrays of one query, placed on the tokens they belong to."""

    arrays: rows_of.Arrays
    speed_cols: dict[int, list[int]] = field(default_factory=dict)  # token -> category per foe
    threats: list[tuple[list[int], list[int], list[float]]] = field(default_factory=list)
    foes: list[tuple[list[int], list[int], list[float]]] = field(default_factory=list)


class Encoder:
    """Observation + candidates to ``EncodedQuery``: pure and deterministic."""

    def __init__(self, game: GameData, config: EncoderConfig) -> None:
        self.game = game
        self.config = config
        self.layout = Layout.of(self.config, game)
        # exact_volatiles: the column of each id (a v1 table is refused by the layout above)
        self._exact_index = (
            {name: i for i, name in enumerate(exact_ids(game))} if _exact(config) else {}
        )
        self.version = encoder_version(game, self.config)

    # -- public --------------------------------------------------------------

    def encode(self, query: Mapping[str, Any]) -> EncodedQuery:
        """Encode ``query["observation"]`` and ``query["candidates"]``; nothing else is read."""
        observation = query["observation"]
        state = reduce_channel(
            observation["player"],
            observation["log"],
            observation["team"]["team"],
            self.game.vocabulary,
        )
        return self._encode_state(query, state, _last_request(observation["log"]))

    def session(self) -> EncoderSession:
        """An incremental encoder: the same tensors as ``encode``, one reduction per channel."""
        return EncoderSession(self)

    def _encode_state(
        self, query: Mapping[str, Any], state: BattleState, last_request: str | None
    ) -> EncodedQuery:
        observation = query["observation"]
        field_features = self._field(state)
        wanted = "preview" if state.request_kind == "preview" else "battle"
        if observation["kind"] != wanted:
            raise EncodingError(
                f"a {observation['kind']} observation at a {state.request_kind} request"
            )
        request = self._request(state, last_request)
        rows = self._rows(query)
        probe = self._probe(query, request, state) if _reads_arrays(self.config) else None
        candidates = [
            self._candidate(c, request, wanted, state, rows, probe) for c in query["candidates"]
        ]
        return self._assemble(observation["kind"], state, field_features, candidates, probe)

    # -- tokens --------------------------------------------------------------

    def _slot_index(self, side: SideState, mon: PokemonState) -> int:
        for position, letter in enumerate(sorted(side.slots)):
            index = side.slots[letter]
            if index is not None and side.roster[index] is mon:
                return position + 1
        return 0

    def _mon_cat(self, side: SideState, mon: PokemonState) -> list[int]:
        game, own = self.game, mon.own
        if mon.item is not None:
            item = game.items.lookup(mon.item)
        else:
            item = NONE if mon.item_known else UNKNOWN
        if mon.ability is not None:
            ability = game.abilities.lookup(mon.ability)
        else:
            ability = NONE if own else UNKNOWN
        shown = mon.moves_own if own and mon.moves_own is not None else mon.moves_revealed
        pad = NONE if own else UNKNOWN
        if len(shown) > self.config.max_moves:
            # The reducer bounds a list at four slots plus Struggle; truncating would hide moves.
            raise EncodingError(f"a Pokémon shows {len(shown)} moves, more than max_moves")
        moves = [game.moves.lookup(m) for m in shown]
        moves += [pad] * (self.config.max_moves - len(moves))
        if mon.gender not in GENDERS:
            raise EncodingError(f"unsupported gender {mon.gender!r}")
        return [
            game.species.lookup(mon.base_species),
            game.species.lookup(mon.species) if mon.species else UNKNOWN,
            game.species.lookup(mon.temp_species) if mon.temp_species else NONE,
            item,
            game.items.lookup(mon.mega_stone) if mon.mega_stone else NONE,
            ability,
            game.statuses.lookup(mon.status) if mon.status else NONE,
            GENDERS.index(mon.gender),
            self._slot_index(side, mon),
            *moves,
            *self._recent_cat(mon),
        ]

    def _recent_cat(self, mon: PokemonState) -> list[int]:
        """The ``recent_history`` category of a Pokémon: its last move (NONE before any)."""
        if not _recent(self.config):
            return []
        last = mon.recent.last_move
        return [NONE if last is None else self.game.moves.lookup_or_unknown(last)]

    def _recent_num(self, mon: PokemonState, turn: int) -> list[float]:
        """The ``recent_history`` numbers of a Pokémon (see ``_recent_mon_num``).

        Ages are turns before the current one (0: this turn). An amount is in the units the
        Pokémon's side shows (the reducer's ``Recent``): damage taken is in its own units, damage
        dealt in its target's. A sum cannot say whom it hit, so dealt is read in the units of the
        usual target, a foe (points of 100) for an own Pokémon and an own Pokémon (exact hit
        points) for an opponent's; an ally hit by a spread move adds its units to the sum.
        """
        config, recent = self.config, mon.recent
        exact = config.stat_scale
        taken = exact if mon.own else PERCENT_SCALE
        dealt = PERCENT_SCALE if mon.own else exact
        target = [0.0] * (len(SIDES) * MAX_SLOT_LETTERS)
        if recent.last_target is not None:
            side, _, letter = recent.last_target.partition(":")
            place = ord(letter) - 97 if len(letter) == 1 else -1
            if side not in SIDES or not 0 <= place < MAX_SLOT_LETTERS:
                raise EncodingError("a recent last_target names no position")
            target[SIDES.index(side) * MAX_SLOT_LETTERS + place] = 1.0
        move_age = 0.0 if recent.last_move_turn is None else turn - recent.last_move_turn
        action_age = 0.0 if recent.last_action_turn is None else turn - recent.last_action_turn
        return [
            recent.active_turns / config.turn_scale,
            move_age / config.turn_scale,
            0.0 if recent.last_action_turn is None else 1.0,
            action_age / config.turn_scale,
            (recent.last_action_order or 0) / ACTION_ORDER_SCALE,
            recent.hp_taken / taken,
            recent.hp_taken_last_turn / taken,
            recent.hp_dealt / dealt,
            recent.hp_dealt_last_turn / dealt,
            recent.protect_streak / STREAK_SCALE,
            *target,
        ]

    def _recent_field(self, state: BattleState) -> list[float]:
        """The ``recent_history`` numbers of the field (see ``_recent_field_num``)."""
        config, turn = self.config, state.turn

        def since(began: int | None) -> float:
            return 0.0 if began is None else (turn - began) / config.turn_scale

        def spread(began: Mapping[str, int]) -> list[float]:
            out = [0.0] * self._condition_width
            for name in sorted(began):
                out[self._condition_column(name)] += since(began[name])
            return out

        return [
            since(state.weather_started),
            since(state.terrain_started),
            *spread(state.field_started),
            *spread(state.own.condition_started),
            *spread(state.opponent.condition_started),
        ]

    def _mon_num(self, mon: PokemonState, turn: int) -> list[float]:
        config = self.config
        fraction = mon.hp_fraction
        stats = mon.stats or {}
        hp_stat = float(mon.hp_den) if mon.own and mon.hp_den else 0.0
        volatiles = [0.0] * self._volatile_width
        for name in sorted(mon.volatiles):
            column = self._volatile_column(name)
            if not self._is_derived(name):  # a numbered or stat-suffixed id is read from `derived`
                volatiles[column] += 1.0
        for condition, suffix in sorted(mon.derived.items()) if _exact(config) else ():
            volatiles[self._volatile_column(condition)] = (
                1.0 + int(suffix) / DERIVED_SCALE if suffix.isdigit() else 1.0
            )
        known_moves = mon.moves_own if mon.moves_own is not None else []
        pp = []
        for i in range(config.max_moves):
            move = known_moves[i] if i < len(known_moves) else None
            base = self.game.move_base_pp[self.game.moves.lookup(move)] if move else 0
            pp.append(mon.pp.get(move, 0) / base if move and base else 0.0)
        return [
            1.0 if mon.own else 0.0,
            0.0 if fraction is None else 1.0,
            0.0 if fraction is None else fraction,
            1.0 if mon.fainted else 0.0,
            0.0 if mon.brought is None else 1.0,
            1.0 if mon.brought else 0.0,
            1.0 if mon.shiny else 0.0,
            1.0 if mon.mega else 0.0,
            1.0 if mon.item_known else 0.0,
            1.0 if mon.item_lost else 0.0,
            1.0 if mon.ability_suppressed else 0.0,
            1.0 if mon.can_mega_evo else 0.0,
            0.0 if mon.level is None else mon.level / 100.0,
            1.0 if mon.stats else 0.0,
            hp_stat / config.stat_scale,
            *(stats.get(name, 0) / config.stat_scale for name in STAT_NAMES),
            *(mon.boosts[name] / 6.0 for name in STAT_KEYS),
            *pp,
            *volatiles,
            *(self._typechange(mon) if _exact(config) else []),
            *(self._recent_num(mon, turn) if _recent(config) else []),
        ]  # fmt: skip

    @property
    def _volatile_width(self) -> int:
        return len(self._exact_index) if _exact(self.config) else self.config.volatile_buckets

    @property
    def _condition_width(self) -> int:
        return len(self._exact_index) if _exact(self.config) else self.config.condition_buckets

    def _volatile_column(self, name: str) -> int:
        """The column of a volatile id: its bucket, or with ``exact_volatiles`` its own column
        (a derived number or stat suffix resolves to the condition id; an id the vocabulary
        does not resolve is refused)."""
        if not _exact(self.config):
            return bucket(name, self.config.volatile_buckets)
        try:
            return self._exact_index[self.game.vocabulary.resolve_volatile(name)]
        except (VocabularyError, KeyError) as error:
            raise EncodingError(f"volatile {name!r} is not in the exact vocabulary") from error

    def _condition_column(self, name: str) -> int:
        """The column of a field or side condition id: its bucket, or its own column."""
        if not _exact(self.config):
            return bucket(name, self.config.condition_buckets)
        found = self._exact_index.get(name)
        if found is None:
            raise EncodingError(f"condition {name!r} is not in the exact vocabulary")
        return found

    def _is_derived(self, name: str) -> bool:
        """Whether ``name`` is a raw numbered or stat-suffixed id (exact_volatiles only). The
        reducer keeps such ids in the set after the engine ended them (`-end|Stockpile`) or
        counted on (`perish3`, `perish2`), so ``PokemonState.derived`` holds the truth."""
        return _exact(self.config) and self.game.vocabulary.resolve_volatile(name) != name

    def _typechange(self, mon: PokemonState) -> list[float]:
        """The ``exact_volatiles`` numbers of a Pokémon: the types a ``typechange`` set and the
        stat a Protosynthesis or Quark Drive boosts (see ``_typechange_num``). All zero while the
        types are the species' own and no such ability is active."""
        out = [0.0] * (2 + len(self.game.types.ids) + len(STAT_NAMES))
        for suffix in mon.derived.values():
            if suffix.isdigit():
                continue
            if suffix not in STAT_NAMES:
                raise EncodingError(f"derived suffix {suffix!r} is neither a number nor a stat")
            out[2 + len(self.game.types.ids) + STAT_NAMES.index(suffix)] = 1.0
        if mon.types is None:
            return out
        out[0] = 1.0
        for name in mon.types:
            if name == TYPELESS:
                out[1] = 1.0
            else:
                found = self.game.types.index.get(name)
                if found is None:
                    raise EncodingError(f"type {name!r} is not in the engine data")
                out[2 + found - RESERVED] = 1.0
        return out

    def _conditions(self, conditions: Mapping[str, int] | set[str]) -> list[float]:
        out = [0.0] * self._condition_width
        counts = (
            conditions if isinstance(conditions, Mapping) else dict.fromkeys(sorted(conditions), 1)
        )
        for name in sorted(counts):
            out[self._condition_column(name)] += float(counts[name])
        return out

    def _field(self, state: BattleState) -> tuple[list[int], list[float]]:
        game = self.game
        if state.request_kind not in REQUEST_KINDS:
            raise EncodingError(f"no decision to encode at a {state.request_kind!r} request")
        cat = [
            game.weathers.lookup(state.weather) if state.weather else NONE,
            game.terrains.lookup(state.terrain) if state.terrain else NONE,
            REQUEST_KINDS.index(state.request_kind) + 1,
        ]
        num = [
            state.turn / self.config.turn_scale,
            1.0 if state.started else 0.0,
            *self._conditions(state.field_conditions),
            *self._conditions(state.own.conditions),
            *self._conditions(state.opponent.conditions),
            *(self._recent_field(state) if _recent(self.config) else []),
        ]
        return cat, num

    # -- the request ---------------------------------------------------------

    def _request(self, state: BattleState, raw: str | None) -> _Request:
        """The player's own last ``|request|``: its Pokémon order and its moves.

        ``rqid`` is never read.
        """
        # The reducer saw a request (_field refused a state without one), so a line exists.
        assert raw is not None
        body = json.loads(raw[len(_REQUEST) :])
        by_name = {mon.name: i for i, mon in enumerate(state.own.roster)}
        out = _Request()
        for entry in body["side"]["pokemon"]:
            # The reducer refused a request naming a Pokémon outside the own team.
            out.order.append(by_name[_SIDE.sub("", str(entry["ident"]), count=1)])
        for active in body.get("active", []):
            out.moves.append([self.game.moves.lookup_or_unknown(m["id"]) for m in active["moves"]])
            out.targets.append([m.get("target") for m in active["moves"]])
        return out

    # -- candidate features --------------------------------------------------

    def _rows(self, query: Mapping[str, Any]) -> dict[rows_of.RowKey, rows_of.Row] | None:
        """The query's ``candidate_features`` rows when the config reads a feature set, else None.

        The query must carry exactly the set the config names (the set and its hash); a query
        with rows under a config without the option, or without them under one with it, is
        refused: nothing is read of rows the checkpoint was not trained with, and nothing is
        filled in for rows it needs.
        """
        wanted = feature_set(self.config.options)
        sent = query.get("candidate_features")
        if wanted is None:
            if sent is not None:
                raise EncodingError(
                    "the query carries candidate_features this encoder does not read"
                )
            return None
        if sent is None:
            raise EncodingError("the query has no candidate_features, which this encoder needs")
        if (sent["feature_set"], sent["feature_set_hash"]) != (
            wanted.feature_set,
            wanted.feature_set_hash,
        ):
            raise EncodingError("the query's candidate_features are of another feature set")
        if not _reads_arrays(self.config) and any(name in sent for name in rows_of.ARRAY_NAMES):
            raise EncodingError("the query carries arrays this feature set does not read")
        try:
            return rows_of.row_index(sent)
        except ValueError as exc:
            raise EncodingError(str(exc)) from exc

    def _own_token(self, request: _Request, position: int) -> int:
        """The token of the own Pokémon at ``position`` of the request's ``side.pokemon``."""
        if not 0 <= position < len(request.order):
            raise EncodingError("a candidate_features row names an own Pokémon the request lacks")
        return request.order[position]

    def _foe_token(self, state: BattleState, loc: int) -> int:
        """The token of the foe at choice location ``loc`` (the engine's numbering)."""
        letters = sorted(state.opponent.slots)
        if not 1 <= loc <= len(letters):
            raise EncodingError("a candidate_features row names a foe location the state lacks")
        index = state.opponent.slots[letters[loc - 1]]
        if index is None:
            raise EncodingError("a candidate_features row names a foe location with no Pokémon")
        return len(state.own.roster) + index

    def _probe(self, query: Mapping[str, Any], request: _Request, state: BattleState) -> _Probe:
        """The arrays of a ``probe-damage/2`` query: each row on the token it belongs to.

        Threat rows of an own Pokémon (Mega flag false) and the foe priority rows are pooled onto
        the tokens by the network; a Mega variant's threats and speeds go to the Mega candidate's
        slot part. A threat whose move has no foe priority row, a row for a Pokémon or foe the
        state lacks, or a location past the format's foes is refused; nothing is filled in.
        """
        try:
            arrays = rows_of.Arrays.of(query["candidate_features"])
        except ValueError as exc:
            raise EncodingError(str(exc)) from exc
        game = self.game
        probe = _Probe(arrays)
        for (own, mega, foe), row in arrays.speeds.items():
            token = self._own_token(request, own)
            if not 1 <= foe <= SLOTS:
                raise EncodingError(
                    "a candidate_features row aims at a foe location the format lacks"
                )
            if not mega:
                cols = probe.speed_cols.setdefault(token, [NONE] * SLOTS)
                cols[foe - 1] = rows_of.speed_category(row)
        for row in arrays.threats:
            foe_row = arrays.foe_priorities.get((row["foe"], row["move"]))
            if foe_row is None and request.moves:
                # A move request prices every threat's foe move; a forced switch asks no move, so
                # its threats carry the NONE priority kind and a zero number.
                raise EncodingError("a threat row has no foe priority row for its move")
            probe.threats.append(
                (
                    [
                        self._own_token(request, row["own"]),
                        self._foe_token(state, row["foe"]),
                        int(row["mega"]),
                    ],
                    [
                        game.moves.lookup(row["move"]),
                        rows_of.probe_category(row),
                        NONE if foe_row is None else rows_of.probe_category(foe_row),
                    ],
                    rows_of.threat_numbers(row, foe_row),
                )
            )
        for row in arrays.foe_priorities.values():
            probe.foes.append(
                (
                    [self._foe_token(state, row["foe"])],
                    [game.moves.lookup(row["move"]), rows_of.probe_category(row)],
                    [rows_of.priority_number(row)],
                )
            )
        return probe

    def _need_speeds(self, probe: _Probe, own: int, mega: bool) -> None:
        """Every foe location the arrays name has a speed row for the own variant, else refused."""
        for loc in probe.arrays.locations:
            if (own, mega, loc) not in probe.arrays.speeds:
                raise EncodingError("a candidate has no candidate_features speed row")

    def _foe_alive(self, state: BattleState, loc: int) -> bool:
        """Whether choice location ``loc`` holds a foe the rows cover.

        The engine's numbering: location L is the foe in active slot L - 1.

        The builder gives a fainted, absent or empty-HP foe no row, so a candidate that aims
        there (the engine lists it) has none to find.
        """
        letters = sorted(state.opponent.slots)
        place = loc - 1
        if not 0 <= place < len(letters):
            return False
        index = state.opponent.slots[letters[place]]
        mon = None if index is None else state.opponent.roster[index]
        return mon is not None and not mon.fainted and bool(mon.hp_fraction)

    def _feature_part(
        self,
        rows: dict[rows_of.RowKey, rows_of.Row],
        request: _Request,
        slot: int,
        action: Any,
        state: BattleState,
    ) -> tuple[list[int], list[float]]:
        """The rows of one move part per foe location; a missing row of a live foe is refused."""
        moves = request.moves[slot]
        move = moves[action.index - 1]
        # An unknown move counts as damaging (the rows are then asked for, never skipped).
        if move >= RESERVED and self.game.move_status[move]:
            return rows_of.no_columns()
        mega = action.event == "mega"
        if action.target is not None:
            # A location past the format's foes names none, so it has no row to find.
            locations = [action.target] if 0 < action.target <= SLOTS else []
        else:
            target_type = (
                request.targets[slot][action.index - 1] if slot < len(request.targets) else None
            )
            locations = (
                list(range(1, SLOTS + 1)) if target_type in rows_of.FOE_SPREAD_TARGET_TYPES else []
            )
        found: list[rows_of.Row | None] = [None] * SLOTS
        for loc in locations:
            found[loc - 1] = rows.get((slot, action.index, loc, mega))
        missing = (
            any(found[loc - 1] is None and self._foe_alive(state, loc) for loc in locations)
            if action.target is not None
            else bool(locations) and all(row is None for row in found)
        )
        if missing:
            raise EncodingError("a damaging move part of a candidate has no candidate_features row")
        return rows_of.part_columns(found)

    # -- candidates ----------------------------------------------------------

    def _token(self, request: _Request, position: int) -> int:
        if not 1 <= position <= len(request.order):
            raise UnsupportedCandidate("a candidate names a team position the request lacks")
        return request.order[position - 1]

    def _shown_species(self, mon: PokemonState | None) -> int:
        """The species index a position shows: NONE for an empty or fainted position.

        A Pokémon in play has been shown by its switch-in (or its Transform copy), so its species
        is public; the roster's base species is the public fallback.
        """
        if mon is None or mon.fainted:
            return NONE
        return self.game.species.lookup(mon.temp_species or mon.species or mon.base_species)

    def _foe_at(self, state: BattleState, place: int) -> PokemonState | None:
        letters = sorted(state.opponent.slots)
        index = state.opponent.slots[letters[place]] if place < len(letters) else None
        return None if index is None else state.opponent.roster[index]

    def _matchup(
        self, move: int, actor: int, request: _Request, slot: int, state: BattleState
    ) -> list[int]:
        """The ``candidate_matchup`` columns of one move slot action (see ``_matchup_columns``).

        Only the species a position shows is read (ADR-022 tier B): never an unrevealed item,
        ability or move, and a type change by a volatile is not tracked. A status move, an
        unknown move and an empty position carry NONE.
        """
        game = self.game
        places = [self._shown_species(self._foe_at(state, i)) for i in range(SLOTS)]
        ally_slot = (slot + 1) % SLOTS  # the other active: doubles has SLOTS == 2
        ally = (
            state.own.roster[request.order[ally_slot]] if ally_slot < len(request.order) else None
        )
        places.append(self._shown_species(ally))
        if move < RESERVED or game.move_status[move]:
            return [NONE] * (MATCHUP_TYPES * len(places) + 1)
        cells = [c for species in places for c in game.matchup_cells(move, species, MATCHUP_TYPES)]
        user = self._shown_species(state.own.roster[actor])
        same = SAME_TYPE_YES if game.move_type[move] in game.species_types[user] else SAME_TYPE_NOT
        return [*cells, same]

    def _target_token(
        self, target: int | None, request: _Request, slot: int, state: BattleState
    ) -> int:
        """The token of the Pokémon a move's target location names, or -1 for none.

        The engine's ``getAtLoc``: a positive location L is the foe in active slot L - 1, a
        negative one the own active slot -L - 1 (the user itself, an empty or a fainted slot is
        none). Untargeted and spread moves carry no location, so they have none.
        """
        if target is None:
            return -1
        if target > 0:
            letters = sorted(state.opponent.slots)
            index = state.opponent.slots[letters[target - 1]] if target <= len(letters) else None
            if index is None or state.opponent.roster[index].fainted:
                return -1
            return len(state.own.roster) + index
        ally = -target - 1
        if ally == slot or ally >= len(request.order):
            return -1
        token = request.order[ally]
        return -1 if state.own.roster[token].fainted else token

    def _candidate(  # noqa: PLR0912, PLR0913, PLR0917
        self,
        candidate: Mapping[str, Any],
        request: _Request,
        kind: str,
        state: BattleState,
        rows: dict[rows_of.RowKey, rows_of.Row] | None = None,
        probe: _Probe | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[int], float, list[list[float]]]:
        parsed: ParsedChoice = parse_choice(candidate["choice"])
        if (parsed.kind == "team") != (kind == "preview"):
            raise EncodingError(f"a {parsed.kind} candidate in a {kind} query")
        config = self.config
        cat = [[0] * len(self.layout.cand_cat) for _ in range(SLOTS)]
        tok = [[-1] * len(self.layout.cand_tok) for _ in range(SLOTS)]
        targeted = len(self.layout.cand_tok) > BASE_CAND_TOKENS
        num: list[list[float]] = [[0.0] * len(self.layout.cand_num) for _ in range(SLOTS)]
        team = [-1] * config.max_picks
        matchup = len(_matchup_columns(config))
        if parsed.kind == "team":
            if len(parsed.team) > config.max_picks:
                raise UnsupportedCandidate("a team order picks more Pokémon than max_picks")
            for place, position in enumerate(parsed.team):
                team[place] = self._token(request, position)
        else:
            for slot, action in enumerate(parsed.slots):
                if action.kind == "pass":
                    continue
                cat[slot][0] = 1 if action.kind == "move" else 2
                tok[slot][0] = self._token(request, slot + 1)
                assert action.index is not None  # a move or switch always has its index
                if action.kind == "switch":
                    tok[slot][1] = self._token(request, action.index)
                    if probe is not None:  # the switch-in's token holds its threats and speeds
                        self._need_speeds(probe, action.index - 1, False)
                    continue
                moves = request.moves[slot] if slot < len(request.moves) else []
                if action.index > min(len(moves), config.max_moves):
                    raise UnsupportedCandidate("a candidate names a move the request lacks")
                cat[slot][1] = moves[action.index - 1]
                cat[slot][2] = action.index
                cat[slot][3] = 0 if action.target is None else TARGET_OFFSET + action.target
                cat[slot][4] = 0 if action.event is None else 1 + EVENTS.index(action.event)
                if targeted:
                    tok[slot][BASE_CAND_TOKENS] = self._target_token(
                        action.target, request, slot, state
                    )
                if matchup:
                    cat[slot][BASE_CAND_COLUMNS : BASE_CAND_COLUMNS + matchup] = self._matchup(
                        cat[slot][1], tok[slot][0], request, slot, state
                    )
                if rows is not None:
                    kinds, num[slot] = self._feature_part(rows, request, slot, action, state)
                    start = BASE_CAND_COLUMNS + matchup
                    cat[slot][start : start + SLOTS] = kinds
                    if probe is not None:
                        self._probe_part(probe, slot, action, cat[slot], num[slot], start + SLOTS)
        return cat, tok, team, 1.0 if candidate["pending"] else 0.0, num

    def _probe_part(  # noqa: PLR0913, PLR0917
        self, probe: _Probe, slot: int, action: Any, cat: list[int], num: list[float], at: int
    ) -> None:
        """A move part's own priority and, for a Mega part, its Mega variant's speed relations."""
        mega = action.event == "mega"
        row = probe.arrays.own_priorities.get((slot, action.index, mega))
        if row is None:
            raise EncodingError("a move part of a candidate has no candidate_features priority row")
        self._need_speeds(probe, slot, mega)
        cat[at] = rows_of.probe_category(row)
        num.append(rows_of.priority_number(row))
        if mega:
            for loc in probe.arrays.locations:  # _need_speeds found a row for each
                cat[at + loc] = rows_of.speed_category(probe.arrays.speeds[(slot, True, loc)])

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        kind: str,
        state: BattleState,
        field_features: tuple[list[int], list[float]],
        candidates: list[
            tuple[list[list[int]], list[list[int]], list[int], float, list[list[float]]]
        ],
        probe: _Probe | None = None,
    ) -> EncodedQuery:
        own = [(state.own, mon) for mon in state.own.roster]
        opponent = [(state.opponent, mon) for mon in state.opponent.roster]
        # An added entry (an Illusion user shown alongside the Pokémon it is shown as) is no
        # extra Pokémon of the team: it does not count against max_team.
        if any(
            sum(not m.phantom for _, m in side) > self.config.max_team for side in (own, opponent)
        ):
            raise EncodingError("a side has more Pokémon than max_team")
        field_cat, field_num = field_features
        rows = [*own, *opponent]
        none = [NONE] * SLOTS  # a token with no speed row (a foe, or no foe at that location)
        n = len(candidates)
        return EncodedQuery(
            kind=kind,
            n_own=len(own),
            mon_cat=torch.tensor(
                [
                    [*self._mon_cat(s, m), *(probe.speed_cols.get(i, none) if probe else [])]
                    for i, (s, m) in enumerate(rows)
                ],
                dtype=torch.int64,
            ),
            mon_num=torch.tensor(
                [self._mon_num(m, state.turn) for _, m in rows], dtype=torch.float32
            ),
            field_cat=torch.tensor(field_cat, dtype=torch.int64),
            field_num=torch.tensor(field_num, dtype=torch.float32),
            cand_cat=torch.tensor([c[0] for c in candidates], dtype=torch.int64).reshape(
                n, SLOTS, len(self.layout.cand_cat)
            ),
            cand_tok=torch.tensor([c[1] for c in candidates], dtype=torch.int64).reshape(
                n, SLOTS, len(self.layout.cand_tok)
            ),
            team_tok=torch.tensor([c[2] for c in candidates], dtype=torch.int64).reshape(
                n, self.config.max_picks
            ),
            pending=torch.tensor([c[3] for c in candidates], dtype=torch.float32),
            cand_num=(
                torch.tensor([c[4] for c in candidates], dtype=torch.float32).reshape(
                    n, SLOTS, len(self.layout.cand_num)
                )
                if self.layout.cand_num
                else None
            ),
            **(self._probe_tensors(probe) if probe is not None else {}),
        )

    def _probe_tensors(self, probe: _Probe) -> dict[str, torch.Tensor]:
        layout = self.layout
        threats, foes = probe.threats, probe.foes
        return {
            "thr_idx": torch.tensor([t[0] for t in threats], dtype=torch.int64).reshape(-1, 3),
            "thr_cat": torch.tensor([t[1] for t in threats], dtype=torch.int64).reshape(
                -1, len(layout.thr_cat)
            ),
            "thr_num": torch.tensor([t[2] for t in threats], dtype=torch.float32).reshape(
                -1, len(layout.thr_num)
            ),
            "foe_idx": torch.tensor([f[0] for f in foes], dtype=torch.int64).reshape(-1, 1),
            "foe_cat": torch.tensor([f[1] for f in foes], dtype=torch.int64).reshape(
                -1, len(layout.foe_cat)
            ),
            "foe_num": torch.tensor([f[2] for f in foes], dtype=torch.float32).reshape(
                -1, len(layout.foe_num)
            ),
        }


class EncoderSession:
    """Encode the queries of one channel with one pass of the reducer over it.

    ``Encoder.encode`` reduces a query's log from its first line. The queries of
    one Episode carry growing prefixes of the same channel, so a session feeds
    the reducer only the lines it has not seen and encodes at once, while the
    state is the one ``reduce_channel`` would return for that prefix. A query
    that does not extend the lines already fed (another player, another team, a
    shorter or different log) starts the reduction again, so the result never
    differs from ``Encoder.encode``; a refusal is raised at the same line and
    leaves the session empty. The tensors are fresh, nothing aliases the state.
    """

    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder
        self._key: tuple[Any, Any] | None = None
        self._reducer: ChannelReducer | None = None
        self._fed: list[str] = []
        self._last_request: str | None = None

    def encode(self, query: Mapping[str, Any]) -> EncodedQuery:
        observation = query["observation"]
        log = observation["log"]
        key = (observation["player"], observation["team"]["team"])
        fed = len(self._fed)
        if self._reducer is None or self._key != key or log[:fed] != self._fed:
            self._reducer = ChannelReducer(key[0], key[1], self._encoder.game.vocabulary)
            self._key = key
            self._fed = []
            self._last_request = None
            fed = 0
        try:
            for line in log[fed:]:
                self._reducer.feed(line)
                self._fed.append(line)
                if line.startswith(_REQUEST):
                    self._last_request = line
            return self._encoder._encode_state(query, self._reducer.state, self._last_request)
        except BaseException:
            self._reducer = None
            self._fed = []
            raise
