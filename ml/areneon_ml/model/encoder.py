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
    Options,
    feature_set,
    is_set,
    with_options,
)
from areneon_ml.reducer import BattleState, ChannelReducer, PokemonState, SideState, reduce_channel
from areneon_ml.reducer.state import STAT_KEYS

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


@dataclass(frozen=True)
class Layout:
    """Names and kinds of every feature column, derived from the configuration."""

    mon_cat: tuple[tuple[str, str], ...]  # (column, vocabulary kind)
    mon_num: tuple[str, ...]
    field_cat: tuple[tuple[str, str], ...]
    field_num: tuple[str, ...]
    cand_cat: tuple[tuple[str, str], ...]
    cand_num: tuple[str, ...] = ()  # numeric columns of a candidate part; only with features
    cand_tok: tuple[str, ...] = ("actor", "switch_in")  # token columns; candidate_target adds one

    @classmethod
    def of(cls, config: EncoderConfig) -> Layout:
        moves = tuple((f"move_{i + 1}", "move") for i in range(config.max_moves))
        buckets = config.condition_buckets
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
                *(f"volatile_{i}" for i in range(config.volatile_buckets)),
            ),
            field_cat=(("weather", "weather"), ("terrain", "terrain"), ("request", "request")),
            field_num=(
                "turn",
                "started",
                *(f"field_{i}" for i in range(buckets)),
                *(f"own_side_{i}" for i in range(buckets)),
                *(f"opponent_side_{i}" for i in range(buckets)),
            ),
            cand_cat=(
                ("action", "action"),
                ("move", "move"),
                ("move_slot", "move_slot"),
                ("target", "target"),
                ("event", "event"),
                *_matchup_columns(config),
                *_feature_columns(config),
            ),
            cand_num=_feature_numbers(config),
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
        "layout": Layout.of(config).as_dict(),
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


class Encoder:
    """Observation + candidates to ``EncodedQuery``: pure and deterministic."""

    def __init__(self, game: GameData, config: EncoderConfig) -> None:
        self.game = game
        self.config = config
        self.layout = Layout.of(self.config)
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
        candidates = [self._candidate(c, request, wanted, state, rows) for c in query["candidates"]]
        return self._assemble(observation["kind"], state, field_features, candidates)

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
        ]

    def _mon_num(self, mon: PokemonState) -> list[float]:
        config = self.config
        fraction = mon.hp_fraction
        stats = mon.stats or {}
        hp_stat = float(mon.hp_den) if mon.own and mon.hp_den else 0.0
        volatiles = [0.0] * config.volatile_buckets
        for name in sorted(mon.volatiles):
            volatiles[bucket(name, config.volatile_buckets)] += 1.0
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
        ]  # fmt: skip

    def _conditions(self, conditions: Mapping[str, int] | set[str]) -> list[float]:
        out = [0.0] * self.config.condition_buckets
        counts = (
            conditions if isinstance(conditions, Mapping) else dict.fromkeys(sorted(conditions), 1)
        )
        for name in sorted(counts):
            out[bucket(name, self.config.condition_buckets)] += float(counts[name])
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
        try:
            return rows_of.row_index(sent)
        except ValueError as exc:
            raise EncodingError(str(exc)) from exc

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

    def _candidate(
        self,
        candidate: Mapping[str, Any],
        request: _Request,
        kind: str,
        state: BattleState,
        rows: dict[rows_of.RowKey, rows_of.Row] | None = None,
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
                    cat[slot][BASE_CAND_COLUMNS + matchup :] = (
                        kinds  # the feature columns come last
                    )
        return cat, tok, team, 1.0 if candidate["pending"] else 0.0, num

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        kind: str,
        state: BattleState,
        field_features: tuple[list[int], list[float]],
        candidates: list[
            tuple[list[list[int]], list[list[int]], list[int], float, list[list[float]]]
        ],
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
        n = len(candidates)
        return EncodedQuery(
            kind=kind,
            n_own=len(own),
            mon_cat=torch.tensor([self._mon_cat(s, m) for s, m in rows], dtype=torch.int64),
            mon_num=torch.tensor([self._mon_num(m) for _, m in rows], dtype=torch.float32),
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
        )


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
