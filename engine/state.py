"""World state: the single source of truth.

Nothing an agent *says* changes the world. Only `apply_action` does.

Deaths, potions, win detection and vote legality live in the `werewolf-engine`
library; this class owns orchestration (phase order, round cap), the speech
phase the library has no concept of, and the public transcript agents read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from werewolf_engine.errors import WerewolfEngineError
from werewolf_engine.phases.day import DayManager

from .rules import (
    MAX_ROUNDS,
    NUM_PLAYERS,
    Game,
    Phase,
    Role,
    Setup,
    Team,
    build_game,
    to_engine_id,
    to_player_id,
)


class ActionError(Exception):
    """A semantically invalid action (voting for a dead player, speaking out of
    turn, ...). The harness turns this into a regeneration, never a crash."""


@dataclass
class Speech:
    round: int
    player_id: int
    order: int
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeathRecord:
    round: int
    player_id: int
    cause: str  # "werewolf" | "witch" | "vote"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckRecord:
    round: int
    target: int
    is_wolf: bool

    def to_dict(self) -> dict:
        return asdict(self)


class _DayVoting(DayManager):
    """The library's vote rules plus abstention.

    `DayManager` requires every living player to vote, but the recovery path
    (retries exhausted -> abstain) needs a living player to end the round
    without a ballot. Everything else -- self-vote ban, dead-target ban,
    double-vote ban, tie resolves to nobody -- is inherited unchanged.
    """

    def abstain(self, voter_id: int) -> None:
        voter = self.game._get_player(voter_id)
        if voter in self._voters:
            self._voters.remove(voter)


@dataclass
class GameState:
    seed: int
    game: Game
    setup: Setup
    round: int = 1
    phase: Phase = Phase.NIGHT
    speeches: list[Speech] = field(default_factory=list)
    votes: dict[int, dict[int, int | None]] = field(default_factory=dict)
    vote_counts: dict[int, dict[int, int]] = field(default_factory=dict)
    exiles: dict[int, int | None] = field(default_factory=dict)
    deaths: list[DeathRecord] = field(default_factory=list)
    seer_checks: list[CheckRecord] = field(default_factory=list)
    night_victims: dict[int, list[int]] = field(default_factory=dict)
    winner: Team | None = None
    speech_cursor: int = 0
    night_records: list[dict] = field(default_factory=list)
    _day: _DayVoting | None = None
    _night: object | None = None

    # ---- construction -------------------------------------------------

    @classmethod
    def new(cls, seed: int) -> "GameState":
        game, setup = build_game(seed)
        return cls(seed=seed, game=game, setup=setup)

    # ---- derived views ------------------------------------------------

    @property
    def roles(self) -> dict[int, Role]:
        return self.setup.roles

    def role_of(self, player_id: int) -> Role:
        return self.setup.roles[player_id]

    def is_alive(self, player_id: int) -> bool:
        return self.game._get_player(to_engine_id(player_id)).is_alive()

    def alive_sorted(self) -> list[int]:
        return sorted(
            to_player_id(p.id) for p in self.game.players if p.is_alive()
        )

    @property
    def alive(self) -> set[int]:
        return set(self.alive_sorted())

    def speech_order_this_round(self) -> list[int]:
        alive = self.alive
        return [p for p in self.setup.speech_order if p in alive]

    def current_speaker(self) -> int | None:
        if self.phase is not Phase.DAY_SPEECH:
            return None
        order = self.speech_order_this_round()
        if self.speech_cursor >= len(order):
            return None
        return order[self.speech_cursor]

    def speeches_of(self, player_id: int, round_no: int) -> list[Speech]:
        return [
            s for s in self.speeches if s.player_id == player_id and s.round == round_no
        ]

    def votes_of_round(self, round_no: int) -> dict[int, int | None]:
        return dict(self.votes.get(round_no, {}))

    def checks_of(self, player_id: int) -> list[CheckRecord]:
        """Seer check history. Private: only ever handed to the seer."""
        if self.role_of(player_id) is not Role.SEER:
            return []
        return list(self.seer_checks)

    # ---- mutation (the only path from an agent to the world) -----------

    def apply_action(self, player_id: int, action: dict) -> dict:
        """`action` is a validated terminal action: {"name": "speak"|"vote", ...}."""
        name = action.get("name")
        if name == "speak":
            return self._apply_speak(player_id, str(action.get("content", "")))
        if name == "vote":
            return self._apply_vote(player_id, action.get("target_id"))
        if name and name.startswith("night_"):
            return self.apply_night_action(player_id, name, action.get("target_id"))
        raise ActionError(f"unknown terminal action: {name!r}")

    def _apply_speak(self, player_id: int, content: str) -> dict:
        if self.phase is not Phase.DAY_SPEECH:
            raise ActionError("cannot speak outside the day speech phase")
        expected = self.current_speaker()
        if expected != player_id:
            raise ActionError(
                f"it is player {expected}'s turn to speak, not player {player_id}"
            )
        speech = Speech(self.round, player_id, self.speech_cursor, content)
        self.speeches.append(speech)
        self.speech_cursor += 1
        if self.speech_cursor >= len(self.speech_order_this_round()):
            self.phase = Phase.DAY_VOTE
            self._day = _DayVoting(self.game)
        return {"accepted": True, "speech_order": speech.order}

    def _apply_vote(self, player_id: int, target_id) -> dict:
        if self.phase is not Phase.DAY_VOTE or self._day is None:
            raise ActionError("cannot vote outside the vote phase")
        if target_id is not None and (
            isinstance(target_id, bool) or not isinstance(target_id, int)
        ):
            # `True` is an int in Python, and `to_engine_id(True)` would quietly
            # become a vote for player 1.
            raise ActionError("vote target must be an integer player id")
        try:
            if target_id is None:
                self._day.abstain(to_engine_id(player_id))
            else:
                self._day.cast_vote(to_engine_id(player_id), to_engine_id(target_id))
        except WerewolfEngineError as exc:
            raise ActionError(_explain(exc, player_id, target_id)) from exc
        self.votes.setdefault(self.round, {})[player_id] = target_id
        return {"accepted": True, "target": target_id}

    # ---- phase transitions (driven by the runner, never by an agent) ----

    # ---- night, one agent turn at a time -------------------------------

    def begin_night(self):
        """Open the night. Actors then act in role-priority order."""
        from .night import NightPhase

        self._night = NightPhase(self)
        return self._night

    def night_actor(self) -> int | None:
        """Whose night turn it is, or None when the night is done."""
        if self._night is None:
            return None
        return self._night.current_actor()

    def night_options(self, player_id: int) -> dict:
        """What this actor may do tonight -- the private half of their view."""
        if self._night is None or self._night.current_actor() != player_id:
            return {"actions": [], "targets": []}
        role = self.role_of(player_id)
        options = {
            "actions": sorted(self._night.allowed_actions()),
            "targets": self._night.legal_targets(player_id),
        }
        if role is Role.WITCH:
            options["victims_tonight"] = self._night.victims_tonight()
            options["antidote_available"] = not getattr(
                self.game._get_player(to_engine_id(player_id)).role, "_heal_used", True
            )
            options["poison_available"] = not getattr(
                self.game._get_player(to_engine_id(player_id)).role, "_kill_used", True
            )
        if role is Role.WOLF:
            options["pack_votes_so_far"] = dict(self._night.wolf_intents)
        return options

    def apply_night_action(self, player_id: int, name: str, target_id) -> dict:
        """The only way a night decision reaches the world."""
        if self.phase is not Phase.NIGHT or self._night is None:
            raise ActionError("it is not night")
        try:
            return self._night.submit(player_id, name, target_id)
        except ValueError as exc:
            raise ActionError(str(exc)) from exc

    def fallback_night_action(self, player_id: int) -> tuple[str, int | None]:
        """What an unsalvageable night turn does instead.

        Seeded and reproducible, and never a no-op for the wolves: a failed
        model call must not turn into "the pack chose not to kill tonight",
        which would silently change the game rather than just the agent.
        """
        if self._night is None:
            return "night_skip", None
        role = self.role_of(player_id)
        legal = self._night.legal_targets(player_id)
        if role is Role.WITCH or not legal:
            return "night_skip", None
        for candidate in self.setup.fallback_order:
            if candidate in legal:
                return ("night_kill" if role is Role.WOLF else "night_check"), candidate
        return "night_skip", None

    def end_night(self) -> list[int]:
        """Resolve the night through the library and open the day."""
        if self._night is None:
            raise ActionError("the night was never opened")
        before = self.alive
        self._night.resolve()
        for record in self._night.records:
            if record["action"] == "night_check":
                self.seer_checks.append(
                    CheckRecord(record["round"], record["target"], bool(record["is_wolf"]))
                )
        self.night_records.extend(self._night.records)
        self._night = None

        died = sorted(before - self.alive)
        for pid in died:
            self.deaths.append(DeathRecord(self.round, pid, self._cause_of(pid)))
        self.night_victims[self.round] = died
        self._check_end()
        if self.phase is not Phase.OVER:
            self.phase = Phase.DAY_SPEECH
            self.speech_cursor = 0
        return died

    def resolve_night(self) -> list[int]:
        """Play the whole night with the fallback policy and no agents.

        This is what the engine's own tests use, and what the runner degrades to
        if a night turn cannot be salvaged. Real games drive the night through
        `begin_night` / `night_actor` / `apply_action` / `end_night` so that
        every night decision is an agent's, with a trace to show for it.
        """
        self.begin_night()
        while (actor := self.night_actor()) is not None:
            name, target = self.fallback_night_action(actor)
            self.apply_night_action(actor, name, target)
        return self.end_night()

    def resolve_vote(self) -> int | None:
        """Count this round's votes through the library, then advance."""
        if self._day is None:
            raise ActionError("vote phase was never opened")
        counts: dict[int, int] = {}
        for target in self.votes.get(self.round, {}).values():
            if target is not None:
                counts[target] = counts.get(target, 0) + 1
        self.vote_counts[self.round] = counts

        eliminated = self._day.resolve()
        exiled = to_player_id(eliminated.id) if eliminated is not None else None
        self.exiles[self.round] = exiled
        if eliminated is not None:
            if eliminated.role:
                eliminated.role.on_player_died(self.game, eliminated, cause="vote")
            self.deaths.append(DeathRecord(self.round, exiled, "vote"))
        self._day = None

        self._check_end()
        if self.phase is not Phase.OVER:
            self.round += 1
            if self.round > MAX_ROUNDS:
                # Time limit: the village failed to find the wolves in time.
                self.winner = Team.WOLF
                self.phase = Phase.OVER
            else:
                self.phase = Phase.NIGHT
                self.speech_cursor = 0
        return exiled

    def _cause_of(self, player_id: int) -> str:
        """Recover a death cause from the library's event stream is overkill;
        night deaths are either the wolf kill or the witch's poison."""
        eng = to_engine_id(player_id)
        witch_kill = None
        for p in self.game.players:
            if p.role and p.role.name == "witch":
                witch_kill = getattr(p.role, "_kill_target_id", None)
        return "witch" if witch_kill == eng else "werewolf"

    def _check_end(self) -> None:
        """Win detection is the library's `_check_game_over`."""
        if self.game._check_game_over():
            self.winner = Team(self.game._winner) if self.game._winner else None
            self.phase = Phase.OVER

    # ---- serialisation -------------------------------------------------

    def ground_truth(self) -> dict:
        return {
            "roles": {str(p): r.value for p, r in self.roles.items()},
            "winner": self.winner.value if self.winner else None,
        }


def _explain(exc: Exception, player_id: int, target_id) -> str:
    """The library's messages are Persian; agents get an actionable English one."""
    text = str(exc)
    if "قبلاً رأی" in text:
        return f"player {player_id} has already voted this round"
    if "نامعتبر" in text:
        return (
            f"player {target_id} is not a legal vote target "
            "(must be alive and not yourself)"
        )
    if "اجازه" in text:
        return f"player {player_id} is not eligible to vote"
    if "یافت نشد" in text:
        return f"player {target_id} does not exist"
    return f"invalid vote: {text}"


__all__ = [
    "ActionError",
    "CheckRecord",
    "DeathRecord",
    "GameState",
    "Speech",
    "MAX_ROUNDS",
    "NUM_PLAYERS",
]
