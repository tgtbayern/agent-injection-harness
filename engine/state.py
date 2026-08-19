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
    SHERIFF_VOTE_WEIGHT,
    Game,
    Phase,
    Role,
    Setup,
    Team,
    Variant,
    build_game,
    get_variant,
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
    # "speech" | "last_words" | "campaign". Last words go in the same
    # transcript as everything else on purpose: they are public, they are
    # retrievable by `query_history`, and they therefore carry a payload
    # exactly like a living player's turn does -- from a speaker who can no
    # longer be questioned.
    kind: str = "speech"

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
    """The library's vote rules plus abstention and a weighted ballot.

    `DayManager` requires every living player to vote, but the recovery path
    (retries exhausted -> abstain) needs a living player to end the round
    without a ballot. Everything else -- self-vote ban, dead-target ban,
    double-vote ban, tie resolves to nobody -- is inherited unchanged.

    The sheriff's ballot is worth 1.5. The library counts one each, so the
    tally is recomputed here when, and only when, a weight is not 1: with no
    sheriff at the table this class defers to `super().resolve()` and the
    8-player games stay bit-identical to what they always were.
    """

    def __init__(self, game, weights: dict[int, float] | None = None):
        super().__init__(game)
        self.weights = weights or {}

    def abstain(self, voter_id: int) -> None:
        voter = self.game._get_player(voter_id)
        if voter in self._voters:
            self._voters.remove(voter)

    def resolve(self):
        if not self.weights:
            return super().resolve()
        if not self.all_votes_submitted():
            return None
        tally: dict[int, float] = {}
        for voter_id, target_id in self._votes.items():
            tally[target_id] = tally.get(target_id, 0.0) + self.weights.get(voter_id, 1.0)
        if not tally:
            return None
        top = max(tally.values())
        leaders = [tid for tid, c in tally.items() if c == top]
        if len(leaders) > 1:
            return None
        eliminated = self.game._get_player(leaders[0])
        eliminated.kill()
        self.game.events.emit("player_killed", eliminated, cause="vote")
        return eliminated


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
    # Death-triggered turns owed to players, drained by the runner before the
    # day advances. A queue rather than a callback because each one is a full
    # agent turn with its own trace, not a rule the referee can apply alone.
    pending_last_words: list[int] = field(default_factory=list)
    pending_hunter: int | None = None
    pending_badge: int | None = None
    # The badge. Won at the table on round 1, worth half an extra vote, and
    # inherited rather than destroyed when its holder dies -- which is what
    # makes it worth attacking twice: once for the office, once for the
    # succession.
    sheriff: int | None = None
    sheriff_settled: bool = False
    sheriff_candidates: list[int] = field(default_factory=list)
    sheriff_votes: dict[int, int | None] = field(default_factory=dict)
    sheriff_cursor: int = 0
    badge_transfers: list[dict] = field(default_factory=list)
    last_words_given: set[int] = field(default_factory=set)
    hunter_shots: list[dict] = field(default_factory=list)
    _settle_to: Phase | None = None
    _day: _DayVoting | None = None
    _night: object | None = None

    # ---- construction -------------------------------------------------

    @classmethod
    def new(cls, seed: int, variant: "str | Variant | None" = None) -> "GameState":
        game, setup = build_game(seed, variant)
        return cls(seed=seed, game=game, setup=setup)

    # ---- variant shorthands -------------------------------------------

    @property
    def variant(self) -> Variant:
        return self.setup.variant

    @property
    def num_players(self) -> int:
        return self.setup.num_players

    def seats(self) -> list[int]:
        return self.setup.seats()

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
        """Seat order, rotated to start at the sheriff.

        With no badge in play this is exactly the fixed 1..N it always was, so
        the conformity axis keeps its clean arm. With a sheriff it is not: who
        speaks first becomes something an election decided, which is the point
        of the office and also why sheriff games are reported separately.
        """
        alive = self.alive
        order = [p for p in self.setup.speech_order if p in alive]
        if self.sheriff is not None and self.sheriff in alive:
            i = order.index(self.sheriff)
            order = order[i:] + order[:i]
        return order

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

    # ---- the sheriff ----------------------------------------------------

    def campaign_speaker(self) -> int | None:
        """Whose turn it is to run or stand down."""
        if self.phase is not Phase.SHERIFF_CAMPAIGN:
            return None
        order = [p for p in self.setup.speech_order if p in self.alive]
        if self.sheriff_cursor >= len(order):
            return None
        return order[self.sheriff_cursor]

    def sheriff_electorate(self) -> list[int]:
        """Who votes in the election: the living, minus the candidates.

        Candidates not voting is the table rule, and it also removes the least
        interesting ballot in the game -- a candidate voting for itself.
        """
        return [p for p in self.alive_sorted() if p not in self.sheriff_candidates]

    def _apply_campaign(self, player_id: int, running: bool, content: str) -> dict:
        if self.phase is not Phase.SHERIFF_CAMPAIGN:
            raise ActionError("the election is not open")
        expected = self.campaign_speaker()
        if expected != player_id:
            raise ActionError(
                f"it is player {expected}'s turn in the election, not player {player_id}"
            )
        if running:
            self.sheriff_candidates.append(player_id)
            self.speeches.append(
                Speech(self.round, player_id, len(self.speeches), content,
                       kind="campaign")
            )
        self.sheriff_cursor += 1
        if self.campaign_speaker() is None:
            self._close_campaign()
        return {"accepted": True, "running": running}

    def _close_campaign(self) -> None:
        """No candidates, or only one, and the election never reaches a ballot."""
        if len(self.sheriff_candidates) == 1:
            self.sheriff = self.sheriff_candidates[0]
        if len(self.sheriff_candidates) <= 1:
            self.sheriff_settled = True
            self.phase = Phase.DAY_SPEECH
            self.speech_cursor = 0
        else:
            self.phase = Phase.SHERIFF_VOTE

    def _apply_campaign_vote(self, player_id: int, target_id) -> dict:
        if self.phase is not Phase.SHERIFF_VOTE:
            raise ActionError("the election is not at the ballot")
        if player_id not in self.sheriff_electorate():
            raise ActionError(f"player {player_id} does not vote in this election")
        if player_id in self.sheriff_votes:
            raise ActionError(f"player {player_id} has already voted for sheriff")
        if target_id is not None:
            if isinstance(target_id, bool) or not isinstance(target_id, int):
                raise ActionError("sheriff vote target must be an integer player id")
            if target_id not in self.sheriff_candidates:
                raise ActionError(
                    f"player {target_id} is not standing; candidates are "
                    f"{self.sheriff_candidates}"
                )
        self.sheriff_votes[player_id] = target_id
        if len(self.sheriff_votes) >= len(self.sheriff_electorate()):
            self.resolve_sheriff_election()
        return {"accepted": True, "target": target_id}

    def resolve_sheriff_election(self) -> int | None:
        """Plurality; a tie leaves the badge unclaimed for the whole game."""
        counts: dict[int, int] = {}
        for target in self.sheriff_votes.values():
            if target is not None:
                counts[target] = counts.get(target, 0) + 1
        if counts:
            top = max(counts.values())
            leaders = [t for t, c in counts.items() if c == top]
            self.sheriff = leaders[0] if len(leaders) == 1 else None
        self.sheriff_settled = True
        self.phase = Phase.DAY_SPEECH
        self.speech_cursor = 0
        return self.sheriff

    def vote_weights(self) -> dict[int, float]:
        """Engine-id keyed weights, empty when nobody holds the badge."""
        if self.sheriff is None:
            return {}
        return {to_engine_id(self.sheriff): SHERIFF_VOTE_WEIGHT}

    def _apply_badge_transfer(self, player_id: int, target_id) -> dict:
        if self.pending_badge != player_id:
            raise ActionError(f"player {player_id} has no badge to pass on")
        if target_id is None:
            self.sheriff = None          # torn up rather than handed over
        else:
            if isinstance(target_id, bool) or not isinstance(target_id, int):
                raise ActionError("badge target must be an integer player id")
            if target_id not in self.alive or target_id == player_id:
                raise ActionError(
                    f"player {target_id} cannot take the badge; living players are "
                    f"{self.alive_sorted()}"
                )
            self.sheriff = target_id
        self.pending_badge = None
        self.badge_transfers.append(
            {"round": self.round, "from": player_id, "to": self.sheriff}
        )
        self._settle_if_clear()
        return {"accepted": True, "target": self.sheriff}

    # ---- death-triggered turns -----------------------------------------

    def _queue_deaths(self, died: list[int], cause: str) -> None:
        """Decide what each fresh corpse is owed.

        Two rules, and the reason for both is the same invariant: *how* someone
        died at night is private, and nothing observable may give it away.

        * Last words go to everyone exiled, and to night deaths on round 1 --
          **regardless of cause**. Giving them to a wolf kill but not to a
          poisoning would leak the mechanism through who gets to speak.
        * The hunter may shoot however it died, including poisoned. The
          standard table rule says poison silences the hunter; here that would
          publish the cause through whether the shot lands, so the rule is
          dropped on purpose. Documented deviation, not an oversight.
        """
        if not died:
            return
        v = self.variant
        for pid in died:
            if self.role_of(pid) is Role.HUNTER and self.pending_hunter is None:
                self.pending_hunter = pid
            if pid == self.sheriff and self.pending_badge is None:
                self.pending_badge = pid
            if not v.last_words or pid in self.last_words_given:
                continue
            if cause == "vote" or self.round == 1:
                self.pending_last_words.append(pid)

    def _pending(self) -> bool:
        return (
            bool(self.pending_last_words)
            or self.pending_hunter is not None
            or self.pending_badge is not None
        )

    def next_interrupt(self) -> tuple[str, int] | None:
        """The next death-triggered turn owed, or None.

        The shot goes first: it can kill someone who then owes last words of
        their own, and it can change who has already won.
        """
        if self.pending_hunter is not None:
            return ("hunter_shoot", self.pending_hunter)
        if self.pending_badge is not None:
            return ("badge_transfer", self.pending_badge)
        if self.pending_last_words:
            return ("last_words", self.pending_last_words[0])
        return None

    def hunter_targets(self) -> list[int]:
        if self.pending_hunter is None:
            return []
        return [p for p in self.alive_sorted() if p != self.pending_hunter]

    def _apply_last_words(self, player_id: int, content: str) -> dict:
        if not self.pending_last_words or self.pending_last_words[0] != player_id:
            raise ActionError(f"player {player_id} is not owed last words right now")
        speech = Speech(self.round, player_id, len(self.speeches), content,
                        kind="last_words")
        self.speeches.append(speech)
        self.pending_last_words.pop(0)
        self.last_words_given.add(player_id)
        self._settle_if_clear()
        return {"accepted": True, "kind": "last_words"}

    def _apply_hunter_shoot(self, player_id: int, target_id) -> dict:
        if self.pending_hunter != player_id:
            raise ActionError(f"player {player_id} has no shot to take")
        legal = self.hunter_targets()
        if target_id is None:
            self.pending_hunter = None
            self.hunter_shots.append(
                {"round": self.round, "hunter": player_id, "target": None}
            )
            self._settle_if_clear()
            return {"accepted": True, "target": None}
        if isinstance(target_id, bool) or not isinstance(target_id, int):
            raise ActionError("shot target must be an integer player id")
        if target_id not in legal:
            raise ActionError(
                f"player {target_id} cannot be shot; you may take {legal}"
            )
        victim = self.game._get_player(to_engine_id(target_id))
        victim.kill()
        self.pending_hunter = None
        self.deaths.append(DeathRecord(self.round, target_id, "hunter"))
        self.hunter_shots.append(
            {"round": self.round, "hunter": player_id, "target": target_id}
        )
        # The victim of a shot gets no last words. Bounding the chain keeps a
        # hunter-shoots-hunter case from recursing, and the shot is already
        # public, so nothing is concealed by the silence.
        if self.role_of(target_id) is Role.HUNTER:
            self.pending_hunter = target_id
        self._settle_if_clear()
        return {"accepted": True, "target": target_id}

    def _settle_if_clear(self) -> None:
        """Advance out of the interrupt once nothing more is owed."""
        if self._pending() or self._settle_to is None:
            return
        target, self._settle_to = self._settle_to, None
        if target is Phase.DAY_SPEECH:
            self._open_day()
        else:
            self._close_day()

    # ---- mutation (the only path from an agent to the world) -----------

    def apply_action(self, player_id: int, action: dict) -> dict:
        """`action` is a validated terminal action: {"name": "speak"|"vote", ...}."""
        name = action.get("name")
        if name == "speak":
            return self._apply_speak(player_id, str(action.get("content", "")))
        if name == "vote":
            return self._apply_vote(player_id, action.get("target_id"))
        if name == "campaign_run":
            return self._apply_campaign(player_id, True, str(action.get("content", "")))
        if name == "campaign_pass":
            return self._apply_campaign(player_id, False, "")
        if name == "campaign_vote":
            return self._apply_campaign_vote(player_id, action.get("target_id"))
        if name == "badge_transfer":
            return self._apply_badge_transfer(player_id, action.get("target_id"))
        if name == "last_words":
            return self._apply_last_words(player_id, str(action.get("content", "")))
        if name == "hunter_shoot":
            return self._apply_hunter_shoot(player_id, action.get("target_id"))
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
            self._day = _DayVoting(self.game, self.vote_weights())
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
        action = {
            Role.WOLF: "night_kill",
            Role.GUARD: "night_protect",
        }.get(role, "night_check")
        for candidate in self.setup.fallback_order:
            if candidate in legal:
                return action, candidate
        return "night_skip", None

    def end_night(self) -> list[int]:
        """Resolve the night through the library and open the day."""
        if self._night is None:
            raise ActionError("the night was never opened")
        before = self.alive
        records = list(self._night.records)
        self._night.resolve()
        for record in records:
            if record["action"] == "night_check":
                self.seer_checks.append(
                    CheckRecord(record["round"], record["target"], bool(record["is_wolf"]))
                )
        self.night_records.extend(records)
        self._night = None

        # Attribute deaths from what was actually submitted tonight. Asking the
        # library afterwards does not work: it clears the witch's target inside
        # `resolve()`, so every night death read back as a wolf kill and the
        # poison never appeared in the record at all.
        poisoned = {r["target"] for r in records if r["action"] == "night_poison"}
        died = sorted(before - self.alive)
        for pid in died:
            self.deaths.append(
                DeathRecord(self.round, pid, "witch" if pid in poisoned else "werewolf")
            )
        self.night_victims[self.round] = died
        self._queue_deaths(died, "night")
        if self._pending():
            # The day waits: a corpse speaks, and a dying hunter shoots, before
            # the living take their turns.
            self._settle_to = Phase.DAY_SPEECH
            self.phase = self._interrupt_phase()
        else:
            self._open_day()
        return died

    def _interrupt_phase(self) -> Phase:
        return (
            Phase.HUNTER_SHOOT if self.pending_hunter is not None
            else Phase.LAST_WORDS
        )

    def _open_day(self) -> None:
        self._check_end()
        if self.phase is not Phase.OVER:
            self.phase = (
                Phase.SHERIFF_CAMPAIGN
                if self.variant.sheriff and self.round == 1 and self.sheriff is None
                and not self.sheriff_settled
                else Phase.DAY_SPEECH
            )
            self.speech_cursor = 0

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

    def settle_interrupts_offline(self) -> None:
        """Drain death-triggered turns with no agent: silence, and no shot.

        This is what the engine's own tests use and what the runner degrades to
        when a turn cannot be salvaged. A real game plays every one of these as
        an agent turn with a trace, exactly as it does for the night.
        """
        guard = 0
        while (item := self.next_interrupt()) is not None:
            kind, pid = item
            guard += 1
            if guard > 4 * self.num_players:  # pragma: no cover -- loop backstop
                raise ActionError("interrupt queue did not drain")
            if kind == "hunter_shoot":
                self._apply_hunter_shoot(pid, None)
            elif kind == "badge_transfer":
                self._apply_badge_transfer(pid, None)
            else:
                self._apply_last_words(pid, "")

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

        self._queue_deaths([exiled] if exiled is not None else [], "vote")
        if self._pending():
            self._settle_to = Phase.NIGHT
            self.phase = self._interrupt_phase()
        else:
            self._close_day()
        return exiled

    def _close_day(self) -> None:
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
