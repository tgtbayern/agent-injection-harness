"""The 12-player table: sheriff, last words, hunter, guard.

These mechanics are not here to make the game richer. Each one changes *who is
worth attacking*: an office concentrates influence, and a corpse gets one turn
nobody can answer. The tests below pin the rules, and two of them pin a
privacy invariant that the mechanics could otherwise break.
"""

from __future__ import annotations

import pytest

from werewolf_harness.engine.rules import Phase, Role
from werewolf_harness.engine.state import ActionError, GameState


def _state(seed=3, variant="12p"):
    return GameState.new(seed, variant=variant)


def _vote_out(state, target):
    """Everyone but the target votes for the target. Self-votes are illegal."""
    for voter in state.alive_sorted():
        if voter == target:
            state.apply_action(voter, {"name": "vote", "target_id": None})
        else:
            state.apply_action(voter, {"name": "vote", "target_id": target})
    return state.resolve_vote()


def _seat_with(state, role):
    return next(p for p, r in state.roles.items() if r is role)


def _run_night(state, choices=None):
    """Play a night with the fallback policy, overriding named actors."""
    choices = choices or {}
    state.begin_night()
    while (actor := state.night_actor()) is not None:
        if actor in choices:
            name, target = choices[actor]
        else:
            name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)
    return state.end_night()


# ---------------------------------------------------------------- variant

def test_twelve_player_variant_has_the_offices_switched_on():
    state = _state()
    assert state.num_players == 12
    assert state.variant.sheriff and state.variant.last_words


def test_eight_player_variant_is_untouched():
    """Every existing result was produced on this. It must not have moved."""
    state = _state(variant="8p")
    assert state.num_players == 8
    assert not state.variant.sheriff and not state.variant.last_words
    _run_night(state)
    assert state.phase is Phase.DAY_SPEECH   # straight to the day, no interrupt
    assert state.sheriff is None and not state.pending_last_words


# ------------------------------------------------------------- last words

def _reach_day(state):
    _run_night(state)
    state.settle_interrupts_offline()
    if state.phase is Phase.SHERIFF_CAMPAIGN:
        while (speaker := state.campaign_speaker()) is not None:
            state.apply_action(speaker, {"name": "campaign_pass"})
    return state


def test_an_exiled_player_gets_last_words():
    state = _reach_day(_state())
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    target = state.alive_sorted()[1]
    exiled = _vote_out(state, target)

    assert exiled == target
    assert state.phase in (Phase.LAST_WORDS, Phase.HUNTER_SHOOT)
    kind, who = state.next_interrupt()
    if kind == "hunter_shoot":            # the exile happened to be the hunter
        state.apply_action(who, {"name": "hunter_shoot", "target_id": None})
        kind, who = state.next_interrupt()
    assert (kind, who) == ("last_words", exiled)

    state.apply_action(exiled, {"name": "last_words", "content": "I was a villager"})
    assert state.next_interrupt() is None
    assert state.phase is Phase.NIGHT     # the day closed only once nothing was owed


def test_last_words_land_in_the_public_transcript():
    """Which is what makes them an attack surface, not a courtesy.

    A dying player's speech is retrievable by `query_history` exactly like a
    living one's -- from a speaker who can no longer be questioned about it.
    """
    state = _reach_day(_state())
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    target = state.alive_sorted()[1]
    _vote_out(state, target)
    while (item := state.next_interrupt()) is not None:
        kind, who = item
        if kind == "last_words":
            state.apply_action(who, {"name": "last_words", "content": "PAYLOAD HERE"})
        elif kind == "hunter_shoot":
            state.apply_action(who, {"name": "hunter_shoot", "target_id": None})
        else:
            state.apply_action(who, {"name": "badge_transfer", "target_id": None})

    spoken = [s for s in state.speeches if s.kind == "last_words"]
    assert spoken and "PAYLOAD HERE" in spoken[0].content
    assert spoken[0].player_id == target


def test_night_deaths_get_last_words_only_on_round_one():
    state = _state(seed=11)
    died = _run_night(state)
    assert died, "seed chosen so somebody dies on night 1"
    assert state.pending_last_words == died

    state.settle_interrupts_offline()
    if state.phase is Phase.SHERIFF_CAMPAIGN:
        while (speaker := state.campaign_speaker()) is not None:
            state.apply_action(speaker, {"name": "campaign_pass"})
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    for voter in state.alive_sorted():
        state.apply_action(voter, {"name": "vote", "target_id": None})
    state.resolve_vote()                      # nobody exiled, round 2 begins

    _run_night(state)
    assert state.pending_last_words == [], "only round 1 night deaths speak"


def test_last_words_do_not_reveal_how_the_night_death_happened():
    """The privacy invariant the mechanic could most easily have broken.

    If a wolf kill spoke and a poisoning did not, the presence of last words
    would publish the cause of death for free -- which is precisely what the
    village is supposed to have to infer.
    """
    state = _state(seed=11)
    witch = _seat_with(state, Role.WITCH)
    victim = next(p for p in state.alive_sorted() if p != witch)
    _run_night(state, choices={witch: ("night_poison", victim)})

    poisoned = [d for d in state.deaths if d.cause == "witch"]
    assert poisoned, "the poison landed"
    for death in state.deaths:
        assert death.player_id in state.pending_last_words

    public = {d["player_id"]: d["cause"] for d in
              __import__("werewolf_harness.engine.visibility", fromlist=["x"])
              .get_visible_state(state, 1)["public"]["dead"]}
    assert set(public.values()) <= {"night"}   # never "witch" or "werewolf"


# ------------------------------------------------------------------ hunter

def test_the_hunter_chooses_its_shot_rather_than_drawing_one():
    """The library kills a random survivor on the hunter's death. Disabled.

    A terminal action with no reasoning behind it has no trace to read, which
    is the one thing this harness cannot have.
    """
    state = _state()
    hunter = _seat_with(state, Role.HUNTER)
    alive_before = state.alive_sorted()
    state.game._get_player(hunter - 1).kill()
    state.role_of(hunter)   # library hook would have fired by now
    assert len(state.alive_sorted()) == len(alive_before) - 1, (
        "only the hunter died; nobody was shot at random"
    )


def test_the_hunter_shot_kills_and_is_recorded():
    state = _state(seed=11)
    hunter = _seat_with(state, Role.HUNTER)
    _run_night(state, choices={
        w: ("night_kill", hunter) for w in state.setup.wolves()
    })
    if hunter not in state.alive:
        assert state.pending_hunter == hunter
        victim = state.hunter_targets()[0]
        state.apply_action(hunter, {"name": "hunter_shoot", "target_id": victim})
        assert victim not in state.alive
        assert state.hunter_shots[-1] == {
            "round": 1, "hunter": hunter, "target": victim
        }


def test_a_shot_must_name_a_living_player():
    state = _state(seed=11)
    hunter = _seat_with(state, Role.HUNTER)
    _run_night(state, choices={w: ("night_kill", hunter) for w in state.setup.wolves()})
    if state.pending_hunter == hunter:
        with pytest.raises(ActionError):
            state.apply_action(hunter, {"name": "hunter_shoot", "target_id": hunter})


# ------------------------------------------------------------------- guard

def test_the_guard_cannot_protect_the_same_player_twice_running():
    state = _state()
    guard = _seat_with(state, Role.GUARD)
    charge = next(p for p in state.alive_sorted() if p != guard)
    _run_night(state, choices={guard: ("night_protect", charge)})
    state.settle_interrupts_offline()
    if state.phase is Phase.SHERIFF_CAMPAIGN:
        while (speaker := state.campaign_speaker()) is not None:
            state.apply_action(speaker, {"name": "campaign_pass"})
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    for voter in state.alive_sorted():
        state.apply_action(voter, {"name": "vote", "target_id": None})
    state.resolve_vote()

    state.begin_night()
    while (actor := state.night_actor()) is not None:
        if actor == guard:
            assert charge not in state._night.legal_targets(guard), (
                "last night's charge is not on the menu"
            )
            with pytest.raises(ActionError):
                state.apply_night_action(guard, "night_protect", charge)
            state.apply_night_action(guard, "night_skip", None)
        else:
            name, target = state.fallback_night_action(actor)
            state.apply_night_action(actor, name, target)


# ----------------------------------------------------------------- sheriff

def _elect(state, candidates, votes):
    while (speaker := state.campaign_speaker()) is not None:
        if speaker in candidates:
            state.apply_action(speaker, {"name": "campaign_run", "content": "vote me"})
        else:
            state.apply_action(speaker, {"name": "campaign_pass"})
    for voter, target in votes.items():
        if state.phase is Phase.SHERIFF_VOTE:
            state.apply_action(voter, {"name": "campaign_vote", "target_id": target})
    return state.sheriff


def test_a_single_candidate_takes_the_badge_without_a_ballot():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    assert state.phase is Phase.SHERIFF_CAMPAIGN
    only = state.alive_sorted()[0]
    _elect(state, {only}, {})
    assert state.sheriff == only
    assert state.phase is Phase.DAY_SPEECH


def test_nobody_stands_and_the_badge_stays_unclaimed():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    _elect(state, set(), {})
    assert state.sheriff is None
    assert state.sheriff_settled and state.phase is Phase.DAY_SPEECH


def test_the_sheriff_speaks_first():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    alive = state.alive_sorted()
    chosen = alive[3]
    _elect(state, {chosen}, {})
    assert state.speech_order_this_round()[0] == chosen
    # ...and the rest is still seat order, rotated rather than reshuffled.
    rotated = state.speech_order_this_round()
    i = alive.index(chosen)
    assert rotated == alive[i:] + alive[:i]


def test_candidates_do_not_vote_in_the_election():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    alive = state.alive_sorted()
    runners = {alive[0], alive[1]}
    while (speaker := state.campaign_speaker()) is not None:
        state.apply_action(
            speaker,
            {"name": "campaign_run", "content": "x"} if speaker in runners
            else {"name": "campaign_pass"},
        )
    assert state.phase is Phase.SHERIFF_VOTE
    assert set(state.sheriff_electorate()) == set(alive) - runners
    with pytest.raises(ActionError):
        state.apply_action(alive[0], {"name": "campaign_vote", "target_id": alive[1]})


def test_the_badge_is_worth_half_a_vote_more():
    """The whole point of the office, and the reason it is worth hijacking."""
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    alive = state.alive_sorted()
    sheriff = alive[0]
    _elect(state, {sheriff}, {})
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})

    a, b = alive[1], alive[2]
    # Split the plain ballots evenly between a and b, with neither voting for
    # itself, so the tally is level before the badge is counted.
    rest = [p for p in state.alive_sorted() if p not in (sheriff, a, b)]
    half = len(rest) // 2
    for voter in rest[:half]:
        state.apply_action(voter, {"name": "vote", "target_id": a})
    for voter in rest[half:]:
        state.apply_action(voter, {"name": "vote", "target_id": b})
    state.apply_action(a, {"name": "vote", "target_id": b})
    state.apply_action(b, {"name": "vote", "target_id": a})
    state.apply_action(sheriff, {"name": "vote", "target_id": a})

    exiled = state.resolve_vote()
    counts = state.vote_counts[1]
    assert counts[a] == counts[b], "a dead heat on heads"
    assert exiled == a, "and the badge breaks it"


def test_the_badge_outlives_its_holder():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    alive = state.alive_sorted()
    sheriff = alive[0]
    _elect(state, {sheriff}, {})
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    _vote_out(state, sheriff)

    kind, who = state.next_interrupt()
    if kind == "hunter_shoot":
        state.apply_action(who, {"name": "hunter_shoot", "target_id": None})
        kind, who = state.next_interrupt()
    assert (kind, who) == ("badge_transfer", sheriff)

    heir = next(p for p in state.alive_sorted() if p != sheriff)
    state.apply_action(sheriff, {"name": "badge_transfer", "target_id": heir})
    assert state.sheriff == heir
    assert state.badge_transfers[-1] == {"round": 1, "from": sheriff, "to": heir}


def test_the_badge_can_be_torn_up_instead():
    state = _state()
    _run_night(state)
    state.settle_interrupts_offline()
    sheriff = state.alive_sorted()[0]
    _elect(state, {sheriff}, {})
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    _vote_out(state, sheriff)
    while (item := state.next_interrupt()) is not None:
        kind, who = item
        if kind == "badge_transfer":
            state.apply_action(who, {"name": "badge_transfer", "target_id": None})
        elif kind == "hunter_shoot":
            state.apply_action(who, {"name": "hunter_shoot", "target_id": None})
        else:
            state.apply_action(who, {"name": "last_words", "content": ""})
    assert state.sheriff is None


# ------------------------------------------------------- end to end

def test_a_twelve_player_game_runs_every_phase_offline():
    """The whole table, driven by the scripted client, with a hard time box.

    The time box is the point of the test as much as the coverage is: the first
    version of this hung, because a phase whose turn cannot fail safely does not
    degrade -- it spins. `recovery.DEFAULT_ACTIONS` is what fixed it, and this
    is what would catch the next task added without one.
    """
    import signal

    from werewolf_harness.evalkit.runner import RunConfig, run_game

    def _timeout(*_a):  # pragma: no cover -- only fires on regression
        raise TimeoutError("a 12-player mock game did not terminate")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(180)
    try:
        log = run_game(RunConfig(seed=7, variant="12p",
                                 guard_layers=("L1", "L2"), attack_enabled=True))
    finally:
        signal.alarm(0)

    assert not log["outcome"]["crashed"], log["outcome"].get("crash_reason")
    assert log["config"]["variant"] == "12p"
    assert len(log["config"]["seat_models"]) == 12

    tasks = {a["task"] for r in log["rounds"] for a in r["agents"]}
    assert {"campaign", "campaign_vote", "speak", "vote"} <= tasks
    assert log["rounds"][0]["sheriff_candidates"], "somebody stood for the badge"


def test_every_task_the_runner_starts_has_a_safe_default():
    """A phase with no default hangs the game rather than degrading."""
    from werewolf_harness.harness.recovery import DEFAULT_ACTIONS

    started_by_runner = {
        "night", "speak", "vote",
        "campaign", "campaign_vote", "last_words", "hunter_shoot", "badge",
    }
    # `night` is special-cased in the loop: the engine hands back a seeded
    # target rather than a constant, so it is not in the table.
    assert started_by_runner - {"night"} <= set(DEFAULT_ACTIONS)
