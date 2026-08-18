"""Night tests.

The night is now a sequence of agent turns rather than a script, so it needs
the same kind of coverage the day has: whose turn it is, what is legal, what
the referee refuses, and what happens when a turn cannot be salvaged.
"""

from __future__ import annotations

import pytest

from werewolf_harness.engine import ActionError, GameState, Phase, Role
from werewolf_harness.harness.agent.tools import build_registry
from werewolf_harness.harness.schema import SchemaError


def _seat(state: GameState, role: Role) -> int:
    return next(p for p, r in state.roles.items() if r is role)


def _open(seed: int = 5) -> GameState:
    state = GameState.new(seed)
    state.begin_night()
    return state


# ---------------------------------------------------------------- order

def test_actors_come_in_role_priority_order():
    """Seer before wolves before witch -- the witch has to know who was hit."""
    state = _open()
    seen = []
    while (actor := state.night_actor()) is not None:
        seen.append(state.role_of(actor))
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)
    roles = [r.value for r in seen]
    assert roles[0] == "seer"
    assert roles.count("werewolf") == 2
    assert roles[-1] == "witch"


def test_villagers_never_get_a_night_turn():
    state = _open()
    while (actor := state.night_actor()) is not None:
        assert state.role_of(actor) is not Role.VILLAGER
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)


def test_acting_out_of_turn_is_refused():
    state = _open()
    actor = state.night_actor()
    other = next(p for p in state.alive_sorted() if p != actor)
    with pytest.raises(ActionError, match="night turn"):
        state.apply_night_action(other, "night_skip", None)


# --------------------------------------------------------------- legality

def test_a_role_cannot_take_another_roles_action():
    state = _open()
    seer = state.night_actor()
    assert state.role_of(seer) is Role.SEER
    with pytest.raises(ActionError, match="not available"):
        state.apply_night_action(seer, "night_kill", 2)


def test_the_seer_cannot_check_itself_or_repeat_a_check():
    state = _open()
    seer = state.night_actor()
    assert seer not in state.night_options(seer)["targets"]
    first = state.night_options(seer)["targets"][0]
    state.apply_night_action(seer, "night_check", first)

    # finish the night and open the next one
    while (actor := state.night_actor()) is not None:
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)
    state.end_night()
    _play_day(state)
    state.begin_night()
    if state.night_actor() == seer:
        assert first not in state.night_options(seer)["targets"]


def test_wolves_cannot_eat_their_own():
    state = _open()
    _advance_to(state, Role.WOLF)
    wolf = state.night_actor()
    packmate = next(p for p in state.setup.wolves() if p != wolf)
    assert packmate not in state.night_options(wolf)["targets"]
    with pytest.raises(ActionError, match="not a legal kill"):
        state.apply_night_action(wolf, "night_kill", packmate)


# ------------------------------------------------------------- pack vote

def test_the_pack_majority_decides_not_the_last_wolf():
    """Without this the second wolf silently overwrites the first."""
    state = _open()
    _advance_to(state, Role.WOLF)
    first_wolf = state.night_actor()
    targets = state.night_options(first_wolf)["targets"]
    a, b = targets[0], targets[1]

    state.apply_night_action(first_wolf, "night_kill", a)
    second_wolf = state.night_actor()
    assert state.role_of(second_wolf) is Role.WOLF
    assert state.night_options(second_wolf)["pack_votes_so_far"] == {first_wolf: a}

    record = state.apply_night_action(second_wolf, "night_kill", b)
    # A 1-1 split goes to whoever named first.
    assert record["pack_choice"] == a


def test_a_unanimous_pack_gets_what_it_asked_for():
    state = _open()
    _advance_to(state, Role.WOLF)
    first = state.night_actor()
    target = state.night_options(first)["targets"][0]
    state.apply_night_action(first, "night_kill", target)
    second = state.night_actor()
    record = state.apply_night_action(second, "night_kill", target)
    assert record["pack_choice"] == target


# ---------------------------------------------------------------- witch

def test_the_witch_sees_tonights_victim_and_nobody_else_does():
    state = _open()
    _advance_to(state, Role.WITCH)
    witch = state.night_actor()
    options = state.night_options(witch)
    assert options["victims_tonight"]
    assert state.night_options(_seat(state, Role.SEER)) == {"actions": [], "targets": []}


def test_the_antidote_only_works_on_tonights_victim():
    state = _open()
    _advance_to(state, Role.WITCH)
    witch = state.night_actor()
    victim = state.night_options(witch)["victims_tonight"][0]
    other = next(p for p in state.alive_sorted() if p not in (victim, witch))
    with pytest.raises(ActionError, match="tonight's victim"):
        state.apply_night_action(witch, "night_save", other)
    state.apply_night_action(witch, "night_save", victim)
    assert state.end_night() == []


def test_poisoning_tonights_victim_is_refused_as_a_waste():
    """The library would read it as a save; naming the two actions separately
    means the ambiguity has to be resolved rather than guessed."""
    state = _open()
    _advance_to(state, Role.WITCH)
    witch = state.night_actor()
    victim = state.night_options(witch)["victims_tonight"][0]
    with pytest.raises(ActionError, match="already dying"):
        state.apply_night_action(witch, "night_poison", victim)


def test_a_spent_potion_is_not_offered_again():
    """The referee must not advertise an illegal move: an agent that takes it
    up burns its entire turn being refused."""
    state = _open()
    _advance_to(state, Role.WITCH)
    witch = state.night_actor()
    victim = state.night_options(witch)["victims_tonight"][0]
    state.apply_night_action(witch, "night_save", victim)
    state.end_night()
    _play_day(state)

    state.begin_night()
    _advance_to(state, Role.WITCH)
    if state.night_actor() == witch:
        assert "night_save" not in state.night_options(witch)["actions"]


# -------------------------------------------------------------- recovery

def test_the_fallback_never_leaves_the_pack_idle():
    """A failed model call must not turn into "the wolves killed nobody",
    which would change the game rather than just the agent."""
    state = _open()
    _advance_to(state, Role.WOLF)
    wolf = state.night_actor()
    name, target = state.fallback_night_action(wolf)
    assert name == "night_kill"
    assert target in state.night_options(wolf)["targets"]


def test_the_fallback_is_reproducible():
    a, b = _open(11), _open(11)
    for state in (a, b):
        _advance_to(state, Role.WOLF)
    assert a.fallback_night_action(a.night_actor()) == b.fallback_night_action(b.night_actor())


# ------------------------------------------------------------ tool layer

def test_night_tools_are_scoped_to_their_role():
    registry = build_registry()
    seer_tools = {t.name for t in registry.visible("seer", "night")}
    assert "night_check" in seer_tools
    assert {"night_kill", "night_save", "night_poison"} & seer_tools == set()
    assert {t.name for t in registry.visible("villager", "night")} & {
        "night_check", "night_kill", "night_save", "night_poison"
    } == set()


def test_day_turns_are_not_shown_night_actions():
    registry = build_registry()
    day = {t.name for t in registry.visible("werewolf", "speak")}
    assert "night_kill" not in day and "speak" in day


def test_calling_a_night_action_by_day_is_rejected():
    registry = build_registry()
    with pytest.raises(SchemaError, match="not available to you on this turn"):
        registry.validate("night_kill", {"target_id": 3}, "werewolf", "vote")


# ------------------------------------------------------------- integration

def test_a_full_night_produces_a_record_per_actor():
    state = _open()
    actors = 0
    while (actor := state.night_actor()) is not None:
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)
        actors += 1
    state.end_night()
    assert len(state.night_records) == actors
    assert state.phase is Phase.DAY_SPEECH
    assert all(r["round"] == 1 for r in state.night_records)


def test_seer_checks_are_recorded_and_truthful():
    state = _open()
    seer = state.night_actor()
    target = state.night_options(seer)["targets"][0]
    state.apply_night_action(seer, "night_check", target)
    while (actor := state.night_actor()) is not None:
        name, tgt = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, tgt)
    state.end_night()
    check = state.seer_checks[0]
    assert check.target == target
    assert check.is_wolf == (state.role_of(target) is Role.WOLF)


def test_a_poisoned_death_is_recorded_as_the_witchs():
    """The library clears the witch's target inside `resolve()`, so reading it
    back afterwards recorded every night death as a wolf kill and the poison
    never appeared in the record at all."""
    for seed in range(60):
        state = _open(seed)
        poisoned = None
        while (actor := state.night_actor()) is not None:
            if state.role_of(actor) is Role.WITCH:
                options = state.night_options(actor)
                target = next(
                    (t for t in options["targets"]
                     if t not in options.get("victims_tonight", [])), None
                ) if "night_poison" in options["actions"] else None
                if target is not None:
                    state.apply_night_action(actor, "night_poison", target)
                    poisoned = target
                    continue
                state.apply_night_action(actor, "night_skip", None)
            else:
                name, tgt = state.fallback_night_action(actor)
                state.apply_night_action(actor, name, tgt)
        state.end_night()
        if poisoned is None:
            continue
        causes = {d.player_id: d.cause for d in state.deaths}
        assert causes.get(poisoned) == "witch", causes
        assert "werewolf" in causes.values(), "the wolf kill should still be a wolf kill"
        return
    pytest.skip("no seed in range produced a poisoning")


def test_night_death_causes_never_reach_the_table():
    """Whether a night death was the wolves or the poison is something the
    village has to infer. It must not arrive for free in a view or a tool."""
    from werewolf_harness.engine import get_visible_state
    from werewolf_harness.harness.agent.tools import ToolContext, build_registry
    from werewolf_harness.harness.agent.belief import BeliefState

    state = _open(0)
    while (actor := state.night_actor()) is not None:
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)
    state.end_night()

    for viewer in state.alive_sorted():
        for entry in get_visible_state(state, viewer)["public"]["dead"]:
            assert entry["cause"] in ("exiled", "night")

    registry = build_registry()
    ctx = ToolContext(state=state, player_id=state.alive_sorted()[0],
                      belief=BeliefState(1, list(range(1, 9))), view={"private": {}})
    text = registry.execute("query_deaths", {}, ctx).observation
    assert "witch" not in text and "werewolf" not in text, text


# ------------------------------------------------------------------ util

def _advance_to(state: GameState, role: Role) -> None:
    while (actor := state.night_actor()) is not None and state.role_of(actor) is not role:
        name, target = state.fallback_night_action(actor)
        state.apply_night_action(actor, name, target)


def _play_day(state: GameState) -> None:
    while (speaker := state.current_speaker()) is not None:
        state.apply_action(speaker, {"name": "speak", "content": "..."})
    for voter in state.alive_sorted():
        state.apply_action(voter, {"name": "vote", "target_id": None})
    state.resolve_vote()
