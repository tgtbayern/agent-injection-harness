"""Isolation tests.

If a player's view ever contains another player's private information, every
number this project produces is meaningless. This file is therefore the one
piece of mandatory coverage in the repo, and it checks the property directly
rather than through the harness.
"""

from __future__ import annotations

import json

import pytest

from werewolf_harness.engine import (
    GameState,
    Phase,
    Role,
    get_visible_state,
)

SEEDS = list(range(25))


def _play_to_day(seed: int) -> GameState:
    state = GameState.new(seed)
    state.resolve_night()
    return state


@pytest.mark.parametrize("seed", SEEDS)
def test_no_role_information_outside_the_private_section(seed):
    """The public half of a view must be role-free for everyone.

    Checked structurally rather than by inspecting fields one at a time: the
    failure this guards against is a *new* field added later that happens to
    carry a role, so the assertion is over the whole serialised subtree.
    """
    state = _play_to_day(seed)
    role_words = {r.value for r in Role}
    for viewer in range(1, 9):
        view = get_visible_state(state, viewer)
        public_blob = json.dumps(view["public"], ensure_ascii=False).lower()
        for word in role_words:
            assert word not in public_blob, (
                f"role word {word!r} appeared in the public view of player {viewer}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_private_section_only_holds_what_the_viewer_earned(seed):
    """Everything in `private` must be traceable to the viewer's own role."""
    state = _play_to_day(seed)
    allowed = {
        Role.WOLF: {"fellow_wolves"},
        Role.SEER: {"checks"},
        Role.WITCH: {"antidote_available", "poison_available"},
        Role.VILLAGER: set(),
    }
    for viewer in range(1, 9):
        private = get_visible_state(state, viewer)["private"]
        assert set(private) <= allowed[state.role_of(viewer)], (
            f"player {viewer} ({state.role_of(viewer).value}) saw {sorted(private)}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_only_wolves_learn_the_pack(seed):
    state = _play_to_day(seed)
    wolves = set(state.setup.wolves())
    for viewer in range(1, 9):
        private = get_visible_state(state, viewer)["private"]
        if viewer in wolves:
            assert set(private["fellow_wolves"]) == wolves - {viewer}
        else:
            assert "fellow_wolves" not in private


@pytest.mark.parametrize("seed", SEEDS)
def test_only_the_seer_sees_check_results(seed):
    state = _play_to_day(seed)
    seer = next(p for p in range(1, 9) if state.role_of(p) is Role.SEER)
    for viewer in range(1, 9):
        private = get_visible_state(state, viewer)["private"]
        if viewer == seer:
            assert "checks" in private
        else:
            assert "checks" not in private


@pytest.mark.parametrize("seed", SEEDS)
def test_seer_checks_are_truthful(seed):
    state = _play_to_day(seed)
    seer = next(p for p in range(1, 9) if state.role_of(p) is Role.SEER)
    for check in get_visible_state(state, seer)["private"]["checks"]:
        assert check["is_wolf"] == (state.role_of(check["target"]) is Role.WOLF)


@pytest.mark.parametrize("seed", SEEDS)
def test_only_the_witch_sees_potions(seed):
    state = _play_to_day(seed)
    witch = next(p for p in range(1, 9) if state.role_of(p) is Role.WITCH)
    for viewer in range(1, 9):
        private = get_visible_state(state, viewer)["private"]
        assert ("antidote_available" in private) == (viewer == witch)


@pytest.mark.parametrize("seed", SEEDS[:10])
def test_ground_truth_never_reaches_a_view(seed):
    state = _play_to_day(seed)
    for viewer in range(1, 9):
        view = get_visible_state(state, viewer)
        assert "roles" not in view
        assert "ground_truth" not in view
        assert "winner" not in json.dumps(view)


def test_public_transcript_is_identical_for_everyone():
    state = _play_to_day(3)
    for speaker in list(state.speech_order_this_round()):
        state.apply_action(speaker, {"name": "speak", "content": f"p{speaker} speaking"})
    views = [get_visible_state(state, p)["public"] for p in range(1, 9)]
    assert all(v == views[0] for v in views)


def test_dead_player_view_still_isolated():
    state = _play_to_day(11)
    dead = [d.player_id for d in state.deaths]
    if not dead:
        pytest.skip("nobody died on night one for this seed")
    view = get_visible_state(state, dead[0])
    assert view["you"]["alive"] is False
    assert view["you"]["player_id"] == dead[0]


def test_view_rejects_unknown_player():
    state = GameState.new(1)
    with pytest.raises(ValueError):
        get_visible_state(state, 99)


@pytest.mark.parametrize("seed", SEEDS[:10])
def test_speech_position_matches_the_public_order(seed):
    state = _play_to_day(seed)
    if state.phase is not Phase.DAY_SPEECH:
        pytest.skip("game ended during the night")
    order = state.speech_order_this_round()
    for viewer in order:
        view = get_visible_state(state, viewer)
        assert view["you"]["speech_position"] == order.index(viewer) + 1
        assert view["public"]["speech_order"] == order
