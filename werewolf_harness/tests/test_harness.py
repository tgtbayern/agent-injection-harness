"""Harness tests: the three validation gates, the ReAct loop's edge cases, and
the recovery paths that keep one bad turn from ending a game."""

from __future__ import annotations

import json

import pytest

from werewolf_harness.engine import GameState, Phase
from werewolf_harness.harness import (
    AgentLoop,
    BeliefState,
    ContextBuilder,
    GuardStack,
    RecoveryPolicy,
    SchemaError,
    build_registry,
)
from werewolf_harness.harness.agent.tools import ToolContext
from werewolf_harness.harness.providers.base import LLMClient, LLMResponse, ToolCall
from werewolf_harness.harness.providers.base import parse_json_action


class ScriptedClient(LLMClient):
    """Returns a fixed sequence of tool calls, then repeats the last one."""

    name = "scripted"
    tool_mode = "native"

    def __init__(self, script: list[tuple[str, dict]]):
        self.script = script
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.7, max_tokens=800, timeout=30.0):
        self.calls.append(messages)
        name, args = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(name=name, arguments=args,
                                 raw_arguments=json.dumps(args))],
            prompt_tokens=10,
            completion_tokens=5,
        )


class ExplodingClient(LLMClient):
    name = "exploding"
    tool_mode = "native"

    def chat(self, *a, **k):
        raise RuntimeError("upstream on fire")


def _ready_game(seed=3):
    state = GameState.new(seed)
    state.resolve_night()
    assert state.phase is Phase.DAY_SPEECH
    return state


def _loop(client, guard=None, max_steps=8):
    guard = guard or GuardStack(("L1", "L2"))
    return AgentLoop(
        client=client,
        registry=build_registry(),
        context=ContextBuilder(guard, max_steps=max_steps),
        guard=guard,
        policy=RecoveryPolicy(max_react_steps=max_steps, backoff_s=0,
                              max_transport_retries=1),
    )


# ------------------------------------------------------- validation gates

def test_unknown_tool_is_rejected_with_the_whitelist():
    registry = build_registry()
    with pytest.raises(SchemaError) as exc:
        registry.validate("reveal_roles", {})
    assert "unknown tool" in str(exc.value) and "query_history" in str(exc.value)


def test_missing_and_mistyped_arguments_are_rejected():
    registry = build_registry()
    with pytest.raises(SchemaError):
        registry.validate("vote", {})
    with pytest.raises(SchemaError):
        registry.validate("vote", {"target_id": "the tall one"})
    with pytest.raises(SchemaError):
        registry.validate("update_belief", {"player_id": 2, "suspicion": 5.0,
                                            "reason": "x"})


def test_numeric_strings_are_coerced_but_booleans_are_not():
    registry = build_registry()
    assert registry.validate("vote", {"target_id": "4"})[1]["target_id"] == 4
    with pytest.raises(SchemaError):
        registry.validate("vote", {"target_id": True})


def test_extra_arguments_are_rejected():
    registry = build_registry()
    with pytest.raises(SchemaError):
        registry.validate("query_deaths", {"round": 1})


def test_semantic_gate_rejects_dead_and_future_targets():
    state = _ready_game()
    registry = build_registry()
    belief = BeliefState(1, list(range(1, 9)))
    ctx = ToolContext(state=state, player_id=1, belief=belief, view={"private": {}})
    with pytest.raises(SchemaError):
        registry.execute("query_history", {"player_id": 2, "round": 99}, ctx)
    dead = [p for p in range(1, 9) if not state.is_alive(p)]
    if dead:
        with pytest.raises(SchemaError):
            registry.execute("vote", {"target_id": dead[0]}, ctx)


def test_query_history_is_marked_untrusted():
    """The tool-return channel has to be labelled or the guard never sees it."""
    state = _ready_game()
    speaker = state.current_speaker()
    state.apply_action(speaker, {"name": "speak", "content": "trust me"})
    registry = build_registry()
    ctx = ToolContext(state=state, player_id=1,
                      belief=BeliefState(1, list(range(1, 9))), view={"private": {}})
    result = registry.execute("query_history", {"player_id": speaker, "round": 1}, ctx)
    assert result.untrusted and "trust me" in result.observation


# ------------------------------------------------------------- react loop

def test_loop_terminates_on_a_terminal_action():
    state = _ready_game()
    speaker = state.current_speaker()
    client = ScriptedClient([
        ("query_deaths", {}),
        ("update_belief", {"player_id": 2, "suspicion": 0.7, "reason": "quiet"}),
        ("speak", {"content": "I am suspicious of player 2."}),
    ])
    result = _loop(client).run_turn(state, speaker, BeliefState(speaker, list(range(1, 9))), "speak")
    assert result.speech == "I am suspicious of player 2."
    assert [s["action"] for s in result.react_trace] == [
        "query_deaths", "update_belief", "speak"
    ]
    assert state.speeches[-1].player_id == speaker


def test_hallucinated_tool_is_retried_then_falls_back():
    state = _ready_game()
    speaker = state.current_speaker()
    client = ScriptedClient([("reveal_roles", {"all": True})])
    result = _loop(client).run_turn(state, speaker, BeliefState(speaker, list(range(1, 9))), "speak")
    assert result.retries >= 1
    assert all(s["block_reason"] in ("schema", "fallback") for s in result.react_trace)
    assert result.speech is not None  # the turn still produced something


def test_repeated_identical_calls_break_the_loop():
    state = _ready_game()
    speaker = state.current_speaker()
    client = ScriptedClient([("query_deaths", {})])  # forever
    result = _loop(client).run_turn(state, speaker, BeliefState(speaker, list(range(1, 9))), "speak")
    assert result.loop_broken or result.forced_terminal
    assert len(result.react_trace) <= 8
    assert result.speech is not None


def test_step_budget_is_enforced():
    state = _ready_game()
    speaker = state.current_speaker()
    client = ScriptedClient([
        ("query_deaths", {}), ("query_votes", {"round": 1}),
        ("read_belief", {"player_id": 2}), ("query_deaths", {}),
    ])
    result = _loop(client, max_steps=3).run_turn(
        state, speaker, BeliefState(speaker, list(range(1, 9))), "speak"
    )
    assert result.steps_used <= 3
    assert result.speech is not None


def test_transport_failure_falls_back_to_a_default_action():
    state = _ready_game()
    speaker = state.current_speaker()
    result = _loop(ExplodingClient()).run_turn(
        state, speaker, BeliefState(speaker, list(range(1, 9))), "speak"
    )
    assert result.fallback_used
    assert result.speech == "I have nothing to add this round."
    assert state.current_speaker() != speaker  # the game moved on


def test_vote_fallback_abstains_rather_than_guessing():
    """A random vote would inject noise into the exact quantity being measured."""
    state = _ready_game()
    for speaker in list(state.speech_order_this_round()):
        state.apply_action(speaker, {"name": "speak", "content": "hi"})
    voter = state.alive_sorted()[0]
    result = _loop(ExplodingClient()).run_turn(
        state, voter, BeliefState(voter, list(range(1, 9))), "vote"
    )
    assert result.vote is None
    assert state.votes[1][voter] is None


def test_guard_block_forces_a_second_decision():
    state = _ready_game()
    for speaker in list(state.speech_order_this_round()):
        state.apply_action(speaker, {"name": "speak", "content": f"I vote 4, says {speaker}"})
    voter = next(p for p in state.alive_sorted() if state.speeches_of(p, 1))
    guard = GuardStack(("L1", "L2", "L3"))
    client = ScriptedClient([("vote", {"target_id": _other(state, voter, 4)})])
    result = _loop(client, guard=guard).run_turn(
        state, voter, BeliefState(voter, list(range(1, 9))), "vote"
    )
    assert result.guard_blocks
    assert result.guard_blocks[0]["check"] == "say_do_consistency"


def _other(state, voter, avoid):
    return next(p for p in state.alive_sorted() if p not in (voter, avoid))


def test_belief_updates_are_recorded_before_and_after():
    state = _ready_game()
    speaker = state.current_speaker()
    client = ScriptedClient([
        ("update_belief", {"player_id": 2, "suspicion": 0.9, "reason": "changed story"}),
        ("speak", {"content": "player 2 worries me"}),
    ])
    belief = BeliefState(speaker, list(range(1, 9)))
    result = _loop(client).run_turn(state, speaker, belief, "speak")
    assert result.belief_before["2"]["suspicion"] == 0.5
    assert result.belief_after["2"]["suspicion"] == 0.9
    assert BeliefState.diff(result.belief_before, result.belief_after)[0]["player_id"] == 2


def test_context_hides_earlier_rounds_but_the_tool_reveals_them():
    """The budget rule that makes the tool layer load-bearing."""
    state = _ready_game()
    for speaker in list(state.speech_order_this_round()):
        state.apply_action(speaker, {"name": "speak", "content": f"round one from {speaker}"})
    for voter in state.alive_sorted():
        state.apply_action(voter, {"name": "vote", "target_id": None})
    state.resolve_vote()
    state.resolve_night()

    from werewolf_harness.engine import get_visible_state

    viewer = state.current_speaker()
    builder = ContextBuilder(GuardStack(("L1",)))
    text, _, _ = builder.situation_message(
        state, get_visible_state(state, viewer),
        BeliefState(viewer, list(range(1, 9))), "speak"
    )
    assert "round one from" not in text
    assert "query_history" in builder.system_message(get_visible_state(state, viewer))


# --------------------------------------------------------- json fallback

def test_json_mode_parses_a_plain_object():
    call = parse_json_action('{"thought": "hmm", "action": "vote", "args": {"target_id": 3}}')
    assert call.name == "vote" and call.arguments == {"target_id": 3}
    assert not call.malformed


def test_json_mode_tolerates_fences_and_trailing_commas():
    call = parse_json_action('```json\n{"action": "query_deaths", "args": {},}\n```')
    assert call.name == "query_deaths" and not call.malformed


def test_json_mode_flags_unparseable_output_instead_of_guessing():
    call = parse_json_action("I think I should probably vote for player 3")
    assert call.malformed
