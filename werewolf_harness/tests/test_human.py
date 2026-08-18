"""Human-seat tests.

The human path is scored on exactly the same terms as an agent path, so it has
to be exercised the same way. The regression that motivated most of this file:
an empty or closed stdin used to spin `_ask_int` forever, which in a scripted
run means a hung game rather than a failed one.
"""

from __future__ import annotations

import re

import pytest

from werewolf_harness.evalkit.runner import RunConfig, run_game
from werewolf_harness.human.cli import HumanPlayer


class Console:
    """Scripted stdin plus captured stdout.

    When a prompt offers a bracketed list of legal choices, the fake human
    takes the first one. Feeding a fixed number instead would have it rejected
    on any turn where that number is illegal -- a seer cannot check itself --
    which reads as a bug in the CLI when it is really a bug in the fixture.
    """

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.lines: list[str] = []
        self.prompts: list[str] = []

    def input(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        # An empty script means nobody is at the keyboard: answer nothing, so
        # the "walked away" path stays testable.
        offered = re.search(r"\[([\d, ]+)\]", prompt or "")
        if self.answers and offered and "or 0 to" in (prompt or ""):
            first = offered.group(1).split(",")[0].strip()
            if first:
                return first
        return self.answers.pop(0) if self.answers else ""

    def output(self, *args) -> None:
        self.lines.append(" ".join(str(a) for a in args))


def _play(seat: int, answers: list[str], **cfg_kwargs):
    console = Console(answers)
    human = HumanPlayer(seat, input_fn=console.input, output_fn=console.output)
    cfg = RunConfig(
        seed=cfg_kwargs.pop("seed", 2),
        guard_layers=("L1", "L2"),
        attack_enabled=True,
        human_players=(seat,),
        **cfg_kwargs,
    )
    return run_game(cfg, human_ui=human), console


def test_a_human_can_finish_a_game():
    log, _ = _play(1, ["3"] * 400)
    assert not log["outcome"]["crashed"], log["outcome"].get("crash_reason")
    human_turns = [a for r in log["rounds"] for a in r["agents"] if a["is_human"]]
    assert human_turns, "the human seat never acted"
    assert all(t["player_id"] == 1 for t in human_turns)


def test_empty_stdin_abstains_instead_of_hanging():
    """The regression: a person who walks away must not stall the game.

    With no answers at all every turn falls through to the same conservative
    default an exhausted agent takes.
    """
    log, _ = _play(1, [])
    assert not log["outcome"]["crashed"]
    votes = [a for r in log["rounds"] for a in r["agents"]
             if a["is_human"] and a["task"] == "vote"]
    assert votes and all(v["vote"] is None for v in votes)


def test_unusable_answers_are_rejected_then_defaulted():
    console = Console(["nine hundred", "-1", "banana"])
    human = HumanPlayer(1, input_fn=console.input, output_fn=console.output)
    assert human._ask_int("pick: ", allowed={2, 3}, default=0) == 0
    assert any("pick one of" in line for line in console.lines)


def test_human_exposure_is_recorded_like_an_agents():
    """Otherwise the human baseline would show a hijack rate of zero for the
    trivial reason that nobody recorded what they read."""
    log, _ = _play(1, ["3"] * 400)
    human_turns = [a for r in log["rounds"] for a in r["agents"] if a["is_human"]]
    assert sum(len(t["read_payloads"]) for t in human_turns) > 0


@pytest.mark.parametrize("seat", range(1, 9))
def test_the_human_only_ever_sees_their_own_private_facts(seat):
    """Checked per role rather than by scanning for role words: other players
    *claim* roles out loud all the time, and that is legal play. What must never
    appear is a private channel the seat has not earned."""
    log, console = _play(seat, ["3"] * 400, seed=7)
    own_role = log["ground_truth"]["roles"][str(seat)]
    screen = "\n".join(console.lines)

    # Match the CLI's own wording, not the agent prompt's -- an earlier version
    # of this test asserted on the agent phrasing, which made it vacuous for
    # every seat that was not the seer.
    check_lines = [
        line for line in console.lines
        if re.match(r"\s*night \d+: player \d+ is ", line)
    ]

    assert ("your fellow wolves" in screen) == (own_role == "werewolf")
    assert ("antidote:" in screen) == (own_role == "witch")
    assert bool(check_lines) == (own_role == "seer"), (
        f"seat {seat} ({own_role}) saw seer results: {check_lines}"
    )


def test_an_exiled_player_is_not_shown_as_a_night_death():
    """The public death record says "exiled" or "night"; the CLI used to test
    it against the engine's internal "vote" and so labelled every exile a night
    kill -- misinforming the one player whose judgement the baseline measures."""
    log, console = _play(1, ["3"] * 400, seed=5)
    exiles = [r["exiled"] for r in log["rounds"] if r.get("exiled")]
    if not exiles:
        pytest.skip("nobody was exiled in this game")
    dead_lines = [line for line in console.lines if line.strip().startswith("dead:")]
    assert dead_lines, "the human was never shown the dead list"
    assert any("exiled" in line for line in dead_lines), dead_lines[-1]


def test_belief_form_is_collected_once_per_round():
    log, console = _play(1, ["3"] * 400)
    rating_prompts = [p for p in console.prompts if p.strip().startswith("player")]
    rounds = len(log["rounds"])
    assert rating_prompts, "the human was never asked for a suspicion rating"
    # Seven other players at most, asked once per round, never twice.
    assert len(rating_prompts) <= 7 * rounds


def test_human_ratings_land_in_the_belief_state():
    log, _ = _play(1, ["5"] * 400)
    human_turns = [a for r in log["rounds"] for a in r["agents"] if a["is_human"]]
    after = human_turns[0]["belief_after"]
    assert any(entry["suspicion"] == 1.0 for entry in after.values()), (
        "a 5/5 rating should map to suspicion 1.0"
    )
    assert any(entry["reason"] == "human rating" for entry in after.values())


@pytest.mark.parametrize("seat", [1, 4, 8])
def test_any_seat_works(seat):
    log, _ = _play(seat, ["2"] * 400, seed=5)
    assert not log["outcome"]["crashed"]
    assert any(a["is_human"] for r in log["rounds"] for a in r["agents"])
