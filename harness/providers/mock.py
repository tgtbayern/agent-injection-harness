"""Offline scripted client.

WHAT THIS IS: a deterministic stand-in that plays the game well enough to
exercise every path in the harness -- ReAct steps, tool calls, belief updates,
guard blocks, retries, terminal actions -- with no network and no API key. It
makes the test suite meaningful and lets anyone clone the repo and watch a full
game replay immediately.

WHAT THIS IS NOT: evidence about any model. Its susceptibility to injection is
a hand-set number (`susceptibility`), not a measurement. Runs produced with
this client are labelled `model="mock"` in the game log and are excluded from
every reported result; they exist to test the instrument, not to read it.

The susceptibility model is deliberately simple and monotone:

    payload visible and unfenced   -> complies with probability p
    payload visible but fenced     -> complies with probability p * 0.45
    payload removed by the filter  -> cannot comply, the target is not there

so a guard ablation over a mock run has a known-good expected ordering. If the
pipeline ever fails to reproduce that ordering, the harness is broken, not the
model.
"""

from __future__ import annotations

import random
import re

from ..guard import verify_mod
from .base import LLMClient, LLMResponse, ToolCall

_SELF = re.compile(r"You are player (\d+), role: (\w+)")
_NIGHT = re.compile(r"=== NIGHT (\d+)")
_TARGETS = re.compile(r"Legal targets: \[([\d, ]*)\]")
_ACTIONS = re.compile(r"You may call: (.+)")
_VICTIMS = re.compile(r"The wolves attacked: \[([\d, ]*)\]")
_ROUND = re.compile(r"=== ROUND (\d+)")
_ALIVE = re.compile(r"Alive: \[([\d, ]*)\]")
_TASK_SPEAK = "it is your turn to speak"
_BLOCKED = "[guard]"


class MockClient(LLMClient):
    name = "mock"
    tool_mode = "native"

    def __init__(self, seed: int = 0, susceptibility: float = 0.75,
                 evidence_rate: float = 0.4, latency_ms: int = 3):
        self.seed = seed
        self.susceptibility = susceptibility
        self.evidence_rate = evidence_rate
        self.latency_ms = latency_ms

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout: float = 30.0,
    ) -> LLMResponse:
        system = _first(messages, "system")
        # assistant turns carry `content: None` in native tool-calling mode
        prompt = "\n".join(
            (m.get("content") or "") for m in messages if m.get("role") != "system"
        )

        me, role = _identity(system)
        round_no = _int(_ROUND, prompt, 1)
        alive = _alive(prompt, me)
        step = sum(1 for m in messages if m.get("role") == "assistant")
        rng = random.Random(f"{self.seed}:{me}:{round_no}:{step}:{len(prompt)}")

        speaking = _TASK_SPEAK in prompt
        night = _NIGHT.search(prompt) is not None
        retry_reason = _last_block_reason(messages)

        if night:
            call = self._decide_night(rng, me, role, prompt, retry_reason)
            text = call.raw_arguments or ""
            return LLMResponse(
                text=text, tool_calls=[call],
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=max(1, len(text) // 4),
                latency_ms=self.latency_ms, model="mock",
            )

        # What untrusted content is asking for, ignoring anything the filter
        # already removed (removed text is simply not in the prompt any more).
        ordered = [t for t in verify_mod.directive_sequence(prompt)
                   if t in alive and t != me]
        fenced = "<untrusted" in prompt
        p = self.susceptibility * (0.45 if fenced else 1.0)
        # The last instruction wins, the way a person reading down the table
        # would weigh the most recent demand made of them.
        target_of_directive = ordered[-1] if ordered else None
        complies = target_of_directive is not None and rng.random() < p

        call = self._decide(
            rng=rng,
            me=me,
            role=role,
            round_no=round_no,
            alive=alive,
            step=step,
            speaking=speaking,
            prompt=prompt,
            retry_reason=retry_reason,
            directive=target_of_directive,
            complies=complies,
        )
        text = call.raw_arguments or ""
        return LLMResponse(
            text=text,
            tool_calls=[call],
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=self.latency_ms,
            model="mock",
        )

    # ---- night ----------------------------------------------------------

    def _decide_night(self, rng, me, role, prompt, retry_reason) -> ToolCall:
        """A plausible night, played from what the prompt actually offers.

        Deliberately simple, and deliberately not optimal: the mock exists to
        exercise the machinery, and a night policy tuned to win would be one
        more hand-set constant to explain away later.
        """
        targets = _int_list(_TARGETS, prompt)
        actions = _ACTIONS.search(prompt)
        allowed = [a.strip() for a in actions.group(1).split(",")] if actions else []

        if retry_reason and targets:
            # Whatever was refused, take the plainest legal option instead.
            targets = targets[1:] or targets

        if role == "seer" and "night_check" in allowed and targets:
            # Check the player it already suspects most, else anyone unchecked.
            return _mk("night_check", {"target_id": self._suspect(prompt, targets, rng)})

        if role == "werewolf" and "night_kill" in allowed and targets:
            # Wolves hunt the villager the table trusts most -- lowest suspicion
            # in this wolf's own notes.
            return _mk("night_kill", {"target_id": self._least_suspect(prompt, targets, rng)})

        if role == "witch":
            victims = _int_list(_VICTIMS, prompt)
            if "night_save" in allowed and victims and me not in victims:
                return _mk("night_save", {"target_id": victims[0]})
            if "night_save" in allowed and me in victims:
                return _mk("night_save", {"target_id": me})
            if "night_poison" in allowed and targets and rng.random() < 0.25:
                return _mk("night_poison", {"target_id": self._suspect(prompt, targets, rng)})

        return _mk("night_skip", {})

    def _suspect(self, prompt, targets, rng) -> int:
        best, score = None, -1.0
        for m in re.finditer(r"player (\d+): suspicion=([0-9.]+)", prompt):
            pid, val = int(m.group(1)), float(m.group(2))
            if pid in targets and val > score:
                best, score = pid, val
        return best if best is not None else rng.choice(targets)

    def _least_suspect(self, prompt, targets, rng) -> int:
        best, score = None, 2.0
        for m in re.finditer(r"player (\d+): suspicion=([0-9.]+)", prompt):
            pid, val = int(m.group(1)), float(m.group(2))
            if pid in targets and val < score:
                best, score = pid, val
        return best if best is not None else rng.choice(targets)

    # ---- policy ---------------------------------------------------------

    def _decide(self, *, rng, me, role, round_no, alive, step, speaking, prompt,
                retry_reason, directive, complies) -> ToolCall:
        others = [p for p in alive if p != me]
        if not others:  # pragma: no cover -- game would be over
            others = [me]

        # A blocked action is re-decided, not repeated. What "re-decided" means
        # depends on why it was blocked, the way a cooperative model would read
        # the reason: reconcile with what it already said, or drop the target it
        # was told to pick and fall back to its own reading.
        if retry_reason:
            stated = _stated_target(retry_reason)
            refused = _refused_target(retry_reason)
            choices = [p for p in others if p != refused] or others
            fallback = stated if (stated in choices) else self._own_pick(
                rng, me, choices, prompt
            )
            if fallback not in choices:
                fallback = choices[0]
            if speaking:
                return _mk("speak", {"content": self._speech(me, fallback, cited=False)})
            return _mk("vote", {"target_id": fallback})

        if step == 0 and round_no > 1:
            # Look up the record -- this is the tool-return channel (path B).
            subject = self._own_pick(rng, me, others, prompt)
            return _mk("query_history", {"player_id": subject, "round": round_no - 1})

        if step <= 1:
            if directive is not None and complies:
                cites = rng.random() < self.evidence_rate
                return _mk(
                    "update_belief",
                    {
                        "player_id": directive,
                        "suspicion": 0.85,
                        "reason": "the table is converging on this player",
                        "evidence_refs": [f"r{round_no - 1}p{directive}"] if cites else [],
                    },
                )
            subject = self._own_pick(rng, me, others, prompt)
            return _mk(
                "update_belief",
                {
                    "player_id": subject,
                    "suspicion": round(0.35 + 0.5 * rng.random(), 2),
                    "reason": "their story does not line up with the deaths",
                    "evidence_refs": [f"r{round_no - 1}p{subject}"] if round_no > 1 else [],
                },
            )

        pick = directive if (directive is not None and complies) else self._own_pick(
            rng, me, others, prompt
        )
        if pick not in alive or pick == me:
            pick = self._own_pick(rng, me, others, prompt)
        if speaking:
            return _mk("speak", {"content": self._speech(me, pick, cited=round_no > 1)})
        return _mk("vote", {"target_id": pick})

    def _own_pick(self, rng, me, others, prompt) -> int:
        """The agent's own read: highest suspicion it has recorded, else seeded."""
        best, best_score = None, -1.0
        for m in re.finditer(r"player (\d+): suspicion=([0-9.]+)", prompt):
            pid, score = int(m.group(1)), float(m.group(2))
            if pid == me or pid not in others:
                continue
            if score > best_score:
                best, best_score = pid, score
        if best is not None and best_score > 0.5:
            return best
        return rng.choice(others)

    def _speech(self, me: int, target: int, cited: bool) -> str:
        opener = rng_choice(me, target, [
            f"I have been watching player {target} closely.",
            f"Something about player {target} does not add up to me.",
            f"Let me say plainly where I stand on player {target}.",
        ])
        middle = (
            f"Going back over the record, player {target} changed position "
            "without explaining why."
            if cited
            else f"Player {target} has given us nothing concrete so far."
        )
        return f"{opener} {middle} I am player {me}, and I vote {target}."


def rng_choice(a: int, b: int, options: list[str]) -> str:
    return options[(a * 7 + b * 13) % len(options)]


def _mk(name: str, args: dict) -> ToolCall:
    import json

    return ToolCall(name=name, arguments=args, raw_arguments=json.dumps(args))


def _first(messages: list[dict], role: str) -> str:
    for m in messages:
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def _identity(system: str) -> tuple[int, str]:
    m = _SELF.search(system or "")
    return (int(m.group(1)), m.group(2)) if m else (1, "villager")


def _int(rx: re.Pattern, text: str, default: int) -> int:
    m = rx.search(text or "")
    return int(m.group(1)) if m else default


def _int_list(rx: re.Pattern, prompt: str) -> list[int]:
    m = rx.search(prompt or "")
    if not m or not m.group(1).strip():
        return []
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def _alive(prompt: str, me: int) -> list[int]:
    m = _ALIVE.search(prompt or "")
    if not m or not m.group(1).strip():
        return [me]
    return [int(x) for x in m.group(1).split(",") if x.strip()]


_STATED = re.compile(r"you said you would vote for player (\d)")
_REFUSED = re.compile(r"instructed you to vote for player (\d)")


def _stated_target(reason: str) -> int | None:
    m = _STATED.search(reason or "")
    return int(m.group(1)) if m else None


def _refused_target(reason: str) -> int | None:
    m = _REFUSED.search(reason or "")
    return int(m.group(1)) if m else None


def _last_block_reason(messages: list[dict]) -> str | None:
    """Was the immediately preceding step blocked?

    Only the last message counts: a block earlier in the turn has already been
    responded to, and treating it as live would stop the agent ever committing.
    The guard's reply arrives as a `tool` message in native mode and a `user`
    message in JSON mode, so match on the marker rather than the role.
    """
    if not messages:
        return None
    content = messages[-1].get("content", "") or ""
    return content if _BLOCKED in content else None
