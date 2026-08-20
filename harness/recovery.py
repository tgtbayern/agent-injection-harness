"""Failure handling.

The point is that one agent failing must never take the game down: a crashed
game is a discarded sample, and discarded samples are how a batch quietly turns
into a biased batch. Every failure mode has a defined fallback, and every
fallback is counted, because "how often did the model need saving" is itself
one of the reported numbers (axis 4).

    timeout / transport error   -> retry with backoff, then abstain for the turn
    malformed output            -> retry with the parse error attached (<=3)
    unknown tool                -> retry with the whitelist attached
    semantically invalid action -> retry with the engine's reason attached
    ReAct loop not terminating  -> force a terminal action
    repeated identical calls    -> break the loop, force a terminal action
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .providers.base import ProviderError

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class RecoveryPolicy:
    max_retries: int = 3
    max_transport_retries: int = 2
    timeout_s: float = 30.0
    backoff_s: float = 2.0
    max_react_steps: int = 8
    loop_repeat_threshold: int = 3
    sleep = staticmethod(time.sleep)


@dataclass
class RecoveryStats:
    retries: int = 0
    transport_retries: int = 0
    timeouts: bool = False
    forced_terminal: bool = False
    loop_broken: bool = False
    fallback_used: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retries": self.retries,
            "transport_retries": self.transport_retries,
            "timeout": self.timeouts,
            "forced_terminal": self.forced_terminal,
            "loop_broken": self.loop_broken,
            "fallback_used": self.fallback_used,
            "errors": self.errors[:10],
        }


def call_model(client, messages, tools, *, policy: RecoveryPolicy, stats: RecoveryStats,
               temperature: float, max_tokens: int):
    """One model call, with bounded retries on transport-level failures.

    Returns None when every attempt failed; the caller then falls back to the
    conservative default for the turn rather than propagating an exception.
    """
    delay = policy.backoff_s
    for attempt in range(policy.max_transport_retries + 1):
        try:
            return client.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=policy.timeout_s,
            )
        except ProviderError as exc:
            retryable = exc.status in RETRYABLE_STATUS or exc.status is None
            stats.errors.append(str(exc)[:200])
            if "timed out" in str(exc).lower():
                stats.timeouts = True
            if not retryable or attempt == policy.max_transport_retries:
                return None
            stats.transport_retries += 1
            policy.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001 -- a client bug must not kill a game
            stats.errors.append(f"{type(exc).__name__}: {exc}"[:200])
            return None
    return None


# The conservative default for each kind of turn. Every task a runner can
# start must appear here: the loop that drives a phase advances only when the
# terminal action lands, so a task with no default does not degrade -- it
# hangs. That is how the 12-player table first failed, and why this is a table
# rather than an if/else that quietly falls through to `vote`.
DEFAULT_ACTIONS: dict[str, tuple[str, dict]] = {
    "speak": ("speak", {"content": "I have nothing to add this round."}),
    "vote": ("vote", {"target_id": None}),
    "campaign": ("campaign_pass", {"reason": "the turn could not be salvaged"}),
    "campaign_vote": ("campaign_vote", {"target_id": None}),
    "last_words": ("last_words", {"content": "I have nothing to add."}),
    "hunter_shoot": ("hunter_hold", {"reason": "the turn could not be salvaged"}),
    "badge": ("badge_tear", {"reason": "the turn could not be salvaged"}),
}


def default_action(task: str, player_id: int) -> tuple[str, dict]:
    """The conservative default when a turn cannot be salvaged.

    Abstention rather than a random vote: a random vote would inject noise into
    exactly the quantity being measured. The same reasoning gives the offices
    their defaults -- not standing, not firing, not passing the badge on -- so
    a broken turn never hands anyone an advantage it did not play for.
    """
    if task not in DEFAULT_ACTIONS:
        raise KeyError(
            f"no default action for task {task!r}; a phase whose turn cannot "
            "fail safely will hang the game rather than degrade"
        )
    return DEFAULT_ACTIONS[task]
