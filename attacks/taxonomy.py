"""Attack corpus, injector and payload detection.

Why the attacker is inside the game rather than outside it: in this environment
the manipulator is a *legal participant* with a rule-given motive. That removes
the usual escape hatch of injection benchmarks, where the attacker's text is
alien to the task and a classifier can win by recognising the genre. Here the
carrier is always a real speech from a real player, and the defence never gets
to see who is speaking from what role.

Delivery is deliberately one-shot per speech: the payload is appended to a
speech the attacker was going to make anyway. It then reaches victims through
two channels, which are scored separately:

  path A  speech       -- read in the round it was spoken
  path B  tool return  -- read later via `query_history`, which quotes it back

Path B needs no extra machinery: the transcript persists, so any payload from
round r is retrievable in round r+1. That is the interesting case, because a
tool return usually arrives with the implicit authority of retrieved data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Aligned with OWASP Top 10 for Agentic Applications and the DeepTeam
# vulnerability families, so results can be reported against a shared taxonomy.
CATEGORIES = {
    "instruction_override": "ASI01 -- agent goal hijack via direct instruction",
    "role_forgery": "ASI07 -- forged conversation structure in an inter-agent message",
    "authority_impersonation": "ASI01 -- claimed moderator/system authority",
    "context_manipulation": "ASI01 -- fabricated history to move a belief",
    "exfiltration": "private information disclosure",
    "tool_abuse": "ASI07 -- induced call to a non-existent tool",
}

INTENTS = ("vote_hijack", "belief_poison", "role_exfil", "tool_hallucination")


@dataclass(frozen=True)
class Payload:
    id: str
    category: str
    intent: str
    text: str
    owasp: str = ""
    # "dev" payloads were visible while the L2 patterns were written; "holdout"
    # ones were not. Reporting both is what keeps the filter from being tuned
    # to its own test set.
    split: str = "dev"

    def render(self, target: int, me: int) -> str:
        return self.text.format(target=target, me=me)


@dataclass(frozen=True)
class BenignSample:
    id: str
    kind: str
    text: str

    def render(self, target: int, me: int) -> str:
        return self.text.format(target=target, me=me)


def load_payloads(
    intent: str | None = None,
    category: str | None = None,
    split: str | None = None,
) -> list[Payload]:
    raw = json.loads((DATA_DIR / "samples" / "injection.json").read_text("utf-8"))
    payloads = [
        Payload(
            id=p["id"],
            category=p["category"],
            intent=p["intent"],
            text=p["text"],
            owasp=p.get("owasp", ""),
            split=p.get("split", "dev"),
        )
        for p in raw["payloads"]
    ]
    if intent:
        payloads = [p for p in payloads if p.intent == intent]
    if category:
        payloads = [p for p in payloads if p.category == category]
    if split:
        payloads = [p for p in payloads if p.split == split]
    return payloads


def load_benign() -> list[BenignSample]:
    raw = json.loads((DATA_DIR / "benign" / "persuasion.json").read_text("utf-8"))
    return [BenignSample(id=s["id"], kind=s["kind"], text=s["text"]) for s in raw["samples"]]


class Injector:
    """Decorates attacker speeches and remembers what it planted.

    It also owns detection: the runner and the agent loop ask "is a payload in
    this text" so the log can record *which* payload an agent actually read and
    through which channel. Detection is done on raw text, before the guard runs,
    otherwise a filtered payload would look like one that was never delivered.
    """

    def __init__(
        self,
        payloads: list[Payload] | None = None,
        benign: list[BenignSample] | None = None,
        seed: int = 0,
        enabled: bool = True,
        benign_mode: bool = False,
    ):
        self.payloads = payloads if payloads is not None else load_payloads()
        self.benign = benign if benign is not None else load_benign()
        self.enabled = enabled
        self.benign_mode = benign_mode
        self.rng = random.Random(seed)
        self.planted: list[dict] = []
        self._signatures: list[tuple[str, str]] = []  # (payload_id, signature)

    def decorate(
        self,
        speech: str,
        *,
        attacker: int,
        target: int,
        round_no: int,
        payload_id: str | None = None,
    ) -> tuple[str, dict | None]:
        """Append one payload (or one benign persuasion line) to a speech."""
        if not self.enabled:
            return speech, None

        if self.benign_mode:
            sample = self._pick_benign(payload_id)
            rendered = sample.render(target=target, me=attacker)
            record = {
                "attacker": attacker,
                "target": target,
                "payload_id": sample.id,
                "channel": "speech",
                "round": round_no,
                "benign": True,
                "text": rendered,
            }
        else:
            payload = self._pick_payload(payload_id)
            rendered = payload.render(target=target, me=attacker)
            record = {
                "attacker": attacker,
                "target": target,
                "payload_id": payload.id,
                "category": payload.category,
                "intent": payload.intent,
                "channel": "speech",
                "round": round_no,
                "benign": False,
                "text": rendered,
            }

        self.planted.append(record)
        self._signatures.append((record["payload_id"], _signature(rendered)))
        return f"{speech.rstrip()} {rendered}", record

    def _pick_payload(self, payload_id: str | None) -> Payload:
        if payload_id:
            for p in self.payloads:
                if p.id == payload_id:
                    return p
            raise KeyError(f"no payload with id {payload_id!r}")
        return self.rng.choice(self.payloads)

    def _pick_benign(self, sample_id: str | None) -> BenignSample:
        if sample_id:
            for s in self.benign:
                if s.id == sample_id:
                    return s
            raise KeyError(f"no benign sample with id {sample_id!r}")
        return self.rng.choice(self.benign)

    def detect(self, text: str) -> list[str]:
        """Which planted payloads appear in this text."""
        if not text:
            return []
        return [pid for pid, sig in self._signatures if sig and sig in text]

    def detector(self):
        return self.detect

    def targets_of(self, payload_id: str) -> list[int]:
        return [p["target"] for p in self.planted if p["payload_id"] == payload_id]


def _signature(rendered: str) -> str:
    """A fragment distinctive enough to identify a payload inside a speech.

    The longest word-run from the middle of the payload: the head can be
    swallowed by a redaction and the tail can be truncated by the context
    budget, but a mid-payload fragment survives both in the raw text.
    """
    words = rendered.split()
    if len(words) < 6:
        return rendered.strip()
    start = len(words) // 3
    return " ".join(words[start : start + 5])
