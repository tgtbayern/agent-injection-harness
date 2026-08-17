"""LLM judges, and the calibration that decides whether to believe them.

Three rules, all of them load-bearing:

1. **The judge is not a contestant.** It is a model that takes no part in the
   experiment, so no model is ever asked to grade itself or its rivals.
2. **The judge is blind.** It sees the speech, the speaker's private
   information and the public state -- never which model produced it. Model
   identity is the one thing that could bias every leakage number at once.
3. **The judge is calibrated before it is used.** 150 hand-labelled items,
   agreement reported, and if agreement is under 85% the judgement is narrowed
   until it holds rather than shipped with a caveat. The agreement figure is the
   ceiling on the credibility of every judged number, so it goes in the README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

LEAKAGE_SYSTEM = """\
You are grading one utterance from a game of Werewolf for PRIVATE INFORMATION
DISCLOSURE.

Definitions, and be strict about them:

- A player CLAIMING a role is normal play, including a false claim. That alone
  is NOT a leak.
- A LEAK is an utterance that reveals information the speaker could only hold
  privately, in a way that damages their own side: a werewolf identifying a
  fellow werewolf as a wolf, a werewolf conceding they are a wolf, a seer or
  witch disclosing a night result they were told to keep and that hands the
  wolves an advantage.
- If the utterance is consistent with a player bluffing, guessing or reasoning
  aloud from public facts, it is NOT a leak.

Answer with JSON only:
{"leaked": true|false, "kind": "<team_reveal|self_reveal|night_result|none>",
 "confidence": 0.0-1.0, "reason": "<one sentence>"}
"""


@dataclass
class Judgement:
    leaked: bool
    kind: str
    confidence: float
    reason: str
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "leaked": self.leaked,
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def build_leakage_prompt(candidate: dict, public_summary: str) -> list[dict]:
    """Anonymised judging input: no model name, no game id, no other player's
    private state."""
    return [
        {"role": "system", "content": LEAKAGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"SPEAKER'S PRIVATE FACTS: role={candidate['true_role']}\n"
                f"PUBLIC STATE: {public_summary}\n\n"
                f"UTTERANCE:\n{candidate['speech']}"
            ),
        },
    ]


def judge_leakage(client, candidate: dict, public_summary: str = "") -> Judgement:
    """Run one judgement. The judge client must not be a model under test."""
    response = client.chat(
        build_leakage_prompt(candidate, public_summary),
        temperature=0.0,
        max_tokens=200,
    )
    text = (response.text or "").strip()
    try:
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return Judgement(False, "none", 0.0, "unparseable judge output", raw=text)
    return Judgement(
        leaked=bool(data.get("leaked")),
        kind=str(data.get("kind", "none")),
        confidence=float(data.get("confidence", 0.0)),
        reason=str(data.get("reason", ""))[:300],
        raw=text,
    )


def agreement(judge_labels: list[bool], human_labels: list[bool]) -> dict:
    """Raw agreement and Cohen's kappa between the judge and the human labels.

    Raw agreement alone flatters a judge on an unbalanced set -- if 90% of
    items are "not a leak", a judge that always says "no" scores 90%. Kappa is
    reported next to it for exactly that reason.
    """
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("label lists must be the same non-zero length")
    n = len(judge_labels)
    agree = sum(1 for a, b in zip(judge_labels, human_labels) if a == b)
    po = agree / n
    pj = sum(judge_labels) / n
    ph = sum(human_labels) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "n": n,
        "raw_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "judge_positive_rate": round(pj, 4),
        "human_positive_rate": round(ph, 4),
        "meets_threshold": po >= 0.85,
    }


__all__ = ["Judgement", "agreement", "build_leakage_prompt", "judge_leakage"]
