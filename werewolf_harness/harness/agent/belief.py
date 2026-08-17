"""Structured long-term memory.

Three layers, kept apart on purpose:

  working memory  -- the current ReAct chain, discarded at the end of the turn
  short-term      -- this round's observations, live in the context window
  long-term       -- this module: one entry per player, carried across rounds

Only conclusions and their evidence are promoted to long-term memory. Keeping
raw transcripts would grow the context linearly with the game; a compressed
"why I suspect 5" survives a whole game in a few hundred tokens.
"""

from __future__ import annotations

from ..schema import BeliefEntry


class BeliefState:
    """player_id -> BeliefEntry, with snapshotting for the log."""

    def __init__(self, owner: int, players: list[int]):
        self.owner = owner
        self.entries: dict[int, BeliefEntry] = {
            p: BeliefEntry(player_id=p, suspicion=0.5, reason="no information yet")
            for p in players
            if p != owner
        }

    def get(self, player_id: int) -> BeliefEntry | None:
        return self.entries.get(player_id)

    def update(
        self,
        player_id: int,
        suspicion: float,
        reason: str,
        round_no: int,
        evidence_refs: list[str] | None = None,
        guessed_role: str | None = None,
    ) -> BeliefEntry:
        entry = self.entries.get(player_id)
        if entry is None:
            entry = BeliefEntry(player_id=player_id, suspicion=suspicion)
            self.entries[player_id] = entry
        entry.suspicion = max(0.0, min(1.0, float(suspicion)))
        entry.reason = reason
        entry.updated_at_round = round_no
        if evidence_refs is not None:
            entry.evidence_refs = list(evidence_refs)
        if guessed_role is not None:
            entry.guessed_role = guessed_role
        return entry

    def most_suspected(self, among: list[int]) -> int | None:
        candidates = [(self.entries[p].suspicion, -p) for p in among if p in self.entries]
        if not candidates:
            return None
        return -max(candidates)[1]

    def snapshot(self) -> dict:
        return {str(p): e.to_dict() for p, e in sorted(self.entries.items())}

    def summarise(self, alive: list[int]) -> str:
        """The compact form injected into the context each turn."""
        rows = []
        for p in sorted(alive):
            e = self.entries.get(p)
            if e is None:
                continue
            role = f", guess={e.guessed_role}" if e.guessed_role else ""
            rows.append(
                f"  player {p}: suspicion={e.suspicion:.2f}{role} "
                f"(r{e.updated_at_round}: {e.reason[:90]})"
            )
        return "\n".join(rows) if rows else "  (no beliefs recorded yet)"

    @staticmethod
    def diff(before: dict, after: dict) -> list[dict]:
        """Which beliefs moved this turn -- axis 3 and the replay UI both need it."""
        changes = []
        for pid, after_entry in after.items():
            before_entry = before.get(pid)
            if before_entry is None:
                changes.append({"player_id": int(pid), "from": None, "to": after_entry["suspicion"]})
            elif abs(before_entry["suspicion"] - after_entry["suspicion"]) > 1e-9:
                changes.append(
                    {
                        "player_id": int(pid),
                        "from": before_entry["suspicion"],
                        "to": after_entry["suspicion"],
                        "reason": after_entry.get("reason", ""),
                    }
                )
        return changes
