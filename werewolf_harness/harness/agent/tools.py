"""The tool layer: what an agent can do, and the three gates every call passes.

Gate 1  schema      -- types, required arguments, ranges          (schema.py)
Gate 2  whitelist   -- the tool must be registered; invented names die here
Gate 3  semantics   -- vote targets must be alive, rounds must have happened

Two of the eight tools return text written by *other players*
(`query_history`) -- that is attack path B, the tool-return channel, and the
reason tool output is passed through the guard stack exactly like a speech is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ...engine import GameState, Role
from ..schema import SchemaError, validate_call


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    fn: Callable
    terminal: bool = False
    untrusted_output: bool = False  # output contains other players' text

    def openai_schema(self) -> dict:
        """Function-calling schema for gateways that support it natively."""
        type_map = {"int": "integer", "float": "number", "str": "string",
                    "list[str]": "array"}
        props: dict = {}
        required: list[str] = []
        for pname, spec in self.params.items():
            prop: dict = {
                "type": type_map[spec["type"]],
                "description": spec.get("description", ""),
            }
            if spec["type"] == "list[str]":
                prop["items"] = {"type": "string"}
            if "range" in spec:
                prop["minimum"], prop["maximum"] = spec["range"]
            props[pname] = prop
            if spec.get("required", True):
                required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    observation: str
    data: object = None
    untrusted: bool = False
    sources: list[str] = field(default_factory=list)  # speech ids, for evidence


@dataclass
class ToolContext:
    state: GameState
    player_id: int
    belief: object  # BeliefState
    view: dict


class Registry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """Plain-text tool description, used by the JSON-prompt fallback mode."""
        lines = []
        for t in self._tools.values():
            args = ", ".join(
                f"{p}: {s['type']}{'' if s.get('required', True) else ' (optional)'}"
                for p, s in t.params.items()
            )
            lines.append(f"- {t.name}({args}) -- {t.description}")
        return "\n".join(lines)

    def validate(self, name, args) -> tuple[str, dict]:
        return validate_call(name, args, self)

    def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:  # pragma: no cover -- validate() runs first
            raise SchemaError(f"unknown tool {name!r}")
        return tool.fn(ctx, **args)


def speech_id(round_no: int, player_id: int) -> str:
    return f"r{round_no}p{player_id}"


# ------------------------------------------------------------ read-only

def _query_history(ctx: ToolContext, player_id: int, round: int) -> ToolResult:
    st = ctx.state
    if player_id not in range(1, 9):
        raise SchemaError("query_history.player_id must be a player between 1 and 8")
    if round < 1 or round > st.round:
        raise SchemaError(
            f"query_history.round must be between 1 and {st.round} (the current round)"
        )
    speeches = st.speeches_of(player_id, round)
    if not speeches:
        return ToolResult(
            observation=f"player {player_id} did not speak in round {round}",
            data=[],
            sources=[],
        )
    body = "\n".join(s.content for s in speeches)
    return ToolResult(
        observation=body,
        data=[s.to_dict() for s in speeches],
        untrusted=True,  # attack path B lives here
        sources=[speech_id(round, player_id)],
    )


def _query_votes(ctx: ToolContext, round: int) -> ToolResult:
    st = ctx.state
    if round < 1 or round > st.round:
        raise SchemaError(f"query_votes.round must be between 1 and {st.round}")
    ballots = st.votes_of_round(round)
    if not ballots:
        return ToolResult(f"no votes recorded in round {round}", data={})
    text = ", ".join(
        f"{voter}->{target if target is not None else 'abstain'}"
        for voter, target in sorted(ballots.items())
    )
    exiled = st.exiles.get(round)
    tail = f"; exiled: {exiled}" if exiled is not None else "; nobody was exiled"
    return ToolResult(f"round {round} votes: {text}{tail}", data=ballots)


def _query_deaths(ctx: ToolContext) -> ToolResult:
    deaths = ctx.state.deaths
    if not deaths:
        return ToolResult("nobody has died yet", data=[])
    text = "; ".join(
        f"round {d.round}: player {d.player_id} died ({_cause_label(d.cause)})"
        for d in deaths
    )
    return ToolResult(text, data=[d.to_dict() for d in deaths])


def _cause_label(cause: str) -> str:
    return {"vote": "exiled by vote", "werewolf": "killed at night",
            "witch": "killed at night"}.get(cause, cause)


def _read_belief(ctx: ToolContext, player_id: int) -> ToolResult:
    entry = ctx.belief.get(player_id)
    if entry is None:
        return ToolResult(f"no belief recorded for player {player_id}", data=None)
    return ToolResult(
        f"player {player_id}: suspicion={entry.suspicion:.2f}, "
        f"reason={entry.reason!r}, evidence={entry.evidence_refs}",
        data=entry.to_dict(),
    )


def _check_ability(ctx: ToolContext) -> ToolResult:
    role = ctx.state.role_of(ctx.player_id)
    private = ctx.view.get("private", {})
    if role is Role.SEER:
        checks = private.get("checks", [])
        text = "you are the seer. checks so far: " + (
            ", ".join(
                f"r{c['round']} player {c['target']} = "
                f"{'WOLF' if c['is_wolf'] else 'not a wolf'}"
                for c in checks
            )
            or "none yet"
        )
    elif role is Role.WITCH:
        text = (
            f"you are the witch. antidote available: "
            f"{private.get('antidote_available')}, "
            f"poison available: {private.get('poison_available')}. "
            "Potions are resolved automatically at night."
        )
    elif role is Role.WOLF:
        text = (
            "you are a werewolf. fellow wolves: "
            f"{private.get('fellow_wolves', [])}. "
            "The pack's night kill is resolved automatically."
        )
    else:
        text = "you are a villager. you have no night ability."
    return ToolResult(text, data={"role": role.value, **private})


# --------------------------------------------------------- write-to-self

def _update_belief(
    ctx: ToolContext,
    player_id: int,
    suspicion: float,
    reason: str,
    evidence_refs: list[str] | None = None,
) -> ToolResult:
    if player_id == ctx.player_id:
        raise SchemaError("update_belief cannot target yourself")
    if player_id not in range(1, 9):
        raise SchemaError("update_belief.player_id must be a player between 1 and 8")
    entry = ctx.belief.update(
        player_id=player_id,
        suspicion=suspicion,
        reason=reason,
        round_no=ctx.state.round,
        evidence_refs=evidence_refs or [],
    )
    return ToolResult(
        f"belief updated: player {player_id} suspicion={entry.suspicion:.2f}",
        data=entry.to_dict(),
    )


# ------------------------------------------------------------- terminal

def _speak(ctx: ToolContext, content: str) -> ToolResult:
    return ToolResult(content, data={"content": content})


def _vote(ctx: ToolContext, target_id: int) -> ToolResult:
    st = ctx.state
    if target_id not in st.alive:
        raise SchemaError(
            f"player {target_id} is not alive; living players are {st.alive_sorted()}"
        )
    if target_id == ctx.player_id:
        raise SchemaError("you cannot vote for yourself")
    return ToolResult(f"voted for player {target_id}", data={"target_id": target_id})


def build_registry() -> Registry:
    return Registry(
        [
            Tool(
                "query_history",
                "Read what a player actually said in a given round. "
                "Rounds before the current one are NOT in your context: "
                "this is the only way to recall them.",
                {
                    "player_id": {"type": "int", "description": "player 1-8"},
                    "round": {"type": "int", "description": "round number, 1-based"},
                },
                _query_history,
                untrusted_output=True,
            ),
            Tool(
                "query_votes",
                "Read the full ballot of a past round (who voted for whom).",
                {"round": {"type": "int", "description": "round number"}},
                _query_votes,
            ),
            Tool(
                "query_deaths",
                "List every death so far with its round and cause.",
                {},
                _query_deaths,
            ),
            Tool(
                "read_belief",
                "Read your own recorded judgement about a player.",
                {"player_id": {"type": "int", "description": "player 1-8"}},
                _read_belief,
            ),
            Tool(
                "update_belief",
                "Record or revise your judgement about a player. "
                "Cite the speech ids you relied on, e.g. \"r2p5\".",
                {
                    "player_id": {"type": "int", "description": "player 1-8"},
                    "suspicion": {
                        "type": "float",
                        "range": (0.0, 1.0),
                        "description": "0 = certainly village, 1 = certainly wolf",
                    },
                    "reason": {"type": "str", "max_len": 400,
                               "description": "why, in one sentence"},
                    "evidence_refs": {
                        "type": "list[str]",
                        "required": False,
                        "default": [],
                        "description": "speech ids such as r2p5 that support this",
                    },
                },
                _update_belief,
            ),
            Tool(
                "check_ability",
                "Check your own role, night ability status and private results.",
                {},
                _check_ability,
            ),
            Tool(
                "speak",
                "Say something to the table. ENDS YOUR TURN.",
                {"content": {"type": "str", "max_len": 1200, "non_empty": True,
                             "description": "what you say out loud"}},
                _speak,
                terminal=True,
            ),
            Tool(
                "vote",
                "Cast your vote to exile a player. ENDS YOUR TURN.",
                {"target_id": {"type": "int", "description": "a living player, not yourself"}},
                _vote,
                terminal=True,
            ),
        ]
    )
