"""The tool layer: what an agent can do, and the three gates every call passes.

Gate 1  schema      -- types, required arguments, ranges          (schema.py)
Gate 2  whitelist   -- the tool must be registered; invented names die here
Gate 3  semantics   -- vote targets must be alive, rounds must have happened

One tool returns text written by *other players* (`query_history`) -- that is
attack path B, the tool-return channel, and the reason tool output is passed
through the guard stack exactly like a speech is.

Terminal actions are scoped by role and by phase: `speak`/`vote` by day, and
`night_check` / `night_kill` / `night_save` / `night_poison` / `night_skip` at
night, each offered only to the role that owns it.
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
    # Which roles may call this at all. Empty means everyone. A tool outside a
    # role's set is not merely refused -- it is never shown, so a villager is
    # never told that `night_kill` exists.
    roles: tuple[str, ...] = ()

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

    def visible(self, role: str | None = None, task: str | None = None) -> list[Tool]:
        """The tools this role may call on this kind of turn.

        Scoping is by omission rather than refusal: a villager is never shown
        that `night_kill` exists, and a day turn is never shown a night action.
        Hiding is the cheaper half of the whitelist gate -- what is never
        offered is rarely invented.
        """
        out = []
        for tool in self._tools.values():
            if tool.roles and role not in tool.roles:
                continue
            is_night_tool = tool.name.startswith("night_")
            if task == "night" and tool.name in ("speak", "vote"):
                continue
            if task is not None and task != "night" and is_night_tool:
                continue
            out.append(tool)
        return out

    def openai_schemas(self, role: str | None = None,
                       task: str | None = None) -> list[dict]:
        return [t.openai_schema() for t in self.visible(role, task)]

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

    def validate(self, name, args, role: str | None = None,
                 task: str | None = None) -> tuple[str, dict]:
        """Gate 2 is role-aware: a tool that exists but is not yours is as
        rejected as one that does not exist."""
        tool = self.get(name)
        if tool is not None and (role is not None or task is not None):
            if tool not in self.visible(role, task):
                allowed = ", ".join(sorted(t.name for t in self.visible(role, task)))
                raise SchemaError(
                    f"{name} is not available to you on this turn; you may call: {allowed}"
                )
        return validate_call(name, args, self)

    def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        if tool is None:  # pragma: no cover -- validate() runs first
            raise SchemaError(f"unknown tool {name!r}")
        return tool.fn(ctx, **args)


# The longest a speech may be. Everything that has to carry a speech intact --
# the context budget for tool returns, above all -- derives from this rather
# than picking its own number and drifting.
SPEECH_MAX_CHARS = 1200


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
            "You spend them yourself, on your night turn."
        )
    elif role is Role.WOLF:
        text = (
            "you are a werewolf. fellow wolves: "
            f"{private.get('fellow_wolves', [])}. "
            "Each of you names a kill at night; the majority decides."
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


# ------------------------------------------------------- night actions

def _night_check(ctx: ToolContext, target_id: int) -> ToolResult:
    record = ctx.state.apply_night_action(ctx.player_id, "night_check", target_id)
    return ToolResult(
        f"you checked player {target_id}: {record['outcome']}",
        data=record,
    )


def _night_kill(ctx: ToolContext, target_id: int) -> ToolResult:
    record = ctx.state.apply_night_action(ctx.player_id, "night_kill", target_id)
    return ToolResult(record["outcome"], data=record)


def _night_save(ctx: ToolContext, target_id: int) -> ToolResult:
    record = ctx.state.apply_night_action(ctx.player_id, "night_save", target_id)
    return ToolResult(f"you saved player {target_id}", data=record)


def _night_poison(ctx: ToolContext, target_id: int) -> ToolResult:
    record = ctx.state.apply_night_action(ctx.player_id, "night_poison", target_id)
    return ToolResult(f"you poisoned player {target_id}", data=record)


def _night_skip(ctx: ToolContext) -> ToolResult:
    record = ctx.state.apply_night_action(ctx.player_id, "night_skip", None)
    return ToolResult("you did nothing tonight", data=record)


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
                {"content": {"type": "str", "max_len": SPEECH_MAX_CHARS, "non_empty": True,
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
            Tool(
                "night_check",
                "SEER ONLY, at night: learn whether a player is a werewolf. "
                "ENDS YOUR TURN.",
                {"target_id": {"type": "int", "description": "a living player you have not checked"}},
                _night_check,
                terminal=True,
                roles=("seer",),
            ),
            Tool(
                "night_kill",
                "WEREWOLF ONLY, at night: name the player the pack should kill. "
                "Your packmates name one too; the majority decides. ENDS YOUR TURN.",
                {"target_id": {"type": "int", "description": "a living non-wolf"}},
                _night_kill,
                terminal=True,
                roles=("werewolf",),
            ),
            Tool(
                "night_save",
                "WITCH ONLY, at night: spend the antidote on tonight's victim. "
                "ENDS YOUR TURN.",
                {"target_id": {"type": "int", "description": "tonight's victim"}},
                _night_save,
                terminal=True,
                roles=("witch",),
            ),
            Tool(
                "night_poison",
                "WITCH ONLY, at night: spend the poison on a living player who is "
                "not already tonight's victim. ENDS YOUR TURN.",
                {"target_id": {"type": "int", "description": "a living player"}},
                _night_poison,
                terminal=True,
                roles=("witch",),
            ),
            Tool(
                "night_skip",
                "At night: do nothing this night. ENDS YOUR TURN.",
                {},
                _night_skip,
                terminal=True,
                roles=("seer", "werewolf", "witch"),
            ),
        ]
    )
