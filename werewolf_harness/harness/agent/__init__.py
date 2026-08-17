"""The agent: a bounded ReAct loop with tools, structured memory and a
context budget. Hand-written rather than framework-driven -- see loop.py."""

from .belief import BeliefState
from .context import ContextBuilder
from .loop import AgentLoop, TurnResult
from .tools import Registry, Tool, ToolContext, ToolResult, build_registry, speech_id

__all__ = [
    "AgentLoop",
    "BeliefState",
    "ContextBuilder",
    "Registry",
    "Tool",
    "ToolContext",
    "ToolResult",
    "TurnResult",
    "build_registry",
    "speech_id",
]
