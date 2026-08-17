"""The harness: everything wrapped around the model.

Boundary rule, enforced by review and by the tests: no game rule lives in this
package, and no model call lives in `engine`. Cross that line and the
experiment stops being interpretable.
"""

from .agent import AgentLoop, BeliefState, ContextBuilder, build_registry
from .guard import GuardStack
from .recovery import RecoveryPolicy
from .schema import BeliefEntry, ReActStep, SchemaError, new_game_log
from .trace import Tracer

__all__ = [
    "AgentLoop",
    "BeliefEntry",
    "BeliefState",
    "ContextBuilder",
    "GuardStack",
    "ReActStep",
    "RecoveryPolicy",
    "SchemaError",
    "Tracer",
    "build_registry",
    "new_game_log",
]
