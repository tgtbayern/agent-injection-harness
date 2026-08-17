"""Architectural boundary tests.

The project's one structural rule: no model call inside `engine`, no game rule
inside `harness`. It is easy to state, easy to violate by accident, and its
violation quietly destroys the interpretability of every result -- so it is
asserted rather than left to review.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
HARNESS = ROOT / "harness"

MODEL_CALL_MARKERS = ("openai", "chat.completions", "llm", "gpt", "claude", "prompt")


def _python_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * (node.level or 0) + (node.module or ""))
    return names


def test_engine_never_imports_the_model_layer():
    for path in _python_files(ENGINE):
        for name in _imports(path):
            assert "providers" not in name, f"{path.name} imports the model layer"
            assert "harness" not in name, f"{path.name} imports the harness"


def test_engine_contains_no_model_call():
    """The referee is if-else, by decision. Anything formalisable belongs in
    code; only ambiguous judgement goes to a model.

    Checked over identifiers in the syntax tree rather than raw text, so that
    prose explaining the rule ("contains no LLM call") does not trip it.
    """
    for path in _python_files(ENGINE):
        tree = ast.parse(path.read_text("utf-8"))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name.lower())
        identifiers |= {name.lower() for name in _imports(path)}
        for marker in MODEL_CALL_MARKERS:
            offenders = [i for i in identifiers if marker in i]
            assert not offenders, f"{path.name} refers to {offenders}"


def test_harness_does_not_reach_past_the_engine_adapter():
    """Game rules reach the harness only through `engine`'s two-method surface;
    importing the rules library directly would put rule logic in the harness."""
    for path in _python_files(HARNESS):
        for name in _imports(path):
            assert not name.startswith("werewolf_engine"), (
                f"{path.name} imports the rules library directly"
            )


def test_engine_public_surface_is_the_documented_two_methods():
    from werewolf_harness.engine import GameState, get_visible_state

    assert hasattr(GameState, "apply_action")
    assert callable(get_visible_state)


def test_every_package_is_importable():
    import importlib

    for module in [
        "werewolf_harness.engine",
        "werewolf_harness.harness",
        "werewolf_harness.harness.agent",
        "werewolf_harness.harness.guard",
        "werewolf_harness.harness.providers",
        "werewolf_harness.attacks",
        "werewolf_harness.evalkit",
        "werewolf_harness.evalkit.metrics",
        "werewolf_harness.evalkit.judges",
        "werewolf_harness.evalkit.stats",
        "werewolf_harness.cli",
    ]:
        importlib.import_module(module)
