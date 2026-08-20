"""FT-E4-S1: retrieve.py / demo must not import graph.reranker."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RETRIEVE = REPO / "src" / "medmemgraph" / "graph" / "retrieve.py"
AGENT = REPO / "src" / "medmemgraph" / "demo" / "agent.py"


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _mentions_reranker_module(name: str) -> bool:
    return any(part == "reranker" for part in name.split("."))


def test_retrieve_py_has_no_reranker_module_import():
    imported = _imported_module_names(RETRIEVE)
    offenders = [n for n in imported if _mentions_reranker_module(n)]
    assert offenders == [], f"retrieve.py imports reranker module(s): {offenders}"
    src = RETRIEVE.read_text(encoding="utf-8")
    assert "from medmemgraph.graph.reranker" not in src
    assert "import medmemgraph.graph.reranker" not in src


def test_comment_containing_rerank_is_not_an_import():
    """The graph span is named rerank; a comment must not trip the check."""
    tree = ast.parse("x = 1  # the rerank span is traverse.rank_paths\n")
    imports = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    assert imports == []


def test_demo_agent_does_not_default_to_finetuned_ce():
    src = AGENT.read_text(encoding="utf-8")
    assert "ms-marco-minilm-l6-v2-ft-" not in src
    assert "DEFAULT_K = 8" in src
