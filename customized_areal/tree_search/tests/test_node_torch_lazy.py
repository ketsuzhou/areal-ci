"""Node + tree_store must import cleanly without torch installed.

The agents/ DAG layer imports Node for the SuperNode.nodes field; if tree_store
hard-imports torch at module top, every torch-free test in agents/ breaks when
torch is absent. This test guards the lazy import by hiding torch from the
import system before importing tree_store.
"""

from __future__ import annotations

import importlib
import sys


def test_tree_store_imports_without_torch(monkeypatch):
    # Block torch from being importable, then force a fresh import of tree_store.
    # If tree_store still has a top-level `import torch`, this raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    # Remove any cached import so the next import re-executes the module body.
    for mod in list(sys.modules):
        if mod.startswith("customized_areal.tree_search.core.tree_store"):
            del sys.modules[mod]
    tree_store = importlib.import_module(
        "customized_areal.tree_search.core.tree_store"
    )
    assert hasattr(tree_store, "Node")
    assert hasattr(tree_store, "MCTSTreeStore")
