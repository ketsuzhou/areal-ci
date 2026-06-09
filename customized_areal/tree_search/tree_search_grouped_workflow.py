"""Compatibility alias for the tree-search grouped workflow module."""

import sys

from customized_areal.tree_search.core import customized_grouped_workflow as _impl

sys.modules[__name__] = _impl
