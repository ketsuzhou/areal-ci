"""Backward-compatible imports for teacher provider symbols."""

from customized_areal.tree_search.distilling.teacher_provider import (
    ExternalTeacherProvider,
    TeacherProvider,
)

__all__ = ["ExternalTeacherProvider", "TeacherProvider"]
