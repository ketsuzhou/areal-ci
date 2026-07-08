"""Tests for the entropy-gated checkpoint decision helper."""

from __future__ import annotations

from customized_areal.tree_search.env_checkpoint import (
    should_create_entropy_checkpoint,
)


def test_entropy_checkpoint_threshold_true():
    assert should_create_entropy_checkpoint(1.2, 1.0) is True


def test_entropy_checkpoint_threshold_false():
    assert should_create_entropy_checkpoint(0.2, 1.0) is False


def test_entropy_checkpoint_skips_missing_logprobs():
    assert should_create_entropy_checkpoint(None, 1.0) is False


def test_entropy_checkpoint_skips_missing_threshold():
    assert should_create_entropy_checkpoint(1.2, None) is False


def test_entropy_checkpoint_boundary_equal():
    assert should_create_entropy_checkpoint(1.0, 1.0) is True
