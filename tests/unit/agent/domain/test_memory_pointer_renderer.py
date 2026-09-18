"""Tests for the size-capped long-term-memory pointer."""
from __future__ import annotations

from src.agent.domain.services.memory_pointer_renderer import MemoryPointerRenderer


def test_empty_vault_renders_empty_hint():
    block = MemoryPointerRenderer.render(0, [])
    assert block.startswith("## Long-term memory")
    assert "Nothing is stored yet" in block
    assert "remember" in block


def test_renders_count_topics_and_recall_instruction():
    block = MemoryPointerRenderer.render(3, ["car", "vacation", None])
    assert block.startswith("## Long-term memory")
    assert "Entries: 3." in block
    assert "Topics: car, vacation." in block
    assert "recall" in block
    # The pointer is a table of contents, never the stored content.
    assert "This is only a table of contents" in block


def test_topics_deduplicate_case_insensitively_and_skip_blanks():
    block = MemoryPointerRenderer.render(4, ["Garden", "garden", "  ", "vacation"])
    assert block.lower().count("garden") == 1
    assert "Topics: Garden, vacation." in block


def test_topics_are_capped_and_truncated():
    many = [f"topic{i}" for i in range(50)]
    block = MemoryPointerRenderer.render(50, many)
    assert "topic14" in block
    assert "topic15" not in block  # at most 15 topics

    long_topic = "lengthy" * 20
    block = MemoryPointerRenderer.render(1, [long_topic])
    assert "…" in block
    assert long_topic not in block


def test_entries_without_topics_render_no_topics_line():
    block = MemoryPointerRenderer.render(2, [None, ""])
    assert "Entries: 2." in block
    assert "Topics:" not in block
