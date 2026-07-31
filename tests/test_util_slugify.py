"""Tests for util.slugify, focused on its use as a filesystem path component."""

import pytest

from trackllm_website.util import slugify


@pytest.mark.parametrize("name", ["..", ".", "...", "...."])
def test_all_dot_names_never_slugify_to_a_traversal_component(name):
    slug = slugify(name)
    assert slug not in (".", ".."), f"{name!r} slugified to a path component {slug!r}"
    assert set(slug) <= set("0123456789abcdef")


def test_truncation_cannot_reintroduce_a_traversal_component():
    # "..foo"[:2] == ".." before the fix
    assert slugify("..foo", max_length=2) not in (".", "..")


@pytest.mark.parametrize(
    "name",
    [
        "deepseek/deepseek-chat-v3-0324#fireworks",
        "openai/gpt-4o",
        "meta-llama/llama-3.1-8b",
        "qwen/qwen3-235b-a22b-2507",
        "z-ai/glm-4.5",
        "anthropic/claude-sonnet-4.5",
        "Fireworks",
        "deepinfra/fp8",
        "x-ai",
        "model.with.dots",
        ".hidden",
        "..leading-dots",
        "trailing..",
        "a.b..c",
    ],
)
def test_ordinary_names_keep_their_slug(name):
    # Regression guard: the traversal fix must only ever touch all-dot names,
    # since a changed slug means an orphaned data directory / dead URL.
    expected = "".join(
        c if (c.isalnum() or c in "._-+=@~,") else ("-" if c == " " else f"{ord(c):02x}")
        for c in name
    )
    assert slugify(name) == expected


def test_hash_suffix_still_appended():
    slug = slugify("openai/gpt-4o", hash_length=4)
    assert slug.startswith("openai2fgpt-4o_")
    assert len(slug.rsplit("_", 1)[1]) == 4
