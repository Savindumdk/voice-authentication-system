"""Unit tests for the vectorized matcher. No GPU/models/DB required."""

import torch

from matching import best_match, rank_matches, _flatten


def test_best_match_identifies_closest_speaker():
    enrolled = {
        "alice": torch.tensor([1.0, 0.0, 0.0]),
        "bob": torch.tensor([0.0, 1.0, 0.0]),
        "carol": torch.tensor([0.0, 0.0, 1.0]),
    }
    probe = torch.tensor([0.9, 0.1, 0.0])
    user, score = best_match(probe, enrolled)
    assert user == "alice"
    assert score > 0.95


def test_rank_matches_is_sorted_descending():
    enrolled = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([0.7, 0.7]),
        "c": torch.tensor([0.0, 1.0]),
    }
    probe = torch.tensor([1.0, 0.0])
    ranked = rank_matches(probe, enrolled)
    scores = [s for _, s in ranked]
    assert ranked[0][0] == "a"
    assert scores == sorted(scores, reverse=True)


def test_matches_legacy_per_pair_cosine():
    """Batched result must equal the original per-user cosine loop."""
    torch.manual_seed(0)
    probe = torch.randn(1, 192)
    enrolled = {f"u{i}": torch.randn(1, 192) for i in range(20)}

    ranked = dict(rank_matches(probe, enrolled))

    p = _flatten(probe)
    p = p / p.norm()
    for uid, emb in enrolled.items():
        e = _flatten(emb)
        e = e / e.norm()
        legacy = torch.dot(p, e).item()
        assert abs(legacy - ranked[uid]) < 1e-5


def test_empty_enrollment_returns_none():
    user, score = best_match(torch.randn(192), {})
    assert user is None
    assert score == -1.0


def test_mismatched_dimensions_are_skipped():
    enrolled = {
        "good": torch.randn(192),
        "bad": torch.randn(128),  # wrong dim, must be ignored not crash
    }
    user, _ = best_match(torch.randn(192), enrolled)
    assert user == "good"
