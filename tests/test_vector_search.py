"""Tests for the Atlas Vector Search query helper. No live Atlas needed."""

import pytest

import database


class _FakeCollection:
    def __init__(self, docs, captured_pipelines):
        self._docs = docs
        self._captured = captured_pipelines

    def aggregate(self, pipeline):
        self._captured.append(pipeline)
        return iter(self._docs)


def test_vector_search_maps_atlas_score_to_raw_cosine(monkeypatch):
    captured = []
    docs = [{"user_id": "alice", "score": 0.95}, {"user_id": "bob", "score": 0.60}]
    monkeypatch.setattr(database, "collection", _FakeCollection(docs, captured))

    ranked = database.vector_search_users([0.1] * 192, limit=5, num_candidates=100)

    # Atlas cosine score -> raw cosine: raw = 2*score - 1.
    assert [u for u, _ in ranked] == ["alice", "bob"]
    assert ranked[0][1] == pytest.approx(0.9)
    assert ranked[1][1] == pytest.approx(0.2)


def test_vector_search_builds_expected_pipeline(monkeypatch):
    captured = []
    monkeypatch.setattr(database, "collection", _FakeCollection([], captured))

    database.vector_search_users([0.0] * 192, limit=3, num_candidates=50,
                                 index_name="my_index")

    stage = captured[0][0]["$vectorSearch"]
    assert stage["index"] == "my_index"
    assert stage["path"] == "embedding"
    assert stage["limit"] == 3
    assert stage["numCandidates"] == 50
    assert len(stage["queryVector"]) == 192


def test_vector_search_returns_empty_when_no_collection(monkeypatch):
    monkeypatch.setattr(database, "collection", None)
    assert database.vector_search_users([0.1] * 192) == []


def test_vector_search_handles_query_error(monkeypatch):
    class Boom:
        def aggregate(self, pipeline):
            raise RuntimeError("index missing")

    monkeypatch.setattr(database, "collection", Boom())
    assert database.vector_search_users([0.1] * 192) == []
