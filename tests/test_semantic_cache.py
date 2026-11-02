"""Tests for src/core/semantic_cache.py: how the sources are saved in Redis and read back.
A tiny fake Redis client stands in for the real one, so no Redis is needed."""
import json
from types import SimpleNamespace

from src.core.semantic_cache import SemanticCache, _decode_sources


class FakeRedis:
    def __init__(self, docs=None):
        self.hashes = {}
        self.expiries = {}
        self.docs = docs or []

    def hset(self, key, mapping):
        self.hashes[key] = mapping

    def expire(self, key, seconds):
        self.expiries[key] = seconds

    def ft(self, name):
        return SimpleNamespace(search=lambda query, query_params=None: SimpleNamespace(docs=self.docs))


def make_cache(docs=None):
    cache = object.__new__(SemanticCache)     # skip __init__: it would connect to a real Redis
    cache.client = FakeRedis(docs)
    cache.threshold = 0.05
    cache.ttl_seconds = 3600
    return cache


# ---------- _decode_sources ------------------------------------------------------

def test_sources_round_trip_through_json_text():
    assert _decode_sources(json.dumps(["a.txt", "b.txt"])) == ["a.txt", "b.txt"]


def test_redis_bytes_are_decoded():
    assert _decode_sources(b'["a.txt"]') == ["a.txt"]


def test_a_missing_or_damaged_value_gives_an_empty_list():
    assert _decode_sources(None) == []                 # an entry saved before sources were kept
    assert _decode_sources("not json") == []
    assert _decode_sources('{"a": 1}') == []           # valid JSON, but not a list
    assert _decode_sources('["a.txt", 5, null]') == ["a.txt"]


# ---------- store and check ------------------------------------------------------

def test_store_saves_the_sources_next_to_the_answer():
    cache = make_cache()
    cache.store("q", "answer", [0.0, 1.0], "context", ["returns_policy.txt"])
    saved = next(iter(cache.client.hashes.values()))
    assert json.loads(saved["sources"]) == ["returns_policy.txt"]
    assert saved["response"] == "answer" and cache.client.expiries


def test_store_without_sources_saves_an_empty_list():
    cache = make_cache()
    cache.store("q", "answer", [0.0])
    assert json.loads(next(iter(cache.client.hashes.values()))["sources"]) == []


def test_check_returns_the_saved_sources():
    doc = SimpleNamespace(response=b"answer", full_context=b"ctx", sources=b'["returns_policy.txt"]', distance="0.01")
    assert make_cache([doc]).check([0.0]) == {
        "response": b"answer", "full_context": b"ctx", "sources": ["returns_policy.txt"]}


def test_check_on_an_entry_saved_before_sources_existed_still_works():
    doc = SimpleNamespace(response=b"answer", full_context=b"ctx", distance="0.01")   # no `sources` attribute
    assert make_cache([doc]).check([0.0])["sources"] == []


def test_check_ignores_a_neighbour_that_is_too_far_away():
    doc = SimpleNamespace(response=b"answer", full_context=b"ctx", sources=b"[]", distance="0.4")
    assert make_cache([doc]).check([0.0]) is None
