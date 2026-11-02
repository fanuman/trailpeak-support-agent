"""Tests for src/core/config.py: defaults, environment overrides, bad values, and that the
limits the API enforces are the ones it tells the web page about."""
import pytest

from src.core import config


# ---------- _setting: default, override, bad value -----------------------------------

def test_a_setting_not_in_the_environment_gets_its_default(monkeypatch):
    monkeypatch.delenv("SOME_LIMIT", raising=False)
    assert config._setting("SOME_LIMIT", 7) == 7


def test_the_environment_overrides_the_default_and_is_converted(monkeypatch):
    monkeypatch.setenv("SOME_LIMIT", "10")
    assert config._setting("SOME_LIMIT", 7) == 10
    monkeypatch.setenv("SOME_LIMIT", " 0.25 ")
    assert config._setting("SOME_LIMIT", 0.5, float) == 0.25
    monkeypatch.setenv("SOME_LIMIT", "30/minute")
    assert config._setting("SOME_LIMIT", "10/minute", str) == "30/minute"


def test_an_empty_value_means_use_the_default(monkeypatch):
    monkeypatch.setenv("SOME_LIMIT", "  ")
    assert config._setting("SOME_LIMIT", 7) == 7


def test_a_value_of_the_wrong_kind_stops_the_app_with_a_clear_message(monkeypatch):
    monkeypatch.setenv("SOME_LIMIT", "abc")
    with pytest.raises(ValueError, match="SOME_LIMIT.*not a valid int"):
        config._setting("SOME_LIMIT", 7)


def test_a_value_below_the_minimum_is_refused(monkeypatch):
    monkeypatch.setenv("SOME_LIMIT", "0")
    with pytest.raises(ValueError, match="SOME_LIMIT.*at least 1"):
        config._setting("SOME_LIMIT", 7, minimum=1)
    assert config._setting("SOME_LIMIT", 7, minimum=0) == 0        # zero is allowed where the minimum is 0


# ---------- the real settings ----------------------------------------------------------

def test_every_setting_is_usable():
    assert config.RETRIEVAL_K >= 1 and config.MAX_AGENT_ITERATIONS >= 1
    assert config.MAX_HISTORY_MESSAGES >= 0 and config.MAX_MESSAGE_CHARS >= 1
    assert config.REWRITE_MAX_TOKENS >= 1 and config.REWRITE_MAX_CHARS >= 1 and config.REWRITE_CLIP_CHARS >= 1
    assert config.CACHE_DISTANCE_THRESHOLD >= 0 and config.CACHE_TTL_SECONDS >= 1
    assert "/" in config.RATE_LIMIT                                  # e.g. "10/minute"
    assert config.MAX_IMAGE_BYTES >= 1 and config.IMAGE_MAX_SIDE >= 64
    assert config.IMAGE_DESCRIBE_MAX_TOKENS >= 1 and config.IMAGE_NOTE_MAX_CHARS >= 1


def test_the_image_types_and_detail_are_fixed_choices():
    assert config.ALLOWED_IMAGE_TYPES == ("image/jpeg", "image/png", "image/webp")
    assert config.IMAGE_DETAIL in ("low", "high")


def test_the_models_are_not_overridable(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "something-else")
    monkeypatch.setenv("DEFAULT_MODEL", "something-else")
    import importlib
    reloaded = importlib.reload(config)
    try:
        assert reloaded.EMBEDDING_MODEL == "text-embedding-3-small" and reloaded.DEFAULT_MODEL == "gpt-4o-mini"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# ---------- one place: the API and the page agree --------------------------------------

def test_the_api_enforces_the_configured_limits():
    from src.api.models import RagRequest
    assert RagRequest.model_fields["history"].metadata[0].max_length == config.MAX_HISTORY_MESSAGES
    assert RagRequest.model_fields["message"].metadata[1].max_length == config.MAX_MESSAGE_CHARS


def test_config_endpoint_reports_the_same_limits_the_api_enforces():
    from src.api.main import app, client_config
    assert "/config" in {route.path for route in app.routes}
    reply = client_config()
    assert reply.max_history_messages == config.MAX_HISTORY_MESSAGES
    assert reply.max_message_chars == config.MAX_MESSAGE_CHARS
    assert reply.max_image_bytes == config.MAX_IMAGE_BYTES and reply.image_max_side == config.IMAGE_MAX_SIDE
    assert reply.image_types == list(config.ALLOWED_IMAGE_TYPES)


def test_a_rejected_request_is_described_but_never_echoed_back():
    import asyncio
    import json
    from fastapi.exceptions import RequestValidationError
    from src.api.main import validation_error

    big = "A" * 5000
    exc = RequestValidationError([{"type": "value_error", "loc": ("body", "image"), "msg": "Value error, bad image",
                                   "input": big, "ctx": {"error": ValueError("bad image")}}])
    response = asyncio.run(validation_error(None, exc))
    assert response.status_code == 422
    assert json.loads(response.body) == {"detail": [{"loc": ["body", "image"], "msg": "Value error, bad image", "type": "value_error"}]}
    assert big.encode() not in response.body
