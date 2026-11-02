# src/core/config.py
#
# Every setting of the app, in one place. Other files import from here; none defines its own copy.
#
# Each setting has a default below. An environment variable with the same name overrides it
# (put it in .env, in docker-compose.yml, or in the ECS task definition), so a limit can be
# changed without editing code. The value is read once, when the app starts. A value that is not
# valid (MAX_HISTORY_MESSAGES=abc, or RETRIEVAL_K=0) stops the app at startup with a clear message.
import os


def _setting(name, default, cast=int, minimum=None):
    """The value of environment variable `name` if it is set, otherwise `default`."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        raise ValueError(f"Setting {name}={raw!r} is not a valid {cast.__name__}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"Setting {name}={raw!r} must be at least {minimum}")
    return value


# ---------- models (fixed on purpose: not overridable) -----------------------------
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"   # changing it makes the stored vectors incompatible until you re-run ingest

# ---------- retrieval and the agent loop -------------------------------------------
RELEVANCE_THRESHOLD = _setting("RELEVANCE_THRESHOLD", 0.68, float, minimum=0)   # a chunk farther than this is "not close"
RETRIEVAL_K = _setting("RETRIEVAL_K", 5, minimum=1)                              # how many chunks a search returns
MAX_AGENT_ITERATIONS = _setting("MAX_AGENT_ITERATIONS", 6, minimum=1)           # tool rounds before giving up

# ---------- what a request may contain (the API checks it, the web page reads it) ----
MAX_MESSAGE_CHARS = _setting("MAX_MESSAGE_CHARS", 2000, minimum=1)       # one question, or one earlier message
MAX_HISTORY_MESSAGES = _setting("MAX_HISTORY_MESSAGES", 6, minimum=0)    # 6 = 3 exchanges; 0 turns memory off
RATE_LIMIT = _setting("RATE_LIMIT", "10/minute", str)                    # per client address, on /ask, /ask/stream

# ---------- images (one per message) ------------------------------------------------
MAX_IMAGE_BYTES = _setting("MAX_IMAGE_BYTES", 2_000_000, minimum=1)           # decoded size; the browser shrinks photos far below this
IMAGE_MAX_SIDE = _setting("IMAGE_MAX_SIDE", 1024, minimum=64)                 # longest side in pixels; the web page resizes to this before sending
IMAGE_DESCRIBE_MAX_TOKENS = _setting("IMAGE_DESCRIBE_MAX_TOKENS", 200, minimum=1)   # a description is a few short lines
IMAGE_NOTE_MAX_CHARS = _setting("IMAGE_NOTE_MAX_CHARS", 500, minimum=1)       # longest description kept for search and for the history
ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")               # fixed on purpose: not overridable
IMAGE_DETAIL = "low"                                                          # "low" = cheap; "high" reads small print better

# ---------- rewriting a follow-up into a standalone question -------------------------
REWRITE_MAX_TOKENS = _setting("REWRITE_MAX_TOKENS", 120, minimum=1)      # a rewritten question is one short line
REWRITE_MAX_CHARS = _setting("REWRITE_MAX_CHARS", 300, minimum=1)        # longest rewrite we will search with
REWRITE_CLIP_CHARS = _setting("REWRITE_CLIP_CHARS", 600, minimum=1)      # kept of each earlier message in the rewrite call

# ---------- the semantic cache ------------------------------------------------------
CACHE_DISTANCE_THRESHOLD = _setting("CACHE_DISTANCE_THRESHOLD", 0.05, float, minimum=0)   # how close counts as the same question
CACHE_TTL_SECONDS = _setting("CACHE_TTL_SECONDS", 3600, minimum=1)                          # how long an answer is kept
