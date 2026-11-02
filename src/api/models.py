from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.core.config import ALLOWED_IMAGE_TYPES, MAX_HISTORY_MESSAGES, MAX_IMAGE_BYTES, MAX_MESSAGE_CHARS
from src.core.images import read_image_data_url

# base64 text is 4/3 the size of the bytes it holds, plus the short "data:image/jpeg;base64," header
MAX_IMAGE_CHARS = MAX_IMAGE_BYTES * 4 // 3 + 64

class CostSummary(BaseModel):
    """What GET /cost returns: totals from logs/cost_log.jsonl (every model call the app has made)."""
    total_calls: int
    total_cost_usd: float
    cost_if_gpt4o_usd: float
    by_stage: dict[str, float]

class ClientConfig(BaseModel):
    """What GET /config returns: the limits the web page has to match. They live in src/core/config.py."""
    max_history_messages: int
    max_message_chars: int
    max_image_bytes: int
    image_max_side: int
    image_types: list[str]

class HistoryMessage(BaseModel):
    """One earlier message of the conversation. Only these two roles are accepted: a client
    can never send a `system` or `tool` message."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content cannot be blank or whitespace-only")
        return v

class RagRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)   # oldest first
    image: Optional[str] = Field(default=None, max_length=MAX_IMAGE_CHARS)   # one photo, as a base64 data URL

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be blank or whitespace-only")
        return v

    @field_validator("image")
    @classmethod
    def is_a_real_image(cls, v):
        if v is not None:
            read_image_data_url(v, MAX_IMAGE_BYTES, ALLOWED_IMAGE_TYPES)   # raises ValueError -> a 422 for the client
        return v

class RagResponse(BaseModel):
    reply: str
    sources: list[str]
    used_fallback: bool
    from_cache: bool = False
    image_note: str = ""   # what the photo showed, in words ("" if no photo was sent)