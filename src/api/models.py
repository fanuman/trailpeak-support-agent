from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_HISTORY_MESSAGES = 6   # 3 question-and-answer exchanges

class CostSummary(BaseModel):
    """What GET /cost returns: totals from logs/cost_log.jsonl (every model call the app has made)."""
    total_calls: int
    total_cost_usd: float
    cost_if_gpt4o_usd: float
    by_stage: dict[str, float]

class HistoryMessage(BaseModel):
    """One earlier message of the conversation. Only these two roles are accepted: a client
    can never send a `system` or `tool` message."""
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content cannot be blank or whitespace-only")
        return v

class RagRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)   # oldest first

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be blank or whitespace-only")
        return v

class RagResponse(BaseModel):
    reply: str
    sources: list[str]
    used_fallback: bool
    from_cache: bool = False