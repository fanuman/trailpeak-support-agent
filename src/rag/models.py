# src/rag/models.py
# The shapes of the data that move around inside the pipeline. No logic lives here.
from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import List


class FinalAnswer(BaseModel):
    answer: str
    sources_used: List[str]


@dataclass
class ToolRequest:
    """One tool call the model asked for. It arrives in pieces while streaming."""
    id: str = ""
    name: str = ""
    arguments: str = ""   # a JSON string, complete only once the model's turn is over


@dataclass
class Turn:
    """One streamed model call, once it has finished."""
    text: str             # what the model wrote, without the SOURCES: footer
    requests: list        # the tools it asked for (ToolRequest objects)
    usage: object         # token counts, or None if the API sent none
    footer: str = None    # the SOURCES: footer it wrote, if footers were asked for


@dataclass
class LoopResult:
    """What _agent_loop hands back when it is finished."""
    text: str             # what the model wrote in its last turn ("" if it hit the cap)
    tool_outputs: list    # the result of every tool call, in order
    hit_cap: bool         # True if it ran out of iterations without writing an answer
    footer: str = None    # the SOURCES: footer of the final turn, if one was asked for and written


@dataclass
class Prepared:
    """Everything decided before the model is called."""
    cached: dict = None           # {"response", "full_context", "sources"} if the cache already has this question
    out_of_scope: bool = False    # True if nothing retrieved is close enough to the question
    chunks: list = field(default_factory=list)
    sources: list = field(default_factory=list)    # one per chunk, so it can repeat
    messages: list = field(default_factory=list)   # the conversation sent to the model
    embedding: list = None

    @property
    def unique_sources(self):
        return list(dict.fromkeys(self.sources))   # no repeats, original order


@dataclass
class SearchResult:
    """What a search finds. The refusal gate uses `closest`, not the hits: in hybrid mode the chunk
    closest by meaning is not always among the hits, and the gate's threshold was measured on it."""
    hits: list        # the chunks the model will read, best first: {"id", "text", "source", "position", "distance"}
    closest: float    # distance of the chunk closest by meaning (the refusal gate checks this)
