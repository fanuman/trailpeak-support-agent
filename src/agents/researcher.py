"""Gathers raw facts for a topic from the product catalog. Deliberately
outputs plain factual notes, not customer-facing prose - keeping this
node's job narrow is what makes the two-agent split actually earn its
keep, rather than being one agent's logic split into two functions."""
from openai import OpenAI
from src.core.config import DEFAULT_MODEL
from src.core.embeddings import get_embedding
from src.core.vectorstore import dense_search
from src.agents.state import ContentState

_client = OpenAI()


def researcher_node(state: ContentState) -> ContentState:
    query_embedding = get_embedding(state["topic"])
    hits = dense_search(query_embedding, k=5)
    chunks = [hit["text"] for hit in hits]
    sources = [hit["source"] for hit in hits]

    prompt = f"""Extract the key facts relevant to this topic from the context below.
Return plain factual bullet points only - no marketing language, no prose, just facts
a writer could later turn into customer-facing copy. If the catalog doesn't fully
support the topic, say so plainly as one of the bullets rather than omitting it.

Topic: {state['topic']}

Context:
{chr(10).join(chunks)}"""

    response = _client.chat.completions.create(
        model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return {**state, "research_notes": response.choices[0].message.content, "sources": list(set(sources))}