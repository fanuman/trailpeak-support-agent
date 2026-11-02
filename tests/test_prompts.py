"""Tests for src/rag/prompts.py: the prompt text the model actually receives."""
from src.rag.prompts import build_answer_prompt, build_rewrite_messages, REWRITE_CLIP, SOURCES_FOOTER_INSTRUCTION


def test_the_default_prompt_is_unchanged_and_has_no_footer_request():
    prompt = build_answer_prompt("hi", ["a chunk"], ["returns_policy.txt"])
    assert prompt.endswith("Question: hi\nAnswer:")   # /ask and the evaluation use exactly this
    assert "SOURCES:" not in prompt


def test_the_footer_request_sits_between_the_question_and_the_answer_prompt():
    prompt = build_answer_prompt("hi", ["a chunk"], ["returns_policy.txt"], sources_footer=True)
    assert prompt.endswith("Question: hi\n\n" + SOURCES_FOOTER_INSTRUCTION + "\n\nAnswer:")


def test_context_chunks_keep_their_source_labels():
    prompt = build_answer_prompt("hi", ["a chunk"], ["returns_policy.txt"], sources_footer=True)
    assert "[Source: returns_policy.txt]\na chunk" in prompt


def test_the_prompt_does_not_ask_for_inline_citations():
    assert "Cite which source" not in build_answer_prompt("hi", ["c"], ["a.txt"])


# ---------- the follow-up rewrite prompt ------------------------------------------

HISTORY = [
    {"role": "user", "content": "Tell me about the SummitCarry"},
    {"role": "assistant", "content": "It is a 65L pack."},
]


def test_the_rewrite_prompt_holds_the_transcript_and_the_last_message():
    system, user = build_rewrite_messages("what about the blue one?", HISTORY)
    assert system["role"] == "system" and user["role"] == "user"
    assert "Customer: Tell me about the SummitCarry\nAssistant: It is a 65L pack." in user["content"]
    assert "Customer's last message: what about the blue one?" in user["content"]


def test_the_rewrite_prompt_forbids_answering_and_inventing():
    instruction = build_rewrite_messages("hi", HISTORY)[0]["content"]
    assert "Do not answer" in instruction and "unchanged" in instruction


def test_long_earlier_messages_are_clipped_in_the_rewrite_prompt():
    long_history = [{"role": "assistant", "content": "x" * (REWRITE_CLIP * 3)}]
    user = build_rewrite_messages("and that?", long_history)[1]["content"]
    assert "x" * REWRITE_CLIP in user and "x" * (REWRITE_CLIP + 1) not in user


def test_the_rewrite_prompt_tells_the_model_to_restate_the_whole_request():
    instruction = build_rewrite_messages("and a stove too?", HISTORY)[0]["content"]
    assert "WHOLE request" in instruction and "full product names" in instruction


def test_the_answer_prompt_forbids_guessing_a_sku():
    prompt = build_answer_prompt("hi", ["a chunk"], ["product_catalog.txt"])
    assert "Never invent a SKU" in prompt and "never use another product's SKU" in prompt
    assert "pass the product's name" in prompt                  # the tool resolves names, so the model never has to guess
    assert "check every one of them" in prompt and "instead of leaving it out" in prompt
    assert prompt.endswith("Question: hi\nAnswer:")      # the ending /ask and the evaluation rely on is unchanged
