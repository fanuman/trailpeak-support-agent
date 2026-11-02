# src/rag/pipeline.py
#
# How to read this file (top to bottom is also the order things happen):
#
#   answer() / answer_stream()   the two public entry points. Each one is short:
#        |
#        +-- _rewrite()          (follow-ups only) restate the question so it stands on its own
#        +-- _prepare()          embed the question, check the cache, retrieve, build the prompt
#        +-- _agent_loop()       let the model call tools until it writes the answer
#        |      +-- _model_turn()   one streamed call to the model
#        |      +-- _run_tool()     run one tool the model asked for
#        +-- _maybe_cache()      remember the answer (only if no live data was used)
#
# The data shapes live in models.py; the wording of the UI trail lives in events.py.
#
# answer_stream() yields (event_name, data) pairs. It knows nothing about HTTP or SSE;
# src/api/streaming.py turns those pairs into the text the browser receives.
import json
import logging
from openai import OpenAI

from src.core.config import DEFAULT_MODEL, RELEVANCE_THRESHOLD, RETRIEVAL_K
from src.core.embeddings import get_embedding
from src.core.vectorstore import get_collection
from src.tools.inventory_tool import TOOL_SCHEMA as AVAILABILITY_SCHEMA, TOOL_FUNCTION as AVAILABILITY_FUNCTION, get_product
from src.tools.calculator_tool import TOOL_SCHEMA as CALC_SCHEMA, TOOL_FUNCTION as CALC_FUNCTION
from src.rag.prompts import build_answer_prompt, build_rewrite_messages, FINAL_ANSWER_INSTRUCTION, SYSTEM_PROMPT
from src.core.semantic_cache import SemanticCache
from src.cost.tracker import cost_tracker
from src.rag.models import FinalAnswer, ToolRequest, Turn, LoopResult, Prepared
from src.rag.events import rewrite_step, search_step, tool_step
from src.rag.sources_footer import SourcesFooter, resolve_sources

from langsmith import traceable

logger = logging.getLogger(__name__)

REWRITE_MAX_TOKENS = 120   # a rewritten question is one short line
REWRITE_MAX_CHARS = 300

CAP_MESSAGE = "I wasn't able to fully work through this request - it may need to be broken into simpler questions."


# ---------- small helpers --------------------------------------------------------

def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def _full_context(chunks, tool_outputs):
    """The text the answer was based on (also used by the evaluation metrics)."""
    return "\n\n".join(list(chunks) + list(tool_outputs))


def _drain(events):
    """Run a generator to the end, ignoring what it yields, and return what it returns.
    answer() uses this: it wants the loop's result, not the live events."""
    try:
        while True:
            next(events)
    except StopIteration as stop:
        return stop.value


class RAGPipeline:
    def __init__(self, model=DEFAULT_MODEL, relevance_threshold=RELEVANCE_THRESHOLD, k=RETRIEVAL_K, max_iterations=6, final_answer_instruction=None, use_cache=True):
        self.client = OpenAI()
        self.model = model
        self.relevance_threshold = relevance_threshold
        self.k = k
        self.max_iterations = max_iterations
        self.tools = [AVAILABILITY_SCHEMA, CALC_SCHEMA]
        self.available_functions = {**AVAILABILITY_FUNCTION, **CALC_FUNCTION}
        self.cache = SemanticCache()
        self.use_cache = use_cache
        self.final_answer_instruction = final_answer_instruction or FINAL_ANSWER_INSTRUCTION
        self.FALLBACK_MESSAGE =  (
            "Hi! I'm the TrailPeak Outdoors assistant. I can help with product details, "
            "current price and stock, shipping/returns/warranty questions, or comparing "
            "items and calculating totals. What are you looking for?"
        )

    # ---------- the two public entry points ---------------------------------------

    @traceable(name="rag_answer")
    def answer(self, query, k=None, history=None):
        """One finished answer as a dict (used by /ask and the evaluation scripts)."""
        prep = self._prepare(query, k, history=history, search_text=self._rewrite(query, history))

        if prep.cached:
            return {"answer": prep.cached["response"], "sources": prep.cached["sources"], "used_fallback": False,
                    "full_context": prep.cached["full_context"], "from_cache": True}

        if prep.out_of_scope:
            return {"answer": self.FALLBACK_MESSAGE, "sources": [], "used_fallback": True,
                    "full_context": _full_context(prep.chunks, []), "from_cache": False}

        # The loop's own last turn is thrown away here: the structured call below
        # writes the answer again, in a shape that also names the sources it used.
        result = _drain(self._agent_loop(prep.messages, final_label="agent_iteration"))
        full_context = _full_context(prep.chunks, result.tool_outputs)

        if result.hit_cap:
            # Report incompleteness honestly rather than guess at an answer built from partial steps
            return {"answer": CAP_MESSAGE, "sources": prep.unique_sources, "used_fallback": True,
                    "full_context": full_context, "from_cache": False}

        prep.messages.append({"role": "user", "content": self.final_answer_instruction})
        final = self.client.chat.completions.parse(model=self.model, messages=prep.messages, response_format=FinalAnswer)
        cost_tracker.record(
            model=self.model,
            input_tokens=final.usage.prompt_tokens,
            output_tokens=final.usage.completion_tokens,
            label="final_answer",
        )
        parsed = final.choices[0].message.parsed

        self._maybe_cache(query, parsed.answer, prep.embedding, result.tool_outputs, full_context, parsed.sources_used, follow_up=bool(history))
        return {"answer": parsed.answer, "sources": parsed.sources_used,
                "used_fallback": False, "full_context": full_context, "from_cache": False}

    def answer_stream(self, query, k=None, history=None):
        """The same question, answered live. Yields (event_name, data) pairs:
          step    - one line of the trail the UI shows: {id, label, detail, status}
          product - a live inventory result, drawn as a card: {sku, name, price, stock, sizes}
          token   - a piece of the answer text: {t}
          done    - the last event: {sources, cached, fallback}
        If something raises, the caller (src/api/streaming.py) turns it into an `error` event."""
        search_text = query
        if history:   # a follow-up: say so in the trail, then restate the question so it can be searched
            yield rewrite_step(status="running")
            search_text = self._rewrite(query, history)
            yield rewrite_step(search_text if search_text != query else "")

        yield search_step("Searching store documents", status="running")
        prep = self._prepare(query, k, footer=True, history=history, search_text=search_text)

        if prep.cached:
            yield search_step("Found a saved answer")
            yield "token", {"t": prep.cached["response"]}
            yield "done", {"sources": prep.cached["sources"], "cached": True, "fallback": False}
            return

        if prep.out_of_scope:
            yield search_step("Searched store documents", "no close match")
            yield "token", {"t": self.FALLBACK_MESSAGE}
            yield "done", {"sources": [], "cached": False, "fallback": True}
            return

        yield search_step("Searched store documents", ", ".join(prep.unique_sources))
        result = yield from self._agent_loop(prep.messages, footer=True)   # streams tokens, steps and product cards

        if result.hit_cap:
            yield "token", {"t": CAP_MESSAGE}
            yield "done", {"sources": prep.unique_sources, "cached": False, "fallback": True}
            return

        sources = resolve_sources(result.footer, prep.unique_sources)
        self._maybe_cache(query, result.text, prep.embedding, result.tool_outputs, _full_context(prep.chunks, result.tool_outputs), sources, follow_up=bool(history))
        yield "done", {"sources": sources, "cached": False, "fallback": False}

    # ---------- steps both entry points share -------------------------------------

    def _rewrite(self, query, history):
        """For a follow-up ("what about the blue one?"), one small model call restates the question so
        it makes sense on its own. That text is what gets embedded, cached and searched.
        No history -> the question unchanged, no call. Any failure -> the question unchanged:
        a problem here must never stop an answer."""
        if not history:
            return query
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=build_rewrite_messages(query, history),
                temperature=0, max_tokens=REWRITE_MAX_TOKENS)
            cost_tracker.record(
                model=self.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                label="rewrite_question",
            )
            lines = (response.choices[0].message.content or "").strip().splitlines()
            rewritten = lines[0].strip().strip('"') if lines else ""
        except Exception:
            logger.exception("Rewriting the follow-up failed; using the question as it was written")
            return query
        return rewritten[:REWRITE_MAX_CHARS] or query

    def _prepare(self, query, k=None, footer=False, history=None, search_text=None):
        """Embed the question, look in the cache, retrieve, and build the prompt.
        footer=True also asks the model to end with a SOURCES: line (streamed answers only).
        `search_text` is what gets embedded and searched (a follow-up's rewrite); the model is still
        shown `query` itself, after the earlier messages in `history`."""
        embedding = get_embedding(search_text or query)

        cached = self.cache.check(query_embedding=embedding) if self.use_cache else None
        if cached:
            return Prepared(embedding=embedding, cached={"response": _text(cached["response"]),
                                                         "full_context": _text(cached["full_context"]),
                                                         "sources": list(cached.get("sources") or [])})

        chunks, sources, distances = self.retrieve(embedding, k=k)
        if self._is_out_of_scope(distances):
            return Prepared(embedding=embedding, out_of_scope=True, chunks=chunks, sources=sources)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history or []]
        messages.append({"role": "user", "content": build_answer_prompt(query, chunks, sources, sources_footer=footer)})
        return Prepared(embedding=embedding, chunks=chunks, sources=sources, messages=messages)

    def _maybe_cache(self, query, text, embedding, tool_outputs, full_context, sources, follow_up=False):
        """Save the answer for next time - but never if live data (stock, prices) went into it, and never
        for a follow-up: an answer written with the conversation in view may lean on it ("yes, that one too")."""
        if not tool_outputs and not follow_up and self.use_cache and text.strip():
            self.cache.store(query, text, embedding, full_context, sources)

    @traceable(name="retrieve")
    def retrieve(self, query_embedding, k=None):
        k = k or self.k
        collection = get_collection()  # fetched fresh, every single call
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["documents"][0], [m["source"] for m in results["metadatas"][0]], results["distances"][0]

    def _is_out_of_scope(self, distances):
        return all(d >= self.relevance_threshold for d in distances)

    # ---------- the ReAct loop (the one and only) ---------------------------------

    def _agent_loop(self, messages, final_label="stream_answer", footer=False):
        """Let the model act and observe until it stops asking for tools, or give up after
        max_iterations (a guard against a runaway loop). `messages` grows as it goes.

        Every model call is streamed, so the turn in which the model stops asking for tools
        IS the final answer - its tokens have already been yielded. Nothing is generated twice.

        Yields the live events (token / step / product) and returns a LoopResult.
        `final_label` is the name the last turn's cost is logged under.
        footer=True hides the SOURCES: line the model was asked to write (see _model_turn)."""
        tool_outputs = []
        for _ in range(self.max_iterations):
            turn = yield from self._model_turn(messages, footer)
            if turn.usage:
                cost_tracker.record(
                    model=self.model,
                    input_tokens=turn.usage.prompt_tokens,
                    output_tokens=turn.usage.completion_tokens,
                    label="agent_iteration" if turn.requests else final_label,
                )

            if not turn.requests:
                return LoopResult(text=turn.text, tool_outputs=tool_outputs, hit_cap=False, footer=turn.footer)

            # remember what the model asked for, then answer each request with a "tool" message
            messages.append({
                "role": "assistant",
                "content": turn.text or None,
                "tool_calls": [{"id": r.id, "type": "function", "function": {"name": r.name, "arguments": r.arguments}}
                               for r in turn.requests],
            })
            for request in turn.requests:
                output = yield from self._run_tool(request, messages)
                tool_outputs.append(output)

        return LoopResult(text="", tool_outputs=tool_outputs, hit_cap=True)

    def _model_turn(self, messages, footer=False):
        """One streamed call to the model. Yields a token event for each piece of text.
        Returns a Turn (text, tool requests, usage, footer).

        With footer=True the model was asked to end with a SOURCES: line. SourcesFooter keeps
        that line out of the tokens (and out of `text`) and hands it back as `footer`. A new
        SourcesFooter is made for every turn, so text a model writes before calling a tool
        can never swallow the final answer.

        A tool request does not arrive whole: an id and name come first, then the JSON
        arguments a few characters at a time. Each piece carries an `index` saying which
        request it belongs to, so the pieces are gathered per index."""
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=self.tools,
            stream=True,
            stream_options={"include_usage": True},  # without this, no usage arrives and no cost is logged
        )
        splitter = SourcesFooter() if footer else None
        text, by_index, usage = "", {}, None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                shown = splitter.feed(delta.content) if splitter else delta.content
                if shown:
                    text += shown
                    yield "token", {"t": shown}
            for piece in delta.tool_calls or []:
                request = by_index.setdefault(piece.index, ToolRequest())
                if piece.id:
                    request.id = piece.id
                if piece.function and piece.function.name:
                    request.name = piece.function.name
                if piece.function and piece.function.arguments:
                    request.arguments += piece.function.arguments

        footer_text = None
        if splitter:
            rest, footer_text = splitter.finish()   # text that was held back but turned out to be plain text
            if rest:
                text += rest
                yield "token", {"t": rest}

        return Turn(text=text, requests=[by_index[i] for i in sorted(by_index)], usage=usage, footer=footer_text)

    def _run_tool(self, request, messages):
        """Run one tool the model asked for. Adds its result to `messages` (so the model sees
        it next turn), yields the trail steps and any product card, and returns the result."""
        args = json.loads(request.arguments)
        yield tool_step(request, args, "running")

        result = self.available_functions[request.name](**args)
        messages.append({"role": "tool", "tool_call_id": request.id, "content": result})

        yield tool_step(request, args, "done", result)
        if request.name == "CheckAvailability":
            product = get_product(args.get("sku", ""))
            if product:
                yield "product", product
        return result
