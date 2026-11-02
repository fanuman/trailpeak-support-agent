from contextlib import asynccontextmanager


from dotenv import load_dotenv
load_dotenv()
from src.core.secrets import load_secret_into_env
load_secret_into_env()

from src.rag.pipeline import RAGPipeline
from src.api.streaming import sse_stream
from src.api.models import ClientConfig, CostSummary, RagRequest, RagResponse
from src.core.config import (ALLOWED_IMAGE_TYPES, IMAGE_MAX_SIDE, MAX_HISTORY_MESSAGES, MAX_IMAGE_BYTES,
                             MAX_MESSAGE_CHARS, RATE_LIMIT)
from src.cost.report import summarize

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request

import os
import logging
from pathlib import Path

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
logger = logging.getLogger(__name__)

rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    rag_pipeline = RAGPipeline()
    yield

app = FastAPI(lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address, storage_uri=f"redis://{REDIS_HOST}:6379/0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(SlowAPIMiddleware)
# The UI is served by this app (same origin), so no CORS is needed by default.
# Only if you host the UI somewhere else: CORS_ORIGINS="https://my-ui.example.com,http://localhost:5173"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    """A rejected request gets a 422 that names what is wrong, but never echoes the request back
    (FastAPI's default does, and a rejected photo would be sent back to the client in full)."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config", response_model=ClientConfig)
def client_config():
    """The limits the web page has to match, so they are defined in one place (src/core/config.py)."""
    return ClientConfig(max_history_messages=MAX_HISTORY_MESSAGES, max_message_chars=MAX_MESSAGE_CHARS,
                        max_image_bytes=MAX_IMAGE_BYTES, image_max_side=IMAGE_MAX_SIDE,
                        image_types=list(ALLOWED_IMAGE_TYPES))


@app.get("/cost", response_model=CostSummary)
def cost():
    """Totals for every model call the app has made, read from logs/cost_log.jsonl
    (the same numbers as `python -m src.cost.report`)."""
    return CostSummary(**summarize())


def _history(rag_request):
    """The earlier messages as plain dicts (oldest first); empty for a first question."""
    return [m.model_dump() for m in rag_request.history]


@app.post("/ask", response_model=RagResponse)
@limiter.limit(RATE_LIMIT)
def ask(request: Request, rag_request: RagRequest):
    try:
        result = rag_pipeline.answer(rag_request.message, history=_history(rag_request), image=rag_request.image)
        return RagResponse(
            reply=result["answer"],
            sources=result["sources"],
            used_fallback=result["used_fallback"],
            from_cache=result["from_cache"],
            image_note=result["image_note"],
        )
    except Exception:
        logger.exception("/ask failed")
        raise HTTPException(status_code=502, detail="Upstream error. Please try again.")


@app.post("/ask/stream")
@limiter.limit(RATE_LIMIT)
def ask_stream(request: Request, rag_request: RagRequest):
    return StreamingResponse(
        sse_stream(rag_pipeline.answer_stream(rag_request.message, history=_history(rag_request), image=rag_request.image)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # stop proxies from buffering the stream
    )


# Serve the web UI from this same app. Must stay LAST: a mount at "/" matches every path,
# so every API route above has to be registered before it.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"   # /app/frontend inside Docker
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
