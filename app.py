import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from support_rag import BackendError, SupportRAG

MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))

os.makedirs("logs", exist_ok=True)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(_fmt)
    _root.addHandler(_console)
_file = RotatingFileHandler("logs/app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
_file.setFormatter(_fmt)
_root.addHandler(_file)
for noisy in ("httpx", "urllib3", "sentence_transformers", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("support_api")

STATE = {"rag": None, "startup_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        STATE["rag"] = SupportRAG()
        log.info("startup ok | backend=%s | chunks=%d | load_ms=%.0f",
                 STATE["rag"].backend.name, STATE["rag"].n_chunks, (time.perf_counter() - t0) * 1000)
    except Exception as e:
        STATE["startup_error"] = type(e).__name__
        log.exception("startup failed - /ask will return 503 until fixed")
    yield
    log.info("shutdown")


app = FastAPI(
    title="Customer Support RAG API",
    description="Experiment 7 - the Experiment 6 RAG pipeline served through FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


# Schemas
class AskRequest(BaseModel):
    question: str = Field(..., description=f"Customer question (1-{MAX_QUESTION_CHARS} characters)",
                          examples=["How many titles can I borrow at a time with Kindle Unlimited?"])

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters (got {len(v)})")
        return v


class AskResponse(BaseModel):
    request_id: str
    answer: str
    intent: str
    used_fallback: bool
    review_note: str
    retries: int
    top_similarity: float
    sources: List[str]
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str | List[str]
    request_id: str


def error(status: int, code: str, detail, rid: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail, "request_id": rid})


# Middleware: request id + response time for EVERY request
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error | rid=%s", rid)
        response = error(500, "internal_error", "Something went wrong. Please try again later.", rid)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s -> %d | %.1f ms | rid=%s", request.method, request.url.path, response.status_code, ms, rid)
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    msgs = [str(e.get("msg", "invalid input")).removeprefix("Value error, ") for e in exc.errors()]
    rid = getattr(request.state, "request_id", "-")
    log.warning("validation failed | rid=%s | %s", rid, msgs)
    return error(422, "invalid_request", msgs, rid)


# Endpoints
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["ops"])
def health(deep: bool = False):
    """Liveness/readiness. `deep=true` also pings the LLM backend (slower)."""
    rag = STATE["rag"]
    body = {
        "status": "ok" if rag else "degraded",
        "api": "running",
        "knowledge_base_loaded": rag is not None,
        "backend": rag.backend.name if rag else None,
        "chunks": rag.n_chunks if rag else 0,
    }
    if rag is None:
        body["startup_error"] = STATE["startup_error"]
    if deep and rag:
        body["backend_reachable"] = rag.backend.ping()
    return JSONResponse(status_code=200 if rag else 503, content=body)


@app.post("/ask", response_model=AskResponse, tags=["support"],
          responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
                     503: {"model": ErrorResponse}})
def ask(body: AskRequest, request: Request):
    """Answer a customer question using only the FAQ knowledge base.
    (Sync `def`: FastAPI runs it in a thread pool, so a slow LLM call doesn't block the server.)"""
    rid = request.state.request_id
    rag = STATE["rag"]
    if rag is None:
        return error(503, "service_unavailable", "The knowledge base is not loaded. Please try again later.", rid)

    t0 = time.perf_counter()
    try:
        result = rag.answer(body.question)
    except BackendError as e:
        ms = (time.perf_counter() - t0) * 1000
        log.error("ask failed | rid=%s | backend_error=%s | %.1f ms", rid, e, ms)   # detail only in logs
        return error(503, "backend_unavailable",
                     "The answer service is temporarily unavailable. Please try again later.", rid)
    except Exception:
        log.exception("ask failed | rid=%s | unexpected error", rid)
        return error(500, "internal_error", "Something went wrong while answering. Please try again later.", rid)

    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("ask ok | rid=%s | q_len=%d | intent=%s | fallback=%s | retries=%d | top=%.3f | %.1f ms",
             rid, len(body.question), result["intent"], result["used_fallback"],
             result["retries"], result["top_similarity"], ms)
    return AskResponse(request_id=rid, latency_ms=ms, **result)
